from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from algorithms.emergency_response import emergency_response
from algorithms.mst import kruskal_mst
from algorithms.shortest_path import (
    alternate_route,
    astar,
    dijkstra,
    least_congested_route,
    time_varying_shortest_path,
)
from models.edge import TimePeriod
from models.graph import Graph
from simulation.traffic_data import TrafficDataStore


ROOT = Path(__file__).parent
STATIC = ROOT / "static"

graph = Graph()
store = TrafficDataStore()


def period_from(value: str | None) -> TimePeriod:
    return TimePeriod.from_string(value or "morning")


def period_index(period: TimePeriod) -> int:
    return {
        TimePeriod.MORNING_PEAK: 0,
        TimePeriod.AFTERNOON: 1,
        TimePeriod.EVENING_PEAK: 2,
        TimePeriod.NIGHT: 3,
    }[period]


def node_payload(node):
    return {
        "id": node.node_id,
        "name": node.name,
        "type": node.node_type.name.replace("_", " ").title(),
        "longitude": node.longitude,
        "latitude": node.latitude,
        "population": node.population,
        "critical": node.is_critical,
    }


def edge_payload(edge, period: TimePeriod):
    ratio = store.get_congestion_ratio(edge.key, edge.capacity, period)
    return {
        "key": edge.key,
        "from": edge.from_id,
        "to": edge.to_id,
        "distance": edge.distance,
        "capacity": edge.capacity,
        "condition": edge.condition,
        "potential": edge.is_potential,
        "cost": edge.cost_millions,
        "maintenance": edge.maint_cost,
        "priority": edge.maint_priority,
        "flow": store.get_flow(edge.key, period),
        "ratio": ratio,
        "level": store.get_congestion_level(edge.key, edge.capacity, period).value,
        "time": store.get_weight(edge.key, edge.capacity, edge.distance, period),
    }


def path_edges(path):
    keys = []
    for a, b in zip(path, path[1:]):
        edge = graph.get_edge(a, b)
        if edge:
            keys.append(edge.key)
    return keys


def route_payload(result):
    return {
        "algorithm": result.algorithm,
        "valid": result.is_valid,
        "path": result.path,
        "pathNames": [graph.nodes[n].name for n in result.path if n in graph.nodes],
        "edges": path_edges(result.path),
        "totalTime": result.total_time_minutes,
        "period": result.time_period.value,
        "nodesVisited": result.nodes_visited,
        "edgesConsidered": result.edges_considered,
    }


def transit_allocation(period: TimePeriod):
    idx = period_index(period)
    routes = store.get_all_routes()
    buses_available, trains_available = store.get_fleet(period)

    bus_routes = [r for r in routes if r.route_type == "bus"]
    total_current_buses = sum(r.buses_assigned for r in bus_routes)
    scale = buses_available / total_current_buses if total_current_buses else 0

    rows = []
    total_capacity = 0
    total_demand = 0
    for route in routes:
        vehicles_per_hour = route.supply_by_period[idx]
        if route.route_type == "bus":
            recommended = max(1, round(route.buses_assigned * scale))
            vehicles_per_hour = max(vehicles_per_hour, recommended)
        else:
            recommended = vehicles_per_hour

        hourly_capacity = vehicles_per_hour * route.vehicle_capacity
        estimated_hourly_demand = route.daily_passengers / 16
        coverage = min(100, hourly_capacity / estimated_hourly_demand * 100)
        total_capacity += hourly_capacity
        total_demand += estimated_hourly_demand
        rows.append({
            "id": route.route_id,
            "type": route.route_type,
            "stops": route.stops,
            "stopNames": [graph.nodes[s].name for s in route.stops if s in graph.nodes],
            "dailyPassengers": route.daily_passengers,
            "vehiclesPerHour": vehicles_per_hour,
            "recommendedVehicles": recommended,
            "vehicleCapacity": route.vehicle_capacity,
            "hourlyCapacity": hourly_capacity,
            "coverage": coverage,
        })

    rows.sort(key=lambda r: (r["type"] != "metro", -r["dailyPassengers"]))
    return {
        "fleet": {"buses": buses_available, "trains": trains_available},
        "totalHourlyCapacity": total_capacity,
        "estimatedHourlyDemand": total_demand,
        "coverage": min(100, total_capacity / total_demand * 100) if total_demand else 0,
        "routes": rows,
    }


def api_network(params):
    period = period_from(params.get("period", ["morning"])[0])
    existing = graph.get_existing_edges()
    report = store.congestion_report(existing, period)
    hotspots = []
    for row in report[:8]:
        clean = dict(row)
        clean["level"] = row["level"].value
        hotspots.append(clean)
    return {
        "summary": {
            "nodes": len(graph.nodes),
            "existingEdges": len(existing),
            "potentialEdges": len(graph.get_potential_edges()),
            "criticalNodes": len(graph.get_critical_nodes()),
            "signals": len(store.get_all_signals()),
            "transitRoutes": len(store.get_all_routes()),
        },
        "nodes": [node_payload(n) for n in graph.nodes.values()],
        "edges": [edge_payload(e, period) for e in graph.get_all_edges()],
        "hotspots": hotspots,
        "period": period.value,
    }


def api_route(params):
    start = params.get("start", ["4"])[0]
    end = params.get("end", ["3"])[0]
    period = period_from(params.get("period", ["morning"])[0])
    algorithm = params.get("algorithm", ["dijkstra"])[0]
    blocked = [b.strip() for b in params.get("blocked", [""])[0].split(",") if b.strip()]

    if algorithm == "astar":
        result = astar(graph, store, start, end, period)
    elif algorithm == "least":
        result = least_congested_route(graph, store, start, end, period)
    elif algorithm == "alternate":
        result = alternate_route(graph, store, start, end, blocked, period)
    elif algorithm == "time":
        hour = int(params.get("hour", ["8"])[0])
        result = time_varying_shortest_path(graph, store, start, end, hour)
    else:
        result = dijkstra(graph, store, start, end, period)

    return route_payload(result)


def api_emergency(params):
    origin = params.get("origin", ["7"])[0]
    period = period_from(params.get("period", ["morning"])[0])
    result = emergency_response(graph, store, origin, period)
    return {
        "origin": origin,
        "hospitalId": result.hospital_id,
        "hospitalName": result.hospital_name,
        "route": route_payload(result.astar_result),
        "travel": result.astar_travel_min,
        "withoutPreemption": result.total_time_without_min,
        "withPreemption": result.total_time_with_min,
        "saved": result.time_saved_min,
        "improvement": result.improvement_pct,
        "signals": [
            {
                "id": e.intersection_id,
                "name": e.intersection_name,
                "edge": e.edge_key,
                "normalWait": e.baseline_wait_sec,
                "preemptedWait": e.preempted_wait_sec,
                "saved": e.time_saved_sec,
            }
            for e in result.preemption_events
        ],
    }


def api_mst(params):
    use_cost = params.get("mode", ["distance"])[0] == "cost"
    include_potential = params.get("potential", ["false"])[0] == "true" or use_cost
    result = kruskal_mst(graph, store, use_cost=use_cost, include_potential=include_potential)
    return {
        "mode": "cost" if use_cost else "distance",
        "edges": [e.key for e in result.edges],
        "forced": [e.key for e in result.critical_forced],
        "totalDistance": result.total_distance_km,
        "totalCost": result.total_cost_megp,
        "maintenance": sum(e.maint_cost for e in result.existing_roads_selected),
        "newRoads": [edge_payload(e, TimePeriod.AFTERNOON) for e in result.new_roads_selected],
        "disconnected": result.disconnected,
    }


def api_transit(params):
    return transit_allocation(period_from(params.get("period", ["morning"])[0]))


ROUTES = {
    "/api/network": api_network,
    "/api/route": api_route,
    "/api/emergency": api_emergency,
    "/api/mst": api_mst,
    "/api/transit": api_transit,
}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ROUTES:
            try:
                payload = ROUTES[parsed.path](parse_qs(parsed.query))
                self.send_json(payload)
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=500)
            return

        target = "index.html" if parsed.path == "/" else parsed.path.lstrip("/")
        path = (STATIC / target).resolve()
        if not str(path).startswith(str(STATIC.resolve())) or not path.exists():
            self.send_error(404)
            return

        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
        }.get(path.suffix, "application/octet-stream")
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


if __name__ == "__main__":
  server = ThreadingHTTPServer(("0.0.0.0", 8000), Handler)
print("Cairo transportation UI running at http://0.0.0.0:8000")
server.serve_forever()
