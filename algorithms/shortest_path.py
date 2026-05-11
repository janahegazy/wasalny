"""
algorithms/shortest_path.py
Cairo Transportation Network — Task 3: Traffic Flow Optimization

Implements shortest path algorithms for Cairo's transportation network:
  1. Dijkstra's Algorithm — standard route planning
  2. A* Search Algorithm — emergency vehicle routing
  3. Time-Varying Dijkstra — accounts for peak/off-peak traffic
  4. Alternate Route — handles road closures
  5. Least Congested Route — minimizes congestion rather than time

Complexity:
  - Dijkstra (min-heap): O((V + E) log V)
  - A* (min-heap):       O((V + E) log V) — faster in practice
  - Space:               O(V)
"""

from __future__ import annotations
import heapq
from typing import Dict, List, Optional, Set, Tuple, Callable

from models.graph import Graph
from models.edge import Edge, TimePeriod
from simulation.traffic_data import TrafficDataStore


# ===========================================================================
# Result Container
# ===========================================================================

class ShortestPathResult:
    """
    Holds the output of a shortest path computation.
    """
    def __init__(
        self,
        algorithm: str,
        path: List[str],
        total_time_minutes: float,
        time_period: TimePeriod,
        nodes_visited: int = 0,
        edges_considered: int = 0,
    ):
        self.algorithm = algorithm
        self.path = path
        self.total_time_minutes = total_time_minutes
        self.time_period = time_period
        self.nodes_visited = nodes_visited
        self.edges_considered = edges_considered

    @property
    def is_valid(self) -> bool:
        return len(self.path) > 1 and self.total_time_minutes < float('inf')

    def summary(self) -> str:
        if not self.is_valid:
            return f"  {self.algorithm}: NO PATH FOUND"

        path_str = " → ".join(self.path[:5])
        if len(self.path) > 5:
            path_str += f" … → {self.path[-1]}"

        return (f"  {self.algorithm:<12}  {self.total_time_minutes:>8.1f} min  "
                f"{self.time_period.name:<14}  {self.nodes_visited:>6} nodes  "
                f"{path_str}")

    def detailed_report(self) -> str:
        lines = [
            "=" * 68,
            f"  Shortest Path Result — {self.algorithm}",
            "=" * 68,
            f"  Time period     : {self.time_period.label()}",
            f"  Total time      : {self.total_time_minutes:.1f} minutes",
            f"  Path length     : {len(self.path)} nodes",
            f"  Nodes explored  : {self.nodes_visited}",
            f"  Path            : {' → '.join(self.path)}",
            "=" * 68,
        ]
        return "\n".join(lines)


# ===========================================================================
# Path Reconstruction Helper
# ===========================================================================

def _reconstruct_path(prev: Dict[str, Optional[str]], start: str, end: str) -> List[str]:
    """
    Walk the predecessor map backwards from 'end' to 'start'
    to reconstruct the shortest path.
    """
    if start not in prev or end not in prev:
        return []

    path = []
    node = end
    while node is not None:
        path.append(node)
        node = prev.get(node)

    path.reverse()

    if not path or path[0] != start:
        return []
    return path


# ===========================================================================
# 1. DIJKSTRA'S ALGORITHM
# ===========================================================================

def dijkstra(
    graph: Graph,
    store: TrafficDataStore,
    start_id: str,
    end_id: str,
    period: TimePeriod = TimePeriod.AFTERNOON,
) -> ShortestPathResult:
    """
    Find the shortest travel-time path using Dijkstra's algorithm.

    Time complexity:  O((V + E) log V)
    Space complexity: O(V)
    """
    # Validate nodes exist
    if start_id not in graph.nodes:
        return ShortestPathResult("Dijkstra", [start_id], float('inf'), period)
    if end_id not in graph.nodes:
        return ShortestPathResult("Dijkstra", [], float('inf'), period)

    # dist[node] = best known travel time from start to node
    dist: Dict[str, float] = {nid: float('inf') for nid in graph.nodes}
    dist[start_id] = 0.0

    # predecessor map for path reconstruction
    prev: Dict[str, Optional[str]] = {nid: None for nid in graph.nodes}

    # Min-heap: (distance, node_id)
    heap = [(0.0, start_id)]

    visited: Set[str] = set()
    nodes_visited = 0
    edges_considered = 0

    while heap:
        current_dist, current = heapq.heappop(heap)

        if current in visited:
            continue
        visited.add(current)
        nodes_visited += 1

        # Early exit when we reach destination
        if current == end_id:
            break

        # Explore neighbors
        for neighbor, edge in graph.get_neighbours(current):
            edges_considered += 1

            if neighbor.node_id in visited:
                continue

            # Get travel time using the traffic data store
            travel_time = store.get_weight(
                edge.key,
                edge.capacity,
                edge.distance,
                period
            )
            new_dist = current_dist + travel_time

            if new_dist < dist[neighbor.node_id]:
                dist[neighbor.node_id] = new_dist
                prev[neighbor.node_id] = current
                heapq.heappush(heap, (new_dist, neighbor.node_id))

    path = _reconstruct_path(prev, start_id, end_id)

    return ShortestPathResult(
        algorithm="Dijkstra",
        path=path,
        total_time_minutes=dist[end_id] if dist[end_id] != float('inf') else -1,
        time_period=period,
        nodes_visited=nodes_visited,
        edges_considered=edges_considered,
    )


# ===========================================================================
# 2. A* SEARCH ALGORITHM (Emergency Vehicle Routing)
# ===========================================================================

def astar(
    graph: Graph,
    store: TrafficDataStore,
    start_id: str,
    end_id: str,
    period: TimePeriod = TimePeriod.MORNING_PEAK,
    heuristic: Optional[Callable[[str, str], float]] = None,
) -> ShortestPathResult:
    """
    A* search for emergency vehicle routing.

    f(n) = g(n) + h(n)
      g(n) = actual travel time from start to n
      h(n) = estimated travel time from n to end (admissible heuristic)
    """
    if start_id not in graph.nodes:
        return ShortestPathResult("A*", [start_id], float('inf'), period)
    if end_id not in graph.nodes:
        return ShortestPathResult("A*", [], float('inf'), period)

    # Default heuristic: Euclidean distance / 60 km/h (admissible)
    def default_heuristic(a: str, b: str) -> float:
        node_a = graph.nodes[a]
        node_b = graph.nodes[b]
        return node_a.euclidean_distance_to(node_b)  # minutes at 60 km/h

    h_func = heuristic or default_heuristic

    g_score: Dict[str, float] = {nid: float('inf') for nid in graph.nodes}
    g_score[start_id] = 0.0

    f_score: Dict[str, float] = {nid: float('inf') for nid in graph.nodes}
    f_score[start_id] = h_func(start_id, end_id)

    prev: Dict[str, Optional[str]] = {nid: None for nid in graph.nodes}

    heap = [(f_score[start_id], start_id)]
    visited: Set[str] = set()
    nodes_visited = 0
    edges_considered = 0

    while heap:
        _, current = heapq.heappop(heap)

        if current in visited:
            continue
        visited.add(current)
        nodes_visited += 1

        if current == end_id:
            break

        for neighbor, edge in graph.get_neighbours(current):
            edges_considered += 1

            if neighbor.node_id in visited:
                continue

            travel_time = store.get_weight(
                edge.key,
                edge.capacity,
                edge.distance,
                period
            )
            tentative_g = g_score[current] + travel_time

            if tentative_g < g_score[neighbor.node_id]:
                g_score[neighbor.node_id] = tentative_g
                f_score[neighbor.node_id] = tentative_g + h_func(neighbor.node_id, end_id)
                prev[neighbor.node_id] = current
                heapq.heappush(heap, (f_score[neighbor.node_id], neighbor.node_id))

    path = _reconstruct_path(prev, start_id, end_id)

    return ShortestPathResult(
        algorithm="A*",
        path=path,
        total_time_minutes=g_score[end_id] if g_score[end_id] != float('inf') else -1,
        time_period=period,
        nodes_visited=nodes_visited,
        edges_considered=edges_considered,
    )


# ===========================================================================
# 3. TIME-VARYING DIJKSTRA
# ===========================================================================

def time_varying_shortest_path(
    graph: Graph,
    store: TrafficDataStore,
    start_id: str,
    end_id: str,
    departure_hour: int,
) -> ShortestPathResult:
    """
    Time-dependent shortest path that selects traffic period based on hour.

    Traffic periods:
      - 7–9 AM   → MORNING_PEAK
      - 16–19    → EVENING_PEAK
      - 19–7     → NIGHT
      - otherwise → AFTERNOON
    """
    if 7 <= departure_hour <= 9:
        period = TimePeriod.MORNING_PEAK
    elif 16 <= departure_hour <= 19:
        period = TimePeriod.EVENING_PEAK
    elif departure_hour >= 19 or departure_hour <= 7:
        period = TimePeriod.NIGHT
    else:
        period = TimePeriod.AFTERNOON

    result = dijkstra(graph, store, start_id, end_id, period)
    result.algorithm = f"TimeVarying({departure_hour}h)"
    return result


# ===========================================================================
# 4. ALTERNATE ROUTE (Road Closure Handling)
# ===========================================================================

def alternate_route(
    graph: Graph,
    store: TrafficDataStore,
    start_id: str,
    end_id: str,
    blocked_edge_keys: List[str],
    period: TimePeriod = TimePeriod.MORNING_PEAK,
) -> ShortestPathResult:
    """
    Find an alternate route when certain roads are blocked.
    """
    blocked_set = set(blocked_edge_keys)

    dist: Dict[str, float] = {nid: float('inf') for nid in graph.nodes}
    dist[start_id] = 0.0

    prev: Dict[str, Optional[str]] = {nid: None for nid in graph.nodes}
    heap = [(0.0, start_id)]
    visited: Set[str] = set()
    nodes_visited = 0
    edges_considered = 0

    while heap:
        current_dist, current = heapq.heappop(heap)

        if current in visited:
            continue
        visited.add(current)
        nodes_visited += 1

        if current == end_id:
            break

        for neighbor, edge in graph.get_neighbours(current):
            edges_considered += 1

            if edge.key in blocked_set:
                continue
            if neighbor.node_id in visited:
                continue

            travel_time = store.get_weight(
                edge.key,
                edge.capacity,
                edge.distance,
                period
            )
            new_dist = current_dist + travel_time

            if new_dist < dist[neighbor.node_id]:
                dist[neighbor.node_id] = new_dist
                prev[neighbor.node_id] = current
                heapq.heappush(heap, (new_dist, neighbor.node_id))

    path = _reconstruct_path(prev, start_id, end_id)

    return ShortestPathResult(
        algorithm=f"AlternateRoute",
        path=path,
        total_time_minutes=dist[end_id] if dist[end_id] != float('inf') else -1,
        time_period=period,
        nodes_visited=nodes_visited,
        edges_considered=edges_considered,
    )


# ===========================================================================
# 5. LEAST CONGESTED ROUTE
# ===========================================================================

def least_congested_route(
    graph: Graph,
    store: TrafficDataStore,
    start_id: str,
    end_id: str,
    period: TimePeriod = TimePeriod.MORNING_PEAK,
) -> ShortestPathResult:
    """
    Find route that minimizes congestion ratio rather than travel time.
    """
    if start_id not in graph.nodes or end_id not in graph.nodes:
        return ShortestPathResult("LeastCongested", [], float('inf'), period)

    def congestion_weight(edge: Edge, p: TimePeriod) -> float:
        return store.get_congestion_ratio(edge.key, edge.capacity, p)

    dist: Dict[str, float] = {nid: float('inf') for nid in graph.nodes}
    dist[start_id] = 0.0
    prev: Dict[str, Optional[str]] = {nid: None for nid in graph.nodes}
    heap = [(0.0, start_id)]
    visited: Set[str] = set()
    nodes_visited = 0
    edges_considered = 0

    while heap:
        current_dist, current = heapq.heappop(heap)

        if current in visited:
            continue
        visited.add(current)
        nodes_visited += 1

        if current == end_id:
            break

        for neighbor, edge in graph.get_neighbours(current):
            edges_considered += 1

            if neighbor.node_id in visited:
                continue

            weight = congestion_weight(edge, period)
            new_dist = current_dist + weight

            if new_dist < dist[neighbor.node_id]:
                dist[neighbor.node_id] = new_dist
                prev[neighbor.node_id] = current
                heapq.heappush(heap, (new_dist, neighbor.node_id))

    path = _reconstruct_path(prev, start_id, end_id)

    # Calculate actual travel time for display
    total_time = 0.0
    for i in range(len(path) - 1):
        edge = graph.get_edge(path[i], path[i + 1])
        if edge:
            total_time += store.get_weight(edge.key, edge.capacity, edge.distance, period)

    return ShortestPathResult(
        algorithm="LeastCongested",
        path=path,
        total_time_minutes=total_time,
        time_period=period,
        nodes_visited=nodes_visited,
        edges_considered=edges_considered,
    )


# ===========================================================================
# 6. ROUTE CACHE (Memoization)
# ===========================================================================

class RouteCache:
    """Memoization cache for route planning."""
    
    def __init__(self):
        self._cache: Dict[tuple, ShortestPathResult] = {}
        self.hits = 0
        self.misses = 0

    def get(self, start: str, end: str, period: TimePeriod) -> Optional[ShortestPathResult]:
        key = (start, end, period)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def store(self, start: str, end: str, period: TimePeriod, result: ShortestPathResult) -> None:
        self._cache[(start, end, period)] = result

    def cached_dijkstra(self, graph: Graph, store: TrafficDataStore,
                        start: str, end: str, period: TimePeriod = TimePeriod.AFTERNOON) -> ShortestPathResult:
        cached = self.get(start, end, period)
        if cached is not None:
            return cached
        result = dijkstra(graph, store, start, end, period)
        self.store(start, end, period, result)
        return result

    def stats(self) -> dict:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {"hits": self.hits, "misses": self.misses, "hit_rate": f"{hit_rate:.1f}%"}


# ===========================================================================
# 7. DEMO FUNCTION
# ===========================================================================

def run_shortest_path_demo(graph: Graph, store: TrafficDataStore) -> dict:
    """Run shortest path demonstrations for Task 3 scenarios."""
    print("\n" + "=" * 68)
    print("  TASK 3 — Traffic Flow Optimization (Shortest Path)")
    print("=" * 68)

    results = {}

    # Scenario A: Time-dependent routing
    print("\n[SCENARIO A] Time-Dependent Route Planning")
    print("  Travel time from New Cairo (4) to Downtown (3):")
    print(f"  {'Period':<16}  {'Time (min)':>12}")
    print("  " + "-" * 30)

    start, end = "4", "3"
    for period in [TimePeriod.MORNING_PEAK, TimePeriod.AFTERNOON, TimePeriod.NIGHT]:
        result = dijkstra(graph, store, start, end, period)
        print(f"  {period.name:<16}  {result.total_time_minutes:>12.1f}")
        results[f"dijkstra_{period.name.lower()}"] = result

    # Scenario C: Road closure
    print("\n[SCENARIO C] Road Closure — Alternate Route")
    print("  Route from Nasr City (2) to Giza (8):")

    start, end = "2", "8"
    period = TimePeriod.MORNING_PEAK

    normal = dijkstra(graph, store, start, end, period)
    print(f"  Normal route:      {normal.total_time_minutes:.1f} min → Path: {normal.path}")

    blocked = alternate_route(graph, store, start, end, ["3-5"], period)
    print(f"  Blocked (road 3-5): {blocked.total_time_minutes:.1f} min → Path: {blocked.path}")

    results["normal_route"] = normal
    results["blocked_route"] = blocked

    # Compare Dijkstra vs A*
    print("\n[COMPARISON] Dijkstra vs A* for Emergency Routing")
    print(f"  {'Route':<30}  {'Algorithm':<10}  {'Time':>6}  {'Nodes':>6}")
    print("  " + "-" * 55)

    test_pairs = [
        ("7", "F9", "6th October → Qasr El Aini"),
        ("13", "F9", "New Capital → Qasr El Aini"),
        ("12", "F10", "Helwan → Maadi Hospital"),
    ]

    for s, e, label in test_pairs:
        dijk = dijkstra(graph, store, s, e, TimePeriod.MORNING_PEAK)
        ast = astar(graph, store, s, e, TimePeriod.MORNING_PEAK)
        print(f"  {label:<30}  {'Dijkstra':<10}  {dijk.total_time_minutes:>5.1f}  {dijk.nodes_visited:>6}")
        print(f"  {'':<30}  {'A*':<10}  {ast.total_time_minutes:>5.1f}  {ast.nodes_visited:>6}")

        results[f"dijkstra_{s}_{e}"] = dijk
        results[f"astar_{s}_{e}"] = ast

    return results