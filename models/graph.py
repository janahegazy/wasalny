"""
models/graph.py
Cairo Transportation Network — Graph Model

The Graph class is the single source of truth for the entire Cairo network.
It builds itself from hardcoded data on construction — no external files,
no dependencies — and exposes a clean, consistent interface that every
algorithm module depends on.

Design decisions:
  — Adjacency list (dict of lists) instead of a matrix: the graph has 25 nodes
    and 40 edges, making it sparse. Adjacency lists give O(V+E) space and
    O(degree) neighbour traversal, both superior to O(V²) matrix for this size.
  — On-demand weight computation: edge weights are NOT stored statically.
    Calling edge.get_weight(period) recomputes travel time from distance,
    speed, and live flow data. This lets any algorithm query any time period
    without rebuilding or copying the graph.
  — Separation of concerns: Node and Edge carry their own logic. Graph
    only manages the adjacency structure and the data-loading pipeline.


"""

from typing import Dict, List, Optional, Tuple
from models.node import Node, NodeType
from models.edge import Edge, TimePeriod


class Graph:
    """
    Weighted, undirected graph of the Greater Cairo transportation network.

    Attributes (read-only after construction)
    ------------------------------------------
    nodes        : dict[str, Node]         — all 25 nodes keyed by ID
    edges        : dict[str, Edge]         — all edges keyed by canonical key
    adjacency    : dict[str, list[Edge]]   — neighbour lists for fast traversal
    """

    def __init__(self):
        self.nodes:     Dict[str, Node]       = {}
        self.edges:     Dict[str, Edge]       = {}
        self.adjacency: Dict[str, List[Edge]] = {}

        # Build the network in strict dependency order
        self._load_nodes()
        self._load_existing_roads()
        self._load_hospital_roads()
        self._load_potential_roads()
        self._load_traffic_flow()
        self._load_speeds()
        self._load_maintenance()

    # ==================================================================
    # Public interface — used by algorithm modules
    # ==================================================================

    def get_node(self, node_id: str) -> Node:
        """Return the Node for a given ID, or raise KeyError."""
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' does not exist in the Cairo network.")
        return self.nodes[node_id]

    def get_edge(self, id_a: str, id_b: str) -> Optional[Edge]:
        """
        Return the Edge between two nodes if it exists, else None.
        Order of arguments does not matter (bidirectional lookup).
        """
        a, b = sorted([id_a, id_b])
        return self.edges.get(f"{a}-{b}")

    def get_neighbours(self, node_id: str) -> List[Tuple[Node, Edge]]:
        """
        Return all (neighbour_node, connecting_edge) pairs for a node.
        This is the primary traversal method used by Dijkstra and A*.
        """
        if node_id not in self.adjacency:
            return []
        result = []
        for edge in self.adjacency[node_id]:
            neighbour_id = edge.to_id if edge.from_id == node_id else edge.from_id
            result.append((self.nodes[neighbour_id], edge))
        return result

    def get_existing_edges(self) -> List[Edge]:
        """Return only real (non-potential) road edges."""
        return [e for e in self.edges.values() if not e.is_potential]

    def get_potential_edges(self) -> List[Edge]:
        """Return only proposed new road edges."""
        return [e for e in self.edges.values() if e.is_potential]

    def get_all_edges(self) -> List[Edge]:
        """Return every edge: existing + potential."""
        return list(self.edges.values())

    def get_critical_nodes(self) -> List[Node]:
        """Return all nodes flagged as critical infrastructure."""
        return [n for n in self.nodes.values() if n.is_critical]

    def remove_edge(self, id_a: str, id_b: str) -> Optional[Edge]:
        """
        Temporarily remove an edge from the graph (road closure simulation).
        Returns the removed Edge so it can be restored later.
        Call restore_edge(edge) to put it back.
        """
        edge = self.get_edge(id_a, id_b)
        if edge is None:
            return None
        key = edge.key
        del self.edges[key]
        self.adjacency[edge.from_id] = [
            e for e in self.adjacency.get(edge.from_id, []) if e.key != key
        ]
        self.adjacency[edge.to_id] = [
            e for e in self.adjacency.get(edge.to_id, []) if e.key != key
        ]
        return edge

    def restore_edge(self, edge: Edge) -> None:
        """Re-insert a previously removed edge (undo road closure)."""
        self._insert_edge(edge)

    def summary(self) -> str:
        """Return a human-readable summary of the graph's contents."""
        existing   = len(self.get_existing_edges())
        potential  = len(self.get_potential_edges())
        critical   = len(self.get_critical_nodes())
        lines = [
            "=" * 60,
            "  Cairo Transportation Network — Graph Summary",
            "=" * 60,
            f"  Nodes (locations)    : {len(self.nodes)}",
            f"    ↳ Critical nodes   : {critical}",
            f"  Edges (roads)        : {existing} existing + {potential} potential",
            f"  Total edges in graph : {len(self.edges)}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def print_summary(self) -> None:
        print(self.summary())

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _insert_node(self, node: Node) -> None:
        self.nodes[node.node_id]     = node
        self.adjacency[node.node_id] = []

    def _insert_edge(self, edge: Edge) -> None:
        """
        Register an edge in the edge dict and both adjacency lists.
        Silently skips if a road with the same canonical key already exists,
        preventing duplicate entries from bidirectional data rows.
        """
        if edge.key in self.edges:
            return  # already registered — bidirectional duplicate in dataset
        self.edges[edge.key] = edge
        self.adjacency.setdefault(edge.from_id, []).append(edge)
        self.adjacency.setdefault(edge.to_id,   []).append(edge)

    # ==================================================================
    # Data loading — Section 1A and 1B
    # ==================================================================

    def _load_nodes(self) -> None:
        """
        Load all 25 nodes: 15 districts + 10 facilities.
        Data source: Section 1A (neighbourhoods) and 1B (facilities).
        """

        # --- 1A: Residential / business districts ---
        # Format: ID, Name, Population, Type, Longitude, Latitude
        districts = [
            ("1",  "Maadi",                      250000, "Residential", 31.25, 29.96),
            ("2",  "Nasr City",                   500000, "Mixed",       31.34, 30.06),
            ("3",  "Downtown Cairo",              100000, "Business",    31.24, 30.04),
            ("4",  "New Cairo",                   300000, "Residential", 31.47, 30.03),
            ("5",  "Heliopolis",                  200000, "Mixed",       31.32, 30.09),
            ("6",  "Zamalek",                      50000, "Residential", 31.22, 30.06),
            ("7",  "6th October City",            400000, "Mixed",       30.98, 29.93),
            ("8",  "Giza",                        550000, "Mixed",       31.21, 29.99),
            ("9",  "Mohandessin",                 180000, "Business",    31.20, 30.05),
            ("10", "Dokki",                       220000, "Mixed",       31.21, 30.03),
            ("11", "Shubra",                      450000, "Residential", 31.24, 30.11),
            ("12", "Helwan",                      350000, "Industrial",  31.33, 29.85),
            ("13", "New Administrative Capital",   50000, "Government",  31.80, 30.02),
            ("14", "Al Rehab",                    120000, "Residential", 31.49, 30.06),
            ("15", "Sheikh Zayed",                150000, "Residential", 30.94, 30.01),
        ]

        for nid, name, pop, ntype, lon, lat in districts:
            self._insert_node(Node(nid, name, NodeType.from_string(ntype), lon, lat, pop))

        # --- 1B: Important facilities ---
        # Format: ID, Name, Type, Longitude, Latitude
        facilities = [
            ("F1",  "Cairo International Airport",  "Airport",     31.41, 30.11),
            ("F2",  "Ramses Railway Station",        "Transit Hub", 31.25, 30.06),
            ("F3",  "Cairo University",              "Education",   31.21, 30.03),
            ("F4",  "Al-Azhar University",           "Education",   31.26, 30.05),
            ("F5",  "Egyptian Museum",               "Tourism",     31.23, 30.05),
            ("F6",  "Cairo International Stadium",   "Sports",      31.30, 30.07),
            ("F7",  "Smart Village",                 "Business",    30.97, 30.07),
            ("F8",  "Cairo Festival City",           "Commercial",  31.40, 30.03),
            ("F9",  "Qasr El Aini Hospital",         "Medical",     31.23, 30.03),
            ("F10", "Maadi Military Hospital",       "Medical",     31.25, 29.95),
        ]

        for nid, name, ntype, lon, lat in facilities:
            self._insert_node(Node(nid, name, NodeType.from_string(ntype), lon, lat))

    # ==================================================================
    # Data loading — Section 2A: Existing roads
    # ==================================================================

    def _load_existing_roads(self) -> None:
        """
        Load the 28 existing road segments from Section 2A.
        Format: FromID, ToID, Distance(km), Capacity(veh/h), Condition(1-10)
        """
        roads = [
            # (from, to, distance_km, capacity_veh_h, condition)
            ("1",  "3",   8.5,  3000, 7),
            ("1",  "8",   6.2,  2500, 6),
            ("2",  "3",   5.9,  2800, 8),
            ("2",  "5",   4.0,  3200, 9),
            ("3",  "5",   6.1,  3500, 7),
            ("3",  "6",   3.2,  2000, 8),
            ("3",  "9",   4.5,  2600, 6),
            ("3",  "10",  3.8,  2400, 7),
            ("4",  "2",  15.2,  3800, 9),
            ("4",  "14",  5.3,  3000, 10),
            ("5",  "11",  7.9,  3100, 7),
            ("6",  "9",   2.2,  1800, 8),
            ("7",  "8",  24.5,  3500, 8),
            ("7",  "15",  9.8,  3000, 9),
            ("8",  "10",  3.3,  2200, 7),
            ("8",  "12", 14.8,  2600, 5),
            ("9",  "10",  2.1,  1900, 7),
            ("10", "11",  8.7,  2400, 6),
            ("11", "F2",  3.6,  2200, 7),
            ("12", "1",  12.7,  2800, 6),
            ("13", "4",  45.0,  4000, 10),
            ("14", "13", 35.5,  3800, 9),
            ("15", "7",   9.8,  3000, 9),
            ("F1", "5",   7.5,  3500, 9),
            ("F1", "2",   9.2,  3200, 8),
            ("F2", "3",   2.5,  2000, 7),
            ("F7", "15",  8.3,  2800, 8),
            ("F8", "4",   6.1,  3000, 9),
        ]

        for from_id, to_id, dist, cap, cond in roads:
            self._insert_edge(Edge(from_id, to_id, dist, cap, cond))

    # ==================================================================
    # Data loading — Section 2B: Hospital access roads
    # ==================================================================

    def _load_hospital_roads(self) -> None:
        """
        Load the 6 hospital-access roads added to make F9 and F10
        reachable for A* emergency routing (Section 2B).
        """
        hospital_roads = [
            ("F9",  "3",   1.2, 1500, 9),
            ("F9",  "6",   2.1, 1200, 8),
            ("F9",  "10",  2.5, 1300, 8),
            ("F10", "1",   1.8, 1400, 9),
            ("F10", "12",  3.2, 1200, 7),
            ("F10", "8",   5.1, 1500, 8),
        ]

        for from_id, to_id, dist, cap, cond in hospital_roads:
            self._insert_edge(Edge(from_id, to_id, dist, cap, cond))

    # ==================================================================
    # Data loading — Section 2C: Potential new roads
    # ==================================================================

    def _load_potential_roads(self) -> None:
        """
        Load the 15 proposed new roads from Section 2C.
        These are NOT in the active adjacency list — they are stored
        in self.edges with is_potential=True so MST can evaluate them
        as expansion candidates without affecting routing algorithms.

        Note: potential roads are inserted into self.edges but NOT into
        self.adjacency, ensuring Dijkstra and A* ignore them automatically.
        """
        potential_roads = [
            # (from, to, distance_km, capacity_veh_h, cost_million_egp)
            ("1",   "4",  22.8, 4000,  450),
            ("1",   "14", 25.3, 3800,  500),
            ("2",   "13", 48.2, 4500,  950),
            ("3",   "13", 56.7, 4500, 1100),
            ("5",   "4",  16.8, 3500,  320),
            ("6",   "8",   7.5, 2500,  150),
            ("7",   "13", 82.3, 4000, 1600),
            ("9",   "11",  6.9, 2800,  140),
            ("10",  "F7", 27.4, 3200,  550),
            ("11",  "13", 62.1, 4200, 1250),
            ("12",  "14", 30.5, 3600,  610),
            ("14",  "5",  18.2, 3300,  360),
            ("15",  "9",  22.7, 3000,  450),
            ("F1",  "13", 40.2, 4000,  800),
            ("F7",  "9",  26.8, 3200,  540),
        ]

        for from_id, to_id, dist, cap, cost in potential_roads:
            edge = Edge(from_id, to_id, dist, cap,
                        condition=10,       # assumed new construction
                        is_potential=True,
                        cost_millions=cost)
            # Only add to edges dict, NOT adjacency — routing ignores these
            self.edges[edge.key] = edge

    # ==================================================================
    # Data loading — Section 3A + 3B: Traffic flow
    # ==================================================================

    def _load_traffic_flow(self) -> None:
        """
        Attach traffic flow per time period to each existing edge.
        Data source: Sections 3A (original roads) and 3B (hospital roads).
        Format: (from_id, to_id, morning, afternoon, evening, night)
        """
        flow_data = [
            # Original roads (Section 3A)
            ("1",  "3",  2800, 1500, 2600,  800),
            ("1",  "8",  2200, 1200, 2100,  600),
            ("2",  "3",  2700, 1400, 2500,  700),
            ("2",  "5",  3000, 1600, 2800,  650),
            ("3",  "5",  3200, 1700, 3100,  800),
            ("3",  "6",  1800, 1400, 1900,  500),
            ("3",  "9",  2400, 1300, 2200,  550),
            ("3",  "10", 2300, 1200, 2100,  500),
            ("4",  "2",  3600, 1800, 3300,  750),
            ("4",  "14", 2800, 1600, 2600,  600),
            ("5",  "11", 2900, 1500, 2700,  650),
            ("6",  "9",  1700, 1300, 1800,  450),
            ("7",  "8",  3200, 1700, 3000,  700),
            ("7",  "15", 2800, 1500, 2600,  600),
            ("8",  "10", 2000, 1100, 1900,  450),
            ("8",  "12", 2400, 1300, 2200,  500),
            ("9",  "10", 1800, 1200, 1700,  400),
            ("10", "11", 2200, 1300, 2100,  500),
            ("11", "F2", 2100, 1200, 2000,  450),
            ("12", "1",  2600, 1400, 2400,  550),
            ("13", "4",  3800, 2000, 3500,  800),
            ("14", "13", 3600, 1900, 3300,  750),
            ("15", "7",  2800, 1500, 2600,  600),
            ("F1", "5",  3300, 2200, 3100, 1200),
            ("F1", "2",  3000, 2000, 2800, 1100),
            ("F2", "3",  1900, 1600, 1800,  900),
            ("F7", "15", 2600, 1500, 2400,  550),
            ("F8", "4",  2800, 1600, 2600,  600),
            # Hospital access roads (Section 3B)
            ("F9",  "3",   800,  600,  750, 200),
            ("F9",  "6",   600,  450,  580, 150),
            ("F9",  "10",  700,  500,  650, 180),
            ("F10", "1",   700,  520,  670, 190),
            ("F10", "12",  550,  400,  520, 140),
            ("F10", "8",   750,  550,  700, 200),
        ]

        for from_id, to_id, morn, aft, eve, night in flow_data:
            edge = self.get_edge(from_id, to_id)
            if edge is not None:
                edge.flow_morning   = morn
                edge.flow_afternoon = aft
                edge.flow_evening   = eve
                edge.flow_night     = night

    # ==================================================================
    # Data loading — Section 4: Speed limits
    # ==================================================================

    def _load_speeds(self) -> None:
        """
        Attach normal and rush-hour speed limits to each edge.
        Data source: Section 4 of the dataset.
        Format: (from_id, to_id, normal_speed_kmh, rush_speed_kmh)
        """
        speed_data = [
            ("1",   "3",   60,  30),
            ("1",   "8",   70,  35),
            ("2",   "3",   60,  25),
            ("2",   "5",   80,  40),
            ("3",   "5",   60,  25),
            ("3",   "6",   50,  20),
            ("3",   "9",   60,  25),
            ("3",   "10",  60,  25),
            ("4",   "2",  100,  60),
            ("4",   "14",  90,  55),
            ("5",   "11",  70,  35),
            ("6",   "9",   50,  20),
            ("7",   "8",  100,  60),
            ("7",   "15",  90,  55),
            ("8",   "10",  60,  30),
            ("8",   "12",  80,  45),
            ("9",   "10",  50,  20),
            ("10",  "11",  60,  30),
            ("11",  "F2",  60,  30),
            ("12",  "1",   80,  45),
            ("13",  "4",  120,  80),
            ("14",  "13", 110,  75),
            ("15",  "7",   90,  55),
            ("F1",  "5",   90,  55),
            ("F1",  "2",   90,  55),
            ("F2",  "3",   50,  20),
            ("F7",  "15",  80,  50),
            ("F8",  "4",   90,  55),
            # Hospital roads
            ("F9",  "3",   50,  30),
            ("F9",  "6",   50,  30),
            ("F9",  "10",  50,  30),
            ("F10", "1",   60,  35),
            ("F10", "12",  60,  35),
            ("F10", "8",   70,  40),
        ]

        for from_id, to_id, normal, rush in speed_data:
            edge = self.get_edge(from_id, to_id)
            if edge is not None:
                edge.normal_speed = normal
                edge.rush_speed   = rush

    # ==================================================================
    # Data loading — Section 5: Maintenance data
    # ==================================================================

    def _load_maintenance(self) -> None:
        """
        Attach maintenance cost and priority score to each existing edge.
        Data source: Section 5 of the dataset.
        Format: (from_id, to_id, condition, cost_million_egp, priority_1_to_5)
        """
        maint_data = [
            ("1",   "3",   7,  12.5, 3),
            ("1",   "8",   6,  18.0, 3),
            ("2",   "3",   8,   8.5, 3),
            ("2",   "5",   9,   6.0, 2),
            ("3",   "5",   7,   9.0, 4),
            ("3",   "6",   8,   5.0, 2),
            ("3",   "9",   6,  14.0, 4),
            ("3",   "10",  7,  10.5, 3),
            ("4",   "2",   9,  20.0, 3),
            ("4",   "14", 10,   4.0, 2),
            ("5",   "11",  7,  11.0, 3),
            ("6",   "9",   8,   4.0, 2),
            ("7",   "8",   8,  30.0, 4),
            ("7",   "15",  9,  12.0, 3),
            ("8",   "10",  7,   6.0, 3),
            ("8",   "12",  5,  35.0, 5),
            ("9",   "10",  7,   4.0, 2),
            ("10",  "11",  6,  18.0, 3),
            ("11",  "F2",  7,   7.0, 3),
            ("12",  "1",   6,  22.0, 4),
            ("13",  "4",  10,  50.0, 4),
            ("14",  "13",  9,  40.0, 3),
            ("15",  "7",   9,  12.0, 3),
            ("F1",  "5",   9,   9.0, 4),
            ("F1",  "2",   8,  11.0, 4),
            ("F2",  "3",   7,   4.0, 5),
            ("F7",  "15",  8,  10.0, 3),
            ("F8",  "4",   9,   7.0, 3),
            # Hospital access roads — all priority 5
            ("F9",  "3",   9,   2.0, 5),
            ("F9",  "6",   8,   3.5, 5),
            ("F9",  "10",  8,   4.0, 5),
            ("F10", "1",   9,   2.5, 5),
            ("F10", "12",  7,   5.0, 5),
            ("F10", "8",   8,   7.0, 5),
        ]

        for from_id, to_id, _cond, cost, priority in maint_data:
            edge = self.get_edge(from_id, to_id)
            if edge is not None:
                edge.maint_cost     = cost
                edge.maint_priority = priority
