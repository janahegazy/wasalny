"""
algorithms/mst.py
Cairo Transportation Network — Task 2: Infrastructure Network Design

Implements Kruskal's Algorithm with critical-node constraints:
  — Critical nodes (hospitals F9, F10; airport F1; transit hub F2;
    New Administrative Capital 13) are GUARANTEED connectivity before
    the cost-minimisation phase runs.

Edge weights:
  — Existing roads  : distance (km) — geometric connectivity cost
  — Potential roads : cost_millions (EGP) — financial construction cost

The two modes are selected by the `use_cost` flag.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing      import Dict, List, Optional, Set, Tuple

from models.graph import Graph
from models.edge  import Edge
from models.node  import Node


# ===========================================================================
# Result container
# ===========================================================================

@dataclass
class MSTResult:
    """
    Holds the output of a single MST run.

    Attributes
    ----------
    algorithm       : name of the algorithm used
    use_cost        : True → weights are EGP cost; False → weights are km
    include_potential: whether proposed roads were eligible
    edges           : list of edges selected into the MST
    total_weight    : sum of weights of selected edges
    critical_forced : edges added in the priority phase (guaranteed critical access)
    disconnected    : node IDs that could not be connected (should be empty)
    """
    algorithm:          str
    use_cost:           bool
    include_potential:  bool
    edges:              List[Edge]      = field(default_factory=list)
    total_weight:       float           = 0.0
    critical_forced:    List[Edge]      = field(default_factory=list)
    disconnected:       List[str]       = field(default_factory=list)

    # -----------------------------------------------------------------------
    # Derived properties
    # -----------------------------------------------------------------------

    @property
    def total_distance_km(self) -> float:
        return sum(e.distance for e in self.edges)

    @property
    def total_cost_megp(self) -> float:
        return sum(e.cost_millions for e in self.edges if e.is_potential)

    @property
    def new_roads_selected(self) -> List[Edge]:
        return [e for e in self.edges if e.is_potential]

    @property
    def existing_roads_selected(self) -> List[Edge]:
        return [e for e in self.edges if not e.is_potential]

    # -----------------------------------------------------------------------
    # Display helpers
    # -----------------------------------------------------------------------

    def weight_label(self) -> str:
        return "Cost (MEGP)" if self.use_cost else "Distance (km)"

    def edge_weight(self, edge: Edge) -> float:
        return edge.cost_millions if self.use_cost else edge.distance

    def summary(self) -> str:
        w = self.weight_label()
        
        # Calculate total costs
        total_maintenance = sum(e.maint_cost for e in self.existing_roads_selected)
        total_construction = sum(e.cost_millions for e in self.new_roads_selected)
        
        lines = [
            "",
            "=" * 68,
            f"  MST Result — {self.algorithm}",
            "=" * 68,
            f"  Weight metric       : {w}",
            f"  Potential roads OK  : {self.include_potential}",
            f"  Edges in MST        : {len(self.edges)}",
            f"  Total distance      : {self.total_distance_km:.1f} km",
            f"  Total maintenance   : {total_maintenance:.1f} M EGP/yr (existing roads)",
            f"  Total construction  : {total_construction:.0f} M EGP (new roads)",
            f"  Total weight        : {self.total_weight:.2f}",
            f"  Critical forced     : {len(self.critical_forced)} edges",
            f"  Disconnected nodes  : {self.disconnected or 'none'}",
            "=" * 68,
        ]
        return "\n".join(lines)

    def print_edges(self) -> None:
        print(self.summary())
        forced_keys = {e.key for e in self.critical_forced}
        
        # Determine column headers based on mode
        if self.use_cost and self.include_potential:
            # Scenario 2: Show cost as primary weight
            print(f"\n  {'Road':<14}  {'Dist':>7}  {'Cost(M)':>9}  {'Maintenance(M)':>13}  {'Type':<12}  {'Note'}")
            print(f"  {'────':<14}  {'────':>7}  {'───────':>9}  {'─────────────':>13}  {'────':<12}  {'────'}")
            for e in self.edges:
                kind = "POTENTIAL" if e.is_potential else "existing"
                note = "* priority *" if e.key in forced_keys else ""
                # For existing roads: show maintenance cost
                # For potential roads: show construction cost
                if e.is_potential:
                    display_cost = e.cost_millions
                    maint = 0
                else:
                    display_cost = e.maint_cost  # annual maintenance cost
                    maint = e.maint_cost
                print(f"  {e.key:<14}  {e.distance:>5.1f}km  "
                      f"{display_cost:>7.0f}M  {maint:>11.1f}M  {kind:<12}  {note}")
        else:
            # Scenario 1: Show distance as primary weight
            print(f"\n  {'Road':<14}  {'Dist':>7}  {'Maint Cost(M)':>13}  {'Type':<12}  {'Note'}")
            print(f"  {'────':<14}  {'────':>7}  {'────────────':>13}  {'────':<12}  {'────'}")
            for e in self.edges:
                kind = "POTENTIAL" if e.is_potential else "existing"
                note = "* priority *" if e.key in forced_keys else ""
                print(f"  {e.key:<14}  {e.distance:>5.1f}km  "
                      f"{e.maint_cost:>11.1f}M  {kind:<12}  {note}")
        
        if self.disconnected:
            print(f"\n  WARNING: These nodes are isolated: {self.disconnected}")


# ===========================================================================
# Union-Find (Disjoint Set Union) — used by Kruskal's
# ===========================================================================

class UnionFind:
    """
    Path-compressed, union-by-rank disjoint set structure.

    Operations:
      find(x)    — O(α(n)) amortised
      union(x,y) — O(α(n)) amortised
    """

    def __init__(self, members: List[str]):
        self._parent: Dict[str, str] = {m: m for m in members}
        self._rank:   Dict[str, int] = {m: 0 for m in members}

    def find(self, x: str) -> str:
        """Return the representative of x's component (with path compression)."""
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])   # path compression
        return self._parent[x]

    def union(self, x: str, y: str) -> bool:
        """
        Merge the components containing x and y.
        Returns True if they were in different components (edge is useful),
        False if they were already connected (edge would create a cycle).
        """
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False   # same component → skip to avoid cycle
        # Union by rank
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1
        return True

    def connected(self, x: str, y: str) -> bool:
        return self.find(x) == self.find(y)


# ===========================================================================
# Priority Phase — guarantee critical node connectivity
# ===========================================================================

# IDs of nodes that must always be reachable
_CRITICAL_IDS: Set[str] = {"F9", "F10", "F1", "F2", "13"}


def _force_critical_edges(
    graph, uf, candidate_edges, store, use_cost
) -> Tuple[List[Edge], float]:
    """
    Pre-connect every critical node before the main MST loop.

    Strategy:
      For each critical node that is not yet connected to the main component,
      find the cheapest edge incident to it (from candidate_edges) and force-add it.
      This guarantees the constraint: all critical nodes appear in the MST.

    Parameters
    ----------
    graph           : the full network
    uf              : UnionFind already initialised with all node IDs
    candidate_edges : sorted edge list (the MST will consume them)
    use_cost        : weight metric

    Returns
    -------
    (forced_edges, forced_weight) — edges added + their total weight
    """
    forced       : List[Edge] = []
    forced_weight: float      = 0.0

    for cid in sorted(_CRITICAL_IDS):
        if cid not in graph.nodes:
            continue  # node not in this graph snapshot — skip
        # Collect all candidate edges touching this critical node
        incident = [
            e for e in candidate_edges
            if e.from_id == cid or e.to_id == cid
        ]
        if not incident:
            continue

        # Sort by weight
        incident.sort(key=lambda e: smart_weight(e, use_cost))

        # Keep trying edges until we find one that connects a new component
        for edge in incident:
            if uf.union(edge.from_id, edge.to_id):
                forced.append(edge)
                forced_weight += smart_weight(edge, use_cost)
                break   # this critical node is now connected

    return forced, forced_weight


# ===========================================================================
# Kruskal's MST (the only algorithm)
# ===========================================================================

def smart_weight(e, use_cost):
    """Calculate edge weight based on mode."""
    if use_cost:
        if e.is_potential:
            return e.cost_millions
        else:
            return e.distance * 10   # scaling factor for existing roads
    else:
        return e.distance


def kruskal_mst(
    graph, store=None, use_cost=False, include_potential=False
) -> MSTResult:
    """
    Build a minimum spanning tree using Kruskal's algorithm.

    Time complexity:
        Sorting   : O(E log E)
        Union-Find: O(E · α(V))   where α is the inverse Ackermann function
        Total     : O(E log E)

    Space complexity: O(V + E)

    Steps:
        1. Collect candidate edges.
        2. Sort by weight (distance or cost).
        3. Priority phase: force-connect all critical nodes.
        4. Main loop: add edges that don't create cycles (Union-Find check).
        5. Report any disconnected nodes.

    Parameters
    ----------
    graph             : the Cairo network
    store             : traffic data store (unused in MST, kept for API consistency)
    use_cost          : True  → weight = edge.cost_millions (EGP)
                        False → weight = edge.distance (km)
    include_potential : if True, proposed new roads are eligible candidates

    Returns
    -------
    MSTResult
    """
    result = MSTResult(
        algorithm         = "Kruskal's",
        use_cost          = use_cost,
        include_potential = include_potential,
    )

    # ── 1. Collect candidates ──────────────────────────────────────────────
    candidates: List[Edge] = list(graph.get_existing_edges())
    if include_potential:
        candidates += list(graph.get_potential_edges())

    # ── 2. Sort by weight ─────────────────────────────────────────────────
    candidates.sort(key=lambda e: smart_weight(e, use_cost))
    
    # ── 3. Initialise Union-Find with all node IDs ─────────────────────────
    all_node_ids = list(graph.nodes.keys())
    uf = UnionFind(all_node_ids)

    # ── 4. Priority phase — guarantee critical connectivity ────────────────
    forced, forced_w = _force_critical_edges(graph, uf, candidates, store, use_cost)
    result.critical_forced = forced
    result.edges           = list(forced)
    result.total_weight    = forced_w

    forced_keys = {e.key for e in forced}

    # ── 5. Main Kruskal loop ───────────────────────────────────────────────
    for edge in candidates:
        if edge.key in forced_keys:
            continue   # already added in priority phase
        if uf.union(edge.from_id, edge.to_id):
            result.edges.append(edge)
            result.total_weight += smart_weight(edge, use_cost)

    # ── 6. Check for disconnected nodes ───────────────────────────────────
    # Nodes that are in the graph but not touched by any MST edge
    mst_nodes: Set[str] = set()
    for e in result.edges:
        mst_nodes.add(e.from_id)
        mst_nodes.add(e.to_id)
    # Only report nodes that have at least one candidate edge available
    reachable_nodes = set()
    for e in candidates:
        reachable_nodes.add(e.from_id)
        reachable_nodes.add(e.to_id)
    result.disconnected = sorted(reachable_nodes - mst_nodes)

    return result


# ===========================================================================
# Cost analysis helper
# ===========================================================================

def cost_analysis(graph: Graph, result: MSTResult) -> None:
    """
    Print a detailed cost analysis comparing the MST network against the
    full existing network.

    Covers:
      — Total road km in MST vs full network
      — Maintenance cost savings (MST subset vs all roads)
      — New construction costs (if potential roads are included)
      — Critical node coverage guarantee
    """
    existing = graph.get_existing_edges()
    total_dist = sum(e.distance for e in existing)
    total_maint = sum(e.maint_cost for e in existing)
    mst_dist = result.total_distance_km
    mst_maint = sum(e.maint_cost for e in result.existing_roads_selected)
    new_cost = sum(e.cost_millions for e in result.new_roads_selected)
    
    print()
    print("=" * 68)
    print("  COST ANALYSIS — MST vs Full Network")
    print("=" * 68)
    print(f"  Full network  : {len(existing):3d} roads, "
          f"{total_dist:.1f} km, {total_maint:.1f} M EGP/yr maintenance")
    print(f"  MST selected  : {len(result.existing_roads_selected):3d} existing roads, "
          f"{mst_dist:.1f} km, {mst_maint:.1f} M EGP/yr maintenance")
    if result.new_roads_selected:
        print(f"  New roads     : {len(result.new_roads_selected):3d} proposed roads, "
              f"construction cost: {new_cost:.0f} M EGP")
        print(f"  Total expansion budget needed: {new_cost:.0f} M EGP")
    print(f"  Maintenance savings: {total_maint - mst_maint:.1f} M EGP/yr "
          f"({(1 - mst_maint/total_maint)*100:.0f}% reduction)")
    print()

    # Critical node coverage
    mst_node_ids: Set[str] = set()
    for e in result.edges:
        mst_node_ids.add(e.from_id)
        mst_node_ids.add(e.to_id)

    print("  Critical Node Coverage:")
    crit_nodes = graph.get_critical_nodes()
    for n in sorted(crit_nodes, key=lambda x: x.node_id):
        covered = "✓ CONNECTED" if n.node_id in mst_node_ids else "✗ ISOLATED"
        forced = " (priority-forced)" if any(
            e.from_id == n.node_id or e.to_id == n.node_id
            for e in result.critical_forced
        ) else ""
        print(f"    [{n.node_id:<4}] {n.name:<35}  {covered}{forced}")
    print()


# ===========================================================================
# Demo entry point
# ===========================================================================

def run_mst_demo(graph: Graph) -> dict:
    """
    Run Kruskal in both modes and return results for visualisation.

    Returns
    -------
    dict with keys:
        'kruskal_dist'  : MSTResult (existing roads, distance)
        'kruskal_cost'  : MSTResult (all roads, construction cost)
    """
    print("\n" + "=" * 68)
    print("  TASK 2 — Infrastructure Network Design (Kruskal's MST)")
    print("=" * 68)

    results = {}

    # ── Mode A: Existing roads, distance weight ────────────────────────────
    print("\n[A] Kruskal's — existing roads, minimize distance")
    r = kruskal_mst(graph, use_cost=False, include_potential=False)
    r.print_edges()
    cost_analysis(graph, r)
    results["kruskal_dist"] = r

    # ── Mode B: All roads, cost weight (expansion planning) ───────────────
    print("\n[B] Kruskal's — all roads, minimize construction cost")
    r = kruskal_mst(graph, use_cost=True, include_potential=True)
    r.print_edges()
    cost_analysis(graph, r)
    results["kruskal_cost"] = r

    return results