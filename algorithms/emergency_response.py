"""
algorithms/emergency_response.py
Cairo Transportation Network — Task 4: Emergency Response System

Flow
----
1. User enters ONLY the origin node (ambulance location).
2. System runs A* to EVERY hospital (F9, F10) and picks the nearest one.
3. Greedy Signal Preemption clears every signalised intersection on the
   chosen path in order — simulating a real emergency corridor.
4. Report compares total response time WITH and WITHOUT preemption,
   showing the improvement per intersection and overall.

Greedy logic
------------
  Intersections are processed in the order the ambulance encounters them
  (greedy: always clear the *next* signal first, never look ahead).

  For each signal:
    baseline_wait  = (cycle - green) / 2   [expected red wait, uniform arrival]
    preempted_wait = preempt_hold / 2      [conservative: half hold on average]
    saved          = baseline_wait - preempted_wait
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from models.graph import Graph
from models.edge import TimePeriod
from simulation.traffic_data import TrafficDataStore, SignalConfig
from algorithms.shortest_path import astar, ShortestPathResult


# ── Hospital node IDs defined in the dataset ────────────────────────────────
HOSPITAL_NODES: Dict[str, str] = {
    "F9":  "Qasr El Aini Hospital (Downtown)",
    "F10": "Maadi Hospital",
}


# ===========================================================================
# Data classes
# ===========================================================================

@dataclass
class SignalPreemptionEvent:
    """One intersection's preemption calculation."""
    intersection_id:    str
    intersection_name:  str
    edge_key:           str      # road segment the signal controls
    normal_cycle_sec:   int
    green_time_sec:     int
    preempt_hold_sec:   int
    baseline_wait_sec:  float    # expected wait WITHOUT preemption
    preempted_wait_sec: float    # expected wait WITH    preemption
    time_saved_sec:     float    # difference


@dataclass
class EmergencyResponseResult:
    """Complete Task-4 result for one origin -> nearest hospital run."""
    origin_id:     str
    hospital_id:   str
    hospital_name: str
    period:        TimePeriod
    astar_result:  ShortestPathResult

    preemption_events:             List[SignalPreemptionEvent] = field(default_factory=list)
    astar_travel_min:              float = 0.0
    total_signal_wait_without_min: float = 0.0
    total_signal_wait_with_min:    float = 0.0

    @property
    def total_time_without_min(self) -> float:
        return self.astar_travel_min + self.total_signal_wait_without_min

    @property
    def total_time_with_min(self) -> float:
        return self.astar_travel_min + self.total_signal_wait_with_min

    @property
    def time_saved_min(self) -> float:
        return self.total_time_without_min - self.total_time_with_min

    @property
    def improvement_pct(self) -> float:
        base = self.total_time_without_min
        if base <= 0:
            return 0.0
        return (self.time_saved_min / base) * 100.0


# ===========================================================================
# Step 1 — Find nearest hospital via A*
# ===========================================================================

def _find_nearest_hospital(
    graph: Graph,
    store: TrafficDataStore,
    origin_id: str,
    period: TimePeriod,
) -> Tuple[str, str, ShortestPathResult]:
    """
    Run A* from origin to each hospital.
    Return (hospital_id, hospital_name, best_astar_result) for the nearest.
    """
    best_hospital_id   = ""
    best_hospital_name = ""
    best_result: Optional[ShortestPathResult] = None

    for h_id, h_name in HOSPITAL_NODES.items():
        if h_id not in graph.nodes:
            continue
        res = astar(graph, store, origin_id, h_id, period)
        if res.is_valid:
            if best_result is None or res.total_time_minutes < best_result.total_time_minutes:
                best_result        = res
                best_hospital_id   = h_id
                best_hospital_name = h_name

    if best_result is None:
        # Fallback — return invalid result for F9
        best_result = astar(graph, store, origin_id, "F9", period)
        best_hospital_id   = "F9"
        best_hospital_name = HOSPITAL_NODES["F9"]

    return best_hospital_id, best_hospital_name, best_result


# ===========================================================================
# Step 2 — Identify signals on the A* path
# ===========================================================================

def _signals_on_path(
    path: List[str],
    store: TrafficDataStore,
) -> List[Tuple[int, SignalConfig]]:
    """
    Walk the path and return (segment_index, SignalConfig) for every
    signalised intersection encountered, in path order.
    Each intersection is included at most once.
    """
    all_signals = store.get_all_signals()

    # build edge-key list in path order
    path_edges: List[str] = []
    for i in range(len(path) - 1):
        a, b = sorted([path[i], path[i + 1]])
        path_edges.append(f"{a}-{b}")

    seen:    set = set()
    matched: List[Tuple[int, SignalConfig]] = []

    for seg_idx, edge_key in enumerate(path_edges):
        for sig in all_signals:
            if sig.intersection_id in seen:
                continue
            if edge_key in sig.connected_roads:
                matched.append((seg_idx, sig))
                seen.add(sig.intersection_id)

    return matched


# ===========================================================================
# Step 3 — Greedy preemption
# ===========================================================================

def _greedy_preempt(
    signals: List[Tuple[int, SignalConfig]],
    period: TimePeriod,
) -> List[SignalPreemptionEvent]:
    """
    Greedy strategy: handle the next intersection the ambulance will
    hit first — no look-ahead, no backtracking.

    baseline_wait  = (cycle - green) / 2
    preempted_wait = preempt_hold    / 2   (conservative estimate)
    saved          = max(baseline - preempted, 0)
    """
    events: List[SignalPreemptionEvent] = []

    for seg_idx, sig in signals:
        green  = sig.green_time(period)
        cycle  = sig.normal_cycle_sec
        red    = max(cycle - green, 0)

        baseline_wait  = red / 2.0
        preempted_wait = sig.preempt_hold_sec / 2.0
        saved          = max(baseline_wait - preempted_wait, 0.0)

        edge_key = sig.connected_roads[0] if sig.connected_roads else "?"

        events.append(SignalPreemptionEvent(
            intersection_id   = sig.intersection_id,
            intersection_name = sig.name,
            edge_key          = edge_key,
            normal_cycle_sec  = cycle,
            green_time_sec    = green,
            preempt_hold_sec  = sig.preempt_hold_sec,
            baseline_wait_sec = baseline_wait,
            preempted_wait_sec= preempted_wait,
            time_saved_sec    = saved,
        ))

    return events


# ===========================================================================
# Public API — single entry point
# ===========================================================================

def emergency_response(
    graph: Graph,
    store: TrafficDataStore,
    origin_id: str,
    period: TimePeriod = TimePeriod.MORNING_PEAK,
) -> EmergencyResponseResult:
    """
    Task 4 full pipeline.

    Parameters
    ----------
    graph     : Graph
    store     : TrafficDataStore
    origin_id : str — ambulance starting node (user input)
    period    : TimePeriod

    Returns
    -------
    EmergencyResponseResult
    """
    # Phase 1 — nearest hospital
    h_id, h_name, astar_result = _find_nearest_hospital(
        graph, store, origin_id, period
    )

    result = EmergencyResponseResult(
        origin_id     = origin_id,
        hospital_id   = h_id,
        hospital_name = h_name,
        period        = period,
        astar_result  = astar_result,
    )

    if not astar_result.is_valid:
        return result

    result.astar_travel_min = astar_result.total_time_minutes

    # Phase 2 — greedy preemption
    raw_signals = _signals_on_path(astar_result.path, store)
    events      = _greedy_preempt(raw_signals, period)
    result.preemption_events = events

    result.total_signal_wait_without_min = (
        sum(e.baseline_wait_sec  for e in events) / 60.0
    )
    result.total_signal_wait_with_min = (
        sum(e.preempted_wait_sec for e in events) / 60.0
    )

    return result


# ===========================================================================
# Report printer
# ===========================================================================

def print_emergency_report(result: EmergencyResponseResult) -> None:
    W    = 72
    SEP  = "=" * W
    DASH = "-" * W
    r    = result.astar_result

    print("\n" + SEP)
    print("  TASK 4 — EMERGENCY RESPONSE SYSTEM")
    print("  A* Nearest-Hospital Routing  +  Greedy Signal Preemption")
    print(SEP)

    print(f"\n  Ambulance origin  : Node {result.origin_id}")
    print(f"  Nearest hospital  : {result.hospital_name}  (Node {result.hospital_id})")
    print(f"  Traffic period    : {result.period.label()}")

    if not r.is_valid:
        print(f"\n  ✗  No path found from {result.origin_id} to any hospital.")
        print(SEP)
        return

    # ── Phase 1 ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  PHASE 1 — A* SHORTEST PATH TO NEAREST HOSPITAL")
    print(f"{'─'*W}")
    print(f"  Path           : {' -> '.join(r.path)}")
    print(f"  Hops           : {len(r.path) - 1} road segment(s)")
    print(f"  Nodes explored : {r.nodes_visited}  (A* efficiency vs Dijkstra)")
    print(f"  Driving time   : {result.astar_travel_min:.2f} min  (signals NOT counted)")

    # ── Phase 2 ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  PHASE 2 — GREEDY SIGNAL PREEMPTION  (encounter order)")
    print(f"{'─'*W}")

    if not result.preemption_events:
        print("  No traffic signals found on this path.")
        print("  The route is signal-free — preemption has no effect here.")
    else:
        print(
            f"  {'#':<3} {'Intersection':<22} "
            f"{'Cycle':>6} {'Green':>6} {'Hold':>5}  "
            f"{'Wait (normal)':>13} {'Wait (preemp)':>13} {'Saved':>7}"
        )
        print("  " + DASH)

        for i, ev in enumerate(result.preemption_events, 1):
            print(
                f"  {i:<3} {ev.intersection_name:<22} "
                f"{ev.normal_cycle_sec:>5}s "
                f"{ev.green_time_sec:>5}s "
                f"{ev.preempt_hold_sec:>4}s  "
                f"{ev.baseline_wait_sec:>12.1f}s "
                f"{ev.preempted_wait_sec:>12.1f}s "
                f"{ev.time_saved_sec:>6.1f}s"
            )

        print("  " + DASH)
        b_tot  = result.total_signal_wait_without_min * 60
        p_tot  = result.total_signal_wait_with_min    * 60
        sv_tot = sum(e.time_saved_sec for e in result.preemption_events)
        print(
            f"  {'':3} {'TOTAL SIGNAL DELAY':<22} "
            f"{'':>6} {'':>6} {'':>5}  "
            f"{b_tot:>12.1f}s "
            f"{p_tot:>12.1f}s "
            f"{sv_tot:>6.1f}s"
        )

    # ── Final comparison ─────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  RESPONSE TIME — BEFORE vs AFTER GREEDY PREEMPTION")
    print(f"{'─'*W}")

    col = 38
    print(
        f"  {'Component':<{col}} "
        f"{'Without Preemption':>18}  {'With Preemption':>15}"
    )
    print("  " + DASH)
    print(
        f"  {'Driving time (A* route)':<{col}} "
        f"{result.astar_travel_min:>17.2f}m  "
        f"{result.astar_travel_min:>14.2f}m"
    )
    print(
        f"  {'Signal waiting time':<{col}} "
        f"{result.total_signal_wait_without_min:>17.2f}m  "
        f"{result.total_signal_wait_with_min:>14.2f}m"
    )
    print("  " + DASH)
    print(
        f"  {'TOTAL RESPONSE TIME':<{col}} "
        f"{result.total_time_without_min:>17.2f}m  "
        f"{result.total_time_with_min:>14.2f}m"
    )
    print("  " + DASH)

    saved_s = result.time_saved_min * 60
    saved_m = result.time_saved_min
    pct     = result.improvement_pct

    print()
    if saved_s > 0:
        print(f"  ✅  Greedy preemption cuts response time by "
              f"{saved_s:.1f} s  ({saved_m:.2f} min)  —  {pct:.1f}% improvement")
        print(f"      Ambulance arrives {saved_m:.2f} min EARLIER.")
    else:
        print("  ℹ   Path has no signal delays — preemption has no measurable effect.")

    print("\n" + SEP)


# ===========================================================================
# Interactive menu (called from main.py when user picks Shortest Path -> 2)
# ===========================================================================

def run_emergency_menu(graph: Graph, store: TrafficDataStore) -> None:
    """
    Task 4 interactive entry point.

    User enters ONLY the ambulance origin node.
    System finds nearest hospital, runs A*, applies Greedy preemption,
    prints full before/after comparison.
    """
    W = 72
    print("\n" + "=" * W)
    print("  TASK 4 — EMERGENCY RESPONSE SYSTEM")
    print("  Auto Nearest-Hospital  +  Greedy Signal Preemption")
    print("=" * W)
    print("""
  How it works:
    1. Enter the ambulance's current node.
    2. A* evaluates every hospital and selects the NEAREST one.
    3. Greedy preemption clears each traffic signal on the path
       in the exact order the ambulance will encounter them.
    4. Total response time is compared BEFORE and AFTER preemption.
""")

    # ── show available hospitals ─────────────────────────────────────────────
    print("  Hospitals in the network:")
    for h_id, h_name in HOSPITAL_NODES.items():
        print(f"    • {h_id:<5} {h_name}")

    # ── time period ──────────────────────────────────────────────────────────
    print("\n  Select traffic period:")
    print("    1. Morning Peak  (07:00-09:00)  <- heaviest congestion")
    print("    2. Afternoon     (09:00-16:00)")
    print("    3. Evening Peak  (16:00-19:00)")
    print("    4. Night         (19:00-07:00)")
    p_map = {
        "1": TimePeriod.MORNING_PEAK,
        "2": TimePeriod.AFTERNOON,
        "3": TimePeriod.EVENING_PEAK,
        "4": TimePeriod.NIGHT,
    }
    p_choice = input("  Enter choice (1-4) [default=1]: ").strip()
    period   = p_map.get(p_choice, TimePeriod.MORNING_PEAK)

    # ── origin node ──────────────────────────────────────────────────────────
    print(f"\n  Available nodes: {sorted(graph.nodes.keys())}")
    origin = input("  Enter ambulance origin node: ").strip()

    if origin not in graph.nodes:
        print(f"  ✗  Node '{origin}' not found. Returning to menu.")
        return

    # ── show A* distance to each hospital so choice is transparent ──────────
    print(f"\n  Calculating A* travel time to each hospital from node {origin}...")
    print(f"  {'Hospital':<40} {'A* Time':>9}  {'Path'}")
    print("  " + "-" * 65)

    for h_id, h_name in HOSPITAL_NODES.items():
        if h_id not in graph.nodes:
            continue
        r = astar(graph, store, origin, h_id, period)
        if r.is_valid:
            path_str = " -> ".join(r.path[:5])
            if len(r.path) > 5:
                path_str += " ..."
            print(f"  {h_name:<40} {r.total_time_minutes:>7.2f}m  {path_str}")
        else:
            print(f"  {h_name:<40}  UNREACHABLE")

    # ── run full pipeline ────────────────────────────────────────────────────
    result = emergency_response(graph, store, origin, period)
    print(f"\n  System selected: {result.hospital_name} ({result.hospital_id}) "
          f"as the nearest hospital.")

    print_emergency_report(result)

    # ── optional: all districts table ───────────────────────────────────────
    again = input(
        "\n  Show comparison table for ALL districts? (y/N): "
    ).strip().lower()
    if again == "y":
        _batch_all_districts(graph, store, period)


# ===========================================================================
# Batch helper
# ===========================================================================

def _batch_all_districts(
    graph: Graph,
    store: TrafficDataStore,
    period: TimePeriod,
) -> None:
    """Run emergency_response for every non-hospital node and tabulate results."""
    W = 82
    print("\n" + "=" * W)
    print(f"  BATCH COMPARISON — All Districts -> Nearest Hospital  ({period.name})")
    print("=" * W)
    print(
        f"  {'Origin':<8} {'Nearest Hospital':<36} "
        f"{'Drive':>7} {'No Preemp':>10} {'Preemp':>9} "
        f"{'Saved(s)':>9} {'Improv%':>8}"
    )
    print("  " + "-" * W)

    non_hosp = [n for n in sorted(graph.nodes.keys()) if n not in HOSPITAL_NODES]

    for origin in non_hosp:
        res = emergency_response(graph, store, origin, period)
        if not res.astar_result.is_valid:
            print(f"  {origin:<8}  UNREACHABLE")
            continue
        saved_s    = res.time_saved_min * 60
        hosp_label = f"{res.hospital_name} ({res.hospital_id})"
        print(
            f"  {origin:<8} {hosp_label:<36} "
            f"{res.astar_travel_min:>6.1f}m "
            f"{res.total_time_without_min:>9.2f}m "
            f"{res.total_time_with_min:>8.2f}m "
            f"{saved_s:>8.1f}s "
            f"{res.improvement_pct:>7.1f}%"
        )

    print("=" * W)