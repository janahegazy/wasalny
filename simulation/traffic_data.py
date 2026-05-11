"""
simulation/traffic_data.py
Cairo Transportation Network — Temporal Traffic Data Store

Purpose
-------
A dedicated data structure for storing and querying all time-varying
traffic data in the network. Algorithm modules should query traffic
through this store rather than reaching into edge attributes directly.

Data model
----------
The store is a two-level lookup table:

    _flow_table  : dict[ edge_key → list[int] ]
                   list is indexed by TimePeriod (0=morning, 1=afternoon,
                   2=evening, 3=night) — O(1) insert and O(1) query.

    _speed_table : dict[ edge_key → (normal_kmh, rush_kmh) ]
                   O(1) query by edge key + period.

    _signal_table: dict[ intersection_id → SignalConfig ]
                   Intersection-level timing data for the Greedy algorithm.

    _schedule_table: nested dict for bus/metro scheduling data (DP).

-------------------------------------------------
Keeping temporal data in the store and static geometry in Edge keeps
responsibilities separate. Algorithms can be given just the store (or
a modified version of it) without touching the graph structure at all.
This enables clean simulation: swap the store, same graph.


"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing      import Dict, List, Optional, Tuple
from models.edge import TimePeriod, CongestionLevel


# ---------------------------------------------------------------------------
# Signal configuration (used by Greedy algorithm)
# ---------------------------------------------------------------------------

@dataclass
class SignalConfig:
    """Traffic signal timing for one intersection."""
    intersection_id:    str
    name:               str
    connected_roads:    List[str]      # list of edge keys
    normal_cycle_sec:   int            # total cycle duration
    green_by_period:    List[int]      # [morning, afternoon, evening, night] green seconds
    preempt_hold_sec:   int            # how long to hold green for emergency vehicle
    recover_cycle_sec:  int            # recovery time after preemption

    def green_time(self, period: TimePeriod) -> int:
        return self.green_by_period[_PERIOD_IDX[period]]


# ---------------------------------------------------------------------------
# Schedule entry (used by DP scheduling)
# ---------------------------------------------------------------------------

@dataclass
class TransitRoute:
    """A single bus or metro line with supply and demand data."""
    route_id:         str
    route_type:       str            # 'bus' or 'metro'
    stops:            List[str]      # node IDs in order
    buses_assigned:   int
    daily_passengers: int
    # Per-period supply: vehicles available per hour in each slot
    supply_by_period: List[int]      # [morning, afternoon, evening, night]
    vehicle_capacity: int            # passengers per vehicle


# ---------------------------------------------------------------------------
# Period index helper (mirrors Edge._PERIOD_INDEX)
# ---------------------------------------------------------------------------

_PERIOD_IDX: Dict[TimePeriod, int] = {
    TimePeriod.MORNING_PEAK: 0,
    TimePeriod.AFTERNOON:    1,
    TimePeriod.EVENING_PEAK: 2,
    TimePeriod.NIGHT:        3,
}

_ALL_PERIODS = [
    TimePeriod.MORNING_PEAK,
    TimePeriod.AFTERNOON,
    TimePeriod.EVENING_PEAK,
    TimePeriod.NIGHT,
]


# ---------------------------------------------------------------------------
# Main data store
# ---------------------------------------------------------------------------

class TrafficDataStore:
    """
    Central store for all time-varying traffic data.

    Usage by algorithm modules
    --------------------------
        store = TrafficDataStore()
        flow  = store.get_flow("3-5", TimePeriod.MORNING_PEAK)   # O(1)
        speed = store.get_speed("3-5", TimePeriod.MORNING_PEAK)  # O(1)
        level = store.get_congestion("3-5", cap, TimePeriod.MORNING_PEAK)
        weight = store.get_weight("3-5", cap, dist, TimePeriod.MORNING_PEAK)

    Simulation support
    ------------------
        # Temporarily inject a surge event on a road
        snapshot = store.snapshot_flow("3-5")
        store.set_flow("3-5", morning=4000, afternoon=3000,
                               evening=3800, night=900)
        # ... run algorithm ...
        store.restore_flow("3-5", snapshot)
    """

    def __init__(self):
        # Core lookup tables — all O(1) access by string key
        self._flow_table:   Dict[str, List[int]]          = {}
        self._speed_table:  Dict[str, Tuple[float, float]] = {}  # (normal, rush)
        self._signal_table: Dict[str, SignalConfig]        = {}
        self._routes:       Dict[str, TransitRoute]        = {}
        self._fleet:        Dict[TimePeriod, Tuple[int,int]] = {}  # (buses, trains)
        self._od_demand:    Dict[Tuple[str,str], int]      = {}    # (origin,dest)->passengers/day

        self._load_all()

    # ------------------------------------------------------------------
    # Public query interface — used by every algorithm
    # ------------------------------------------------------------------

    def get_flow(self, edge_key: str, period: TimePeriod) -> int:
        """O(1) flow lookup. Returns 0 if edge has no flow data."""
        row = self._flow_table.get(edge_key)
        if row is None:
            return 0
        return row[_PERIOD_IDX[period]]

    def get_speed(self, edge_key: str, period: TimePeriod) -> float:
        """O(1) speed lookup. Returns normal or rush speed based on period."""
        normal, rush = self._speed_table.get(edge_key, (60.0, 30.0))
        return rush if period.is_rush else normal

    def get_congestion_ratio(self, edge_key: str, capacity: int,
                             period: TimePeriod) -> float:
        """Flow-to-capacity ratio for the given period. O(1)."""
        if capacity == 0:
            return 1.0
        return self.get_flow(edge_key, period) / capacity

    def get_congestion_level(self, edge_key: str, capacity: int,
                             period: TimePeriod) -> CongestionLevel:
        """CongestionLevel enum for the given edge and period. O(1)."""
        return CongestionLevel.from_ratio(
            self.get_congestion_ratio(edge_key, capacity, period)
        )

    def get_weight(self, edge_key: str, capacity: int,
                   distance: float, period: TimePeriod) -> float:
        """
        Travel-time weight in minutes. O(1).
        Formula: (distance / speed) * 60 * congestion_factor
        """
        speed = self.get_speed(edge_key, period)
        if speed <= 0:
            speed = 10.0
        ratio  = self.get_congestion_ratio(edge_key, capacity, period)
        factor = CongestionLevel.from_ratio(ratio).factor()
        return (distance / speed) * 60.0 * factor

    def get_all_periods_weight(self, edge_key: str, capacity: int,
                               distance: float) -> Dict[TimePeriod, float]:
        """Return travel times for all four periods. Useful for reports."""
        return {p: self.get_weight(edge_key, capacity, distance, p)
                for p in _ALL_PERIODS}

    def get_signal(self, intersection_id: str) -> Optional[SignalConfig]:
        """Return the SignalConfig for an intersection, or None."""
        return self._signal_table.get(intersection_id)

    def get_all_signals(self) -> List[SignalConfig]:
        return list(self._signal_table.values())

    def get_route(self, route_id: str) -> Optional[TransitRoute]:
        return self._routes.get(route_id)

    def get_all_routes(self) -> List[TransitRoute]:
        return list(self._routes.values())

    def get_fleet(self, period: TimePeriod) -> Tuple[int, int]:
        """Return (available_buses, available_trains) for a period."""
        return self._fleet.get(period, (0, 0))

    def get_od_demand(self, origin: str, dest: str) -> int:
        """Daily passenger demand between an OD pair (symmetric)."""
        return self._od_demand.get((origin, dest),
               self._od_demand.get((dest, origin), 0))

    def get_all_od_demands(self) -> Dict[Tuple[str,str], int]:
        return dict(self._od_demand)

    # ------------------------------------------------------------------
    # Simulation mutation interface
    # ------------------------------------------------------------------

    def set_flow(self, edge_key: str, morning: int = 0, afternoon: int = 0,
                 evening: int = 0, night: int = 0) -> None:
        """Overwrite the flow for one edge (for event injection)."""
        self._flow_table[edge_key] = [morning, afternoon, evening, night]

    def snapshot_flow(self, edge_key: str) -> List[int]:
        """Return a copy of the current flow row for later restore."""
        return list(self._flow_table.get(edge_key, [0, 0, 0, 0]))

    def restore_flow(self, edge_key: str, snapshot: List[int]) -> None:
        """Restore a previously snapshotted flow row."""
        self._flow_table[edge_key] = list(snapshot)

    def apply_surge(self, edge_key: str, period: TimePeriod,
                    multiplier: float) -> List[int]:
        """
        Multiply traffic flow on one edge for one period by `multiplier`.
        Returns the snapshot before the change so it can be restored.
        Useful for simulating a special event or accident overspill.
        """
        snap = self.snapshot_flow(edge_key)
        row  = list(snap)
        idx  = _PERIOD_IDX[period]
        row[idx] = int(row[idx] * multiplier)
        self._flow_table[edge_key] = row
        return snap

    # ------------------------------------------------------------------
    # Congestion report — used by main.py and Scenario demos
    # ------------------------------------------------------------------

    def congestion_report(self, edges, period: TimePeriod) -> List[dict]:
        """
        Build a sorted congestion report for a list of edges.
        Each edge in `edges` must have .key, .capacity, .distance attributes.
        Returns list of dicts sorted by congestion ratio descending.
        """
        rows = []
        for e in edges:
            ratio = self.get_congestion_ratio(e.key, e.capacity, period)
            level = CongestionLevel.from_ratio(ratio)
            rows.append({
                "edge":    e.key,
                "ratio":   ratio,
                "level":   level,
                "flow":    self.get_flow(e.key, period),
                "cap":     e.capacity,
                "weight":  self.get_weight(e.key, e.capacity, e.distance, period),
            })
        rows.sort(key=lambda r: r["ratio"], reverse=True)
        return rows

    # ------------------------------------------------------------------
    # Data loading — all from the dataset
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        self._load_flow()
        self._load_speeds()
        self._load_signals()
        self._load_routes()
        self._load_fleet()
        self._load_od_demand()

    def _load_flow(self) -> None:
        """
        Section 3A + 3B — traffic flow per edge per period.
        Stored as { edge_key: [morning, afternoon, evening, night] }.
        """
        raw = [
            # (from, to,  morning, afternoon, evening, night)
            # --- Section 3A: original roads ---
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
            # --- Section 3B: hospital access roads ---
            ("F9",  "3",   800,  600,  750, 200),
            ("F9",  "6",   600,  450,  580, 150),
            ("F9",  "10",  700,  500,  650, 180),
            ("F10", "1",   700,  520,  670, 190),
            ("F10", "12",  550,  400,  520, 140),
            ("F10", "8",   750,  550,  700, 200),
        ]
        for a, b, *flows in raw:
            key = "-".join(sorted([a, b]))
            self._flow_table[key] = flows

    def _load_speeds(self) -> None:
        """Section 4 — speed limits per road."""
        raw = [
            ("1",  "3",   60,  30), ("1",  "8",   70,  35),
            ("2",  "3",   60,  25), ("2",  "5",   80,  40),
            ("3",  "5",   60,  25), ("3",  "6",   50,  20),
            ("3",  "9",   60,  25), ("3",  "10",  60,  25),
            ("4",  "2",  100,  60), ("4",  "14",  90,  55),
            ("5",  "11",  70,  35), ("6",  "9",   50,  20),
            ("7",  "8",  100,  60), ("7",  "15",  90,  55),
            ("8",  "10",  60,  30), ("8",  "12",  80,  45),
            ("9",  "10",  50,  20), ("10", "11",  60,  30),
            ("11", "F2",  60,  30), ("12", "1",   80,  45),
            ("13", "4",  120,  80), ("14", "13", 110,  75),
            ("15", "7",   90,  55), ("F1", "5",   90,  55),
            ("F1", "2",   90,  55), ("F2", "3",   50,  20),
            ("F7", "15",  80,  50), ("F8", "4",   90,  55),
            ("F9",  "3",  50,  30), ("F9",  "6",  50,  30),
            ("F9",  "10", 50,  30), ("F10", "1",  60,  35),
            ("F10", "12", 60,  35), ("F10", "8",  70,  40),
        ]
        for a, b, normal, rush in raw:
            key = "-".join(sorted([a, b]))
            self._speed_table[key] = (float(normal), float(rush))

    def _load_signals(self) -> None:
        """Section 6A/6B/6C — traffic signal data for 10 major intersections."""
        raw_signals = [
            # (id, name, roads, cycle, [morning,aft,eve,night] green, hold, recover)
            ("I1",  "Tahrir Square",
             ["1-3","2-3","3-6","3-9","3-10"], 120,
             [90, 45, 85, 30], 15, 45),
            ("I2",  "Nasr City Junction",
             ["2-3","2-5","2-4"], 100,
             [80, 40, 75, 25], 12, 40),
            ("I3",  "Giza Square",
             ["1-8","7-8","8-10","8-12"], 110,
             [85, 42, 80, 28], 13, 42),
            ("I4",  "Mohandessin Center",
             ["3-9","6-9","9-10"], 90,
             [70, 35, 65, 20], 10, 35),
            ("I5",  "Dokki Junction",
             ["3-10","8-10","9-10","10-11"], 100,
             [75, 38, 70, 22], 11, 38),
            ("I6",  "Heliopolis Hub",
             ["2-5","3-5","5-11","5-F1"], 110,
             [80, 40, 78, 25], 12, 40),
            ("I7",  "Ramses Junction",
             ["10-11","11-F2","3-F2"], 100,
             [75, 38, 72, 22], 11, 38),
            ("I8",  "October-Giza Ring",
             ["7-8","7-15","15-7"], 120,
             [85, 42, 80, 28], 13, 42),
            ("I9",  "New Cairo Gate",
             ["2-4","4-14","4-F8"], 100,
             [80, 40, 75, 25], 12, 40),
            ("I10", "Airport Junction",
             ["5-F1","2-F1"], 90,
             [70, 45, 68, 30], 10, 35),
        ]
        for sid, name, roads, cycle, greens, hold, recover in raw_signals:
            self._signal_table[sid] = SignalConfig(
                intersection_id  = sid,
                name             = name,
                connected_roads  = roads,
                normal_cycle_sec = cycle,
                green_by_period  = greens,
                preempt_hold_sec = hold,
                recover_cycle_sec= recover,
            )

    def _load_routes(self) -> None:
        """Sections 7A, 7B, 8A, 8B — transit routes with supply per period."""
        # Metro lines: peak frequency 3-4 min, off-peak 6-8 min, capacity 1000-1200
        metro_raw = [
            # (id, stops, daily_pass, peak_freq, offpeak_freq, cap)
            ("M1", ["12","1","3","F2","11"], 1500000, 3, 6,  1200),
            ("M2", ["11","F2","3","10","8"], 1200000, 3, 6,  1200),
            ("M3", ["F1","5","2","3","9"],    800000, 4, 8,  1000),
        ]
        # Supply = 60 / frequency (trains per hour per direction)
        for rid, stops, daily, peak_f, off_f, cap in metro_raw:
            peak_supply   = int(60 / peak_f)
            offpeak_supply= int(60 / off_f)
            self._routes[rid] = TransitRoute(
                route_id        = rid,
                route_type      = "metro",
                stops           = stops,
                buses_assigned  = 0,
                daily_passengers= daily,
                supply_by_period= [peak_supply, offpeak_supply, peak_supply, offpeak_supply],
                vehicle_capacity= cap,
            )

        # Bus routes: peak/off-peak frequency, 80 passengers per bus
        bus_raw = [
            # (id, stops, buses, daily, peak_freq, offpeak_freq)
            ("B1",  ["1","3","6","9"],       25, 35000, 10, 20),
            ("B2",  ["7","15","8","10","3"], 30, 42000,  8, 15),
            ("B3",  ["2","5","F1"],          20, 28000, 12, 25),
            ("B4",  ["4","14","2","3"],      22, 31000, 10, 20),
            ("B5",  ["8","12","1"],          18, 25000, 15, 30),
            ("B6",  ["11","5","2"],          24, 33000, 10, 20),
            ("B7",  ["13","4","14"],         15, 21000, 20, 40),
            ("B8",  ["F7","15","7"],         12, 17000, 25, 45),
            ("B9",  ["1","8","10","9","6"],  28, 39000, 10, 20),
            ("B10", ["F8","4","2","5"],      20, 28000, 12, 25),
        ]
        for rid, stops, buses, daily, peak_f, off_f in bus_raw:
            peak_supply   = int(60 / peak_f)
            offpeak_supply= int(60 / off_f)
            self._routes[rid] = TransitRoute(
                route_id        = rid,
                route_type      = "bus",
                stops           = stops,
                buses_assigned  = buses,
                daily_passengers= daily,
                supply_by_period= [peak_supply, offpeak_supply, peak_supply, offpeak_supply],
                vehicle_capacity= 80,
            )

    def _load_fleet(self) -> None:
        """Section 8C — total available fleet per time period."""
        self._fleet = {
            TimePeriod.MORNING_PEAK: (180, 24),
            TimePeriod.AFTERNOON:    (140, 18),
            TimePeriod.EVENING_PEAK: (175, 22),
            TimePeriod.NIGHT:        (100, 12),
        }

    def _load_od_demand(self) -> None:
        """Section 7C — daily passenger demand between OD pairs."""
        raw = [
            ("3",  "5",  15000), ("1",  "3",  12000), ("2",  "3",  18000),
            ("F2", "11", 25000), ("F1", "3",  20000), ("7",  "3",  14000),
            ("4",  "3",  16000), ("8",  "3",  22000), ("3",  "9",  13000),
            ("5",  "2",  17000), ("11", "3",  24000), ("12", "3",  11000),
            ("1",  "8",   9000), ("7",  "F7", 18000), ("4",  "F8", 12000),
            ("13", "3",   8000), ("14", "4",   7000),
        ]
        for o, d, demand in raw:
            self._od_demand[(o, d)] = demand
