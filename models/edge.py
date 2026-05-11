"""
models/edge.py
Cairo Transportation Network — Edge Model

Represents a bidirectional road segment between two nodes.
Edges carry multiple data layers:
  — static geometry   (distance, capacity, condition)
  — time-varying flow (traffic volume per time period)
  — speed limits      (normal + rush-hour, used for travel-time computation)
  — maintenance data  (cost + priority, used by DP Knapsack)
  — a potential-road flag for MST cost analysis

All travel-time computation lives here so every algorithm that
needs a weight simply calls edge.get_weight(period) and gets back
minutes — no unit conversion scattered across the codebase.


"""

from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Time-period enumeration
# ---------------------------------------------------------------------------

class TimePeriod(Enum):
    """
    The four daily traffic windows defined in the dataset.

    Each period maps to a different traffic-flow column and, consequently,
    to a different congestion factor and travel time.
    """
    MORNING_PEAK  = "morning"    # 07:00 – 09:00  heavy congestion
    AFTERNOON     = "afternoon"  # 09:00 – 16:00  normal flow
    EVENING_PEAK  = "evening"    # 16:00 – 19:00  heavy congestion
    NIGHT         = "night"      # 19:00 – 07:00  light traffic

    @classmethod
    def from_string(cls, label: str) -> "TimePeriod":
        mapping = {
            "morningpeak":  cls.MORNING_PEAK,
            "morning":      cls.MORNING_PEAK,
            "morning_peak": cls.MORNING_PEAK,
            "afternoon":    cls.AFTERNOON,
            "eveningpeak":  cls.EVENING_PEAK,
            "evening":      cls.EVENING_PEAK,
            "evening_peak": cls.EVENING_PEAK,
            "night":        cls.NIGHT,
        }
        key = label.strip().lower().replace(" ", "")
        if key not in mapping:
            raise ValueError(f"Unknown time period: '{label}'")
        return mapping[key]

    def label(self) -> str:
        labels = {
            TimePeriod.MORNING_PEAK: "Morning Peak  (07:00 – 09:00)",
            TimePeriod.AFTERNOON:    "Afternoon     (09:00 – 16:00)",
            TimePeriod.EVENING_PEAK: "Evening Peak  (16:00 – 19:00)",
            TimePeriod.NIGHT:        "Night         (19:00 – 07:00)",
        }
        return labels[self]

    @property
    def is_rush(self) -> bool:
        return self in (TimePeriod.MORNING_PEAK, TimePeriod.EVENING_PEAK)


# ---------------------------------------------------------------------------
# Congestion thresholds (from Section 3C of the dataset)
# ---------------------------------------------------------------------------

class CongestionLevel(Enum):
    CLEAR    = "clear"     # ratio < 0.75  → factor = 1.0
    MODERATE = "moderate"  # ratio >= 0.75 → factor = 1.8
    SEVERE   = "severe"    # ratio >= 0.90 → factor = 3.0

    def factor(self) -> float:
        return {
            CongestionLevel.CLEAR:    1.0,
            CongestionLevel.MODERATE: 1.8,
            CongestionLevel.SEVERE:   3.0,
        }[self]

    @classmethod
    def from_ratio(cls, ratio: float) -> "CongestionLevel":
        if ratio >= 0.90:
            return cls.SEVERE
        if ratio >= 0.75:
            return cls.MODERATE
        return cls.CLEAR


# ---------------------------------------------------------------------------
# Edge (road segment)
# ---------------------------------------------------------------------------

class Edge:
    """
    A weighted, bidirectional road connecting two nodes.

    Parameters
    ----------
    from_id      : str   — ID of the first endpoint node
    to_id        : str   — ID of the second endpoint node
    distance     : float — length in km
    capacity     : int   — maximum flow in vehicles/hour
    condition    : int   — road condition score 1–10 (10 = perfect)
    is_potential : bool  — True for roads that don't exist yet (2C dataset)
    cost_millions: float — construction cost in million EGP (potential roads only)

    Optional (set after construction by Graph._load_*):
    normal_speed   : float — km/h under free-flow conditions
    rush_speed     : float — km/h during peak hours
    flow_morning   : int   — measured traffic flow during morning peak (veh/h)
    flow_afternoon : int
    flow_evening   : int
    flow_night     : int
    maint_cost     : float — annual maintenance cost (million EGP)
    maint_priority : int   — maintenance priority score 1–5
    """

    def __init__(
        self,
        from_id:       str,
        to_id:         str,
        distance:      float,
        capacity:      int,
        condition:     int,
        is_potential:  bool  = False,
        cost_millions: float = 0.0,
    ):
        # Core identity
        self.from_id      = from_id.strip()
        self.to_id        = to_id.strip()
        self.distance     = distance      # km
        self.capacity     = capacity      # veh/h
        self.condition    = condition     # 1–10
        self.is_potential = is_potential  # doesn't physically exist yet
        self.cost_millions = cost_millions

        # Speed limits (populated by Graph._load_speeds)
        self.normal_speed: float = 60.0   # sensible default
        self.rush_speed:   float = 30.0

        # Traffic flow per period (populated by Graph._load_traffic_flow)
        self.flow_morning:   int = 0
        self.flow_afternoon: int = 0
        self.flow_evening:   int = 0
        self.flow_night:     int = 0

        # Maintenance data (populated by Graph._load_maintenance)
        self.maint_cost:     float = 0.0
        self.maint_priority: int   = 1

    # ------------------------------------------------------------------
    # Canonical edge key — lets Graph look up edges in O(1)
    # ------------------------------------------------------------------

    @property
    def key(self) -> str:
        """
        A consistent, order-independent identifier for this road.
        Always returns the lexicographically smaller node ID first,
        so edge (A→B) and edge (B→A) share the same key.
        """
        a, b = sorted([self.from_id, self.to_id])
        return f"{a}-{b}"

    # ------------------------------------------------------------------
    # Flow accessor
    # ------------------------------------------------------------------

    def get_flow(self, period: TimePeriod) -> int:
        """Return the measured traffic volume for the given time period."""
        return {
            TimePeriod.MORNING_PEAK: self.flow_morning,
            TimePeriod.AFTERNOON:    self.flow_afternoon,
            TimePeriod.EVENING_PEAK: self.flow_evening,
            TimePeriod.NIGHT:        self.flow_night,
        }[period]

    # ------------------------------------------------------------------
    # Congestion computation
    # ------------------------------------------------------------------

    def congestion_level(self, period: TimePeriod) -> CongestionLevel:
        """Classify the current congestion based on flow-to-capacity ratio."""
        if self.capacity == 0:
            return CongestionLevel.SEVERE
        ratio = self.get_flow(period) / self.capacity
        return CongestionLevel.from_ratio(ratio)

    def congestion_ratio(self, period: TimePeriod) -> float:
        """Return the raw flow / capacity ratio (0.0 – 1.0+)."""
        if self.capacity == 0:
            return 1.0
        return self.get_flow(period) / self.capacity

    # ------------------------------------------------------------------
    # Travel time — the edge weight used by all routing algorithms
    # ------------------------------------------------------------------

    def get_weight(self, period: TimePeriod) -> float:
        """
        Compute travel time in minutes for the given time period.

        Formula (from Section 3C of the dataset):
            travel_time = (distance / speed) * 60 * congestion_factor

        Speed selection:
          — Morning Peak / Evening Peak → rush_speed
          — Afternoon / Night           → normal_speed

        Returns
        -------
        float — travel time in minutes (always > 0)
        """
        is_rush = period in (TimePeriod.MORNING_PEAK, TimePeriod.EVENING_PEAK)
        speed   = self.rush_speed if is_rush else self.normal_speed

        # Guard against zero-speed data issues
        if speed <= 0:
            speed = 10.0

        base_time         = (self.distance / speed) * 60.0
        congestion_factor = self.congestion_level(period).factor()

        return base_time * congestion_factor

    def get_weight_summary(self, period: TimePeriod) -> dict:
        """
        Return a full diagnostic breakdown of the weight computation.
        Used by main.py to display per-road detail in routing results.
        """
        is_rush = period in (TimePeriod.MORNING_PEAK, TimePeriod.EVENING_PEAK)
        speed   = self.rush_speed if is_rush else self.normal_speed
        if speed <= 0:
            speed = 10.0

        level = self.congestion_level(period)
        return {
            "edge":              self.key,
            "distance_km":       self.distance,
            "speed_kmh":         speed,
            "flow_veh_h":        self.get_flow(period),
            "capacity_veh_h":    self.capacity,
            "congestion_ratio":  round(self.congestion_ratio(period), 3),
            "congestion_level":  level.name,
            "congestion_factor": level.factor(),
            "travel_time_min":   round(self.get_weight(period), 2),
        }

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        kind = "POTENTIAL" if self.is_potential else "road"
        return (
            f"Edge({self.from_id}↔{self.to_id}, "
            f"{self.distance} km, cap={self.capacity}, "
            f"cond={self.condition}/10, [{kind}])"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return NotImplemented
        return self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __lt__(self, other: "Edge") -> bool:
        """Allow edges to be sorted by distance (used in Kruskal's)."""
        return self.distance < other.distance
