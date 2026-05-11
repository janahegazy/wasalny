"""
models/node.py
Cairo Transportation Network — Node Model

Represents a single location in the Greater Cairo network.
A node can be a residential district, a business hub, or a
critical facility (hospital, airport, transit hub).



"""

from enum import Enum, auto


# ---------------------------------------------------------------------------
# Node type classification
# ---------------------------------------------------------------------------

class NodeType(Enum):
    """
    Categorical label for every node in the network.
    Used by MST to identify which nodes need priority connectivity,
    and by A* to locate the nearest medical facility.
    """
    RESIDENTIAL  = auto()
    MIXED        = auto()
    BUSINESS     = auto()
    INDUSTRIAL   = auto()
    GOVERNMENT   = auto()
    MEDICAL      = auto()   # F9, F10  — must be reachable from every district
    AIRPORT      = auto()   # F1       — critical hub
    TRANSIT_HUB  = auto()   # F2       — critical hub
    EDUCATION    = auto()
    TOURISM      = auto()
    SPORTS       = auto()
    COMMERCIAL   = auto()

    @classmethod
    def from_string(cls, label: str) -> "NodeType":
        """Parse the raw string from the dataset into an enum value."""
        mapping = {
            "residential": cls.RESIDENTIAL,
            "mixed":       cls.MIXED,
            "business":    cls.BUSINESS,
            "industrial":  cls.INDUSTRIAL,
            "government":  cls.GOVERNMENT,
            "medical":     cls.MEDICAL,
            "airport":     cls.AIRPORT,
            "transit hub": cls.TRANSIT_HUB,
            "education":   cls.EDUCATION,
            "tourism":     cls.TOURISM,
            "sports":      cls.SPORTS,
            "commercial":  cls.COMMERCIAL,
        }
        key = label.strip().lower()
        if key not in mapping:
            raise ValueError(f"Unrecognised node type: '{label}'")
        return mapping[key]


# ---------------------------------------------------------------------------
# The node itself
# ---------------------------------------------------------------------------

class Node:
    """
    A vertex in the Cairo transportation graph.

    Attributes
    ----------
    node_id    : str   — unique identifier (e.g. '1', 'F9', 'F1')
    name       : str   — human-readable name
    node_type  : NodeType
    longitude  : float — WGS-84 longitude (used for A* heuristic)
    latitude   : float — WGS-84 latitude  (used for A* heuristic)
    population : int   — resident population (0 for non-residential nodes)
    is_critical: bool  — True if the node is a hospital, airport, or transit hub
    """

    # Nodes that must always be reachable — enforced by MST's priority phase
    _CRITICAL_IDS = frozenset({"F9", "F10", "F1", "F2", "13"})

    def __init__(
        self,
        node_id:   str,
        name:      str,
        node_type: NodeType,
        longitude: float,
        latitude:  float,
        population: int = 0,
    ):
        self.node_id    = node_id.strip()
        self.name       = name.strip()
        self.node_type  = node_type
        self.longitude  = longitude
        self.latitude   = latitude
        self.population = population

        # A node is critical if it appears in the hard-coded set OR
        # if it is typed as a medical facility or major transit node.
        self.is_critical: bool = (
            self.node_id in self._CRITICAL_IDS
            or self.node_type in (NodeType.MEDICAL, NodeType.AIRPORT, NodeType.TRANSIT_HUB)
        )

    # ------------------------------------------------------------------
    # Geometry helpers — used by the A* heuristic in shortest_path.py
    # ------------------------------------------------------------------

    def euclidean_distance_to(self, other: "Node") -> float:
        """
        Return the straight-line (aerial) distance to another node in km.

        Uses a flat-earth approximation scaled by 111 km/degree.
        This is admissible as a heuristic: road distance is always >= aerial distance.

        Parameters
        ----------
        other : Node — the destination node

        Returns
        -------
        float — aerial distance in kilometres
        """
        SCALE = 111.0  # km per degree (approximate for Cairo's latitude)
        delta_lon = (self.longitude - other.longitude) * SCALE
        delta_lat = (self.latitude  - other.latitude)  * SCALE
        return (delta_lon ** 2 + delta_lat ** 2) ** 0.5

    # ------------------------------------------------------------------
    # Dunder methods — useful for debugging and set operations
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        critical_flag = " [CRITICAL]" if self.is_critical else ""
        return (
            f"Node({self.node_id!r}, {self.name!r}, "
            f"{self.node_type.name}{critical_flag})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.node_id == other.node_id

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __lt__(self, other: "Node") -> bool:
        """
        Tie-breaking comparator for priority queues.
        When two nodes have identical f-scores in A*, Python needs a
        way to break the tie without comparing full Node objects.
        """
        return self.node_id < other.node_id
