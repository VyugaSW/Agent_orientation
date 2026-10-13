import math
import numpy as np
from typing import List


class Neuron:
    def __init__(self, x: float, y: float, offset_x: float = 0.0, offset_y: float = 0.0):
        # Physical position in 2D space
        self.position = np.array([x, y], dtype=float)

        # Base offset relative to the membrane center (used for reset_grid and morphology updates)
        self.base_offset = np.array([offset_x, offset_y], dtype=float)

        # Toroidal anchor in world coordinates (ensures proper homeostasis after teleportation)
        self.anchor = np.array([x, y], dtype=float)

        # Topological connections
        self.neighbors: List["Neuron"] = []

        # Environmental and instinctual modifiers
        self.k_fear: float = 1.0    # Increases apparent distance (> 1.0 in danger zones)
        self.k_desire: float = 1.0  # Decreases apparent distance (< 1.0 in zones of interest)

    def add_neighbor(self, neuron: "Neuron") -> None:
        """Adds a bidirectional topological link."""
        if neuron not in self.neighbors:
            self.neighbors.append(neuron)
            neuron.neighbors.append(self)

    def get_euclidean_distance(self, target_coords: np.ndarray) -> float:
        """Calculates 2D Euclidean distance using fast math.hypot."""
        dx = float(self.position[0] - target_coords[0])
        dy = float(self.position[1] - target_coords[1])
        return math.hypot(dx, dy)

    def get_modified_distance(self, target_coords: np.ndarray) -> float:
        """
        Calculates apparent distance applying both fear and desire multipliers.
        Clamps minimum base distance under fear to prevent zero-multiplication artifacts.
        """
        base_distance = self.get_euclidean_distance(target_coords)
        if self.k_fear > 1.0 and base_distance < 0.5:
            base_distance = 0.5
        return base_distance * self.k_fear * self.k_desire

    def pull_towards(self, target_coords: np.ndarray, weight: float) -> None:
        """Shifts the neuron's physical position towards the target."""
        self.position += (target_coords - self.position) * weight