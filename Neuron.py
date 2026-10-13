import numpy as np
from typing import List


class Neuron:
    def __init__(self, x: float, y: float):
        # Physical position in the 2D space
        self.position = np.array([x, y], dtype=float)

        # Topological connections
        self.neighbors: List["Neuron"] = []

        # Environmental and instinctual modifiers
        self.k_fear = 1.0  # Increases apparent distance (> 1.0 in danger zones)
        self.k_desire = 1.0  # Decreases apparent distance (< 1.0 in zones of interest)

    def add_neighbor(self, neuron: "Neuron") -> None:
        """Adds a bidirectional topological link."""
        if neuron not in self.neighbors:
            self.neighbors.append(neuron)
            neuron.neighbors.append(self)

    def get_euclidean_distance(self, target_coords: np.ndarray) -> float:
        """Calculates standard Euclidean distance."""
        return float(np.linalg.norm(self.position - target_coords))

    def get_modified_distance(self, target_coords: np.ndarray) -> float:
        """
        Calculates apparent distance applying both fear and desire multipliers.
        """
        base_distance = self.get_euclidean_distance(target_coords)
        dist = base_distance * self.k_fear * self.k_desire

        if self.k_fear > 1.0:
            dist += (self.k_fear - 1.0) * 100.0

        return dist

    def pull_towards(self, target_coords: np.ndarray, weight: float) -> None:
        """Shifts the neuron's position towards the target."""
        direction_vector = target_coords - self.position
        self.position += direction_vector * weight
