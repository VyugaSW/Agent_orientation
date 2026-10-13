import numpy as np
from abc import ABC, abstractmethod
from typing import List
from Neuron import Neuron


class InfluenceEntity(ABC):
    def __init__(self, x: float, y: float, radius: float, peak_strength: float):
        self.position = np.array([x, y], dtype=float)
        self.radius = radius
        # peak_strength determines the maximum impact at the absolute center of the entity
        self.peak_strength = peak_strength

    @abstractmethod
    def apply_influence(self, neuron: Neuron, distance: float) -> None:
        """
        Calculates and applies the specific multiplier to the neuron based on distance.
        """
        pass


class DangerEntity(InfluenceEntity):
    def __init__(self, x: float, y: float, radius: float, peak_strength: float):
        super().__init__(x, y, radius, peak_strength)
        self.influence_radius = radius * 2.0

    def apply_influence(self, neuron: Neuron, distance: float) -> None:
        if distance < self.influence_radius:
            intensity = self.peak_strength * (1.0 - (distance / self.influence_radius))
            neuron.k_fear = max(neuron.k_fear, 1.0 + intensity)

            if distance > 0.01:
                repel_vector = neuron.position - self.position
                normalized_repel = repel_vector / distance

                push_force = 5 * (1.0 - (distance / self.radius))

                neuron.position += normalized_repel * push_force


class AttractorEntity(InfluenceEntity):
    def apply_influence(self, neuron: Neuron, distance: float) -> None:
        if distance < self.radius:
            # For desires, strength reduces the multiplier below 1.0
            # Example: peak_strength = 0.9 => desire drops to 0.1 at the center (highly attractive)
            intensity = self.peak_strength * (1.0 - (distance / self.radius))

            # Clamp at 0.01 to prevent distance from becoming mathematically zero or negative
            calculated_desire = max(0.01, 1.0 - intensity)

            # We use min() so if multiple food sources overlap, the spot becomes even more attractive
            neuron.k_desire = min(neuron.k_desire, calculated_desire)
