import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
from Neuron import Neuron


class InfluenceEntity(ABC):
    def __init__(
        self,
        x: float,
        y: float,
        radius: float,
        peak_strength: float,
        vx: float = 0.0,
        vy: float = 0.0,
        blink_period: int = 0,
        blink_offset: int = 0,
    ):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([vx, vy], dtype=float)
        self.radius = float(radius)
        self.influence_radius = float(radius)
        # peak_strength determines the maximum impact at the absolute center of the entity
        self.peak_strength = float(peak_strength)

        # Dynamic blinking (appearing/disappearing) state
        self.blink_period = int(blink_period)
        self.tick_counter = int(blink_offset)
        self.is_active: bool = True

    def step(self, bounds: Tuple[float, float, float, float] = (3.0, 77.0, 3.0, 77.0)) -> bool:
        """
        Updates entity position and blinking state.
        Returns True if the spatial hash buckets need to be updated.
        """
        changed = False

        # Handle appearing / disappearing cycle
        if self.blink_period > 0:
            self.tick_counter += 1
            new_active = ((self.tick_counter // self.blink_period) % 2) == 0
            if new_active != self.is_active:
                self.is_active = new_active
                changed = True

        # Handle physical movement and wall bouncing
        if self.velocity[0] != 0.0 or self.velocity[1] != 0.0:
            self.position += self.velocity
            min_x, max_x, min_y, max_y = bounds

            if self.position[0] - self.radius < min_x:
                self.position[0] = min_x + self.radius
                self.velocity[0] *= -1.0
            elif self.position[0] + self.radius > max_x:
                self.position[0] = max_x - self.radius
                self.velocity[0] *= -1.0

            if self.position[1] - self.radius < min_y:
                self.position[1] = min_y + self.radius
                self.velocity[1] *= -1.0
            elif self.position[1] + self.radius > max_y:
                self.position[1] = max_y - self.radius
                self.velocity[1] *= -1.0

            changed = True

        return changed

    @abstractmethod
    def apply_influence(self, neuron: Neuron, distance: float) -> None:
        """
        Calculates and applies the specific multiplier to the neuron based on distance.
        """
        pass


class DangerEntity(InfluenceEntity):
    def __init__(
        self,
        x: float,
        y: float,
        radius: float,
        peak_strength: float,
        influence_multiplier: float = 1.0,
        vx: float = 0.0,
        vy: float = 0.0,
        blink_period: int = 0,
        blink_offset: int = 0,
    ):
        super().__init__(
            x,
            y,
            radius,
            peak_strength,
            vx=vx,
            vy=vy,
            blink_period=blink_period,
            blink_offset=blink_offset,
        )
        self.influence_radius = self.radius * influence_multiplier

    def apply_influence(self, neuron: Neuron, distance: float) -> None:
        if not self.is_active:
            return

        if distance < self.influence_radius:
            intensity = self.peak_strength * (1.0 - (distance / self.influence_radius))
            neuron.k_fear = max(neuron.k_fear, 1.0 + intensity)

            if 0.01 < distance < self.radius:
                repel_vector = neuron.position - self.position
                normalized_repel = repel_vector / distance
                push_force = max(0.0, 5.0 * (1.0 - (distance / self.radius)))
                neuron.position += normalized_repel * push_force


class AttractorEntity(InfluenceEntity):
    def apply_influence(self, neuron: Neuron, distance: float) -> None:
        if not self.is_active:
            return

        if distance < self.radius:
            # For desires, strength reduces the multiplier below 1.0
            intensity = self.peak_strength * (1.0 - (distance / self.radius))

            # Clamp at 0.01 to prevent distance from becoming mathematically zero or negative
            calculated_desire = max(0.01, 1.0 - intensity)

            # Use min() so overlapping attractors reinforce each other
            neuron.k_desire = min(neuron.k_desire, calculated_desire)