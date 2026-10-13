import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional

from Neuron import Neuron
from Orientation_membrane import Orientation_membrane
from InfluenceEntity import InfluenceEntity, DangerEntity, AttractorEntity


class EnvironmentStrategy(ABC):
    @abstractmethod
    def update_neurons_cost(self, neurons: List[Neuron]) -> None:
        """
        Receives a list of neurons and updates their k_fear and k_desire
        based on the environment's internal representation (Hash, Costmap, etc.).
        """
        pass


class SpatialHashEnvironment(EnvironmentStrategy):
    def __init__(self, sector_size: float):
        self.sector_size = sector_size
        # The environment holds a list of all active dynamic entities
        self.active_entities: List[InfluenceEntity] = []

    def add_entity(self, entity: InfluenceEntity) -> None:
        self.active_entities.append(entity)

    def update_neurons_cost(self, neurons: List[Neuron]) -> None:
        """
        Checks interactions between neurons and dynamic entities.
        (A true spatial hash would partition active_entities into grid buckets to avoid O(N*M) checks).
        """
        for neuron in neurons:
            # 1. Reset state before calculating new frame's influences
            neuron.k_fear = 1.0
            neuron.k_desire = 1.0

            # 2. Iterate through objects (in a fully optimized C++ version, only objects in the same/adjacent hash buckets)
            for entity in self.active_entities:
                # Calculate physical Euclidean distance from neuron to the object's center
                dist = neuron.get_euclidean_distance(entity.position)

                # 3. If the neuron is inside the object's radius, apply its specific math
                if dist < entity.radius:
                    entity.apply_influence(neuron, dist)


class CostmapEnvironment(EnvironmentStrategy):
    def __init__(self, map_matrix: np.ndarray, resolution: float):
        self.map_matrix = map_matrix
        self.resolution = resolution

    def update_neurons_cost(self, neurons: List[Neuron]) -> None:
        """
        O(1) approach: Directly accesses a static 2D array.
        Best for static geometry (walls, terrain).
        """
        # Determine max indices to prevent out-of-bounds errors
        max_y = self.map_matrix.shape[0] - 1
        max_x = self.map_matrix.shape[1] - 1

        for neuron in neurons:
            # Convert physical coordinates to matrix indices
            idx_x = int(neuron.position[0] / self.resolution)
            idx_y = int(neuron.position[1] / self.resolution)

            # Check if the neuron is within the bounds of the costmap
            if 0 <= idx_x <= max_x and 0 <= idx_y <= max_y:
                cost = self.map_matrix[idx_y][idx_x]

                # Logic: values >= 1.0 are penalties, values < 1.0 are attractors
                if cost >= 1.0:
                    neuron.k_fear = cost
                    neuron.k_desire = 1.0
                else:
                    neuron.k_fear = 1.0
                    # Clamp at 0.01 to prevent distance from becoming zero
                    neuron.k_desire = max(0.01, cost)
            else:
                # If the neuron goes off the map, treat the void as an absolute wall
                neuron.k_fear = 100.0
                neuron.k_desire = 1.0
