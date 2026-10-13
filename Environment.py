import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple

from Neuron import Neuron
from InfluenceEntity import InfluenceEntity, DangerEntity, AttractorEntity


class EnvironmentStrategy(ABC):
    @abstractmethod
    def update_neurons_cost(self, neurons: List[Neuron]) -> None:
        """
        Receives a list of neurons and updates their k_fear and k_desire
        based on the environment's internal representation (Hash, Costmap, etc.).
        """
        pass

    @abstractmethod
    def get_nearby_obstacles(
        self, position: np.ndarray, query_radius: float
    ) -> List[InfluenceEntity]:
        """Returns active hazardous entities within query_radius of the given position."""
        pass

    def step_dynamic_entities(self) -> None:
        """Advances moving or blinking entities in the environment."""
        pass


class SpatialHashEnvironment(EnvironmentStrategy):
    def __init__(self, sector_size: float = 20.0):
        self.sector_size = max(1.0, float(sector_size))
        self.active_entities: List[InfluenceEntity] = []
        self._buckets: Dict[Tuple[int, int], List[InfluenceEntity]] = {}
        self._indexed_count: int = 0

    def add_entity(self, entity: InfluenceEntity) -> None:
        self.active_entities.append(entity)
        if getattr(entity, "is_active", True):
            self._insert_into_buckets(entity)
        self._indexed_count = len(self.active_entities)

    def remove_entity(self, entity: InfluenceEntity) -> None:
        """Removes a disappearing entity and updates the spatial hash."""
        if entity in self.active_entities:
            self.active_entities.remove(entity)
            self.rebuild_spatial_hash()

    def step_dynamic_entities(self) -> None:
        """Steps all dynamic entities and rebuilds buckets if any moved or blinked."""
        any_changed = False
        for entity in self.active_entities:
            if entity.step():
                any_changed = True
        if any_changed:
            self.rebuild_spatial_hash()

    def rebuild_spatial_hash(self) -> None:
        """Rebuilds the spatial hash bucket table for all currently active entities."""
        self._buckets.clear()
        for entity in self.active_entities:
            if getattr(entity, "is_active", True):
                self._insert_into_buckets(entity)
        self._indexed_count = len(self.active_entities)

    def _insert_into_buckets(self, entity: InfluenceEntity) -> None:
        r = max(entity.radius, getattr(entity, "influence_radius", entity.radius))
        min_bx = int(math.floor((entity.position[0] - r) / self.sector_size))
        max_bx = int(math.floor((entity.position[0] + r) / self.sector_size))
        min_by = int(math.floor((entity.position[1] - r) / self.sector_size))
        max_by = int(math.floor((entity.position[1] + r) / self.sector_size))

        for bx in range(min_bx, max_bx + 1):
            for by in range(min_by, max_by + 1):
                key = (bx, by)
                if key not in self._buckets:
                    self._buckets[key] = []
                self._buckets[key].append(entity)

    def _ensure_synced(self) -> None:
        if len(self.active_entities) != self._indexed_count:
            self.rebuild_spatial_hash()

    def update_neurons_cost(self, neurons: List[Neuron]) -> None:
        """O(1) average lookup using spatial hash grid buckets."""
        self._ensure_synced()

        for neuron in neurons:
            neuron.k_fear = 1.0
            neuron.k_desire = 1.0

            bx = int(math.floor(neuron.position[0] / self.sector_size))
            by = int(math.floor(neuron.position[1] / self.sector_size))

            candidates = self._buckets.get((bx, by), ())
            for entity in candidates:
                if not getattr(entity, "is_active", True):
                    continue
                dist = neuron.get_euclidean_distance(entity.position)
                if dist < entity.influence_radius:
                    entity.apply_influence(neuron, dist)

    def get_nearby_obstacles(
        self, position: np.ndarray, query_radius: float
    ) -> List[InfluenceEntity]:
        self._ensure_synced()
        min_bx = int(math.floor((position[0] - query_radius) / self.sector_size))
        max_bx = int(math.floor((position[0] + query_radius) / self.sector_size))
        min_by = int(math.floor((position[1] - query_radius) / self.sector_size))
        max_by = int(math.floor((position[1] + query_radius) / self.sector_size))

        seen: Set[int] = set()
        result: List[InfluenceEntity] = []
        for bx in range(min_bx, max_bx + 1):
            for by in range(min_by, max_by + 1):
                for entity in self._buckets.get((bx, by), ()):
                    eid = id(entity)
                    if (
                        eid not in seen
                        and isinstance(entity, DangerEntity)
                        and getattr(entity, "is_active", True)
                    ):
                        seen.add(eid)
                        result.append(entity)
        return result


class CostmapEnvironment(EnvironmentStrategy):
    def __init__(self, map_matrix: np.ndarray, resolution: float):
        self.map_matrix = map_matrix
        self.resolution = float(resolution)
        self.active_entities: List[InfluenceEntity] = []

    def update_neurons_cost(self, neurons: List[Neuron]) -> None:
        """
        O(1) approach: Directly accesses a static 2D array.
        Best for static geometry (walls, terrain).
        """
        max_y = self.map_matrix.shape[0] - 1
        max_x = self.map_matrix.shape[1] - 1

        for neuron in neurons:
            idx_x = int(math.floor(neuron.position[0] / self.resolution))
            idx_y = int(math.floor(neuron.position[1] / self.resolution))

            if 0 <= idx_x <= max_x and 0 <= idx_y <= max_y:
                cost = float(self.map_matrix[idx_y][idx_x])
                if cost >= 1.0:
                    neuron.k_fear = cost
                    neuron.k_desire = 1.0
                else:
                    neuron.k_fear = 1.0
                    neuron.k_desire = max(0.01, cost)
            else:
                # If the neuron goes off the map, treat the void as an absolute wall
                neuron.k_fear = 100.0
                neuron.k_desire = 1.0

    def get_nearby_obstacles(
        self, position: np.ndarray, query_radius: float
    ) -> List[InfluenceEntity]:
        return []