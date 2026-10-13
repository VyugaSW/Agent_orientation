import math
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from Neuron import Neuron


class TopologyStrategy(ABC):
    @abstractmethod
    def build_grid(
        self, rows: int, cols: int, cell_size: float, start_x: float, start_y: float
    ) -> List[List[Neuron]]:
        pass

    def get_periods(self, rows: int, cols: int, cell_size: float) -> Tuple[float, float]:
        """Returns (radar_width, radar_height) for proper toroidal wrap-around."""
        return cols * cell_size, rows * cell_size


class ToroidalSquareTopology(TopologyStrategy):
    """4-way connected square grid with Toroidal (wrap-around) links."""

    def build_grid(
        self, rows: int, cols: int, cell_size: float, start_x: float, start_y: float
    ) -> List[List[Neuron]]:
        grid: List[List[Neuron]] = []
        center_r = (rows - 1) / 2.0
        center_c = (cols - 1) / 2.0

        # Step 1: Physical placement
        for r in range(rows):
            row_neurons: List[Neuron] = []
            for c in range(cols):
                off_x = (c - center_c) * cell_size
                off_y = (r - center_r) * cell_size
                row_neurons.append(
                    Neuron(start_x + off_x, start_y + off_y, off_x, off_y)
                )
            grid.append(row_neurons)

        # Step 2: Toroidal topological linking
        for r in range(rows):
            for c in range(cols):
                current = grid[r][c]
                current.add_neighbor(grid[r][(c + 1) % cols])
                current.add_neighbor(grid[(r + 1) % rows][c])

        return grid


class ToroidalHexagonalTopology(TopologyStrategy):
    """6-way connected hexagonal grid with Toroidal (wrap-around) links."""

    def build_grid(
        self, rows: int, cols: int, cell_size: float, start_x: float, start_y: float
    ) -> List[List[Neuron]]:
        if rows % 2 != 0:
            raise ValueError("ToroidalHexagonalTopology requires an even count of rows.")

        grid: List[List[Neuron]] = []
        center_r = (rows - 1) / 2.0
        center_c = (cols - 1) / 2.0
        row_height = cell_size * (math.sqrt(3.0) / 2.0)

        # Step 1: Physical placement
        for r in range(rows):
            row_neurons: List[Neuron] = []
            row_offset = (cell_size / 2.0) if r % 2 != 0 else 0.0
            for c in range(cols):
                off_x = (c - center_c) * cell_size + row_offset
                off_y = (r - center_r) * row_height
                row_neurons.append(
                    Neuron(start_x + off_x, start_y + off_y, off_x, off_y)
                )
            grid.append(row_neurons)

        # Step 2: Toroidal topological linking (3 forward connections yield 6 neighbors total)
        for r in range(rows):
            for c in range(cols):
                current = grid[r][c]
                right_c = (c + 1) % cols
                current.add_neighbor(grid[r][right_c])

                bottom_r = (r + 1) % rows
                if r % 2 == 0:
                    bottom_left_c = (c - 1) % cols
                    bottom_right_c = c
                else:
                    bottom_left_c = c
                    bottom_right_c = (c + 1) % cols

                current.add_neighbor(grid[bottom_r][bottom_left_c])
                current.add_neighbor(grid[bottom_r][bottom_right_c])

        return grid

    def get_periods(self, rows: int, cols: int, cell_size: float) -> Tuple[float, float]:
        row_height = cell_size * (math.sqrt(3.0) / 2.0)
        return cols * cell_size, rows * row_height


class ParameterizedMorphologyTopology(TopologyStrategy):
    """
    Decorator over any base TopologyStrategy (Square or Hexagonal) controlled by a genome:
    - scale_x, scale_y: anisotropic stretching along X and Y axes
    - curvature: crescent/parabolic bending of the membrane front
    - foveation: radial density concentration towards the center (gamma > 1) or periphery (gamma < 1)
    """

    def __init__(
        self,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        curvature: float = 0.0,
        foveation: float = 1.0,
        base_topology: Optional[TopologyStrategy] = None,
    ):
        # Unwrap nested decorators to preserve the original underlying topology
        while isinstance(base_topology, ParameterizedMorphologyTopology):
            base_topology = base_topology.base_topology

        self.base_topology: TopologyStrategy = (
            base_topology if base_topology is not None else ToroidalHexagonalTopology()
        )
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.curvature = curvature
        self.foveation = foveation

    def build_grid(
        self, rows: int, cols: int, cell_size: float, start_x: float, start_y: float
    ) -> List[List[Neuron]]:
        if isinstance(self.base_topology, ToroidalHexagonalTopology) and rows % 2 != 0:
            rows += 1

        grid = self.base_topology.build_grid(rows, cols, cell_size, start_x, start_y)
        max_radius = max(1e-5, math.hypot(cols * cell_size * 0.5, rows * cell_size * 0.5))

        for r in range(len(grid)):
            for c in range(len(grid[r])):
                n = grid[r][c]
                ox, oy = n.base_offset[0], n.base_offset[1]

                # Non-linear radial foveation
                rad = math.hypot(ox, oy)
                if rad > 1e-5 and abs(self.foveation - 1.0) > 1e-3:
                    norm_r = rad / max_radius
                    factor = (norm_r ** self.foveation) / norm_r
                    ox *= factor
                    oy *= factor

                # Anisotropic scaling and parabolic curvature
                ox_new = ox * self.scale_x + self.curvature * (oy / max_radius) ** 2 * cell_size
                oy_new = oy * self.scale_y

                n.base_offset[0] = ox_new
                n.base_offset[1] = oy_new
                n.position[0] = start_x + ox_new
                n.position[1] = start_y + oy_new
                n.anchor[:] = n.position

        return grid

    def get_periods(self, rows: int, cols: int, cell_size: float) -> Tuple[float, float]:
        w, h = self.base_topology.get_periods(rows, cols, cell_size)
        return w * self.scale_x, h * self.scale_y