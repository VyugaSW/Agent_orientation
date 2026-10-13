import numpy as np
from abc import ABC, abstractmethod
from typing import List

from Neuron import Neuron


class TopologyStrategy(ABC):
    @abstractmethod
    def build_grid(
        self, rows: int, cols: int, cell_size: float, start_x: float, start_y: float
    ) -> List[List["Neuron"]]:
        pass


# 4-way connected square grid with Toroidal (wrap-around) links
class ToroidalSquareTopology(TopologyStrategy):
    def build_grid(
        self, rows: int, cols: int, cell_size: float, start_x: float, start_y: float
    ) -> List[List["Neuron"]]:
        grid = []

        center_r = (rows - 1) / 2.0
        center_c = (cols - 1) / 2.0

        # Step 1: Physical placement
        for r in range(rows):
            row_neurons = []
            for c in range(cols):
                pos_x = start_x + (c - center_c) * cell_size
                pos_y = start_y + (r - center_r) * cell_size
                row_neurons.append(Neuron(pos_x, pos_y))
            grid.append(row_neurons)

        # Step 2: Toroidal topological linking
        for r in range(rows):
            for c in range(cols):
                current = grid[r][c]

                # Right neighbor (wraps around to 0 if at the right edge)
                right_neighbor = grid[r][(c + 1) % cols]
                current.add_neighbor(right_neighbor)

                # Bottom neighbor (wraps around to 0 if at the bottom edge)
                bottom_neighbor = grid[(r + 1) % rows][c]
                current.add_neighbor(bottom_neighbor)

        return grid



# 6-way connected hexagonal grid with Toroidal (wrap-around) links
class ToroidalHexagonalTopology(TopologyStrategy):
    def build_grid(
        self, rows: int, cols: int, cell_size: float, start_x: float, start_y: float
    ) -> List[List["Neuron"]]:
        if rows % 2 != 0:
            raise ValueError("ToroidalHexagonalTopology requres even count of rows.")
            
        grid = []
        center_r = (rows - 1) / 2.0
        center_c = (cols - 1) / 2.0
        row_height = cell_size * (np.sqrt(3) / 2.0)

        # Step 1: Physical placement
        for r in range(rows):
            row_neurons = []
            row_offset = (cell_size / 2.0) if r % 2 != 0 else 0.0
            for c in range(cols):
                pos_x = start_x + (c - center_c) * cell_size + row_offset
                pos_y = start_y + (r - center_r) * row_height
                row_neurons.append(Neuron(pos_x, pos_y))
            grid.append(row_neurons)

        # Step 2: Toroidal topological linking (3 forward connections yield 6 neighbors total)
        for r in range(rows):
            for c in range(cols):
                current = grid[r][c]

                # Right neighbor (wraps around to 0 if at the right edge)
                right_c = (c + 1) % cols
                current.add_neighbor(grid[r][right_c])

                # Bottom neighbors (wraps around to 0 if at the bottom edge)
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