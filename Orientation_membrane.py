import numpy as np
from typing import List, Tuple, Optional

from Topology import TopologyStrategy, ToroidalSquareTopology
from Neuron import Neuron


class Orientation_membrane:
    def __init__(
        self,
        rows: int,
        cols: int,
        cell_size: float,
        start_x: float,
        start_y: float,
        topology: TopologyStrategy,
    ):
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.topology = topology
        self.center = np.array([start_x, start_y], dtype=float)

        # Delegate initialization to the injected topology strategy
        self.grid: List[List[Neuron]] = self.topology.build_grid(
            rows, cols, cell_size, start_x, start_y
        )

    def find_bmu(
        self, target_coords: np.ndarray, neurons: list = None
    ) -> Optional[Neuron]:
        """
        Finds the Best Matching Unit (BMU) for a given stimulus target.
        Utilizes the modified distance that accounts for k_fear and k_desire.
        """
        best_neuron = None
        min_dist = float("inf")

        # for r in range(self.rows):
        #     for c in range(self.cols):
        #         neuron = self.grid[r][c]
        #         dist = neuron.get_modified_distance(target_coords)

        #         if dist < min_dist:
        #             min_dist = dist
        #             best_neuron = neuron

        for n in neurons:
            neuron = n
            dist = neuron.get_modified_distance(target_coords)

            if dist < min_dist:
                min_dist = dist
                best_neuron = neuron

        return best_neuron

    def adapt_membrane(
        self,
        target_coords: np.ndarray,
        learning_rate: float = 0.1,
        radius_hops: int = 2,
        neurons: list = None,
    ) -> None:
        """
        Applies SOM pull to the BMU and its topological neighbors.
        Uses BFS with a physical distance cut-off to prevent Toroidal phantom links
        from ripping the physical mesh apart.
        """
        bmu = self.find_bmu(target_coords, neurons)
        if not bmu:
            return

        # Queue stores tuples of (Neuron, topological_distance_from_BMU)
        queue = [(bmu, 0)]
        visited = set([bmu])

        # Max acceptable physical distance between neighbors before considering it a phantom link
        # Cell size * 1.5 covers standard spacing and slight deformations, but blocks wrap-around links
        phantom_link_threshold = self.cell_size * 1.5

        while queue:
            current_neuron, hop_dist = queue.pop(0)

            # Calculate weight (Gaussian falloff based on topological hops)
            # hop_dist == 0 (BMU) gets weight 1.0 * learning_rate
            # hop_dist == 1 gets weight 0.5 * learning_rate, etc.
            falloff = np.exp(-(hop_dist**2) / 2.0)
            weight = learning_rate * falloff

            # Pull the current neuron physically towards the target
            current_neuron.pull_towards(target_coords, weight)

            # Stop propagating if we reached the edge of the influence radius
            if hop_dist >= radius_hops:
                continue

            # Check neighbors
            for neighbor in current_neuron.neighbors:
                if neighbor not in visited:
                    # RULE: The Scissors (Physical Cut-off)
                    # Measure pure Euclidean distance between connected nodes
                    dist = current_neuron.get_euclidean_distance(neighbor.position)

                    if dist < phantom_link_threshold:
                        visited.add(neighbor)
                        queue.append((neighbor, hop_dist + 1))

    def scroll_forward(self, step_vector: np.ndarray) -> None:
        """
        Implements Toroidal O(1) memory shift.
        We do not move array elements or change topological pointers.
        We only teleport the physical coordinates of neurons that fall behind.
        """
        # 1. Update the absolute center of the radar
        self.center += step_vector

        # 2. Calculate physical boundaries of the radar
        half_width = (self.cols / 2.0) * self.cell_size
        half_height = (self.rows / 2.0) * self.cell_size

        radar_width = self.cols * self.cell_size
        radar_height = self.rows * self.cell_size

        # 3. Teleport trailing neurons to the leading edge
        for r in range(self.rows):
            for c in range(self.cols):
                neuron = self.grid[r][c]

                # Teleport along X-axis
                if neuron.position[0] < self.center[0] - half_width:
                    neuron.position[0] += radar_width
                    neuron.k_fear = 1.0
                    neuron.k_desire = 1.0
                elif neuron.position[0] > self.center[0] + half_width:
                    neuron.position[0] -= radar_width
                    neuron.k_fear = 1.0
                    neuron.k_desire = 1.0

                # Teleport along Y-axis
                if neuron.position[1] < self.center[1] - half_height:
                    neuron.position[1] += radar_height
                    neuron.k_fear = 1.0
                    neuron.k_desire = 1.0
                elif neuron.position[1] > self.center[1] + half_height:
                    neuron.position[1] -= radar_height
                    neuron.k_fear = 1.0
                    neuron.k_desire = 1.0

    def reset_grid(self) -> None:
        """Restores the physical grid to a perfect un-deformed state around the center."""
        center_r = (self.rows - 1) / 2.0
        center_c = (self.cols - 1) / 2.0
        for r in range(self.rows):
            for c in range(self.cols):
                neuron = self.grid[r][c]
                neuron.position[0] = self.center[0] + (c - center_c) * self.cell_size
                neuron.position[1] = self.center[1] + (r - center_r) * self.cell_size
                neuron.k_fear = 1.0
                neuron.k_desire = 1.0

    def apply_homeostasis(self, relaxation_rate: float = 0.2) -> None:
        """
        Anchor force: Pulls each neuron back to its ideal geometric position
        relative to the membrane's absolute center (the agent).
        Prevents infinite drift and structural collapse.
        """
        center_r = (self.rows - 1) / 2.0
        center_c = (self.cols - 1) / 2.0

        for r in range(self.rows):
            for c in range(self.cols):
                neuron = self.grid[r][c]

                # Calculate where this node SHOULD be mathematically
                ideal_x = self.center[0] + (c - center_c) * self.cell_size
                ideal_y = self.center[1] + (r - center_r) * self.cell_size
                ideal_pos = np.array([ideal_x, ideal_y], dtype=float)

                # Slowly drag the node back to its ideal position
                # This fights back against the BMU pull and obstacle repulsion
                neuron.position += (ideal_pos - neuron.position) * relaxation_rate
