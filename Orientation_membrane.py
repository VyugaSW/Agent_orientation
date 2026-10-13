from collections import deque
import numpy as np
from typing import List, Optional, Tuple, Set

from Topology import TopologyStrategy
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
        self.cell_size = float(cell_size)
        self.topology = topology
        self.center = np.array([start_x, start_y], dtype=float)

        # Delegate initialization to the injected topology strategy
        self.grid: List[List[Neuron]] = self.topology.build_grid(
            rows, cols, self.cell_size, start_x, start_y
        )
        self.flat_neurons: List[Neuron] = [
            neuron for row in self.grid for neuron in row
        ]

    def rebuild(
        self,
        rows: int,
        cols: int,
        cell_size: float,
        topology: Optional[TopologyStrategy] = None,
    ) -> None:
        """Rebuilds the membrane with new dimensions or topology around the current center."""
        self.rows = rows
        self.cols = cols
        self.cell_size = float(cell_size)
        if topology is not None:
            self.topology = topology
        self.grid = self.topology.build_grid(
            rows, cols, self.cell_size, float(self.center[0]), float(self.center[1])
        )
        self.flat_neurons = [neuron for row in self.grid for neuron in row]

    def find_bmu(
        self, target_coords: np.ndarray, neurons: Optional[List[Neuron]] = None
    ) -> Optional[Neuron]:
        """
        Finds the Best Matching Unit (BMU) for a given stimulus target.
        Utilizes the modified distance that accounts for k_fear and k_desire.
        """
        candidates = neurons if neurons is not None else self.flat_neurons
        best_neuron: Optional[Neuron] = None
        min_dist = float("inf")

        for neuron in candidates:
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
        neurons: Optional[List[Neuron]] = None,
        bmu: Optional[Neuron] = None,
    ) -> None:
        """
        Applies SOM pull to the BMU and its topological neighbors using an O(1) deque BFS.
        Uses a physical distance cut-off (The Scissors Rule) to prevent Toroidal phantom links
        from ripping the physical mesh apart.
        """
        if bmu is None:
            bmu = self.find_bmu(target_coords, neurons)
        if bmu is None:
            return

        queue = deque([(bmu, 0)])
        visited: Set[Neuron] = {bmu}
        phantom_link_threshold = self.cell_size * 1.5

        while queue:
            current_neuron, hop_dist = queue.popleft()

            # Gaussian falloff based on topological hops
            falloff = np.exp(-(hop_dist ** 2) / 2.0)
            weight = learning_rate * falloff
            current_neuron.pull_towards(target_coords, weight)

            if hop_dist >= radius_hops:
                continue

            for neighbor in current_neuron.neighbors:
                if neighbor not in visited:
                    # RULE: The Scissors (Physical Cut-off)
                    dist = current_neuron.get_euclidean_distance(neighbor.position)
                    if dist < phantom_link_threshold:
                        visited.add(neighbor)
                        queue.append((neighbor, hop_dist + 1))

    def scroll_forward(self, step_vector: np.ndarray) -> None:
        """
        Implements Toroidal O(N) memory shift.
        Synchronously teleports both physical coordinates and ideal lattice anchors
        of trailing neurons to the leading edge.
        """
        self.center += step_vector
        for neuron in self.flat_neurons:
            neuron.anchor += step_vector

        radar_width, radar_height = self.topology.get_periods(
            self.rows, self.cols, self.cell_size
        )
        half_width = radar_width / 2.0
        half_height = radar_height / 2.0

        cx, cy = float(self.center[0]), float(self.center[1])

        for neuron in self.flat_neurons:
            # Teleport physical position along X-axis
            if neuron.position[0] < cx - half_width:
                neuron.position[0] += radar_width
                neuron.k_fear = 1.0
                neuron.k_desire = 1.0
            elif neuron.position[0] > cx + half_width:
                neuron.position[0] -= radar_width
                neuron.k_fear = 1.0
                neuron.k_desire = 1.0

            # Teleport physical position along Y-axis
            if neuron.position[1] < cy - half_height:
                neuron.position[1] += radar_height
                neuron.k_fear = 1.0
                neuron.k_desire = 1.0
            elif neuron.position[1] > cy + half_height:
                neuron.position[1] -= radar_height
                neuron.k_fear = 1.0
                neuron.k_desire = 1.0

            # Synchronously wrap the lattice anchor around the torus
            if neuron.anchor[0] < cx - half_width:
                neuron.anchor[0] += radar_width
            elif neuron.anchor[0] > cx + half_width:
                neuron.anchor[0] -= radar_width

            if neuron.anchor[1] < cy - half_height:
                neuron.anchor[1] += radar_height
            elif neuron.anchor[1] > cy + half_height:
                neuron.anchor[1] -= radar_height

    def reset_grid(self) -> None:
        """Restores the physical grid to an undeformed state around the current center."""
        for neuron in self.flat_neurons:
            neuron.position[:] = self.center + neuron.base_offset
            neuron.anchor[:] = neuron.position
            neuron.k_fear = 1.0
            neuron.k_desire = 1.0

    def apply_homeostasis(self, relaxation_rate: float = 0.2) -> None:
        """
        Anchor force: Pulls each neuron back towards its current toroidal lattice anchor.
        Prevents structural collapse without snapping teleported neurons back across the grid.
        """
        for neuron in self.flat_neurons:
            neuron.position += (neuron.anchor - neuron.position) * relaxation_rate

    def compute_homology_betti(
        self, fear_threshold: float = 2.0
    ) -> Tuple[int, int, List[List[Neuron]]]:
        """
        Computes simplicial Betti numbers (beta_0, beta_1) of the safe membrane subgraph:
        - beta_0: number of connected safe components (homological obstacle-avoidance classes)
        - beta_1: number of independent cycles (mesh integrity via Euler characteristic)
        """
        safe_set: Set[Neuron] = {
            n for n in self.flat_neurons if n.k_fear <= fear_threshold
        }
        if not safe_set:
            return 0, 0, []

        phantom_threshold = self.cell_size * 1.5
        visited: Set[Neuron] = set()
        components: List[List[Neuron]] = []
        edge_count = 0

        for neuron in safe_set:
            for nb in neuron.neighbors:
                if nb in safe_set and id(neuron) < id(nb):
                    if neuron.get_euclidean_distance(nb.position) < phantom_threshold:
                        edge_count += 1

            if neuron not in visited:
                comp: List[Neuron] = []
                q = deque([neuron])
                visited.add(neuron)
                while q:
                    curr = q.popleft()
                    comp.append(curr)
                    for nb in curr.neighbors:
                        if nb in safe_set and nb not in visited:
                            if curr.get_euclidean_distance(nb.position) < phantom_threshold:
                                visited.add(nb)
                                q.append(nb)
                components.append(comp)

        beta_0 = len(components)
        beta_1 = max(0, edge_count - len(safe_set) + beta_0)
        return beta_0, beta_1, components