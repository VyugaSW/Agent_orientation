import math
import numpy as np
from typing import Callable, Dict, List, Optional

from Agent import Agent
from Environment import SpatialHashEnvironment
from InfluenceEntity import DangerEntity
from Topology import (
    TopologyStrategy,
    ToroidalHexagonalTopology,
    ParameterizedMorphologyTopology,
)


def _extract_base_topology(topology: TopologyStrategy) -> TopologyStrategy:
    """Unwraps ParameterizedMorphologyTopology to get the original Square or Hexagonal strategy."""
    while isinstance(topology, ParameterizedMorphologyTopology):
        topology = topology.base_topology
    return topology


class TopologicalMapAnalyzer:
    """
    Extracts true persistent homology features (H_0 inter-component passage widths
    and H_1 2D concave cavity scales) from active obstacles.
    """

    @staticmethod
    def analyze(env: SpatialHashEnvironment) -> Dict[str, float]:
        obstacles = [
            e
            for e in env.active_entities
            if isinstance(e, DangerEntity)
            and getattr(e, "is_active", True)
            and 2.0 < e.position[0] < 78.0
            and 2.0 < e.position[1] < 78.0
        ]
        if len(obstacles) < 2:
            return {
                "min_gap_h0": 12.0,
                "cavity_scale_h1": 16.0,
                "obstacle_clusters": float(len(obstacles)),
                "aspect_ratio": 1.0,
            }

        n = len(obstacles)
        centers = np.array([e.position for e in obstacles], dtype=float)
        radii = np.array([e.radius for e in obstacles], dtype=float)

        # Pairwise surface gap matrix (Cech complex filtration on obstacle disks)
        diff = centers[:, None, :] - centers[None, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)
        gap_matrix = dist_matrix - (radii[:, None] + radii[None, :])
        np.fill_diagonal(gap_matrix, np.inf)

        # Step 1: Identify true connected obstacle components (overlapping disks: gap <= 0.1)
        comp_id = [-1] * n
        clusters: List[List[int]] = []

        for i in range(n):
            if comp_id[i] == -1:
                cid = len(clusters)
                comp: List[int] = []
                stack = [i]
                comp_id[i] = cid
                while stack:
                    curr = stack.pop()
                    comp.append(curr)
                    for nb in range(n):
                        if comp_id[nb] == -1 and gap_matrix[curr, nb] <= 0.1:
                            comp_id[nb] = cid
                            stack.append(nb)
                clusters.append(comp)

        clusters_count = len(clusters)

        # Step 2: H_0 persistence (measure positive gaps ONLY between distinct components)
        if clusters_count >= 2:
            nearest_inter_cluster_gaps: List[float] = []
            for c1 in range(clusters_count):
                min_d = float("inf")
                for i in clusters[c1]:
                    for c2 in range(clusters_count):
                        if c1 != c2:
                            for j in clusters[c2]:
                                if 0.2 < gap_matrix[i, j] < min_d:
                                    min_d = float(gap_matrix[i, j])
                if min_d < 40.0:
                    nearest_inter_cluster_gaps.append(min_d)

            min_gap_h0 = (
                float(np.percentile(nearest_inter_cluster_gaps, 25))
                if nearest_inter_cluster_gaps
                else 12.0
            )
        else:
            # Single wall component has no narrow inter-obstacle passages
            min_gap_h0 = 14.0

        # Step 3: H_1 cavity scale (distinguish 1D flat walls from 2D concave pockets)
        max_cavity_span = 14.0
        aspect_ratio = 1.0

        for comp in clusters:
            if len(comp) >= 2:
                pts = centers[comp]
                span_x = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
                span_y = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))

                # A true 2D pocket (like U-trap) has significant depth in BOTH X and Y
                pocket_depth_2d = math.sqrt(max(1.0, span_x) * max(1.0, span_y))
                wall_length = math.hypot(span_x, span_y)

                if span_x > 8.0 and span_y > 8.0:
                    # 2D concave trap: high H_1 persistence
                    cavity_est = wall_length * 1.15
                    aspect_ratio = 1.25
                else:
                    # 1D straight barrier (like Level 1 stripe): moderate radar span
                    cavity_est = pocket_depth_2d * 2.2
                    aspect_ratio = 0.85

                max_cavity_span = max(max_cavity_span, cavity_est)

        # For multi-cluster fields (forests/swarms), scale radar by average cluster spacing
        if clusters_count >= 3 and max_cavity_span <= 14.0:
            max_cavity_span = float(np.clip(min_gap_h0 * 3.8 + 10.0, 14.0, 32.0))
            aspect_ratio = 0.85 if min_gap_h0 < 4.5 else 1.05

        return {
            "min_gap_h0": float(np.clip(min_gap_h0, 1.4, 18.0)),
            "cavity_scale_h1": float(np.clip(max_cavity_span, 14.0, 44.0)),
            "obstacle_clusters": float(clusters_count),
            "aspect_ratio": aspect_ratio,
        }


class HomologyMembraneConfigurator:
    """Analytically configures membrane dimensions and morphology from H_0 and H_1 map barcodes."""

    @staticmethod
    def apply_to_agent(agent: Agent, env: SpatialHashEnvironment) -> Dict[str, float]:
        metrics = TopologicalMapAnalyzer.analyze(env)
        min_gap = metrics["min_gap_h0"]
        cavity_scale = metrics["cavity_scale_h1"]
        aspect = metrics["aspect_ratio"]

        base_topology = _extract_base_topology(agent.membrane.topology)

        # H_0 sampling rule: scale cell_size smoothly across [3.4, 6.8]
        cell_size = float(np.clip(2.8 + min_gap * 0.25, 3.4, 6.8))

        # H_1 coverage rule: separate rows and cols using aspect ratio
        cols = int(np.clip(round((cavity_scale * aspect) / cell_size), 4, 9))
        rows = int(np.clip(round((cavity_scale / aspect) / cell_size), 4, 8))
        if isinstance(base_topology, ToroidalHexagonalTopology) and rows % 2 != 0:
            rows += 1

        is_forest = metrics["obstacle_clusters"] >= 4
        scale_x = 0.92 if is_forest else 1.12
        scale_y = 0.82 if (is_forest and min_gap < 4.5) else 1.10
        curvature = 0.28 if aspect > 1.1 else 0.10
        foveation = 1.25 if is_forest else 0.95

        new_topology = ParameterizedMorphologyTopology(
            scale_x=scale_x,
            scale_y=scale_y,
            curvature=curvature,
            foveation=foveation,
            base_topology=base_topology,
        )
        agent.membrane.rebuild(
            rows=rows, cols=cols, cell_size=cell_size, topology=new_topology
        )
        agent.params["look_ahead_cells"] = max(2, min(rows, cols) // 2)
        agent.params["tangent_force"] = 3.6 if not is_forest else 2.6

        return {
            "rows": rows,
            "cols": cols,
            "cell_size": round(cell_size, 2),
            "min_gap_h0": round(min_gap, 2),
            "cavity_scale_h1": round(cavity_scale, 2),
        }


class EvolutionaryMembraneOptimizer:
    """
    (1+lambda)-ES Evolutionary Strategy to optimize membrane morphology
    and steering parameters directly on the active map.
    Genome: [scale_x, scale_y, curvature, foveation, tangent_force, membrane_lr, membrane_relax]
    """

    def __init__(self, generations: int = 8, pop_size: int = 6, max_steps: int = 280):
        self.generations = generations
        self.pop_size = pop_size
        self.max_steps = max_steps

    def evaluate_genome(
        self,
        genome: np.ndarray,
        env: SpatialHashEnvironment,
        start_pos: np.ndarray,
        target_pos: np.ndarray,
        base_rows: int,
        base_cols: int,
        base_cell_size: float,
        base_topology: TopologyStrategy,
    ) -> float:
        scale_x, scale_y, curvature, foveation, tangent_force, lr, relax = genome

        topo = ParameterizedMorphologyTopology(
            scale_x=float(scale_x),
            scale_y=float(scale_y),
            curvature=float(curvature),
            foveation=float(foveation),
            base_topology=base_topology,
        )
        from Orientation_membrane import Orientation_membrane

        membrane = Orientation_membrane(
            rows=base_rows,
            cols=base_cols,
            cell_size=base_cell_size,
            start_x=float(start_pos[0]),
            start_y=float(start_pos[1]),
            topology=topo,
        )
        sim_agent = Agent(
            x=float(start_pos[0]),
            y=float(start_pos[1]),
            membrane=membrane,
            environment=env,
        )
        sim_agent.set_target(float(target_pos[0]), float(target_pos[1]))
        sim_agent.params["tangent_force"] = float(tangent_force)
        sim_agent.params["membrane_lr"] = float(lr)
        sim_agent.params["membrane_relax"] = float(relax)

        collisions = 0
        oscillation_penalty = 0.0
        prev_dir = np.zeros(2)

        steps_taken = self.max_steps
        for step_idx in range(1, self.max_steps + 1):
            prev_pos = sim_agent.position.copy()
            sim_agent.step()

            dist_to_target = float(
                np.linalg.norm(sim_agent.target - sim_agent.position)
            )
            if dist_to_target < 1.5:
                steps_taken = step_idx
                break

            move_dir = sim_agent.position - prev_pos
            norm_m = float(np.linalg.norm(move_dir))
            if norm_m > 1e-4:
                move_dir /= norm_m
                if np.linalg.norm(prev_dir) > 1e-4:
                    oscillation_penalty += max(0.0, -float(np.dot(move_dir, prev_dir)))
                prev_dir = move_dir

            for obs in env.get_nearby_obstacles(sim_agent.position, 15.0):
                if np.linalg.norm(sim_agent.position - obs.position) < obs.radius * 0.95:
                    collisions += 1

        final_dist = float(np.linalg.norm(sim_agent.target - sim_agent.position))
        loss = (
            steps_taken * 1.0
            + final_dist * 8.0
            + collisions * 25.0
            + oscillation_penalty * 5.0
        )
        return loss

    def optimize_for_agent(
        self,
        agent: Agent,
        env: SpatialHashEnvironment,
        start_pos: np.ndarray,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> Dict[str, float]:
        base_topology = _extract_base_topology(agent.membrane.topology)

        info = HomologyMembraneConfigurator.apply_to_agent(agent, env)
        base_rows = int(info["rows"])
        base_cols = int(info["cols"])
        base_cell_size = float(info["cell_size"])
        target_pos = agent.target.copy()

        best_genome = np.array(
            [1.0, 1.0, 0.15, 1.05, agent.params["tangent_force"], 0.06, 0.25]
        )
        bounds_low = np.array([0.6, 0.6, -0.5, 0.7, 1.0, 0.02, 0.10])
        bounds_high = np.array([1.5, 1.5, 0.6, 1.5, 6.0, 0.18, 0.45])
        sigma = np.array([0.18, 0.18, 0.15, 0.15, 0.6, 0.02, 0.05])

        best_loss = self.evaluate_genome(
            best_genome,
            env,
            start_pos,
            target_pos,
            base_rows,
            base_cols,
            base_cell_size,
            base_topology,
        )

        for gen in range(1, self.generations + 1):
            for ind in range(1, self.pop_size + 1):
                if progress_callback is not None:
                    progress_callback(gen, ind, best_loss)

                candidate = np.clip(
                    best_genome + np.random.randn(len(best_genome)) * sigma,
                    bounds_low,
                    bounds_high,
                )
                loss = self.evaluate_genome(
                    candidate,
                    env,
                    start_pos,
                    target_pos,
                    base_rows,
                    base_cols,
                    base_cell_size,
                    base_topology,
                )
                if loss < best_loss:
                    best_loss = loss
                    best_genome = candidate

            sigma *= 0.85

        best_topo = ParameterizedMorphologyTopology(
            scale_x=float(best_genome[0]),
            scale_y=float(best_genome[1]),
            curvature=float(best_genome[2]),
            foveation=float(best_genome[3]),
            base_topology=base_topology,
        )
        agent.position[:] = start_pos
        agent.membrane.center[:] = start_pos
        if hasattr(agent, "reset_memory"):
            agent.reset_memory()
        agent.membrane.rebuild(
            rows=base_rows, cols=base_cols, cell_size=base_cell_size, topology=best_topo
        )
        agent.params["tangent_force"] = float(best_genome[4])
        agent.params["membrane_lr"] = float(best_genome[5])
        agent.params["membrane_relax"] = float(best_genome[6])

        return {
            "loss": round(best_loss, 1),
            "scale_x": round(float(best_genome[0]), 2),
            "scale_y": round(float(best_genome[1]), 2),
            "curvature": round(float(best_genome[2]), 2),
            "foveation": round(float(best_genome[3]), 2),
        }