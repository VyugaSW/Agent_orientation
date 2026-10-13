import unittest
import numpy as np

from Neuron import Neuron
from Topology import (
    ToroidalSquareTopology,
    ToroidalHexagonalTopology,
    ParameterizedMorphologyTopology,
)
from Orientation_membrane import Orientation_membrane
from InfluenceEntity import DangerEntity, AttractorEntity
from Environment import CostmapEnvironment, SpatialHashEnvironment
from Agent import Agent
from TopologyOptimizer import (
    TopologicalMapAnalyzer,
    HomologyMembraneConfigurator,
    EvolutionaryMembraneOptimizer,
)


class TestNeuron(unittest.TestCase):
    def test_modified_distance(self):
        """Test distance metric distortion under k_fear and k_desire influence."""
        n = Neuron(0.0, 0.0)
        target = np.array([3.0, 4.0])  # Euclidean distance = 5.0

        self.assertAlmostEqual(n.get_modified_distance(target), 5.0)

        n.k_fear = 10.0
        self.assertAlmostEqual(n.get_modified_distance(target), 50.0)

        n.k_fear = 1.0
        n.k_desire = 0.1
        self.assertAlmostEqual(n.get_modified_distance(target), 0.5)


class TestTopology(unittest.TestCase):
    def test_toroidal_links(self):
        """Test Toroidal wrap-around linking."""
        strategy = ToroidalSquareTopology()
        grid = strategy.build_grid(
            rows=3, cols=3, cell_size=1.0, start_x=0.0, start_y=0.0
        )

        top_left = grid[0][0]
        top_right = grid[0][2]
        bottom_left = grid[2][0]

        # Right neighbor of the right edge should be the left edge (wrap-around X)
        self.assertIn(top_left, top_right.neighbors)
        # Bottom neighbor of the bottom edge should be the top edge (wrap-around Y)
        self.assertIn(top_left, bottom_left.neighbors)


class TestInfluenceEntity(unittest.TestCase):
    def test_danger_entity(self):
        """Test linear falloff of the danger zone."""
        danger = DangerEntity(x=10.0, y=10.0, radius=10.0, peak_strength=100.0)
        neuron = Neuron(15.0, 10.0)  # Distance 5.0 (half of the radius)

        danger.apply_influence(neuron, distance=5.0)
        # 100 * (1 - 5/10) = 50. Final k_fear = 1.0 + 50.0 = 51.0
        self.assertAlmostEqual(neuron.k_fear, 51.0)

    def test_attractor_entity(self):
        """Test attractor zone calculation with clamping."""
        attractor = AttractorEntity(x=0.0, y=0.0, radius=10.0, peak_strength=0.9)
        neuron = Neuron(0.0, 0.0)  # Distance 0.0 (epicenter)

        attractor.apply_influence(neuron, distance=0.0)
        # 0.9 * (1 - 0/10) = 0.9. calculated_desire = 1.0 - 0.9 = 0.1
        self.assertAlmostEqual(neuron.k_desire, 0.1)

    def test_dynamic_movement_and_blinking(self):
        """Test moving and disappearing/reappearing obstacle mechanics."""
        entity = DangerEntity(
            x=20.0, y=20.0, radius=5.0, peak_strength=50.0, vx=1.0, vy=-0.5, blink_period=2
        )
        self.assertTrue(entity.is_active)
        entity.step()
        self.assertAlmostEqual(entity.position[0], 21.0)
        self.assertAlmostEqual(entity.position[1], 19.5)
        entity.step()
        # After 2 ticks (blink_period=2), entity toggles to inactive
        self.assertFalse(entity.is_active)


class TestEnvironment(unittest.TestCase):
    def test_costmap_environment(self):
        """Test O(1) costmap evaluation and out-of-bounds handling."""
        map_matrix = np.ones((10, 10), dtype=float)
        map_matrix[5][5] = 50.0  # Obstacle
        map_matrix[1][1] = 0.5   # Zone of interest

        env = CostmapEnvironment(map_matrix, resolution=1.0)

        n_wall = Neuron(5.1, 5.1)
        n_good = Neuron(1.1, 1.1)
        n_void = Neuron(100.0, 100.0)  # Out of bounds

        env.update_neurons_cost([n_wall, n_good, n_void])

        self.assertEqual(n_wall.k_fear, 50.0)
        self.assertEqual(n_good.k_desire, 0.5)
        self.assertEqual(n_void.k_fear, 100.0)  # Out-of-bounds protection triggered


class TestOrientationMembrane(unittest.TestCase):
    def setUp(self):
        self.topology = ToroidalSquareTopology()
        self.membrane = Orientation_membrane(
            rows=5,
            cols=5,
            cell_size=2.0,
            start_x=0.0,
            start_y=0.0,
            topology=self.topology,
        )

    def test_scroll_forward_teleportation(self):
        """Test Toroidal memory shift and homeostasis anchor consistency."""
        step = np.array([2.0, 0.0])

        # The left edge of the radar is at X = -4.0 before the step
        left_edge_neuron = self.membrane.grid[2][0]
        self.assertEqual(left_edge_neuron.position[0], -4.0)

        self.membrane.scroll_forward(step)

        # Radar width = 5 * 2 = 10.0. New coordinate: -4.0 + 10.0 = 6.0.
        self.assertEqual(left_edge_neuron.position[0], 6.0)
        self.assertEqual(left_edge_neuron.k_fear, 1.0)

        # Verify homeostasis does NOT pull the teleported neuron back to the left edge
        self.membrane.apply_homeostasis(relaxation_rate=0.5)
        self.assertEqual(left_edge_neuron.position[0], 6.0)

    def test_adapt_membrane_phantom_cut_off(self):
        """Test the Scissors Rule: adaptation pull must stop at Toroidal phantom links."""
        target = np.array([0.0, 0.0])
        self.membrane.adapt_membrane(target, learning_rate=0.5, radius_hops=4)

        dist = self.membrane.grid[2][0].get_euclidean_distance(
            self.membrane.grid[2][4].position
        )
        self.assertGreater(dist, self.membrane.cell_size * 2)


class TestAgent(unittest.TestCase):
    def test_agent_step(self):
        """Integration test: a single logical tick of the agent controller."""
        env = SpatialHashEnvironment(sector_size=10.0)
        membrane = Orientation_membrane(
            rows=3,
            cols=3,
            cell_size=2.0,
            start_x=0.0,
            start_y=0.0,
            topology=ToroidalSquareTopology(),
        )
        agent = Agent(x=0.0, y=0.0, membrane=membrane, environment=env)

        agent.set_target(10.0, 0.0)
        agent.step()

        self.assertGreater(agent.position[0], 0.0)
        self.assertGreater(agent.membrane.center[0], 0.0)


class TestTopologyAndHomologyOptimizer(unittest.TestCase):
    def test_homology_configurator_and_betti(self):
        """Test persistent homology map analysis and membrane Betti numbers."""
        env = SpatialHashEnvironment(sector_size=20.0)
        membrane = Orientation_membrane(
            rows=4,
            cols=5,
            cell_size=5.0,
            start_x=10.0,
            start_y=50.0,
            topology=ToroidalHexagonalTopology(),
        )
        agent = Agent(x=10.0, y=50.0, membrane=membrane, environment=env)
        build_level_2_u_trap(env, agent)

        info = HomologyMembraneConfigurator.apply_to_agent(agent, env)
        self.assertGreaterEqual(info["rows"], 4)
        self.assertEqual(info["rows"] % 2, 0)

        b0, b1, _ = agent.membrane.compute_homology_betti()
        self.assertGreaterEqual(b0, 1)
        self.assertGreaterEqual(b1, 0)


def add_arena_borders(env: SpatialHashEnvironment) -> None:
    step = 3.0
    radius = 2.0
    strength = 70.0

    for y in range(0, 81, int(step)):
        env.add_entity(
            DangerEntity(x=0.0, y=float(y), radius=radius, peak_strength=strength)
        )
        env.add_entity(
            DangerEntity(x=80.0, y=float(y), radius=radius, peak_strength=strength)
        )

    for x in range(0, 81, int(step)):
        env.add_entity(
            DangerEntity(x=float(x), y=0.0, radius=radius, peak_strength=strength)
        )
        env.add_entity(
            DangerEntity(x=float(x), y=80.0, radius=radius, peak_strength=strength)
        )


def build_level_1_stripe(env: SpatialHashEnvironment, agent: Agent) -> None:
    """Level 1: A solid vertical wall (red stripe) blocking the direct path."""
    env.active_entities.clear()
    add_arena_borders(env)

    # Create a continuous wall by overlapping danger entities
    for y in range(15, 86, 5):
        env.add_entity(DangerEntity(x=40.0, y=float(y), radius=8.0, peak_strength=80.0))

    env.rebuild_spatial_hash()
    agent.position = np.array([10.0, 50.0], dtype=float)
    agent.membrane.center = np.array([10.0, 50.0], dtype=float)
    agent.set_target(70.0, 50.0)
    agent.membrane.reset_grid()


def build_level_2_u_trap(env: SpatialHashEnvironment, agent: Agent) -> None:
    """Level 2: A U-shaped trap to test if the algorithm escapes local minima."""
    env.active_entities.clear()
    add_arena_borders(env)

    # Back wall
    for y in range(30, 71, 5):
        env.add_entity(DangerEntity(x=50.0, y=float(y), radius=7.0, peak_strength=80.0))
    # Top wall
    for x in range(30, 51, 5):
        env.add_entity(DangerEntity(x=float(x), y=30.0, radius=7.0, peak_strength=80.0))
    # Bottom wall
    for x in range(30, 51, 5):
        env.add_entity(DangerEntity(x=float(x), y=70.0, radius=7.0, peak_strength=80.0))

    env.rebuild_spatial_hash()
    agent.position = np.array([20.0, 50.0], dtype=float)
    agent.membrane.center = np.array([20.0, 50.0], dtype=float)
    agent.set_target(70.0, 50.0)
    agent.membrane.reset_grid()


def build_level_3_forest(env: SpatialHashEnvironment, agent: Agent) -> None:
    """Level 3: Scattered obstacles forcing the membrane to thread the needle."""
    env.active_entities.clear()
    add_arena_borders(env)

    positions = [(25, 40), (25, 60), (45, 50), (45, 30), (45, 70), (60, 45)]
    for pos in positions:
        env.add_entity(
            DangerEntity(
                x=float(pos[0]), y=float(pos[1]), radius=9.3, peak_strength=50.0
            )
        )

    env.rebuild_spatial_hash()
    agent.position = np.array([10.0, 50.0], dtype=float)
    agent.membrane.center = np.array([10.0, 50.0], dtype=float)
    agent.set_target(75.0, 50.0)
    agent.membrane.reset_grid()


def build_level_4_something(env: SpatialHashEnvironment, agent: Agent) -> None:
    """Level 4: Complex obstacle field."""
    env.active_entities.clear()
    add_arena_borders(env)

    positions = [(40, 40), (25, 60), (50, 50), (10, 30), (30, 70), (60, 60)]
    for pos in positions:
        env.add_entity(
            DangerEntity(
                x=float(pos[0]), y=float(pos[1]), radius=10.0, peak_strength=70.0
            )
        )

    env.rebuild_spatial_hash()
    agent.position = np.array([10.0, 50.0], dtype=float)
    agent.membrane.center = np.array([10.0, 50.0], dtype=float)
    agent.set_target(75.0, 50.0)
    agent.membrane.reset_grid()


def build_level_5_dynamic_swarm(env: SpatialHashEnvironment, agent: Agent) -> None:
    """
    Level 5: Dynamic Swarm.
    Combines vertically patrolling obstacles, diagonally bouncing hazards,
    and blinking (appearing/disappearing) gate obstacles.
    """
    env.active_entities.clear()
    add_arena_borders(env)

    # 1. Vertically patrolling obstacles crossing the agent's main corridor
    env.add_entity(
        DangerEntity(x=26.0, y=25.0, radius=7.5, peak_strength=75.0, vx=0.0, vy=0.32)
    )
    env.add_entity(
        DangerEntity(x=42.0, y=65.0, radius=8.0, peak_strength=75.0, vx=0.0, vy=-0.36)
    )
    env.add_entity(
        DangerEntity(x=58.0, y=35.0, radius=7.5, peak_strength=75.0, vx=0.18, vy=0.28)
    )

    # 2. Blinking (appearing/disappearing) obstacles acting as dynamic gates
    env.add_entity(
        DangerEntity(
            x=34.0,
            y=48.0,
            radius=8.5,
            peak_strength=85.0,
            blink_period=85,
            blink_offset=0,
        )
    )
    env.add_entity(
        DangerEntity(
            x=50.0,
            y=45.0,
            radius=8.5,
            peak_strength=85.0,
            blink_period=95,
            blink_offset=45,
        )
    )
    env.add_entity(
        DangerEntity(
            x=50.0,
            y=60.0,
            radius=7.5,
            peak_strength=80.0,
            vx=-0.15,
            vy=0.15,
            blink_period=110,
            blink_offset=20,
        )
    )

    env.rebuild_spatial_hash()
    agent.position = np.array([10.0, 50.0], dtype=float)
    agent.membrane.center = np.array([10.0, 50.0], dtype=float)
    agent.set_target(75.0, 50.0)
    agent.membrane.reset_grid()


def main():
    from Visualizer import PygameVisualizer

    # 1. Initialize the Environment
    env = SpatialHashEnvironment(sector_size=20.0)

    # 2. Initialize the Membrane (Radar)
    topology = ToroidalHexagonalTopology()
    membrane = Orientation_membrane(
        rows=4, cols=5, cell_size=6.5, start_x=10.0, start_y=50.0, topology=topology
    )

    # 3. Initialize the Agent
    agent = Agent(x=10.0, y=50.0, membrane=membrane, environment=env)

    # 4. Map the level loader functions to keyboard keys (1 to 5)
    level_loaders = {
        1: lambda: build_level_1_stripe(env, agent),
        2: lambda: build_level_2_u_trap(env, agent),
        3: lambda: build_level_3_forest(env, agent),
        4: lambda: build_level_4_something(env, agent),
        5: lambda: build_level_5_dynamic_swarm(env, agent),
    }

    # Load Level 1 by default
    level_loaders[1]()

    # 5. Start the Visualizer
    app = PygameVisualizer(
        agent=agent, environment=env, width=800, height=800, pixels_per_meter=10.0
    )
    app.run(level_loaders)


if __name__ == "__main__":
    main()