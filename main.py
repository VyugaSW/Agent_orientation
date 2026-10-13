import unittest
import numpy as np

# Assuming the files are named exactly as in your project
from Neuron import Neuron
from Topology import ToroidalSquareTopology, ToroidalHexagonalTopology
from Orientation_membrane import Orientation_membrane
from InfluenceEntity import DangerEntity, AttractorEntity
from Environment import CostmapEnvironment, SpatialHashEnvironment
from Agent import Agent
from Visualizer import PygameVisualizer


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
        # 3x3 Grid
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


class TestEnvironment(unittest.TestCase):
    def test_costmap_environment(self):
        """Test O(1) costmap evaluation and out-of-bounds handling."""
        map_matrix = np.ones((10, 10), dtype=float)
        map_matrix[5][5] = 50.0  # Obstacle
        map_matrix[1][1] = 0.5  # Zone of interest

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
        # Force initialization of center until the bug is fixed in the main code
        self.membrane.center = np.array([0.0, 0.0], dtype=float)

    def test_scroll_forward_teleportation(self):
        """Test Toroidal memory shift (teleportation of trailing neurons)."""
        # Agent steps right by 1 cell
        step = np.array([2.0, 0.0])

        # The left edge of the radar is at X = -4.0 before the step
        left_edge_neuron = self.membrane.grid[2][0]
        self.assertEqual(left_edge_neuron.position[0], -4.0)

        self.membrane.scroll_forward(step)

        # After the step, the center shifted by +2.0.
        # The old left edge (-4.0) is now behind the visibility bound and must teleport forward.
        # Radar width = 5 * 2 = 10.0. New coordinate: -4.0 + 10.0 = 6.0.
        self.assertEqual(left_edge_neuron.position[0], 6.0)
        self.assertEqual(left_edge_neuron.k_fear, 1.0)  # Memory wipe check

    def test_adapt_membrane_phantom_cut_off(self):
        """Test the Scissors Rule: adaptation pull must stop at Toroidal phantom links."""
        target = np.array([0.0, 0.0])
        # Trigger pull with a 4-hop radius to reach the grid edges
        self.membrane.adapt_membrane(target, learning_rate=0.5, radius_hops=4)

        # Node on the left edge (-4, 0) and right edge (4, 0) are physically distant
        # Thanks to BFS with distance limit (phantom_link_threshold), they will not pull each other.
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
        membrane.center = np.array([0.0, 0.0], dtype=float)
        agent = Agent(x=0.0, y=0.0, membrane=membrane, environment=env)

        agent.set_target(10.0, 0.0)  # Target is far to the right
        agent.step()

        # The agent must step towards the target
        self.assertGreater(agent.position[0], 0.0)
        # The radar center must shift following the agent
        self.assertGreater(agent.membrane.center[0], 0.0)


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

    # Reset agent position and target
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

    agent.position = np.array([20.0, 50.0], dtype=float)
    agent.membrane.center = np.array([20.0, 50.0], dtype=float)
    agent.set_target(70.0, 50.0)
    agent.membrane.reset_grid()


def build_level_3_forest(env: SpatialHashEnvironment, agent: Agent) -> None:
    """Level 3: Scattered obstacles forcing the membrane to thread the needle."""
    env.active_entities.clear()
    add_arena_borders(env)

    # Randomly placed distinct obstacles
    positions = [(25, 40), (25, 60), (45, 50), (45, 30), (45, 70), (60, 45)]
    for pos in positions:
        env.add_entity(
            DangerEntity(
                x=float(pos[0]), y=float(pos[1]), radius=9.3, peak_strength=50.0
            )
        )

    agent.position = np.array([10.0, 50.0], dtype=float)
    agent.membrane.center = np.array([10.0, 50.0], dtype=float)
    agent.set_target(75.0, 50.0)
    agent.membrane.reset_grid()


def build_level_4_something(env: SpatialHashEnvironment, agent: Agent) -> None:
    """Level 4: Something"""
    env.active_entities.clear()
    add_arena_borders(env)

    positions = [(40, 40), (25, 60), (50, 50), (10, 30), (30, 70), (60, 60)]
    for pos in positions:
        env.add_entity(
            DangerEntity(
                x=float(pos[0]), y=float(pos[1]), radius=10.0, peak_strength=70.0
            )
        )

    agent.position = np.array([10.0, 50.0], dtype=float)
    agent.membrane.center = np.array([10.0, 50.0], dtype=float)
    agent.set_target(75.0, 50.0)
    agent.membrane.reset_grid()


def main():
    # 1. Initialize the Environment
    env = SpatialHashEnvironment(sector_size=20.0)

    # 2. Initialize the Membrane (Radar)
    # Increased size to 21x21 for better visibility of deformation around large obstacles
    # topology = ToroidalSquareTopology()
    topology = ToroidalHexagonalTopology()
    membrane = Orientation_membrane(
        rows=4, cols=5, cell_size=6.5, start_x=10.0, start_y=10.0, topology=topology
    )

    # 3. Initialize the Agent
    agent = Agent(x=10.0, y=50.0, membrane=membrane, environment=env)

    # 4. Map the level loader functions to keyboard keys
    level_loaders = {
        1: lambda: build_level_1_stripe(env, agent),
        2: lambda: build_level_2_u_trap(env, agent),
        3: lambda: build_level_3_forest(env, agent),
        4: lambda: build_level_4_something(env, agent),
    }

    # Load Level 1 by default
    level_loaders[1]()

    # 5. Start the Visualizer
    app = PygameVisualizer(
        agent=agent, environment=env, width=800, height=800, pixels_per_meter=10.0
    )

    # Pass the level dictionary to the run loop
    app.run(level_loaders)


if __name__ == "__main__":
    main()
