import pygame
import numpy as np
import sys

from Agent import Agent
from Environment import SpatialHashEnvironment
from InfluenceEntity import DangerEntity, AttractorEntity


class PygameVisualizer:
    def __init__(
        self,
        agent: Agent,
        environment: SpatialHashEnvironment,
        width: int = 800,
        height: int = 800,
        pixels_per_meter: float = 10.0,
    ):
        """
        Initializes the Pygame rendering engine for the SOM algorithm.
        pixels_per_meter: Scaling factor to convert physical coordinates to screen pixels.
        """
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Scrolling SOM: Living Membrane Navigation")

        self.clock = pygame.time.Clock()
        self.agent = agent
        self.environment = environment
        self.scale = pixels_per_meter

        # Colors (RGB)
        self.BG_COLOR = (30, 30, 30)  # Dark gray
        self.GRID_COLOR = (100, 200, 255)  # Light blue for neutral membrane
        self.FEAR_COLOR = (255, 50, 50)  # Red for high k_fear nodes
        self.AGENT_COLOR = (50, 255, 50)  # Green for the agent
        self.TARGET_COLOR = (255, 200, 0)  # Yellow for the destination
        self.DANGER_COLOR = (150, 30, 30)  # Dark red for obstacles

    def _to_screen(self, x: float, y: float) -> tuple:
        """Converts physical world coordinates to screen pixel coordinates."""
        # We center the view somewhat, but for simplicity, map directly scaled
        return (int(x * self.scale), int(y * self.scale))

    def draw_environment(self) -> None:
        """Draws all static and dynamic entities present in the environment."""
        for entity in self.environment.active_entities:
            screen_pos = self._to_screen(entity.position[0], entity.position[1])
            screen_radius = int(entity.radius * self.scale)

            if isinstance(entity, DangerEntity):
                # Draw a semi-transparent danger zone (using a surface)
                surface = pygame.Surface(
                    (screen_radius * 2, screen_radius * 2), pygame.SRCALPHA
                )
                pygame.draw.circle(
                    surface,
                    (*self.DANGER_COLOR, 100),
                    (screen_radius, screen_radius),
                    screen_radius,
                )
                self.screen.blit(
                    surface,
                    (screen_pos[0] - screen_radius, screen_pos[1] - screen_radius),
                )
                # Draw solid core
                pygame.draw.circle(self.screen, self.DANGER_COLOR, screen_pos, 4)

    def draw_membrane(self) -> None:
        """Draws the Toroidal SOM grid. Avoids drawing phantom wrap-around links."""
        membrane = self.agent.membrane
        phantom_threshold = membrane.cell_size * 1.5

        # 1. Draw topological connections (Edges)
        for r in range(membrane.rows):
            for c in range(membrane.cols):
                neuron = membrane.grid[r][c]
                pos1 = self._to_screen(neuron.position[0], neuron.position[1])

                for neighbor in neuron.neighbors:
                    # Prevent drawing lines for Toroidal wrap-around links across the screen
                    dist = neuron.get_euclidean_distance(neighbor.position)
                    if dist < phantom_threshold:
                        pos2 = self._to_screen(
                            neighbor.position[0], neighbor.position[1]
                        )
                        pygame.draw.line(self.screen, (70, 100, 120), pos1, pos2, 1)

        # 2. Draw neurons (Nodes) highlighting their state
        for r in range(membrane.rows):
            for c in range(membrane.cols):
                neuron = membrane.grid[r][c]
                pos = self._to_screen(neuron.position[0], neuron.position[1])

                # Color interpolates to red based on k_fear intensity
                if neuron.k_fear > 1.0:
                    intensity = min(255, int((neuron.k_fear - 1.0) * 10))
                    color = (intensity, max(0, 200 - intensity), 100)
                else:
                    color = self.GRID_COLOR

                pygame.draw.circle(self.screen, color, pos, 3)

    def draw_agent_and_target(self) -> None:
        """Renders the agent and the global target marker."""
        agent_pos = self._to_screen(self.agent.position[0], self.agent.position[1])
        pygame.draw.circle(self.screen, self.AGENT_COLOR, agent_pos, 6)

        target_pos = self._to_screen(self.agent.target[0], self.agent.target[1])
        pygame.draw.circle(self.screen, self.TARGET_COLOR, target_pos, 8)

        # Draw a line showing the direct global intent
        pygame.draw.line(self.screen, (50, 100, 50), agent_pos, target_pos, 1)

    def run(self, level_loaders) -> None:
        """Main visualization and physics loop."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Left click sets a new target
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    world_x = mouse_x / self.scale
                    world_y = mouse_y / self.scale
                    self.agent.set_target(world_x, world_y)
                elif event.type == pygame.KEYDOWN:
                    # Switch levels using 1, 2, 3 keys
                    if event.key == pygame.K_1 and 1 in level_loaders:
                        level_loaders[1]()
                    elif event.key == pygame.K_2 and 2 in level_loaders:
                        level_loaders[2]()
                    elif event.key == pygame.K_3 and 3 in level_loaders:
                        level_loaders[3]()
                    elif event.key == pygame.K_4 and 4 in level_loaders:
                        level_loaders[4]()

            self.agent.step()

            self.screen.fill(self.BG_COLOR)
            self.draw_environment()
            self.draw_membrane()
            self.draw_agent_and_target()

            pygame.display.flip()
            self.clock.tick(50)

        pygame.quit()
        sys.exit() 
