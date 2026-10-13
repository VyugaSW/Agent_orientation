import pygame
import numpy as np
import sys
from typing import Dict, Tuple, Callable

from Agent import Agent
from Environment import SpatialHashEnvironment
from InfluenceEntity import DangerEntity, AttractorEntity
from TopologyOptimizer import (
    HomologyMembraneConfigurator,
    EvolutionaryMembraneOptimizer,
)


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
        pygame.display.set_caption(
            "Scrolling SOM: Living Membrane [1-5: Levels | H: Homology | E: Evolution]"
        )

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 15)
        self.agent = agent
        self.environment = environment
        self.scale = pixels_per_meter

        # Colors (RGB)
        self.BG_COLOR = (30, 30, 30)
        self.GRID_COLOR = (100, 200, 255)
        self.AGENT_COLOR = (50, 255, 50)
        self.TARGET_COLOR = (255, 200, 0)
        self.DANGER_COLOR = (150, 30, 30)
        self.ATTRACTOR_COLOR = (30, 160, 90)
        self.MEMORY_COLOR = (160, 80, 255)

        self._circle_cache: Dict[
            Tuple[int, Tuple[int, int, int, int]], pygame.Surface
        ] = {}
        self.status_message: str = (
            "Keys [1-5]: Levels (5=Dynamic) | [H]: Homology | [E]: Evolution"
        )
        self.current_level: int = 1
        self.level_start_pos = np.copy(self.agent.position)

    def _to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Converts physical world coordinates to screen pixel coordinates."""
        return (int(x * self.scale), int(y * self.scale))

    def _get_cached_circle_surface(
        self, screen_radius: int, rgba: Tuple[int, int, int, int]
    ) -> pygame.Surface:
        key = (screen_radius, rgba)
        if key not in self._circle_cache:
            surf = pygame.Surface(
                (screen_radius * 2, screen_radius * 2), pygame.SRCALPHA
            )
            pygame.draw.circle(
                surf, rgba, (screen_radius, screen_radius), screen_radius
            )
            self._circle_cache[key] = surf
        return self._circle_cache[key]

    def draw_environment(self) -> None:
        """Draws all static, moving, and blinking entities present in the environment."""
        for entity in self.environment.active_entities:
            screen_pos = self._to_screen(entity.position[0], entity.position[1])
            screen_radius = max(1, int(entity.radius * self.scale))

            # If a blinking obstacle is currently inactive (disappeared), draw a faint ghost outline
            if not getattr(entity, "is_active", True):
                pygame.draw.circle(
                    self.screen, (80, 50, 50), screen_pos, screen_radius, 1
                )
                continue

            if isinstance(entity, DangerEntity):
                surf = self._get_cached_circle_surface(
                    screen_radius, (*self.DANGER_COLOR, 100)
                )
                self.screen.blit(
                    surf,
                    (screen_pos[0] - screen_radius, screen_pos[1] - screen_radius),
                )
                pygame.draw.circle(self.screen, self.DANGER_COLOR, screen_pos, 4)

                # Draw velocity vector indicator for moving obstacles
                if hasattr(entity, "velocity") and (
                    entity.velocity[0] != 0.0 or entity.velocity[1] != 0.0
                ):
                    vel_end = self._to_screen(
                        entity.position[0] + entity.velocity[0] * 18.0,
                        entity.position[1] + entity.velocity[1] * 18.0,
                    )
                    pygame.draw.line(
                        self.screen, (255, 120, 120), screen_pos, vel_end, 2
                    )
            elif isinstance(entity, AttractorEntity):
                surf = self._get_cached_circle_surface(
                    screen_radius, (*self.ATTRACTOR_COLOR, 90)
                )
                self.screen.blit(
                    surf,
                    (screen_pos[0] - screen_radius, screen_pos[1] - screen_radius),
                )
                pygame.draw.circle(self.screen, self.ATTRACTOR_COLOR, screen_pos, 4)

    def draw_memory_trail(self) -> None:
        """Draws the agent's decaying spatial memory trail (repulsive breadcrumbs)."""
        for cx, cy, strength in self.agent.memory_trail:
            pos = self._to_screen(cx, cy)
            radius_px = max(2, int(min(8.0, strength * 0.35)))
            pygame.draw.circle(self.screen, self.MEMORY_COLOR, pos, radius_px, 1)

    def draw_membrane(self) -> None:
        """Draws the Toroidal SOM grid. Avoids drawing phantom wrap-around links."""
        membrane = self.agent.membrane
        phantom_threshold = membrane.cell_size * 1.5

        for neuron in membrane.flat_neurons:
            pos1 = self._to_screen(neuron.position[0], neuron.position[1])
            for neighbor in neuron.neighbors:
                if id(neuron) < id(neighbor):
                    dist = neuron.get_euclidean_distance(neighbor.position)
                    if dist < phantom_threshold:
                        pos2 = self._to_screen(
                            neighbor.position[0], neighbor.position[1]
                        )
                        pygame.draw.line(self.screen, (70, 100, 120), pos1, pos2, 1)

        for neuron in membrane.flat_neurons:
            pos = self._to_screen(neuron.position[0], neuron.position[1])
            if neuron.k_fear > 1.05:
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
        pygame.draw.line(self.screen, (50, 100, 50), agent_pos, target_pos, 1)

    def draw_hud(self) -> None:
        """Renders telemetry, memory count, and homology metrics at the top of the screen."""
        b0, b1 = self.agent.last_betti
        m = self.agent.membrane
        mem_cnt = len(self.agent.memory_trail)
        hud_line1 = (
            f"Lvl: {self.current_level} | Grid: {m.rows}x{m.cols} (c={m.cell_size:.1f}) "
            f"| Betti: B0={b0},B1={b1} | Mem: {mem_cnt} | FPS: {int(self.clock.get_fps())}"
        )
        surf1 = self.font.render(hud_line1, True, (220, 220, 220))
        surf2 = self.font.render(self.status_message, True, (150, 230, 150))
        self.screen.blit(surf1, (10, 8))
        self.screen.blit(surf2, (10, 26))

    def _on_evolution_progress(self, gen: int, ind: int, best_loss: float) -> None:
        pygame.event.pump()
        self.status_message = (
            f"Evolving (1+lambda)-ES: Gen {gen}/8, Ind {ind}/6 | Best Loss: {best_loss:.1f}"
        )
        self.screen.fill(self.BG_COLOR)
        self.draw_environment()
        self.draw_agent_and_target()
        self.draw_hud()
        pygame.display.flip()

    def run(self, level_loaders: Dict[int, Callable[[], None]]) -> None:
        """Main visualization and physics loop."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    self.agent.set_target(mouse_x / self.scale, mouse_y / self.scale)
                elif event.type == pygame.KEYDOWN:
                    for key_num, pg_key in enumerate(
                        [
                            pygame.K_1,
                            pygame.K_2,
                            pygame.K_3,
                            pygame.K_4,
                            pygame.K_5,
                        ],
                        start=1,
                    ):
                        if event.key == pg_key and key_num in level_loaders:
                            self.current_level = key_num
                            level_loaders[key_num]()
                            self.agent.reset_memory()
                            self.level_start_pos = np.copy(self.agent.position)
                            self.status_message = f"Loaded Level {key_num}"

                    if event.key == pygame.K_h:
                        info = HomologyMembraneConfigurator.apply_to_agent(
                            self.agent, self.environment
                        )
                        self.status_message = (
                            f"[Homology] H0_gap={info['min_gap_h0']} H1_cav={info['cavity_scale_h1']} "
                            f"-> {info['rows']}x{info['cols']}, cell={info['cell_size']}"
                        )
                    elif event.key == pygame.K_e:
                        opt = EvolutionaryMembraneOptimizer(generations=8, pop_size=6)
                        res = opt.optimize_for_agent(
                            self.agent,
                            self.environment,
                            self.level_start_pos,
                            progress_callback=self._on_evolution_progress,
                        )
                        self.agent.reset_memory()
                        self.status_message = (
                            f"[Evolved] loss={res['loss']} scale=({res['scale_x']},{res['scale_y']}) "
                            f"curv={res['curvature']} fov={res['foveation']}"
                        )

            # Advance moving and blinking obstacles before stepping the agent
            self.environment.step_dynamic_entities()
            self.agent.step()

            self.screen.fill(self.BG_COLOR)
            self.draw_environment()
            self.draw_memory_trail()
            self.draw_membrane()
            self.draw_agent_and_target()
            self.draw_hud()

            pygame.display.flip()
            self.clock.tick(50)

        pygame.quit()
        sys.exit()