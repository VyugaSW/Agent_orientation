from collections import deque
import math
import numpy as np
from typing import Deque, List, Optional, Tuple

from Neuron import Neuron
from Orientation_membrane import Orientation_membrane
from Environment import EnvironmentStrategy


class Agent:
    """
    An autonomous agent that navigates a 2D environment using a Self-Organizing Map (SOM) 
    membrane for obstacle avoidance and pathfinding. The agent incorporates spatial memory 
    to detect and escape cyclical loops or dead ends.
    """

    def __init__(
        self,
        x: float,
        y: float,
        membrane: Orientation_membrane,
        environment: EnvironmentStrategy,
    ):
        """
        Initializes the agent with its starting position, sensory membrane, and environment.

        Args:
            x (float): Initial X coordinate in the world.
            y (float): Initial Y coordinate in the world.
            membrane (Orientation_membrane): The SOM grid used for spatial perception.
            environment (EnvironmentStrategy): The environment interface containing obstacles.
        """
        self.position = np.array([x, y], dtype=float)
        self.membrane = membrane
        self.environment = environment

        self.target = np.array([x, y], dtype=float)
        self.dodge_direction: Optional[str] = None
        self.dodge_persist_counter: int = 0
        self.prev_steer_direction = np.array([0.0, 0.0], dtype=float)
        
        # Tracks how many consecutive ticks the agent has failed to move significantly
        self.stagnation_counter: int = 0
        self.last_betti: Tuple[int, int] = (1, 0)

        # Spatial memory (Tabu / Pheromone breadcrumbs): stores [x, y, strength]
        self.memory_trail: Deque[List[float]] = deque(maxlen=90)
        
        # Sliding window of recent positions to detect multi-step oscillation loops
        self.position_history: Deque[np.ndarray] = deque(maxlen=40)
        self.is_in_loop: bool = False

        # Hyperparameters for navigation, membrane adaptation, and memory mechanics
        self.params = {
            "look_ahead_cells": max(
                1, min(self.membrane.rows, self.membrane.cols) // 2
            ),
            "reaction_pad_mult": 0.5,
            "tangent_force": 3.0,
            "step_size": 0.25,
            "membrane_lr": 0.05,
            "membrane_relax": 0.25,
            "max_turn_angle": 0.08,
            "stagnation_threshold": 5,
            "dodge_persistence": 15,
            "memory_radius": 8.0,
            "memory_decay": 0.985,
            "memory_strength": 15.0,
        }

    def set_target(self, x_0: float, y_0: float) -> None:
        """
        Assigns a new global target for the agent and resets its spatial memory.

        Args:
            x_0 (float): Target X coordinate.
            y_0 (float): Target Y coordinate.
        """
        self.target = np.array([x_0, y_0], dtype=float)
        self.reset_memory()

    def reset_memory(self) -> None:
        """
        Clears the spatial breadcrumb trail, trajectory history, and obstacle dodging states.
        Typically called when a new target is set or the agent is externally teleported.
        """
        self.memory_trail.clear()
        self.position_history.clear()
        self.dodge_direction = None
        self.dodge_persist_counter = 0
        self.stagnation_counter = 0
        self.is_in_loop = False

    def _update_spatial_memory_and_detect_loops(self) -> None:
        """
        Manages the agent's spatial memory system:
        1. Analyzes a sliding window of the last 40 positions to detect if the agent is stuck in a loop.
        2. Decays the repulsive strength of existing spatial breadcrumbs.
        3. Drops a new repulsive breadcrumb if the agent has moved a sufficient distance.
        """
        self.position_history.append(self.position.copy())
        self.is_in_loop = False
        
        # Detect stagnation loops by comparing total travel distance vs net displacement
        if len(self.position_history) == self.position_history.maxlen:
            oldest_pos = self.position_history[0]
            net_displacement = math.hypot(
                float(self.position[0] - oldest_pos[0]),
                float(self.position[1] - oldest_pos[1]),
            )
            expected_travel = len(self.position_history) * self.params["step_size"]
            
            # If net displacement is unusually low compared to expected steps, the agent is oscillating
            if net_displacement < expected_travel * 0.28:
                self.is_in_loop = True
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = max(0, self.stagnation_counter - 1)

        # Apply exponential decay to all active breadcrumbs and remove depleted ones
        decay = self.params["memory_decay"]
        surviving_trail: Deque[List[float]] = deque(maxlen=self.memory_trail.maxlen)
        for crumb in self.memory_trail:
            crumb[2] *= decay
            if crumb[2] > 0.5:
                surviving_trail.append(crumb)
        self.memory_trail = surviving_trail

        # Determine if the agent has traveled far enough to drop a new breadcrumb
        min_crumb_spacing = 1.5
        should_drop = False
        if not self.memory_trail:
            should_drop = True
        else:
            last_x, last_y, _ = self.memory_trail[-1]
            if (
                math.hypot(
                    float(self.position[0] - last_x), float(self.position[1] - last_y)
                )
                >= min_crumb_spacing
            ):
                should_drop = True

        # Drop a new breadcrumb; increase its strength if currently dodging or stuck in a loop
        if should_drop:
            base_strength = self.params["memory_strength"]
            if self.is_in_loop:
                base_strength *= 2.2
            elif self.dodge_direction is not None:
                base_strength *= 1.3
            self.memory_trail.append(
                [float(self.position[0]), float(self.position[1]), float(base_strength)]
            )

    def _apply_spatial_memory_to_neurons(self, neurons: List[Neuron]) -> None:
        """
        Increases the 'k_fear' (repulsion multiplier) of membrane neurons that physically overlap 
        with the agent's recent spatial breadcrumbs.

        Args:
            neurons (List[Neuron]): The list of neurons to evaluate and update.
        """
        # Ignore the 5 most recent breadcrumbs directly under the agent to prevent self-repulsion
        if len(self.memory_trail) <= 5:
            return

        mem_radius = self.params["memory_radius"]
        active_crumbs = list(self.memory_trail)[:-5]
        
        # Amplify the fear multiplier if the agent realizes it's stuck in a loop
        weight_mult = 1.4 if self.is_in_loop else 0.65

        for neuron in neurons:
            nx, ny = float(neuron.position[0]), float(neuron.position[1])
            memory_fear = 0.0
            
            for cx, cy, strength in active_crumbs:
                dist = math.hypot(nx - cx, ny - cy)
                if dist < mem_radius:
                    # Quadratic falloff for smoother spatial gradient
                    falloff = (1.0 - (dist / mem_radius)) ** 2
                    memory_fear += strength * falloff * weight_mult

            if memory_fear > 0.0:
                neuron.k_fear += memory_fear

    def _get_sensory_edge(self, movement_vector: np.ndarray) -> List[Neuron]:
        """
        Filters the membrane to return only the neurons located in the front hemisphere 
        relative to the agent's current movement direction.

        Args:
            movement_vector (np.ndarray): The current steering direction vector.

        Returns:
            List[Neuron]: A subset of neurons facing the movement direction.
        """
        norm = float(np.linalg.norm(movement_vector))
        if norm < 0.001:
            return list(self.membrane.flat_neurons)

        direction = movement_vector / norm
        edge_neurons: List[Neuron] = []
        
        # Widen the sensory cone if trapped in a loop to detect escape routes to the sides
        dot_threshold = -0.5 if self.is_in_loop else -0.1

        for neuron in self.membrane.flat_neurons:
            rel_pos = neuron.position - self.position
            # Dot product checks alignment between neuron's relative position and agent's heading
            if float(np.dot(rel_pos, direction)) > dot_threshold:
                edge_neurons.append(neuron)

        # Fallback to the whole membrane if no neurons passed the filter
        return edge_neurons if edge_neurons else list(self.membrane.flat_neurons)

    def _apply_stagnation_noise(self, local_stimulus: np.ndarray) -> np.ndarray:
        """
        Adds isotropic random noise to the local stimulus coordinate if the agent has been 
        stuck for too long. This helps break symmetry and escape local minima.

        Args:
            local_stimulus (np.ndarray): The current target coordinate for the BMU search.

        Returns:
            np.ndarray: The potentially perturbed target coordinate.
        """
        stagnation_threshold = self.params["stagnation_threshold"]

        if self.stagnation_counter >= stagnation_threshold:
            noise_vector = np.random.randn(2)
            norm = float(np.linalg.norm(noise_vector))
            if norm > 0.001:
                noise_vector /= norm

            # Noise amplitude grows linearly the longer the agent remains stuck
            momentum_multiplier = min(
                3.0, 1.0 + (self.stagnation_counter - stagnation_threshold) * 0.15
            )
            noise_magnitude = self.params["step_size"] * momentum_multiplier
            
            return local_stimulus + noise_vector * noise_magnitude

        return local_stimulus

    def _calculate_local_stimulus(
        self,
        base_direction: np.ndarray,
        distance_to_target: float,
        look_ahead_distance: float,
        turn_dampener: float,
    ) -> Tuple[np.ndarray, List[Neuron]]:
        """
        Synthesizes the optimal steering direction by combining the base target direction, 
        obstacle repulsion, obstacle tangents, and spatial memory repulsion. 
        Calculates the local stimulus point used to attract the SOM membrane.

        Args:
            base_direction (np.ndarray): Normalized vector pointing directly to the global target.
            distance_to_target (float): Distance remaining to the global target.
            look_ahead_distance (float): How far ahead the agent projects its stimulus.
            turn_dampener (float): Scalar to smoothly reduce violent steering when close to target.

        Returns:
            Tuple[np.ndarray, List[Neuron]]: The calculated physical coordinate of the local stimulus 
                                             and the list of neurons selected for evaluation.
        """
        steer_direction = np.copy(base_direction)
        is_dodging_this_tick = False

        # Define an expanded search radius to anticipate incoming obstacles
        query_radius = (
            look_ahead_distance * (1.0 + self.params["reaction_pad_mult"]) + 15.0
        )
        nearby_obstacles = self.environment.get_nearby_obstacles(
            self.position, query_radius
        )

        for entity in nearby_obstacles:
            # Predict obstacle position 6 ticks ahead if it possesses a velocity attribute
            obs_vel = getattr(entity, "velocity", np.zeros(2))
            predicted_pos = entity.position + obs_vel * 6.0

            diff = self.position - predicted_pos
            dist_to_entity = math.hypot(float(diff[0]), float(diff[1]))
            
            # The critical distance at which the agent starts executing avoidance maneuvers
            reaction_zone = (
                entity.radius + look_ahead_distance * self.params["reaction_pad_mult"]
            )

            if 1e-5 < dist_to_entity < reaction_zone:
                is_dodging_this_tick = True
                repel = diff / dist_to_entity

                # Calculate left and right tangential evasion vectors perpendicular to the repulsion vector
                tangent_left = np.array([-repel[1], repel[0]])
                tangent_right = np.array([repel[1], -repel[0]])

                # Decide dodge direction based on which tangent aligns better with the base direction
                if self.dodge_direction is None:
                    dot_left = float(np.dot(tangent_left, base_direction))
                    dot_right = float(np.dot(tangent_right, base_direction))

                    # Include a small hysteresis buffer (0.05) to favor 'right' to prevent jittering
                    if abs(dot_left - dot_right) < 0.05:
                        self.dodge_direction = "right"
                    elif dot_left > dot_right:
                        self.dodge_direction = "left"
                    else:
                        self.dodge_direction = "right"

                best_tangent = (
                    tangent_left if self.dodge_direction == "left" else tangent_right
                )
                intensity = 1.0 - (dist_to_entity / reaction_zone)
                
                # Blend the chosen tangent into the overall steering vector
                steer_direction += best_tangent * (
                    intensity * self.params["tangent_force"] * turn_dampener
                )

        # Maintain dodge persistence to prevent jittering or trapping when scraping against long walls
        if is_dodging_this_tick:
            self.dodge_persist_counter = int(self.params.get("dodge_persistence", 15))
        else:
            if self.dodge_persist_counter > 0 and self.is_in_loop:
                self.dodge_persist_counter -= 1
            else:
                self.dodge_persist_counter = 0
                self.dodge_direction = None

        # Apply direct steering repulsion away from heavily visited breadcrumb clusters when looping
        if len(self.memory_trail) > 6:
            mem_repel = np.zeros(2, dtype=float)
            mem_radius = self.params["memory_radius"] * 1.2
            
            for cx, cy, strength in list(self.memory_trail)[:-6]:
                dx = float(self.position[0] - cx)
                dy = float(self.position[1] - cy)
                d = math.hypot(dx, dy)
                if 1e-4 < d < mem_radius:
                    # Inverse square falloff weighted by the historical breadcrumb strength
                    w = ((1.0 - d / mem_radius) ** 2) * (strength / 20.0)
                    mem_repel[0] += (dx / d) * w
                    mem_repel[1] += (dy / d) * w
                    
            repel_gain = 1.3 if self.is_in_loop else 0.45
            steer_direction += mem_repel * repel_gain

        # Normalize the final steering vector
        steer_norm = float(np.linalg.norm(steer_direction))
        if steer_norm > 1e-5:
            steer_direction /= steer_norm
        else:
            steer_direction = np.copy(base_direction)

        # Constrain the maximum turn angle in SO(2) to ensure smooth kinematic trajectories
        max_angle = self.params.get("max_turn_angle", 0.08)
        prev_norm = float(np.linalg.norm(self.prev_steer_direction))

        if prev_norm > 0.001 and distance_to_target > look_ahead_distance:
            dot_prod = float(
                np.clip(np.dot(self.prev_steer_direction, steer_direction), -1.0, 1.0)
            )
            angle = math.acos(dot_prod)

            if angle > max_angle:
                # Calculate the 2D cross product to determine the sign of the required rotation
                cross_prod = (
                    self.prev_steer_direction[0] * steer_direction[1]
                    - self.prev_steer_direction[1] * steer_direction[0]
                )
                sign = 1.0 if cross_prod > 0 else -1.0
                theta = sign * max_angle
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)

                # Rotate the previous steering direction by the maximum allowed angle
                steer_direction = np.array(
                    [
                        self.prev_steer_direction[0] * cos_t
                        - self.prev_steer_direction[1] * sin_t,
                        self.prev_steer_direction[0] * sin_t
                        + self.prev_steer_direction[1] * cos_t,
                    ]
                )

        self.prev_steer_direction = np.copy(steer_direction)

        # Extract the front-facing sensory edge and update environmental costs
        neurons_to_evaluate = self._get_sensory_edge(steer_direction)
        self.environment.update_neurons_cost(neurons_to_evaluate)
        self._apply_spatial_memory_to_neurons(neurons_to_evaluate)

        # Compute topological Betti numbers (beta_0, beta_1) for analysis and hud rendering
        beta_0, beta_1, _ = self.membrane.compute_homology_betti()
        self.last_betti = (beta_0, beta_1)

        # Project the local stimulus physically forward along the steering vector
        projection_distance = min(look_ahead_distance, distance_to_target)
        local_stimulus = self.position + (steer_direction * projection_distance)
        local_stimulus = self._apply_stagnation_noise(local_stimulus)

        return local_stimulus, neurons_to_evaluate

    def step(self) -> None:
        """
        Executes a single logical tick of the agent's navigation loop:
        1. Checks distance to target.
        2. Calculates the optimal steering stimulus.
        3. Finds the Best Matching Unit (BMU) on the SOM membrane.
        4. Moves the agent towards the BMU and resolves hard physical collisions.
        5. Updates spatial memory and topological bounds.
        6. Adapts and scrolls the SOM membrane.
        """
        movement_vector = self.target - self.position
        distance_to_target = float(np.linalg.norm(movement_vector))

        # Stop condition
        if distance_to_target < 0.1:
            return

        # Clear memory trail if the agent was externally teleported (e.g., during level switch)
        if self.position_history:
            jump = float(np.linalg.norm(self.position - self.position_history[-1]))
            if jump > self.params["step_size"] * 8.0:
                self.reset_memory()

        base_direction = movement_vector / distance_to_target
        look_ahead_distance = self.membrane.cell_size * self.params["look_ahead_cells"]
        
        # Dampen drastic turns when approaching the global target
        turn_dampener = min(1.0, distance_to_target / max(1e-5, look_ahead_distance))

        local_stimulus, neurons_to_evaluate = self._calculate_local_stimulus(
            base_direction, distance_to_target, look_ahead_distance, turn_dampener
        )

        # Identify the Best Matching Unit mapping to the local stimulus
        bmu = self.membrane.find_bmu(local_stimulus, neurons_to_evaluate)

        if bmu is not None:
            previous_position = np.copy(self.position)

            step_size = min(self.params["step_size"], distance_to_target)
            step_direction = bmu.position - self.position
            step_norm = float(np.linalg.norm(step_direction))

            if step_norm > 0.001:
                step_direction /= step_norm
                self.position += step_direction * step_size

                # Hard collision resolution: Physically push the agent outside of any intersected obstacles
                nearby_obstacles = self.environment.get_nearby_obstacles(
                    self.position, 20.0
                )
                for entity in nearby_obstacles:
                    diff = self.position - entity.position
                    dist_to_entity = math.hypot(float(diff[0]), float(diff[1]))
                    if 1e-5 < dist_to_entity < entity.radius:
                        repel = diff / dist_to_entity
                        self.position += repel * (entity.radius - dist_to_entity) * 0.85

                actual_shift = self.position - previous_position

                # Update stagnation metrics if the physical shift was extremely small
                if float(np.linalg.norm(actual_shift)) < (
                    self.params["step_size"] * 0.15
                ):
                    self.stagnation_counter += 1

                self._update_spatial_memory_and_detect_loops()

                # Adapt the membrane structure towards the local stimulus
                self.membrane.adapt_membrane(
                    local_stimulus,
                    learning_rate=self.params["membrane_lr"],
                    radius_hops=max(
                        1, min(self.membrane.rows, self.membrane.cols) // 2
                    ),
                    neurons=neurons_to_evaluate,
                    bmu=bmu,
                )

                # Execute O(N) toroidal scroll based on actual movement
                self.membrane.scroll_forward(actual_shift)

                # Re-anchor the membrane nodes elastically
                self.membrane.apply_homeostasis(
                    relaxation_rate=self.params["membrane_relax"]
                )
            else:
                self.stagnation_counter += 1
