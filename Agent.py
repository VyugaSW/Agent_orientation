import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from Neuron import Neuron
from Orientation_membrane import Orientation_membrane
from Environment import EnvironmentStrategy, DangerEntity, AttractorEntity


class Agent:
    def __init__(
        self,
        x: float,
        y: float,
        membrane: Orientation_membrane,
        environment: EnvironmentStrategy,
    ):
        self.position = np.array([x, y], dtype=float)
        self.membrane = membrane
        self.environment = environment

        self.target = np.array([x, y], dtype=float)
        self.dodge_direction = None
        self.prev_steer_direction = np.array([0.0, 0.0])
        self.stagnation_counter = 0

        self.params = {
            "look_ahead_cells": max(
                1, min(self.membrane.rows, self.membrane.cols) // 2
            ),
            "reaction_pad_mult": 0.5,
            "tangent_force": 3.0,
            "step_size": 0.25,
            "membrane_lr": 0.05,
            "membrane_relax": 0.25,
            "max_turn_angle": 0.05,
            "stagnation_threshold": 3,
        }

    def set_target(self, x_0: float, y_0: float) -> None:
        self.target = np.array([x_0, y_0], dtype=float)

    def _get_sensory_edge(self, movement_vector: np.ndarray) -> List[Neuron]:
        norm = np.linalg.norm(movement_vector)
        if norm < 0.001:
            edge_neurons = []
            for r in range(self.membrane.rows):
                for c in range(self.membrane.cols):
                    edge_neurons.append(self.membrane.grid[r][c])
            return edge_neurons

        direction = movement_vector / norm
        edge_neurons = []

        for r in range(self.membrane.rows):
            for c in range(self.membrane.cols):
                neuron = self.membrane.grid[r][c]
                rel_pos = neuron.position - self.position

                if np.dot(rel_pos, direction) > -0.1:
                    edge_neurons.append(neuron)

        return edge_neurons

    def _apply_stagnation_noise(self, local_stimulus: np.ndarray) -> np.ndarray:  
        stagnation_threshold = self.params["stagnation_threshold"]

        if self.stagnation_counter >= self.params["stagnation_threshold"]:
            noise_vector = np.random.randn(2)
            norm = np.linalg.norm(noise_vector)
            if norm > 0.001:
                noise_vector /= norm
            
            momentum_multiplier = 1.0 + (self.stagnation_counter - stagnation_threshold) * 0.15
            momentum_multiplier = min(momentum_multiplier, 3.0)
            
            noise_magnitude = self.params["step_size"] * momentum_multiplier
            
            return local_stimulus + noise_vector * noise_magnitude
            print("stagnation!\n")
        else:
            self.stagnation_counter += 1
            
        return local_stimulus

    def _calculate_local_stimulus(
        self,
        base_direction: np.ndarray,
        distance_to_target: float,
        look_ahead_distance: float,
        turn_dampener: float,
    ) -> Tuple[np.ndarray, List[Neuron]]:

        steer_direction = np.copy(base_direction)
        is_dodging_this_tick = False

        for entity in self.environment.active_entities:
            dist_to_entity = np.linalg.norm(self.position - entity.position)
            reaction_zone = (
                entity.radius + look_ahead_distance * self.params["reaction_pad_mult"]
            )

            if dist_to_entity < reaction_zone:
                is_dodging_this_tick = True
                repel = (self.position - entity.position) / dist_to_entity

                tangent_left = np.array([-repel[1], repel[0]])
                tangent_right = np.array([repel[1], -repel[0]])

                if self.dodge_direction is None:
                    dot_left = np.dot(tangent_left, base_direction)
                    dot_right = np.dot(tangent_right, base_direction)

                    if abs(dot_left - dot_right) < 0.05:
                        self.dodge_direction = "right"
                    elif dot_left > dot_right:
                        self.dodge_direction = "left"
                    else:
                        self.dodge_direction = "right"

                if self.dodge_direction == "left":
                    best_tangent = tangent_left
                else:
                    best_tangent = tangent_right

                intensity = 1.0 - (dist_to_entity / reaction_zone)

                steer_direction += best_tangent * (
                    intensity * self.params["tangent_force"] * turn_dampener
                )

        if not is_dodging_this_tick:
            self.dodge_direction = None

        steer_direction = steer_direction / np.linalg.norm(steer_direction)

        max_angle = self.params.get("max_turn_angle", 0.15)

        if (
            hasattr(self, "prev_steer_direction")
            and np.linalg.norm(self.prev_steer_direction) > 0.001
            and distance_to_target > look_ahead_distance
        ):
            dot_prod = np.clip(
                np.dot(self.prev_steer_direction, steer_direction), -1.0, 1.0
            )
            angle = np.arccos(dot_prod)

            if angle > max_angle:
                cross_prod = (
                    self.prev_steer_direction[0] * steer_direction[1]
                    - self.prev_steer_direction[1] * steer_direction[0]
                )
                sign = 1.0 if cross_prod > 0 else -1.0

                theta = sign * max_angle
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)

                steer_direction = np.array(
                    [
                        self.prev_steer_direction[0] * cos_t
                        - self.prev_steer_direction[1] * sin_t,
                        self.prev_steer_direction[0] * sin_t
                        + self.prev_steer_direction[1] * cos_t,
                    ]
                )

            angle = np.arccos(dot_prod)

        self.prev_steer_direction = np.copy(steer_direction)

        neurons_to_evaluate = self._get_sensory_edge(steer_direction)
        self.environment.update_neurons_cost(neurons_to_evaluate)

        projection_distance = min(look_ahead_distance, distance_to_target)
        local_stimulus = self.position + (steer_direction * projection_distance)

        local_stimulus = self._apply_stagnation_noise(local_stimulus)

        return local_stimulus, neurons_to_evaluate

    def step(self) -> None:
        movement_vector = self.target - self.position
        distance_to_target = np.linalg.norm(movement_vector)

        if distance_to_target < 0.1:
            return

        base_direction = movement_vector / distance_to_target
        look_ahead_distance = self.membrane.cell_size * self.params["look_ahead_cells"]
        turn_dampener = min(1.0, distance_to_target / look_ahead_distance)

        local_stimulus, neurons_to_evaluate = self._calculate_local_stimulus(
            base_direction, distance_to_target, look_ahead_distance, turn_dampener
        )

        bmu = self.membrane.find_bmu(local_stimulus, neurons_to_evaluate)

        if bmu:
            previous_position = np.copy(self.position)

            step_size = min(self.params["step_size"], distance_to_target)
            step_direction = bmu.position - self.position

            if np.linalg.norm(step_direction) > 0.001:
                step_direction = step_direction / np.linalg.norm(step_direction)
                self.position += step_direction * step_size

                for entity in self.environment.active_entities:
                    dist_to_entity = np.linalg.norm(self.position - entity.position)
                    if dist_to_entity < entity.radius:
                        repel = (self.position - entity.position) / dist_to_entity
                        self.position += repel * (entity.radius - dist_to_entity) * 0.5

                actual_shift = self.position - previous_position

                if np.linalg.norm(actual_shift) >= (self.params["step_size"] * 0.1):
                    self.stagnation_counter = 0

                self.membrane.adapt_membrane(
                    local_stimulus,
                    learning_rate=self.params["membrane_lr"],
                    radius_hops=min(self.membrane.rows, self.membrane.cols) // 2,
                    neurons=neurons_to_evaluate,
                )

                self.membrane.scroll_forward(actual_shift)

                self.membrane.apply_homeostasis(
                    relaxation_rate=self.params["membrane_relax"]
                )
