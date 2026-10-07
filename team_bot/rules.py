from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Rules:
    field_width: float = 100.0
    field_height: float = 140.0
    goal_width: float = 36.0
    player_radius: float = 3.0
    player_speed: float = 4.0
    ball_radius: float = 1.5
    ball_speed: float = 8.0
    possession_radius: float = 5.0
    kick_distances: tuple[float, ...] = (32.0, 64.0, 96.0)
    obstacle_count: int = 6
    obstacle_width: float = 12.0
    obstacle_height: float = 8.0
    maximum_iterations: int = 400
    maximum_goals: int = 7
    possession_limit_iterations: int = 10
    loose_ball_restart_iterations: int = 20
    initial_possessor: str = "player_1"

    def validate(self):
        pass
