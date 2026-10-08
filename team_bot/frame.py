"""Canonicalisation to an always-attack-UP frame.

The field is exactly symmetric under ``y -> field_height - y`` (verified in
IMPLEMENTATION_NOTES), so player_2's problem becomes player_1's problem under
that reflection. Canonicalising means the planner has a single code path and
both sides are tested by every change.

The reflection flips handedness, so UP_RIGHT <-> DOWN_RIGHT on the way out;
``physics.FLIP_MOVE`` holds the mapping. x is never touched.
"""
from __future__ import annotations

from typing import Any

from . import physics as ph


class View:
    """A canonical, flat snapshot of one observation. Attack is always +y."""

    __slots__ = (
        "flipped", "me", "opp", "ball", "ball_vel", "status", "owner",
        "remaining", "steps", "loose_steps", "obstacles", "iteration",
        "max_iterations", "score_me", "score_opp", "seed", "player_id", "raw",
    )

    def __init__(self) -> None:
        self.flipped = False
        self.me = (0.0, 0.0)
        self.opp = (0.0, 0.0)
        self.ball = (0.0, 0.0)
        self.ball_vel = (0.0, 0.0)
        self.status = "stationary"
        self.owner = None          # "me", "opp" or None
        self.remaining = 0.0
        self.steps = 0
        self.loose_steps = 0
        self.obstacles = ()
        self.iteration = 0
        self.max_iterations = ph.MAX_ITERATIONS
        self.score_me = 0
        self.score_opp = 0
        self.seed = 0
        self.player_id = "player_1"
        self.raw = None


def canonicalize(observation: dict[str, Any]) -> View:
    view = View()
    player_id = observation["player_id"]
    opponent_id = observation["opponent_id"]
    state = observation["state"]
    players = state["players"]
    ball = state["ball"]

    flipped = observation.get("attack_direction") != "UP"
    view.flipped = flipped
    view.player_id = player_id
    view.raw = observation

    height = ph.FIELD_H
    if flipped:
        def fy(value: float) -> float:
            return height - value
    else:
        def fy(value: float) -> float:
            return value

    me = players[player_id]
    opp = players[opponent_id]
    view.me = (float(me["x"]), fy(float(me["y"])))
    view.opp = (float(opp["x"]), fy(float(opp["y"])))
    view.ball = (float(ball["x"]), fy(float(ball["y"])))

    velocity = ball.get("velocity") or {}
    vx = float(velocity.get("x", 0.0))
    vy = float(velocity.get("y", 0.0))
    view.ball_vel = (vx, -vy if flipped else vy)

    view.status = ball.get("status", "stationary")
    possession = ball.get("possession")
    view.owner = "me" if possession == player_id else ("opp" if possession == opponent_id else None)
    view.remaining = float(ball.get("remaining_kick_distance", 0.0) or 0.0)
    view.steps = int(ball.get("possession_steps", 0) or 0)
    view.loose_steps = int(ball.get("loose_ball_steps", 0) or 0)

    rects = []
    for obstacle in state.get("obstacles", ()):  
        ox = float(obstacle["x"])
        oy = float(obstacle["y"])
        ow = float(obstacle["width"])
        oh = float(obstacle["height"])
        rects.append((ox, height - oy - oh if flipped else oy, ow, oh))
    view.obstacles = tuple(rects)

    view.iteration = int(state.get("iteration", 0) or 0)
    view.max_iterations = int(state.get("maximum_iterations", ph.MAX_ITERATIONS) or ph.MAX_ITERATIONS)
    score = state.get("score") or {}
    view.score_me = int(score.get(player_id, 0) or 0)
    view.score_opp = int(score.get(opponent_id, 0) or 0)
    view.seed = int(state.get("seed", 0) or 0)
    return view


def decanonicalize(action: dict[str, Any], flipped: bool) -> dict[str, Any]:
    """Map a canonical-frame action back to the engine's frame."""
    if not flipped:
        return action
    out = {"move": ph.FLIP_MOVE[action["move"]]}
    kick = action.get("kick")
    if kick is not None:
        out["kick"] = {"direction": ph.FLIP_MOVE[kick["direction"]], "power": kick["power"]}
    return out
