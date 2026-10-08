"""Exact stdlib replica of the engine's ball and movement physics.

Every function here mirrors a specific routine in ``soccer_env/engine.py`` and
is held to it by ``tools/diff_physics.py``. Constants come from
``config/game.json`` and are re-asserted against the observation's ``field``
block at runtime, because ``GameConfig()``'s dataclass defaults differ from the
official file.

Mirrored routines:
    circle_hits_rect   <- SoccerEnv._circle_hits_rectangle
    valid_center       <- SoccerEnv._valid_player_position
    goal_scorer        <- SoccerEnv._goal_scorer
    _bounce_walls      <- SoccerEnv._bounce_from_walls
    _bounce_obstacles  <- SoccerEnv._bounce_from_obstacles
    advance_one_step   <- SoccerEnv._move_ball
    kick_spawn         <- SoccerEnv._start_kick
"""
from __future__ import annotations

from math import ceil, hypot
from typing import Any

FIELD_W = 100.0
FIELD_H = 140.0
GOAL_W = 36.0
GOAL_LEFT = (FIELD_W - GOAL_W) / 2.0      # 32.0
GOAL_RIGHT = GOAL_LEFT + GOAL_W           # 68.0
PLAYER_R = 3.0
PLAYER_SPEED = 4.0
BALL_R = 1.5
BALL_SPEED = 8.0
POSSESSION_R = 5.0
KICK_DISTANCES = (32.0, 64.0, 96.0)
POSSESSION_LIMIT = 10
LOOSE_RESTART = 20
MAX_ITERATIONS = 400
MAX_GOALS = 7

# Derived
CATCH_R = BALL_R + PLAYER_R               # 4.5  interception radius
TACKLE_R = 2.0 * PLAYER_R + 0.15          # 6.15
CONTACT_R = 2.0 * PLAYER_R                # 6.0
KICK_CLEARANCE = PLAYER_R + BALL_R + 0.05  # 4.55
SUBSTEP_LIMIT = max(0.25, BALL_R * 0.45)  # 0.675
OBSTACLE_NUDGE = min(0.05, BALL_R / 10.0)  # 0.05
GOAL_TOP_Y = FIELD_H - BALL_R             # 138.5  ball centre scores at/above
GOAL_BOTTOM_Y = BALL_R                    # 1.5

ROOT2_2 = 0.7071067811865476

# Canonical move order. Index 0 is STAY; the engine's DIRECTIONS dict order.
MOVES = (
    "STAY", "UP", "UP_RIGHT", "RIGHT", "DOWN_RIGHT",
    "DOWN", "DOWN_LEFT", "LEFT", "UP_LEFT",
)
KICK_DIRS = MOVES[1:]
# Unit vectors, exactly as ``_unit(DIRECTIONS[name])`` produces them.
UNIT = {
    "STAY": (0.0, 0.0),
    "UP": (0.0, 1.0),
    "UP_RIGHT": (ROOT2_2, ROOT2_2),
    "RIGHT": (1.0, 0.0),
    "DOWN_RIGHT": (ROOT2_2, -ROOT2_2),
    "DOWN": (0.0, -1.0),
    "DOWN_LEFT": (-ROOT2_2, -ROOT2_2),
    "LEFT": (-1.0, 0.0),
    "UP_LEFT": (-ROOT2_2, ROOT2_2),
}
# Step offsets a move produces (unit * player_speed).
STEP = {name: (vec[0] * PLAYER_SPEED, vec[1] * PLAYER_SPEED) for name, vec in UNIT.items()}

FLIP_MOVE = {
    "STAY": "STAY", "UP": "DOWN", "UP_RIGHT": "DOWN_RIGHT", "RIGHT": "RIGHT",
    "DOWN_RIGHT": "UP_RIGHT", "DOWN": "UP", "DOWN_LEFT": "UP_LEFT",
    "LEFT": "LEFT", "UP_LEFT": "DOWN_LEFT",
}


def assert_official(field: dict[str, Any]) -> bool:
    """Confirm the observation's field block matches our hard-coded constants."""
    return (
        abs(float(field["width"]) - FIELD_W) < 1e-9
        and abs(float(field["height"]) - FIELD_H) < 1e-9
        and abs(float(field["goal_width"]) - GOAL_W) < 1e-9
        and abs(float(field["player_radius"]) - PLAYER_R) < 1e-9
        and abs(float(field["player_speed"]) - PLAYER_SPEED) < 1e-9
    )


def circle_hits_rect(cx: float, cy: float, radius: float, rect: tuple) -> bool:
    """Mirror of ``SoccerEnv._circle_hits_rectangle`` (strict ``<``)."""
    rx, ry, rw, rh = rect
    nx = rx if cx < rx else (rx + rw if cx > rx + rw else cx)
    ny = ry if cy < ry else (ry + rh if cy > ry + rh else cy)
    dx = cx - nx
    dy = cy - ny
    return dx * dx + dy * dy < radius * radius


def valid_center(x: float, y: float, obstacles: tuple) -> bool:
    """Mirror of ``SoccerEnv._valid_player_position``. Boundary touching is legal."""
    if x - PLAYER_R < 0.0 or x + PLAYER_R > FIELD_W:
        return False
    if y - PLAYER_R < 0.0 or y + PLAYER_R > FIELD_H:
        return False
    for rect in obstacles:
        rx, ry, rw, rh = rect
        nx = rx if x < rx else (rx + rw if x > rx + rw else x)
        ny = ry if y < ry else (ry + rh if y > ry + rh else y)
        dx = x - nx
        dy = y - ny
        if dx * dx + dy * dy < 9.0:  # PLAYER_R ** 2
            return False
    return True


def goal_scorer(x: float, y: float) -> int:
    """0 = no goal, 1 = top goal (player_1 scores), -1 = bottom goal."""
    if GOAL_LEFT <= x <= GOAL_RIGHT:
        if y + BALL_R >= FIELD_H:
            return 1
        if y - BALL_R <= 0.0:
            return -1
    return 0


def kick_spawn(cx: float, cy: float, direction: str) -> tuple[float, float]:
    """Ball centre right after ``_start_kick`` from a post-move player centre."""
    ux, uy = UNIT[direction]
    return (cx + ux * KICK_CLEARANCE, cy + uy * KICK_CLEARANCE)


def _bounce_walls(x: float, y: float, vx: float, vy: float):
    """Mirror of ``_bounce_from_walls``. Returns (x, y, vx, vy, bounced)."""
    bounced = False
    if x - BALL_R < 0.0 or x + BALL_R > FIELD_W:
        vx = -vx
        x = BALL_R if x < BALL_R else (FIELD_W - BALL_R if x > FIELD_W - BALL_R else x)
        bounced = True
    if y - BALL_R < 0.0 or y + BALL_R > FIELD_H:
        vy = -vy
        y = BALL_R if y < BALL_R else (FIELD_H - BALL_R if y > FIELD_H - BALL_R else y)
        bounced = True
    return x, y, vx, vy, bounced


def _bounce_obstacles(px: float, py: float, cx: float, cy: float,
                      vx: float, vy: float, obstacles: tuple):
    """Mirror of ``_bounce_from_obstacles``. Handles only the first hit obstacle."""
    for rect in obstacles:
        if not circle_hits_rect(cx, cy, BALL_R, rect):
            continue
        rx, ry, rw, rh = rect
        left = rx - BALL_R
        right = rx + rw + BALL_R
        bottom = ry - BALL_R
        top = ry + rh + BALL_R
        crossed_x = px <= left or px >= right
        crossed_y = py <= bottom or py >= top
        if crossed_x:
            vx = -vx
        if crossed_y:
            vy = -vy
        if not crossed_x and not crossed_y:
            vx = -vx
            vy = -vy
        length = hypot(vx, vy)
        if length == 0.0:
            ux = uy = 0.0
        else:
            ux = vx / length
            uy = vy / length
        return px + ux * OBSTACLE_NUDGE, py + uy * OBSTACLE_NUDGE, vx, vy, True
    return cx, cy, vx, vy, False


def advance_one_step(bx: float, by: float, vx: float, vy: float, remaining: float,
                     obstacles: tuple, catchers: tuple = ()):
    """Exact mirror of one ``SoccerEnv._move_ball()`` call.

    ``catchers`` is a tuple of ``(key, x, y)`` checked in order, mirroring the
    engine's ``for player in PLAYERS`` interception loop (player_1 first).

    Returns a dict with the post-call ball state plus the sub-step path.
    """
    path: list = []
    if remaining <= 0.0 or (vx == 0.0 and vy == 0.0):
        return {"x": bx, "y": by, "vx": vx, "vy": vy, "remaining": remaining,
                "scorer": 0, "caught": None, "path": path, "stopped": False}

    travel = BALL_SPEED if remaining > BALL_SPEED else remaining
    substeps = ceil(travel / SUBSTEP_LIMIT)
    if substeps < 1:
        substeps = 1
    step_distance = travel / substeps

    for _ in range(substeps):
        length = hypot(vx, vy)
        if length == 0.0:
            ux = uy = 0.0
        else:
            ux = vx / length
            uy = vy / length
        px, py = bx, by
        cx = px + ux * step_distance
        cy = py + uy * step_distance

        scorer = goal_scorer(cx, cy)
        if scorer:
            path.append((cx, cy))
            return {"x": cx, "y": cy, "vx": 0.0, "vy": 0.0, "remaining": 0.0,
                    "scorer": scorer, "caught": None, "path": path, "stopped": True}

        cx, cy, vx, vy, _ = _bounce_walls(cx, cy, vx, vy)
        cx, cy, vx, vy, _ = _bounce_obstacles(px, py, cx, cy, vx, vy, obstacles)
        bx, by = cx, cy
        remaining = remaining - step_distance
        if remaining < 0.0:
            remaining = 0.0
        path.append((bx, by))

        for key, hx, hy in catchers:
            dx = bx - hx
            dy = by - hy
            if dx * dx + dy * dy <= 20.25:  # CATCH_R ** 2
                return {"x": hx, "y": hy, "vx": 0.0, "vy": 0.0, "remaining": 0.0,
                        "scorer": 0, "caught": key, "path": path, "stopped": True}

    stopped = False
    if remaining <= 1e-9:
        remaining = 0.0
        vx = vy = 0.0
        stopped = True
    return {"x": bx, "y": by, "vx": vx, "vy": vy, "remaining": remaining,
            "scorer": 0, "caught": None, "path": path, "stopped": stopped}


def simulate_ball(bx: float, by: float, ux: float, uy: float, remaining: float,
                  obstacles: tuple, max_steps: int = 14, catchers: tuple = ()):
    """Roll a loose ball forward, one engine step at a time.

    Returns ``(steps, outcome)`` where ``steps`` is a list (one entry per engine
    step) of sub-step ``(x, y)`` lists, and ``outcome`` is a dict with
    ``scorer`` (0 / 1 / -1), ``caught``, ``x``, ``y``, ``n_steps`` and
    ``starts`` -- the ``(x, y, vx, vy)`` the ball had entering each engine
    step, which is exactly what a chasing opponent observes.
    """
    vx = ux * BALL_SPEED
    vy = uy * BALL_SPEED
    steps: list = []
    starts: list = []
    for _ in range(max_steps):
        starts.append((bx, by, vx, vy))
        out = advance_one_step(bx, by, vx, vy, remaining, obstacles, catchers)
        steps.append(out["path"])
        bx, by, vx, vy, remaining = out["x"], out["y"], out["vx"], out["vy"], out["remaining"]
        if out["scorer"] or out["caught"] is not None or out["stopped"]:
            return steps, {"scorer": out["scorer"], "caught": out["caught"],
                           "x": bx, "y": by, "n_steps": len(steps), "starts": starts}
        if remaining <= 0.0:
            break
    return steps, {"scorer": 0, "caught": None, "x": bx, "y": by,
                   "n_steps": len(steps), "starts": starts}


def segment_point_min_distance(ax: float, ay: float, bxx: float, byy: float,
                               px: float, py: float) -> float:
    """Shortest distance from (px, py) to the segment (ax, ay)-(bxx, byy)."""
    dx = bxx - ax
    dy = byy - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-18:
        return hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return hypot(px - (ax + t * dx), py - (ay + t * dy))


def ray_rect_blocked(ax: float, ay: float, ux: float, uy: float, length: float,
                     rect: tuple) -> bool:
    """True if a ball of radius BALL_R travelling the ray would touch ``rect``.

    Slab test against the rectangle inflated by BALL_R. Used only as a fast
    pre-filter; the exact simulation is always the authority.
    """
    rx, ry, rw, rh = rect
    minx = rx - BALL_R
    maxx = rx + rw + BALL_R
    miny = ry - BALL_R
    maxy = ry + rh + BALL_R
    t0 = 0.0
    t1 = length
    for origin, direction, lo, hi in ((ax, ux, minx, maxx), (ay, uy, miny, maxy)):
        if -1e-12 < direction < 1e-12:
            if origin <= lo or origin >= hi:
                return False
            continue
        inv = 1.0 / direction
        ta = (lo - origin) * inv
        tb = (hi - origin) * inv
        if ta > tb:
            ta, tb = tb, ta
        if ta > t0:
            t0 = ta
        if tb < t1:
            t1 = tb
        if t0 > t1:
            return False
    return True
