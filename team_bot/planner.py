"""Tactical planner: exhaustive move+kick search over the exact physics.

All geometry is in the canonical attack-UP frame produced by ``frame.py``.

Timing model used throughout (derived from ``SoccerEnv.step``, which moves the
players *before* the ball):

    step T   : we move to p', the kick fires, the ball runs ball-step 0
    step T+k : we have moved k more times, the ball runs ball-step k

So at ball-step ``k`` our own reach from ``p'`` is ``4.5 + 4k``, while the
opponent -- which also moved during the kick step -- has a reach of
``4.5 + 4(k + 1)`` from the position we *observed*. That extra move is the
correction recorded in IMPLEMENTATION_NOTES; using ``4.5 + 4k`` for the
opponent would call reachable shots safe.
"""
from __future__ import annotations

from math import hypot
from time import perf_counter
from typing import Any

from . import opponent_model as om
from . import physics as ph

CATCH = ph.CATCH_R
SPEED = ph.PLAYER_SPEED
BIG = 1e9

# Contact resolution runs *before* the kick fires and can cancel or redirect our
# move, so the ball then spawns from somewhere other than the post-move position
# a proof assumed. The engine resolves contact only when the two proposed
# positions are closer than 2*player_radius, and the opponent's proposed position
# lies within player_speed of the one we observed; so if the post-move position
# is at least this far from the observed opponent, contact is impossible and the
# position is guaranteed.
CONTACT_SAFE = ph.CONTACT_R + ph.PLAYER_SPEED   # 10.0
GAP_CAP = 40.0   # larger than any coverable deficit on a 100 x 140 field


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------

def legal_moves(x: float, y: float, obstacles: tuple) -> list:
    """(move_name, px, py) for every move whose target the engine accepts.

    An illegal target makes the engine leave the player where it is, so such a
    move is behaviourally STAY -- except that it still counts as non-STAY for
    tackling, which ``defend_choice`` exploits.
    """
    out = [("STAY", x, y)]
    for name in ph.KICK_DIRS:
        dx, dy = ph.STEP[name]
        px = x + dx
        py = y + dy
        if ph.valid_center(px, py, obstacles):
            out.append((name, px, py))
    return out


def first_reach(steps: list, px: float, py: float,
                base: float, per_step: float) -> tuple:
    """Earliest ``(ball_step, sub_step)`` at which the growing reach disc covers
    the ball, or ``None``.

    Step granularity is not enough for release planning: the engine tests
    interception once per *sub-step*, so two players that both "arrive at step
    0" are not tied -- the one whose disc covers an earlier sub-step takes the
    ball. Comparing at step granularity made the planner rate a kick launched
    straight into an opponent 7 units away as an even race.
    """
    for k, path in enumerate(steps):
        radius = base + per_step * k
        limit = radius * radius
        for index in range(len(path)):
            bx, by = path[index]
            dx = bx - px
            dy = by - py
            if dx * dx + dy * dy <= limit:
                return (k, index)
    return None


def first_reach_step(steps: list, px: float, py: float,
                     base: float, per_step: float,
                     stop_before: int = -1) -> int:
    """Earliest ball-step index whose path enters the growing reach disc.

    Returns -1 when the player never reaches the ball. The disc radius at
    ball-step ``k`` is ``base + per_step * k``.
    """
    for k, path in enumerate(steps):
        if stop_before >= 0 and k >= stop_before:
            return -1
        radius = base + per_step * k
        limit = radius * radius
        for (bx, by) in path:
            dx = bx - px
            dy = by - py
            if dx * dx + dy * dy <= limit:
                return k
    return -1


def path_clearance(steps: list, px: float, py: float,
                   base: float, per_step: float,
                   goal_step: int, goal_index: int) -> float:
    """Smallest (distance - reach) over every sub-step strictly before the goal.

    A positive result means the player provably cannot touch the ball before it
    crosses the line. Sub-step positions inside one ball-step are collinear, so
    the discrete minimum is what the engine actually tests.
    """
    worst = BIG
    for k, path in enumerate(steps):
        if k > goal_step:
            break
        radius = base + per_step * k
        last = goal_index if k == goal_step else len(path)
        for index in range(last):
            bx, by = path[index]
            slack = hypot(bx - px, by - py) - radius
            if slack < worst:
                worst = slack
                if worst <= -BIG:
                    return worst
    return worst


# Lattice reach. A player moves by one of eight vectors of length 4 (or stays),
# so after m moves it lies inside the Minkowski sum of m copies of their convex
# hull: the regular octagon with vertices at 4m along the axes and diagonals.
# That octagon is strictly inside the 4m disc -- its apothem is 4m*cos(22.5deg),
# 7.6% shorter -- so a disc-based proof rejects shots no opponent can actually
# reach. Most bots (the rivals included) reason with Euclidean distance/4; the
# geometry they misjudge is exactly what an octagon proof can certify.
_OCT_C = 0.9238795325112867   # cos(pi/8)
_OCT_S = 0.3826834323650898   # sin(pi/8)


def octagon_gap(qx: float, qy: float, reach: float) -> float:
    """Lower bound on the distance from q to the octagon of vertex-radius ``reach``.

    Both terms are valid lower bounds -- the disc contains the octagon, and the
    octagon lies inside each of its eight edge half-planes -- so their maximum is
    too. It is exact along every axis/diagonal (disc term) and along every edge
    normal (half-plane term), and never overstates the gap, which is what a
    *proof* needs.
    """
    a = qx if qx >= 0.0 else -qx
    b = qy if qy >= 0.0 else -qy
    disc = hypot(qx, qy) - reach
    p1 = a * _OCT_C + b * _OCT_S
    p2 = a * _OCT_S + b * _OCT_C
    plane = (p1 if p1 > p2 else p2) - reach * _OCT_C
    return disc if disc > plane else plane


def path_clearance_oct(steps: list, px: float, py: float, moves_offset: int,
                       goal_step: int, goal_index: int) -> float:
    """``path_clearance`` for an opponent, using lattice (octagon) reach.

    At ball-step k the opponent has made k + moves_offset moves; it catches if
    its body comes within CATCH of a sub-step. Returns min over pre-goal
    sub-steps of (gap to its reachable octagon) - CATCH.
    """
    worst = BIG
    for k, path in enumerate(steps):
        if k > goal_step:
            break
        reach = SPEED * (k + moves_offset)
        last = goal_index if k == goal_step else len(path)
        for index in range(last):
            bx, by = path[index]
            slack = octagon_gap(bx - px, by - py, reach) - CATCH
            if slack < worst:
                worst = slack
    return worst


def line_blocked(ax: float, ay: float, bx: float, by: float,
                 obstacles: tuple, radius: float, skip: int = -1) -> bool:
    """True if a disc of ``radius`` cannot slide along the segment a->b.

    ``skip`` excludes one obstacle index. A detour waypoint sits just outside a
    rectangle's inflated corner, so a leg ending there always grazes that same
    rectangle; without the exclusion every detour is rejected and the caller
    falls back to a wildly pessimistic estimate -- which is how the agent came
    to abandon a loose ball 11 units away.
    """
    dx = bx - ax
    dy = by - ay
    length = hypot(dx, dy)
    if length < 1e-9:
        return False
    ux = dx / length
    uy = dy / length
    for index, rect in enumerate(obstacles):
        if index == skip:
            continue
        rx, ry, rw, rh = rect
        t0 = 0.0
        t1 = length
        hit = True
        for origin, direction, lo, hi in (
            (ax, ux, rx - radius, rx + rw + radius),
            (ay, uy, ry - radius, ry + rh + radius),
        ):
            if -1e-12 < direction < 1e-12:
                if origin <= lo or origin >= hi:
                    hit = False
                    break
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
                hit = False
                break
        if hit:
            return True
    return False


def travel_steps(x: float, y: float, tx: float, ty: float, obstacles: tuple,
                 memo: dict | None = None) -> float:
    """Obstacle-aware estimate of the player-steps needed to reach a target.

    Straight-line when the lane is clear; otherwise the best two-leg detour via
    an inflated corner of the blocking rectangle. The obstacles are six small
    axis-aligned boxes, so a single waypoint is effectively always enough and
    this costs a fraction of a grid search.
    """
    if memo is not None:
        key = (round(x, 1), round(y, 1), round(tx, 1), round(ty, 1))
        hit = memo.get(key)
        if hit is not None:
            return hit
        value = travel_steps(x, y, tx, ty, obstacles)
        memo[key] = value
        return value
    direct = hypot(tx - x, ty - y)
    if not line_blocked(x, y, tx, ty, obstacles, ph.PLAYER_R):
        return direct / SPEED
    best = BIG
    pad = ph.PLAYER_R + 0.6
    for index, rect in enumerate(obstacles):
        rx, ry, rw, rh = rect
        for cx, cy in (
            (rx - pad, ry - pad), (rx + rw + pad, ry - pad),
            (rx - pad, ry + rh + pad), (rx + rw + pad, ry + rh + pad),
        ):
            if not ph.valid_center(cx, cy, obstacles):
                continue
            if line_blocked(x, y, cx, cy, obstacles, ph.PLAYER_R, index):
                continue
            if line_blocked(cx, cy, tx, ty, obstacles, ph.PLAYER_R, index):
                continue
            total = hypot(cx - x, cy - y) + hypot(tx - cx, ty - cy)
            if total < best:
                best = total
    if best >= BIG:
        # Still no two-leg route: assume a detour of roughly one obstacle
        # diagonal rather than the old, far more pessimistic guess.
        return (direct + 14.0) / SPEED
    return best / SPEED


# --------------------------------------------------------------------------
# shot search
# --------------------------------------------------------------------------

def _goal_substep(steps: list) -> tuple:
    """(step_index, substep_index) of the scoring sub-step, or (-1, -1)."""
    if not steps:
        return (-1, -1)
    last = len(steps) - 1
    path = steps[last]
    if not path:
        return (-1, -1)
    bx, by = path[-1]
    if ph.goal_scorer(bx, by) == 1:
        return (last, len(path) - 1)
    return (-1, -1)


def _prefix(steps: list, n_steps: int) -> list:
    """Steps of a lower-power kick. Valid because 32/64/96 are exact multiples
    of ball_speed 8, so every engine step is a full 12 x 0.6667 step and a
    shorter budget truncates the path without changing its sub-step grid."""
    return steps[:n_steps]


POWER_STEPS = (4, 8, 12)   # 32/8, 64/8, 96/8


def shot_options(view, margin_required: float, cache: dict,
                 params: dict | None = None,
                 model_modes: tuple | None = None,
                 model_leads: tuple | None = None,
                 deadline: float | None = None) -> tuple:
    """Search 9 moves x 8 directions x 3 powers for scoring kicks.

    Two tiers are returned together, proven ones first:

    * **tier 1 (proven)** -- clearance beats ``margin_required`` against the
      conservative free-space bound *and* the post-move position is beyond
      ``CONTACT_SAFE`` of the opponent, so no opponent of any skill reaches it.
      The contact condition is not cosmetic: an audit of 1,745 tier-1 shots
      found 12 blocked, every one of them in a step where contact resolution had
      moved the kicker away from the position the proof assumed.
    * **tier 2 (opportunistic)** -- the bound says "maybe", but the replica of
      the kit's chase rule in ``opponent_model`` says this opponent does not
      actually get there. Worth taking because a blocked shot is cheap: every
      scoring trajectory runs toward the opponent's end, so a block leaves them
      with the ball a field-length away from our goal.

    ``best_by_move`` maps a move to the best clearance seen from it (often
    negative) and is the gradient the dribble climbs.
    """
    ox, oy = view.opp
    mx, my = view.me
    obstacles = view.obstacles
    guaranteed = []
    best_by_move = {}
    use_model = bool(params) and params["use_model"] > 0.5
    octagon = bool(params) and params["reach_octagon"] > 0.5
    model_floor = params["model_floor"] if params else 0.0
    modes = model_modes if model_modes is not None else (0, 1, 2)
    leads = model_leads if model_leads is not None else (1.0, 2.0)

    for name, px, py in legal_moves(mx, my, obstacles):
        if deadline is not None and best_by_move and perf_counter() > deadline:
            break
        move_best = -BIG
        # Only a position contact cannot disturb may carry a *proof*; otherwise
        # the shot is still considered, but as an opportunistic tier-2 candidate.
        contact_free = hypot(px - ox, py - oy) >= CONTACT_SAFE
        for direction in ph.KICK_DIRS:
            sx, sy = ph.kick_spawn(px, py, direction)
            # Cheap exact prune: straight up is the shortest route to the line.
            if ph.GOAL_TOP_Y - sy > ph.KICK_DISTANCES[2]:
                continue
            key = (round(sx, 6), round(sy, 6), direction)
            cached = cache.get(key)
            if cached is None:
                ux, uy = ph.UNIT[direction]
                steps, outcome = ph.simulate_ball(sx, sy, ux, uy, ph.KICK_DISTANCES[2],
                                                  obstacles, max_steps=13)
                cached = (steps, outcome["starts"])
                cache[key] = cached
            steps, starts = cached
            for power_index in (0, 1, 2):
                sub = _prefix(steps, POWER_STEPS[power_index])
                goal_step, goal_index = _goal_substep(sub)
                if goal_step < 0:
                    continue
                # We must not intercept our own shot before it crosses.
                mine = path_clearance(sub, px, py, CATCH, 0.0, goal_step, goal_index)
                if mine <= 0.0:
                    continue
                if octagon:
                    clear = path_clearance_oct(sub, ox, oy, 1, goal_step, goal_index)
                else:
                    clear = path_clearance(sub, ox, oy, CATCH + SPEED, SPEED,
                                           goal_step, goal_index)
                if clear > move_best:
                    move_best = clear
                if clear > margin_required and contact_free:
                    guaranteed.append({
                        "move": name, "direction": direction,
                        "power": power_index + 1, "clear": clear,
                        "steps_to_goal": goal_step, "px": px, "py": py, "tier": 1,
                    })
                elif use_model and clear > model_floor:
                    if not om.any_predicted_catch(
                            sub, starts, ox, oy, mx, my, obstacles,
                            view.flipped, goal_step, goal_index, modes, leads):
                        guaranteed.append({
                            "move": name, "direction": direction,
                            "power": power_index + 1, "clear": clear,
                            "steps_to_goal": goal_step, "px": px, "py": py, "tier": 2,
                        })
                break   # lowest power that scores on this ray is enough
        best_by_move[name] = move_best

    guaranteed.sort(key=lambda item: (item["tier"], -item["clear"],
                                      item["steps_to_goal"], item["power"]))
    return guaranteed, best_by_move


SEEK_DIRS = ("UP", "UP_LEFT", "UP_RIGHT")
SEEK_XS = tuple(6.0 + 4.0 * i for i in range(23))      # 6 .. 94
SEEK_YS = tuple(90.0 + 4.0 * j for j in range(12))     # 90 .. 134


def spot_clearance(px: float, py: float, ox: float, oy: float,
                   obstacles: tuple, cache: dict) -> float:
    """Best worst-case shot clearance from a standing spot, opponent fixed.

    The same bound shot_options uses, restricted to the three upward rays;
    ball paths are cached by spawn point, so on the fixed seek grid each ray
    is simulated once per match and only the cheap clearance pass repeats.
    """
    best = -BIG
    for direction in SEEK_DIRS:
        sx, sy = ph.kick_spawn(px, py, direction)
        key = (round(sx, 6), round(sy, 6), direction)
        cached = cache.get(key)
        if cached is None:
            ux, uy = ph.UNIT[direction]
            steps, outcome = ph.simulate_ball(sx, sy, ux, uy, ph.KICK_DISTANCES[2],
                                              obstacles, max_steps=13)
            cached = (steps, outcome["starts"])
            cache[key] = cached
        steps = cached[0]
        for power_index in (0, 1, 2):
            sub = _prefix(steps, POWER_STEPS[power_index])
            goal_step, goal_index = _goal_substep(sub)
            if goal_step < 0:
                continue
            if path_clearance(sub, px, py, CATCH, 0.0, goal_step, goal_index) <= 0.0:
                break
            clear = path_clearance(sub, ox, oy, CATCH + SPEED, SPEED, goal_step, goal_index)
            if clear > best:
                best = clear
            break
    return best


def seek_target(view, cache: dict, w_travel: float,
                deadline: float | None = None) -> tuple | None:
    """The standing spot that best trades shot clearance against travel.

    A stalled attack is a local optimum of the one-move dribble: every
    neighbouring square shoots worse than this one, so the carrier stands
    still and self-passes for the rest of the match (22 of 120 matches
    against one rival ended 0-0 that way). This scores every spot of a coarse
    grid in front of goal against the defender's *current* square, so the
    carrier gets a destination instead of a gradient. Cached per defender
    square, so a parked keeper costs one evaluation.
    """
    ox, oy = view.opp
    mx, my = view.me
    obstacles = view.obstacles
    okey = ("seek", round(ox, 1), round(oy, 1))
    table = cache.get(okey)
    if table is None:
        # The first build simulates every ray (about 0.2 s); the deadline lets
        # it spread over a few ticks, since the ball paths it has already
        # simulated stay cached and only the cheap clearance pass repeats.
        table = []
        for gy in SEEK_YS:
            if deadline is not None and perf_counter() > deadline:
                return None
            for gx in SEEK_XS:
                if not ph.valid_center(gx, gy, obstacles):
                    continue
                clear = spot_clearance(gx, gy, ox, oy, obstacles, cache)
                if clear > -BIG / 2:
                    table.append((clear, gx, gy))
        cache[okey] = table
    best = None
    best_value = -BIG
    memo = cache.setdefault("travel_memo", {})
    for clear, gx, gy in table:
        # Getting there gives the defender time; never value a spot the
        # defender reaches first.
        theirs = hypot(gx - ox, gy - oy)
        if theirs < hypot(gx - mx, gy - my) * 0.5 + CONTACT_SAFE:
            continue
        mine = travel_steps(mx, my, gx, gy, obstacles, memo)
        value = (clear if clear < 20.0 else 20.0) - w_travel * mine
        if value > best_value:
            best_value = value
            best = (gx, gy, clear)
    return best


# --------------------------------------------------------------------------
# release / pass search
# --------------------------------------------------------------------------

def release_options(view, params: dict, cache: dict,
                    deadline: float | None = None) -> list:
    """Kicks that are not goals, ranked by how safely they advance the ball.

    Scored on who provably arrives first, how far up the pitch the ball ends
    up, and whether it strays toward our own goal.
    """
    ox, oy = view.opp
    mx, my = view.me
    obstacles = view.obstacles
    w_prog = params["w_release_prog"]
    w_first = params["w_release_first"]
    w_own = params["w_release_own"]
    w_immediate = params["w_release_immediate"]
    w_lose = params["w_release_lose"]
    clamp = params["release_clamp"]
    immediate_steps = int(params["immediate_steps"])
    out = []
    memo: dict = {}

    for name, px, py in legal_moves(mx, my, obstacles):
        if deadline is not None and out and perf_counter() > deadline:
            break
        contact_free = hypot(px - ox, py - oy) >= CONTACT_SAFE
        for direction in ph.KICK_DIRS:
            sx, sy = ph.kick_spawn(px, py, direction)
            key = (round(sx, 6), round(sy, 6), direction)
            cached = cache.get(key)
            if cached is None:
                ux, uy = ph.UNIT[direction]
                steps, outcome = ph.simulate_ball(sx, sy, ux, uy, ph.KICK_DISTANCES[2],
                                                  obstacles, max_steps=13)
                cached = (steps, outcome["starts"])
                cache[key] = cached
            steps = cached[0]
            for power_index in (0, 1, 2):
                sub = _prefix(steps, POWER_STEPS[power_index])
                if not sub or not sub[-1]:
                    continue
                goal_step, goal_index = _goal_substep(sub)
                if goal_step >= 0:
                    continue   # handled by shot_options
                end_x, end_y = sub[-1][-1]
                # Own goal: reject outright, and reject with a margin whenever
                # contact could move us before the kick fires.
                #
                # The exact test alone was not enough. Contact resolution runs
                # before the kick, so a `players_separated` push shifts the
                # spawn point and with it the whole straight trajectory; a path
                # simulated as missing the mouth by a metre then went in. Three
                # own goals in a 240-match census were all of exactly this form:
                # in contact, deep in our own half, kicking DOWN. Shifting the
                # spawn by d shifts the path by d, so widening the mouth test by
                # the largest plausible displacement is a sound bound.
                margin = 0.0 if contact_free else ph.CONTACT_R
                own_goal = False
                for path in sub:
                    for bx, by in path:
                        if (by - ph.BALL_R <= margin
                                and ph.GOAL_LEFT - margin <= bx <= ph.GOAL_RIGHT + margin):
                            own_goal = True
                            break
                    if own_goal:
                        break
                if own_goal:
                    continue

                my_arrival = first_reach(sub, px, py, CATCH, SPEED)
                opp_arrival = first_reach(sub, ox, oy, CATCH + SPEED, SPEED)
                my_step = my_arrival[0] if my_arrival else -1
                opp_step = opp_arrival[0] if opp_arrival else -1
                # Kicking into a nearby opponent is the one release mistake that
                # is always severe: it concedes the ball exactly where we stand.
                giveaway = (
                    opp_arrival is not None
                    and opp_arrival[0] <= immediate_steps
                    and (my_arrival is None or opp_arrival < my_arrival)
                )
                # Whoever is nearer the resting point claims it within 5.0.
                my_settle = travel_steps(px, py, end_x, end_y, obstacles, memo)
                opp_settle = travel_steps(ox, oy, end_x, end_y, obstacles, memo) - 1.0

                if my_step < 0:
                    my_time = my_settle
                else:
                    my_time = float(my_step)
                if opp_step < 0:
                    opp_time = opp_settle
                else:
                    opp_time = float(opp_step)

                first = opp_time - my_time
                progress = end_y - my
                # Retention must outrank raw distance. `first` is clamped to +-6
                # while progress is unbounded, so a 96-unit kick scored about +30
                # of progress against at most -12 of risk: the planner reliably
                # chose long kicks the opponent collected. Against the kit's bots
                # that cost nothing (they collected 9% of our releases); against a
                # competent rival they collected 51%, and 75% of goals conceded
                # began with them claiming a loose ball.
                # The clamp used to be a hard +-6, which made a release the
                # opponent cannot reach for fifteen steps score exactly like one
                # it reaches in seven. Retention is the bottleneck on scoring --
                # reaching a shooting position takes about three possession
                # cycles, so cycle survival enters the goal rate cubed -- so the
                # margin is worth preferring, not just the sign of it.
                score = w_prog * progress + w_first * max(-clamp, min(clamp, first))
                if first <= 0.0:
                    score -= w_lose
                if end_y < params["own_half_guard"] and first <= 0.0:
                    score -= w_own
                if giveaway:
                    score -= w_immediate
                out.append({
                    "move": name, "direction": direction, "power": power_index + 1,
                    "score": score, "first": first, "end_x": end_x, "end_y": end_y,
                    "my_time": my_time, "opp_time": opp_time,
                })
    out.sort(key=lambda item: -item["score"])
    return out


# --------------------------------------------------------------------------
# loose-ball interception
# --------------------------------------------------------------------------

def intercept_plan(view, params: dict) -> dict:
    """Simulate the live ball and decide whether we win the race for it.

    The worst-case opponent bound (``4.5 + 4(k+1)``) is right for *proving* a
    shot unblockable, but using it here made the agent concede races it would
    have won: measured against the organizer, 63% of goals conceded began with
    a loose ball the agent declined to contest. Races therefore use a tuned
    lead ``opp_lead_loose`` instead of the full extra move, plus a
    ``contest_slack`` tie-break, since contesting and losing costs little.
    """
    lead = params["opp_lead_loose"] * SPEED
    slack = params["contest_slack"]
    if view.player_id == "player_2":
        # Interception and stationary claims are resolved player_1 first, so
        # every exact tie in a race goes against player_2.
        slack += params["p2_slack_delta"]
    bx, by = view.ball
    vx, vy = view.ball_vel
    mx, my = view.me
    ox, oy = view.opp
    obstacles = view.obstacles
    length = hypot(vx, vy)

    if view.remaining <= 0.0 or length < 1e-9:
        end_x, end_y = bx, by
        steps = []
    else:
        ux = vx / length
        uy = vy / length
        steps, _ = ph.simulate_ball(bx, by, ux, uy, view.remaining, obstacles, max_steps=14)
        end_x, end_y = (steps[-1][-1] if steps and steps[-1] else (bx, by))

    my_step = first_reach_step(steps, mx, my, CATCH, SPEED) if steps else -1
    opp_step = first_reach_step(steps, ox, oy, CATCH + lead, SPEED) if steps else -1

    my_settle = travel_steps(mx, my, end_x, end_y, obstacles)
    opp_settle = travel_steps(ox, oy, end_x, end_y, obstacles)
    my_time = float(my_step) if my_step >= 0 else my_settle
    opp_time = float(opp_step) if opp_step >= 0 else opp_settle

    target = (end_x, end_y)
    if my_step >= 0:
        # Head for the earliest sub-step we can actually meet.
        radius = CATCH + SPEED * my_step
        limit = radius * radius
        for px, py in steps[my_step]:
            if (px - mx) ** 2 + (py - my) ** 2 <= limit:
                target = (px, py)
                break
    return {
        "we_win": my_time <= opp_time + slack,
        "my_time": my_time, "opp_time": opp_time,
        "target": target, "end": (end_x, end_y), "steps": steps,
    }


# --------------------------------------------------------------------------
# defending
# --------------------------------------------------------------------------

def threat_set(view, params: dict, cache: dict,
               deadline: float | None = None) -> list:
    """Opponent kicks that would score in our goal, with their sub-step paths."""
    ox, oy = view.opp
    obstacles = view.obstacles
    threats = []
    for name, px, py in legal_moves(ox, oy, obstacles):
        if deadline is not None and perf_counter() > deadline:
            break
        for direction in ph.KICK_DIRS:
            sx, sy = ph.kick_spawn(px, py, direction)
            if sy - ph.GOAL_BOTTOM_Y > ph.KICK_DISTANCES[2]:
                continue
            key = ("T", round(sx, 6), round(sy, 6), direction)
            cached = cache.get(key)
            if cached is None:
                ux, uy = ph.UNIT[direction]
                steps, outcome = ph.simulate_ball(sx, sy, ux, uy, ph.KICK_DISTANCES[2],
                                                  obstacles, max_steps=13)
                cached = (steps, outcome["starts"])
                cache[key] = cached
            steps = cached[0]
            for power_index in (0, 1, 2):
                sub = _prefix(steps, POWER_STEPS[power_index])
                if not sub or not sub[-1]:
                    continue
                bx, by = sub[-1][-1]
                if ph.goal_scorer(bx, by) != -1:
                    continue
                threats.append({"steps": sub, "n": len(sub) - 1,
                                "index": len(sub[-1]) - 1})
                break
    return threats


def defend_choice(view, params: dict, cache: dict,
                  deadline: float | None = None) -> dict:
    """Pick a move (and maybe a tackle) while the opponent holds the ball."""
    mx, my = view.me
    ox, oy = view.opp
    obstacles = view.obstacles
    threats = threat_set(view, params, cache, deadline)
    shadow = shadow_point(ox, oy, params)
    can_tackle = view.steps >= 3

    w_threat = params["w_threat"]
    w_pos = params["w_pos"]
    tackle_bonus = params["tackle_bonus"]
    # Pressing gradient. The shadow point alone is a step function: it parks us a
    # fixed distance off the carrier and the tackle bonus only pays once we are
    # already inside 6.15, which never happens. Against the kit's bots that is
    # harmless because they release the ball almost immediately. A rival that
    # *carries* walked the ball 30 units into a shooting position while we
    # retreated in front of it through six consecutive tackleable steps. This
    # term gives closing down a continuous reward.
    w_press = params["w_press"]
    # Continuous threat cost. Counting unreachable lanes gives the keeper an
    # integer signal that a one-unit coverage term (4.7 per unit of distance)
    # swamps: moving 4 units to cover one more lane cost ~19 and gained 0.5.
    # Summing *how far* each lane lies outside our reach gives a real gradient
    # toward the lanes we cannot yet cover, which is what an exact shooter
    # exploits -- it shoots its own lane or banks, not at the goal centre.
    continuous = params["threat_continuous"] > 0.5
    # Goal-mouth coverage. The shadow point is anchored to the *carrier* at a
    # fixed standoff, and tuning drove its weight to ~0.05, which left no term at
    # all that keeps us between the ball and our own goal. This one is anchored to
    # our own mouth instead: the ideal spot sits on the line from the ball to the
    # centre of our goal, `cover_depth` out from the goal line. Classic keeping,
    # and complementary to pressing rather than in tension with it.
    w_cover = params["w_cover"]
    cover_depth = params["cover_depth"]
    cover_x, cover_y = _cover_point(view, cover_depth)
    if params["cover_anchor"] > 0.5 and threats:
        # Anchor the keeper to where the opponent's *actual* scoring trajectories
        # cross our holding depth, instead of to the goal centre. The centre line
        # is exactly where the kit's bots aim (direction_toward(width/2 - x)),
        # which is why tuning against them made it look right; an exact shooter
        # goes straight down its own x or banks off a wall. Measured against one:
        # the keeper sat at x 51-55 while a straight-down shot went in at x 62,
        # 6.8 units off the line and outside the 4.5 interception radius.
        crossings = _threat_crossings(threats, cover_depth)
        if crossings:
            crossings.sort()
            cover_x = crossings[len(crossings) // 2]
            cover_y = cover_depth

    best = None
    best_score = -BIG
    # Non-STAY moves count for tackling even when the engine rejects the target,
    # so consider every direction and fall back to the stay-put position.
    for name in ph.MOVES:
        if name == "STAY":
            px, py = mx, my
        else:
            dx, dy = ph.STEP[name]
            px = mx + dx
            py = my + dy
            if not ph.valid_center(px, py, obstacles):
                px, py = mx, my
        unreachable = 0
        uncovered = 0.0
        for threat in threats:
            if continuous:
                gap = _threat_gap(threat, px, py)
                if gap > 0.0:
                    # Cap it: a lane that scores on the ball's first sub-step has
                    # no pre-goal sub-steps, so its gap is BIG. Uncapped, one such
                    # lane drove every candidate below the -BIG floor and
                    # defend_choice returned None (caught by the harness).
                    uncovered += gap if gap < GAP_CAP else GAP_CAP
                    unreachable += 1
            # Our reach at ball-step k from the post-move position is 4.5 + 4k.
            elif first_reach_step(threat["steps"], px, py, CATCH, SPEED,
                                  stop_before=threat["n"] + 1) < 0:
                unreachable += 1
        to_carrier = hypot(px - ox, py - oy)
        score = (-w_threat * (uncovered if continuous else unreachable)
                 - w_pos * hypot(px - shadow[0], py - shadow[1])
                 - w_press * to_carrier
                 - w_cover * hypot(px - cover_x, py - cover_y))
        if can_tackle and name != "STAY" and to_carrier <= ph.TACKLE_R:
            score += tackle_bonus
        # Stay goal-side of the carrier.
        if py > oy:
            score -= params["w_goalside"]
        if score > best_score:
            best_score = score
            best = (name, px, py)

    # No kick is attached to a tackle. A tackle requires ending within
    # 6.15 of the carrier, which is inside CONTACT_SAFE, so the spawn position
    # of an attached kick can never be guaranteed -- and measurement showed the
    # path fired 3 times in 360 matches, so there was nothing to protect.
    if best is None:
        # Defensive: never let an all-equal or non-finite score set crash the tick.
        return {"move": "STAY"}
    return {"move": best[0]}


def _threat_gap(threat: dict, px: float, py: float) -> float:
    """How far a scoring trajectory stays outside our reach (<= 0: we cover it).

    Our reach at ball-step k from the post-move square is the lattice octagon
    of k moves plus the 4.5 catch disc -- the octagon, not the disc, because for
    *covering* we want the true reach, and the disc overstates it off-axis.
    """
    best = BIG
    last_step = threat["n"]
    for k, path in enumerate(threat["steps"]):
        if k > last_step:
            break
        reach = SPEED * k
        stop = threat["index"] if k == last_step else len(path)
        for index in range(stop):
            bx, by = path[index]
            gap = octagon_gap(bx - px, by - py, reach) - CATCH
            if gap < best:
                best = gap
                if best <= 0.0:
                    return best
    return best


def _threat_crossings(threats: list, depth: float) -> list:
    """x where each scoring threat path first reaches y <= depth (our end).

    Paths that never come that shallow (they score from further out than our
    holding depth, which cannot happen for depth >= 1.5, or are cut short) fall
    back to their goal-line crossing.
    """
    out = []
    for threat in threats:
        hit = None
        last = None
        for path in threat["steps"]:
            for bx, by in path:
                last = (bx, by)
                if by <= depth:
                    hit = bx
                    break
            if hit is not None:
                break
        if hit is None and last is not None:
            hit = last[0]
        if hit is not None:
            out.append(hit)
    return out


def _cover_point(view, depth: float) -> tuple:
    """Where a keeper should stand: on the ball-to-own-goal line, `depth` out.

    Our goal is the bottom of the canonical frame, so the mouth centre is (50, 0).
    """
    bx, by = view.ball
    gx, gy = 50.0, 0.0
    dx = bx - gx
    dy = by - gy
    length = hypot(dx, dy)
    if length < 1e-6:
        return (gx, depth)
    scale = depth / length
    return (gx + dx * scale, gy + dy * scale)


def shadow_point(anchor_x: float, anchor_y: float, params: dict) -> tuple:
    """A goal-side holding spot between a threat anchor and our goal centre.

    ``d_shadow`` alone put the defender a fixed short distance from the
    carrier, which meant charging a carrier 70 units away and abandoning the
    goal. ``shadow_frac`` scales the standoff with the carrier's distance from
    our goal, so the defender presses when the threat is real and drops when it
    is not. ``shadow_frac = 0`` reproduces the fixed-distance behaviour.
    """
    gx, gy = 50.0, 0.0
    dx = gx - anchor_x
    dy = gy - anchor_y
    length = hypot(dx, dy)
    distance = params["d_shadow"] + params["shadow_frac"] * length
    if distance > length:
        distance = length
    if length < 1e-6:
        return (anchor_x, max(ph.PLAYER_R, anchor_y - params["d_shadow"]))
    return (anchor_x + dx / length * distance, anchor_y + dy / length * distance)


# --------------------------------------------------------------------------
# movement
# --------------------------------------------------------------------------

def move_toward(view, tx: float, ty: float, banned: tuple = ()) -> str:
    """Best legal move toward a target, routed around obstacles if needed."""
    mx, my = view.me
    obstacles = view.obstacles
    waypoint = (tx, ty)
    if line_blocked(mx, my, tx, ty, obstacles, ph.PLAYER_R):
        best = BIG
        pad = ph.PLAYER_R + 0.6
        for index, rect in enumerate(obstacles):
            rx, ry, rw, rh = rect
            for cx, cy in (
                (rx - pad, ry - pad), (rx + rw + pad, ry - pad),
                (rx - pad, ry + rh + pad), (rx + rw + pad, ry + rh + pad),
            ):
                if not ph.valid_center(cx, cy, obstacles):
                    continue
                if line_blocked(mx, my, cx, cy, obstacles, ph.PLAYER_R, index):
                    continue
                if line_blocked(cx, cy, tx, ty, obstacles, ph.PLAYER_R, index):
                    continue
                total = hypot(cx - mx, cy - my) + hypot(tx - cx, ty - cy)
                if total < best:
                    best = total
                    waypoint = (cx, cy)
    wx, wy = waypoint
    best_name = "STAY"
    best_cost = hypot(mx - wx, my - wy)
    for name, px, py in legal_moves(mx, my, obstacles):
        if name == "STAY" or name in banned:
            continue
        cost = hypot(px - wx, py - wy)
        if cost < best_cost - 1e-9:
            best_cost = cost
            best_name = name
    if best_name == "STAY":
        # Nothing improves the distance; take any legal move that does not
        # worsen it much, so we never freeze in front of an obstacle.
        for name, px, py in legal_moves(mx, my, obstacles):
            if name == "STAY" or name in banned:
                continue
            if hypot(px - wx, py - wy) <= best_cost + SPEED * 0.5:
                return name
    return best_name
