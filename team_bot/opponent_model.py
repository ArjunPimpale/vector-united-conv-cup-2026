"""Replica of the chase rule every readable opponent in the kit shares.

`reference_bot.choose_action`, `submission_kit.tactical_action`,
`soccer_env.practice_action` and the organizer bot's unseen-state / safety-guard
fallback all pick their off-ball move the same way: aim at the ball plus a
one-step velocity lead (``aggressive_action`` uses two), convert the delta to one
of eight directions through ``direction_toward``, then snap it to the legal
direction best aligned with that preference via ``safe_move``.

Replicating it turns "could the opponent possibly reach this shot" into "will
this particular opponent actually reach it". The conservative free-space bound
in ``planner.shot_options`` stays the gate for *guaranteed* goals; this model
only unlocks an extra, explicitly opportunistic tier -- a shot that gets blocked
is cheap, because it is always blocked in the opponent's own half.

Everything here is derived from observations only.
"""
from __future__ import annotations

from math import hypot

from . import physics as ph

# safe_move's obstacle test is stricter than the engine's: radius + 0.25.
BOT_CLEARANCE = ph.PLAYER_R + 0.25


def _bot_move_is_safe(x: float, y: float, obstacles: tuple) -> bool:
    if x - ph.PLAYER_R < 0.0 or x + ph.PLAYER_R > ph.FIELD_W:
        return False
    if y - ph.PLAYER_R < 0.0 or y + ph.PLAYER_R > ph.FIELD_H:
        return False
    for rect in obstacles:
        if ph.circle_hits_rect(x, y, BOT_CLEARANCE, rect):
            return False
    return True


def _direction_toward(dx: float, dy: float, dead_zone: float) -> str:
    horizontal = "" if abs(dx) <= dead_zone else ("RIGHT" if dx > 0 else "LEFT")
    vertical = "" if abs(dy) <= dead_zone else ("UP" if dy > 0 else "DOWN")
    if vertical and horizontal:
        return vertical + "_" + horizontal
    return vertical or horizontal or "STAY"


def _order(flipped: bool) -> tuple:
    """safe_move resolves ties by ``max`` over the engine's MOVES order, which
    is not symmetric under the y-flip, so player_2's tie-breaks differ. Iterate
    in the engine's order expressed in our canonical frame."""
    if not flipped:
        return ph.MOVES
    return tuple(ph.FLIP_MOVE[name] for name in ph.MOVES)


def chase_move(x: float, y: float, target_x: float, target_y: float,
               obstacles: tuple, flipped: bool, dead_zone: float = 0.6) -> str:
    """What ``safe_move(direction_toward(target - me))`` would return."""
    preferred = _direction_toward(target_x - x, target_y - y, dead_zone)
    if flipped:
        engine_preferred = ph.FLIP_MOVE[preferred]
        pref_vec = ph.UNIT[engine_preferred]
        pref_vec = (pref_vec[0], -pref_vec[1])
    else:
        pref_vec = ph.UNIT[preferred]
    # The bots compare against the raw (+-1) vectors, not unit-normalised ones;
    # for a dot-product ranking the positive scale factor is irrelevant.
    best_name = "STAY"
    best_key = None
    for name in _order(flipped):
        if name == "STAY":
            continue
        dx, dy = ph.STEP[name]
        if not _bot_move_is_safe(x + dx, y + dy, obstacles):
            continue
        vec = ph.UNIT[name]
        key = (vec[0] * pref_vec[0] + vec[1] * pref_vec[1], name == preferred)
        if best_key is None or key > best_key:
            best_key = key
            best_name = name
    return best_name


def first_target(mode: int, ox: float, oy: float,
                 holder_x: float, holder_y: float) -> tuple:
    """Where each opponent family aims on the step it still sees us carrying.

    Once the ball is loose every bot in the kit chases ``ball + lead*velocity``
    identically, so the kick step is the only place the families differ:

    0 -- goal-side press (``reference``, ``practice``, ``starter``, and the
         organizer bot's fallback): a (+-3, +3) offset from the carrier.
    1 -- direct charge (``aggressive``, and ``counter`` inside 32 units).
    2 -- goal-side drop (``counter`` beyond 32 units): retreat to the midpoint
         between the carrier and its own goal.
    """
    if mode == 0:
        return (holder_x + (-3.0 if holder_x > ph.FIELD_W / 2.0 else 3.0),
                holder_y + 3.0)
    if mode == 1:
        return (holder_x, holder_y)
    return ((holder_x + ph.FIELD_W / 2.0) / 2.0, (holder_y + ph.FIELD_H) / 2.0)


def any_predicted_catch(steps: list, starts: list, ox: float, oy: float,
                        holder_x: float, holder_y: float, obstacles: tuple,
                        flipped: bool, goal_step: int, goal_index: int,
                        modes: tuple = (0, 1, 2), leads: tuple = (1.0, 2.0)) -> bool:
    """True if any of the given opponent families would reach the ball in time.

    With all three families the test is sound but pessimistic -- a *retreating*
    opponent sits deeper, which is exactly where the shot is going, so demanding
    that it miss too costs real goals (measured -0.30 GD). ``Tracker`` narrows
    ``modes``/``leads`` to the family this opponent demonstrably is, and the
    full set stays the fallback whenever the evidence is thin.
    """
    for mode in modes:
        for lead in leads:
            if predicted_catch(steps, starts, ox, oy, holder_x, holder_y,
                               obstacles, flipped, lead, goal_step, goal_index,
                               mode):
                return True
    return False


class Tracker:
    """Scores the replicas against the opponent's realised moves.

    Two independent signals, both read from plain observations:

    * while **we** carry the ball, the three ``first_target`` modes disagree,
      so those ticks identify the family;
    * while the ball is **loose**, every family chases ``ball + lead*velocity``
      and only the lead (1 for most, 2 for ``aggressive``) differs.

    A family is adopted only once it is uniquely best over enough samples;
    until then the planner keeps the pessimistic full ensemble.
    """

    __slots__ = ("mode_hits", "mode_n", "lead_hits", "lead_n")

    def __init__(self) -> None:
        self.mode_hits = [0, 0, 0]
        self.mode_n = 0
        self.lead_hits = [0, 0]
        self.lead_n = 0

    def reset(self) -> None:
        self.mode_hits = [0, 0, 0]
        self.mode_n = 0
        self.lead_hits = [0, 0]
        self.lead_n = 0

    def _predict_position(self, px: float, py: float, tx: float, ty: float,
                          obstacles: tuple, flipped: bool) -> tuple:
        move = chase_move(px, py, tx, ty, obstacles, flipped)
        dx, dy = ph.STEP[move]
        nx = px + dx
        ny = py + dy
        if _bot_move_is_safe(nx, ny, obstacles):
            return (nx, ny)
        return (px, py)

    def update(self, previous, current) -> None:
        """Compare each replica's prediction with what the opponent actually did."""
        if previous is None:
            return
        obstacles = previous.obstacles
        ox, oy = previous.opp
        ax, ay = current.opp
        mx, my = previous.me
        # Contact resolution can move a player in a direction it never asked
        # for, so those ticks carry no information about its policy.
        if hypot(ox - mx, oy - my) < ph.CONTACT_R + ph.PLAYER_SPEED + 0.5:
            return
        if previous.owner == "me":
            for mode in (0, 1, 2):
                tx, ty = first_target(mode, ox, oy, mx, my)
                nx, ny = self._predict_position(ox, oy, tx, ty, obstacles, current.flipped)
                if hypot(nx - ax, ny - ay) < 0.75:
                    self.mode_hits[mode] += 1
            self.mode_n += 1
        elif previous.owner is None and previous.remaining > 0.0:
            bx, by = previous.ball
            vx, vy = previous.ball_vel
            for index, lead in enumerate((1.0, 2.0)):
                nx, ny = self._predict_position(
                    ox, oy, bx + lead * vx, by + lead * vy, obstacles, current.flipped)
                if hypot(nx - ax, ny - ay) < 0.75:
                    self.lead_hits[index] += 1
            self.lead_n += 1

    def chaser_fit(self, min_samples: int) -> float:
        """How well the best chase replica explains the opponent's movement.

        This doubles as a strength detector, and a sharp one. Measured over one
        full match per opponent, the best family's hit rate is 0.96-1.00 for
        every bot in the kit but 0.03 and 0.40 for the two strong rival
        submissions: a planner that computes an interception point simply does
        not move like a bot that walks at ``ball + lead*velocity``.

        Returns -1.0 while the evidence is too thin to judge.
        """
        if self.mode_n < min_samples:
            return -1.0
        return max(self.mode_hits) / float(self.mode_n)

    def modes(self, min_samples: int, threshold: float) -> tuple:
        if self.mode_n < min_samples:
            return (0, 1, 2)
        best = max(self.mode_hits)
        if best < threshold * self.mode_n:
            return (0, 1, 2)
        winners = tuple(i for i, hits in enumerate(self.mode_hits) if hits == best)
        return winners if len(winners) < 3 else (0, 1, 2)

    def leads(self, min_samples: int, threshold: float) -> tuple:
        if self.lead_n < min_samples:
            return (1.0, 2.0)
        best = max(self.lead_hits)
        if best < threshold * self.lead_n:
            return (1.0, 2.0)
        if self.lead_hits[0] == self.lead_hits[1]:
            return (1.0, 2.0)
        return (1.0,) if self.lead_hits[0] > self.lead_hits[1] else (2.0,)


def predicted_catch(steps: list, starts: list, ox: float, oy: float,
                    holder_x: float, holder_y: float, obstacles: tuple,
                    flipped: bool, lead: float = 1.0,
                    goal_step: int = -1, goal_index: int = -1,
                    mode: int = 0) -> bool:
    """True if a chase-rule opponent would reach the ball before it scores."""
    px, py = ox, oy
    target_x, target_y = first_target(mode, ox, oy, holder_x, holder_y)
    for index, path in enumerate(steps):
        if index == 0:
            move = chase_move(px, py, target_x, target_y, obstacles, flipped)
        else:
            bx, by, vx, vy = starts[index]
            move = chase_move(px, py, bx + lead * vx, by + lead * vy,
                              obstacles, flipped)
        dx, dy = ph.STEP[move]
        nx = px + dx
        ny = py + dy
        if _bot_move_is_safe(nx, ny, obstacles):
            px, py = nx, ny
        last = goal_index if index == goal_step else len(path)
        for sub in range(last):
            bx, by = path[sub]
            if hypot(bx - px, by - py) <= ph.CATCH_R:
                return True
        if index == goal_step:
            return False
    return False
