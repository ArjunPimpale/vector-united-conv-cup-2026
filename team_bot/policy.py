"""Decision state machine.

Four regimes, selected by who holds the ball:

    OWN              we hold it  -> guaranteed shot, else safe release, else carry
    OPP              they hold it-> threat-minimising defence and tackles
    LOOSE_MOVING     ball in flight -> race for the earliest reachable sub-step
    LOOSE_STATIONARY ball at rest   -> race for the claim, else hold goal-side

Everything is deterministic: no RNG anywhere, so a seed replays exactly.
Parameters live in ``models/params.json`` and are tuned offline.
"""
from __future__ import annotations

import json
from math import hypot
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

from . import physics as ph
from . import planner as pl
from .opponent_model import Tracker
from .frame import canonicalize, decanonicalize

DEFAULTS = {
    "shoot_margin": 0.5,
    "shoot_margin_close": 2.0,
    "close_opponent": 12.0,
    "vuln_distance": 10.2,
    "force_release_step": 8,
    "d_shadow": 9.0,
    "w_shot": 0.9,
    "w_prog": 1.0,
    "w_risk": 9.0,
    "w_crowd": 0.6,
    "w_center": 0.10,
    "w_threat": 6.0,
    "w_pos": 0.25,
    "w_goalside": 3.0,
    "tackle_bonus": 2.5,
    "w_release_prog": 0.10,
    "w_release_first": 2.2,
    "w_release_own": 8.0,
    "own_half_guard": 45.0,
    "corridor_half": 4.5,
    "opp_lead_loose": 0.35,
    "contest_slack": 1.0,
    "p2_slack_delta": 0.0,
    # A carrier that taps the ball a few units and re-collects it is "loose"
    # every other tick. The loose-ball holding spot and the defensive one
    # differ, so the keeper flip-flopped between them for the whole match
    # (UP, DOWN, UP, ...), and one rival held the ball 400 ticks against it.
    # When the opponent will re-collect within this many units of where it
    # stands, defend as if it still had the ball. 0 disables it.
    "loose_defend_radius": 0.0,
    "shadow_frac": 0.0,
    "release_threshold": 0.0,
    "use_model": 1.0,
    "reach_octagon": 0.0,
    "model_floor": -14.0,
    "model_min_samples": 6.0,
    "model_threshold": 0.8,
    "w_release_immediate": 25.0,
    "immediate_steps": 1.0,
    "release_good": 99.0,
    "release_scan_step": 3.0,
    "w_lane": 0.0,
    "lane_width": 14.0,
    "osc_travel_min": 8.0,
    "osc_net_limit": 0.0,
    "w_mouth": 0.6,
    "goal_half": 18.0,
    "stay_penalty": 0.05,
    # Adaptive conservatism. The loose-ball and release parameters above were
    # tuned against the kit's bots, which chase badly. Against a competent
    # planner the same values concede races and hand over 51% of releases
    # (9% against kit bots), and 75% of goals conceded then begin with the
    # opponent claiming a stationary loose ball. When the tracker cannot explain
    # the opponent as a chase replica, assume an optimal interceptor and fall
    # back to the worst-case bound. Measured chase-replica fit: 0.96-1.00 for
    # every kit bot, 0.03 and 0.40 for the two strong rivals.
    "w_press": 0.0,
    "w_cover": 0.0,
    "cover_depth": 22.0,
    "cover_anchor": 0.0,
    "threat_continuous": 0.0,
    "w_release_lose": 0.0,
    "release_clamp": 6.0,
    # Restart-loop variation. Every restart hands the conceding side a bit-exact
    # identical state, so a deterministic policy replays whatever line it just
    # conceded from, until the goal cap. Measured against a strong rival: 25% of
    # matches conceded three or more goals after one identical eight-action
    # opening, and one opening was replayed six times in a single match.
    # Indexing the opening by goals conceded breaks the cycle while staying
    # perfectly deterministic -- the same seed still replays exactly.
    "restart_variation": 0.0,
    # Restart replay. A restart is bit-identical every time (the engine resets
    # positions, ball and counters; only the clock and the score move on), so
    # against a deterministic opponent a line that scored once from a restart
    # scores again if we repeat it. Our own actions are what we record, keyed by
    # who holds the ball at the restart; the replay is followed only while every
    # observation matches the recording exactly, and is dropped for good at the
    # first difference, after which the planner takes over as usual.
    "restart_replay": 0.0,
    # Only short lines are worth repeating: the first goal of a match often
    # comes before the tracker has learned the opponent, and replaying that
    # slow line cost the planner's faster later goals (7 -> 3 in one match).
    "replay_max_len": 60.0,
    # Stall breaker for our own possession: after this many ticks without the
    # opponent touching the ball, with the score level or behind, the dribble
    # heads for the best shooting spot instead of climbing a flat gradient.
    # 0 disables it. Ahead, holding the ball is the right result, so never then.
    "seek_after": 0.0,
    "seek_travel": 1.0,
    "seek_min_clear": -6.0,
    "w_seek": 1.0,
    # Stall breaker for the opponent's possession: a carrier that self-passes
    # and never shoots holds a level match to 0-0 while our keeper waits in the
    # mouth. After this many ticks without our touching the ball, level or
    # behind, press the carrier instead. 0 disables it.
    "press_after": 0.0,
    "stall_press": 3.0,
    "stall_cover": 0.3,
    # Only press a carrier this far from our goal line (canonical y): out of
    # shooting range, so leaving the mouth costs nothing immediate.
    "press_min_y": 70.0,
    "strong_fit_max": 0.70,
    "strong_lead": 1.0,
    "strong_contest_slack": 0.0,
    "strong_release_threshold": 0.0,
    "budget_s": 0.18,
}


REPLAY_MAX = 4000   # ticks recorded per restart line; a match is far shorter


def _coarse_direction(dx: float, dy: float, dead_zone: float) -> str:
    """Delta -> one of the nine move names. Used only by the fallback policy."""
    horizontal = "" if abs(dx) <= dead_zone else ("RIGHT" if dx > 0 else "LEFT")
    vertical = "" if abs(dy) <= dead_zone else ("UP" if dy > 0 else "DOWN")
    if vertical and horizontal:
        return vertical + "_" + horizontal
    return vertical or horizontal or "STAY"


class Policy:
    """Physics-exact tactical planner. Stdlib only, no learned weights."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = dict(DEFAULTS)
        if params:
            for key, value in params.items():
                if key in self.params:
                    self.params[key] = float(value)
        self.reset_match()

    @classmethod
    def load(cls, path: str | Path) -> "Policy":
        """Read tuned parameters. ``Path.read_text`` only -- ``open`` is banned."""
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return cls()
        if isinstance(raw, dict):
            params = raw.get("params")
            if isinstance(params, dict):
                return cls(params)
        return cls()

    def reset_match(self) -> None:
        self.cache: dict = {}
        self.obstacle_key = None
        self.constants_ok = True
        self.recent: list = []
        self.last_moves: list = []
        self.tick = 0
        self.tracker = Tracker()
        self.previous = None
        self.previous_score = (0, 0)
        self.conceded_seen = 0
        self.warned = False
        self.proven = False
        self.book: dict = {}
        self.since_opp = 0
        self.since_me = 0
        self.att_best = -1.0
        self.att_stall = 0
        self.def_best = 1e9
        self.def_stall = 0
        self.recording = None
        self.replay = None
        self.replay_index = 0
        self.pending_signature = None

    # ------------------------------------------------------------------
    @staticmethod
    def fallback_action(observation: dict[str, Any]) -> dict[str, Any]:
        """Config-independent safety policy.

        The planner hard-codes the official constants because the observation
        never sends ball speed, kick distances or the possession radius. If the
        five geometry fields it *does* send disagree with ``game.json``, the
        replica would be mis-calibrated, so fall back to something that derives
        everything it needs from the observation itself: chase, then shoot at
        the goal. Weaker, but never wrong.
        """
        player_id = observation["player_id"]
        state = observation["state"]
        field = state["field"]
        me = state["players"][player_id]
        ball = state["ball"]
        width = float(field["width"])
        height = float(field["height"])
        up = observation.get("attack_direction") == "UP"
        if ball.get("possession") == player_id:
            target_x, target_y = width / 2.0, (height if up else 0.0)
            distance = abs(target_y - me["y"])
            powers = observation["action_space"]["kick"]["power"]
            power = max(powers) if distance > 40.0 else min(powers)
            direction = _coarse_direction(target_x - me["x"], target_y - me["y"], 6.0)
            if direction == "STAY":
                direction = "UP" if up else "DOWN"
            return {"move": direction, "kick": {"direction": direction, "power": power}}
        tx = float(ball["x"])
        ty = float(ball["y"])
        velocity = ball.get("velocity") or {}
        if ball.get("status") == "moving":
            tx += float(velocity.get("x", 0.0))
            ty += float(velocity.get("y", 0.0))
        return {"move": _coarse_direction(tx - me["x"], ty - me["y"], 0.6)}

    # ------------------------------------------------------------------
    def choose_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        started = perf_counter()
        view = canonicalize(observation)

        # The obstacle layout is fixed for a match; drop caches if it changes
        # (a fresh match reusing this object) so stale paths cannot leak.
        key = (view.obstacles, view.flipped)
        if key != self.obstacle_key:
            self.cache = {}
            self.obstacle_key = key
            self.recent = []
            self.last_moves = []
            self.previous = None
            self.previous_score = (view.score_me, view.score_opp)
            self.tracker.reset()
            self.book = {}
            self.recording = None
            self.replay = None
            field = observation["state"].get("field") or {}
            self.constants_ok = ph.assert_official(field)
            if not self.constants_ok and not self.warned:
                self.warned = True
                print("planner: field constants differ from game.json; "
                      "using the config-independent fallback policy",
                      file=sys.stderr, flush=True)
        self.tick += 1

        # Positions jump at a restart, so those transitions teach nothing.
        score = (view.score_me, view.score_opp)
        if score == self.previous_score:
            self.tracker.update(self.previous, view)
            self.since_opp = 0 if view.owner == "opp" else self.since_opp + 1
            self.since_me = 0 if view.owner == "me" else self.since_me + 1
        else:
            self.previous_score = score
            self.previous = None
            self.since_opp = 0
            self.since_me = 0
        self._track_stall(view)

        action = None
        if self.params["restart_replay"] > 0.5 and self.constants_ok:
            action = self._replay_step(view)
        if action is None:
            action = self._decide(view, started)

        move = action.get("move", "STAY")
        if move not in ph.UNIT:
            action = {"move": "STAY"}
        kick = action.get("kick")
        if isinstance(kick, dict):
            if kick.get("direction") not in ph.KICK_DIRS or kick.get("power") not in (1, 2, 3):
                action = {"move": action["move"]}
        self.recent.append(view.me)
        if len(self.recent) > 10:
            self.recent.pop(0)
        self.last_moves.append(action["move"])
        if len(self.last_moves) > 10:
            self.last_moves.pop(0)
        self.previous = view
        recording = self.recording
        if recording is not None and self.pending_signature is not None:
            if len(recording["acts"]) < REPLAY_MAX:
                kick = action.get("kick")
                recording["sigs"].append(self.pending_signature)
                recording["acts"].append(
                    {"move": action["move"], "kick": dict(kick)} if isinstance(kick, dict)
                    else {"move": action["move"]})
            else:
                self.recording = None
        return decanonicalize(action, view.flipped)

    def _track_stall(self, view) -> None:
        """Ticks since the side in possession last gained ground.

        A spell runs from first touch until the other side touches the ball
        (or a goal); the stall clock counts ticks since the carrier last
        reached a new best y, more than 2 units better. A normal attack
        gains ground every few ticks; a deadlock does not.
        """
        if self.since_opp == 0:          # the opponent's spell (or a restart)
            self.att_best = -1.0
            self.att_stall = 0
        elif view.owner == "me" or self.att_best >= 0.0:
            y = view.me[1] if view.owner == "me" else self.att_best
            if y > self.att_best + 2.0:
                self.att_best = y
                self.att_stall = 0
            else:
                self.att_stall += 1
        if self.since_me == 0:           # our spell (or a restart)
            self.def_best = 1e9
            self.def_stall = 0
        elif view.owner == "opp" or self.def_best < 1e8:
            y = view.opp[1] if view.owner == "opp" else self.def_best
            if y < self.def_best - 2.0:
                self.def_best = y
                self.def_stall = 0
            else:
                self.def_stall += 1

    # ------------------------------------------------------------------
    @staticmethod
    def _signature(view) -> tuple:
        return (view.me, view.opp, view.ball, view.ball_vel, view.owner,
                view.status, view.steps, view.loose_steps, round(view.remaining, 9))

    @staticmethod
    def _is_restart(view) -> bool:
        """Any restart state: kick-off, or the tick after a goal, either holder."""
        return (view.steps == 0 and view.owner is not None
                and view.ball_vel == (0.0, 0.0)
                and abs(view.me[0] - 50.0) < 1e-9 and abs(view.me[1] - 35.0) < 1e-9
                and abs(view.opp[0] - 50.0) < 1e-9 and abs(view.opp[1] - 105.0) < 1e-9)

    def _replay_step(self, view):
        """Keep the restart book, and return the recorded action while it applies."""
        score = (view.score_me, view.score_opp)
        recording = self.recording
        if recording is not None and score != recording["score"]:
            start_me, start_opp = recording["score"]
            if (score[0] > start_me and score[1] == start_opp and recording["acts"]
                    and len(recording["acts"]) <= self.params["replay_max_len"]):
                # We scored from this restart without conceding: keep the line.
                self.book[recording["key"]] = (recording["sigs"], recording["acts"])
            elif recording.get("replayed") and score[1] > start_opp:
                # A replayed line that ended in a goal against us no longer
                # reproduces; drop it so restart_variation applies again.
                self.book.pop(recording["key"], None)
            self.recording = None
            self.replay = None
        signature = self._signature(view)
        if self._is_restart(view):
            self.replay = self.book.get(view.owner)
            self.recording = {"key": view.owner, "score": score, "sigs": [], "acts": [],
                              "replayed": self.replay is not None}
            self.replay_index = 0
        self.pending_signature = signature if self.recording is not None else None
        replay = self.replay
        if replay is None:
            return None
        sigs, acts = replay
        index = self.replay_index
        if index < len(sigs) and sigs[index] == signature:
            self.replay_index = index + 1
            act = acts[index]
            kick = act.get("kick")
            return {"move": act["move"], "kick": dict(kick)} if kick else {"move": act["move"]}
        self.replay = None
        return None





    # ------------------------------------------------------------------
    def _decide(self, view, started: float) -> dict[str, Any]:
        if not self.constants_ok:
            return self.fallback_action(view.raw)
        # Soft budget. The runner allows 2.0 s and the search costs under 100 ms
        # here, but the organizer's machine is unknown, so every search can bail
        # out with the best candidate it has already fully verified rather than
        # risk a late line.
        deadline = started + self.params["budget_s"]
        self.proven = False
        if view.owner == "me":
            base = self._attack(view, started, deadline)
        elif view.owner == "opp":
            params, _ = self._defend_params(view)
            base = pl.defend_choice(view, params, self.cache, deadline)
        else:
            base = self._loose(view, started, deadline)
        return base

    # ------------------------------------------------------------------
    def _defend_params(self, view) -> tuple:
        """Defensive parameters for this tick, and whether the stall press is on.

        Shared by the held-ball and loose-ball paths: a carrier that taps the
        ball and re-collects it is loose every other tick, and scoring those
        ticks with a different objective made the keeper flip-flop.
        """
        params = self._effective_params()
        if (params["press_after"] > 0.0 and self.def_stall >= params["press_after"]
                and view.opp[1] >= params["press_min_y"]
                and view.score_me <= view.score_opp):
            params = dict(params)
            params["w_press"] = params["stall_press"]
            params["w_cover"] = params["w_cover"] * params["stall_cover"]
            return params, True
        return params, False

    def _effective_params(self) -> dict[str, Any]:
        """Parameters for this tick, with strong-opponent overrides applied."""
        params = self.params
        fit = self.tracker.chaser_fit(int(params["model_min_samples"]))
        if fit < 0.0 or fit >= params["strong_fit_max"]:
            return params
        tuned = dict(params)
        tuned["opp_lead_loose"] = params["strong_lead"]
        tuned["contest_slack"] = params["strong_contest_slack"]
        tuned["release_threshold"] = params["strong_release_threshold"]
        return tuned

    def _attack(self, view, started: float, deadline: float) -> dict[str, Any]:
        params = self._effective_params()
        mx, my = view.me
        ox, oy = view.opp
        opponent_distance = hypot(ox - mx, oy - my)

        # Contact can displace us before the kick fires, so demand a wider
        # clearance when the opponent is near enough to shove us.
        required = params["shoot_margin"]
        if opponent_distance <= params["close_opponent"]:
            required = params["shoot_margin_close"]

        samples = int(params["model_min_samples"])
        threshold = params["model_threshold"]
        guaranteed, best_by_move = pl.shot_options(
            view, required, self.cache, params,
            self.tracker.modes(samples, threshold),
            self.tracker.leads(samples, threshold), deadline)
        if guaranteed:
            pick = guaranteed[0]
            # Tier 1 is a proof, not an estimate; never route it through search.
            self.proven = pick["tier"] == 1
            return {"move": pick["move"],
                    "kick": {"direction": pick["direction"], "power": pick["power"]}}

        vulnerable = view.steps >= 3 and opponent_distance <= params["vuln_distance"]
        must_release = view.steps >= params["force_release_step"]

        scan = view.steps >= params["release_scan_step"]
        if vulnerable or must_release or scan:
            options = pl.release_options(view, params, self.cache, deadline)
            if options:
                best = options[0]
                # When forced, take the best available. When at risk, release
                # unless it simply hands the ball over. Otherwise release only
                # for a clearly winning pass -- carrying all the way to the
                # forced-release step leaves no good option left to choose.
                bar = (params["release_threshold"] if (vulnerable or must_release)
                       else params["release_good"])
                if must_release or best["first"] > bar:
                    return {"move": best["move"],
                            "kick": {"direction": best["direction"], "power": best["power"]}}
            if must_release:
                return {"move": "UP", "kick": {"direction": "UP", "power": 2}}

        return self._dribble(view, best_by_move, deadline)





    @staticmethod
    def _at_restart(view) -> bool:
        """True on the one tick that follows a goal, holding the ball.

        Canonicalisation puts the restart spots at (50, 35) for the carrier and
        (50, 105) for the defender whichever side we are, so one test covers both.
        """
        return (view.owner == "me" and view.steps == 0
                and abs(view.me[0] - 50.0) < 1e-6 and abs(view.me[1] - 35.0) < 1e-6
                and abs(view.opp[0] - 50.0) < 1e-6 and abs(view.opp[1] - 105.0) < 1e-6)

    def _dribble(self, view, best_by_move: dict,
                 deadline: float | None = None) -> dict[str, Any]:
        params = self.params
        mx, my = view.me
        ox, oy = view.opp
        obstacles = view.obstacles
        w_shot = params["w_shot"]
        w_prog = params["w_prog"]
        w_risk = params["w_risk"]
        w_crowd = params["w_crowd"]
        w_center = params["w_center"]
        w_lane = params["w_lane"]
        w_mouth = params["w_mouth"]
        will_be_vulnerable = view.steps + 1 >= 3

        seek = None
        if (params["seek_after"] > 0.0 and self.att_stall >= params["seek_after"]
                and view.score_me <= view.score_opp):
            seek = pl.seek_target(view, self.cache, params["seek_travel"], deadline)
            if seek is not None and seek[2] < params["seek_min_clear"]:
                seek = None
        w_seek = params["w_seek"]
        memo = self.cache.setdefault("travel_memo", {})
        if seek is not None:
            seek_from = pl.travel_steps(mx, my, seek[0], seek[1], obstacles, memo)

        banned = self._oscillation_ban()
        ranked: list = []
        for name, px, py in pl.legal_moves(mx, my, obstacles):
            if name in banned:
                continue
            clearance = best_by_move.get(name, -pl.BIG)
            if clearance > -pl.BIG / 2:
                shot_term = max(-40.0, min(40.0, clearance))
            else:
                deficit = ph.GOAL_TOP_Y - (py + ph.KICK_CLEARANCE) - ph.KICK_DISTANCES[2]
                shot_term = -40.0 - (deficit if deficit > 0.0 else 0.0)
            progress = py - my
            if seek is not None:
                progress = w_seek * ph.PLAYER_SPEED * (
                    seek_from - pl.travel_steps(px, py, seek[0], seek[1], obstacles, memo))
            distance = hypot(px - ox, py - oy)
            risk = 0.0
            if will_be_vulnerable and distance <= ph.TACKLE_R + ph.PLAYER_SPEED:
                risk += 1.0
            if distance < ph.CONTACT_R + ph.PLAYER_SPEED:
                risk += 0.4
            crowd = 0.0
            if px < 9.0:
                crowd += 9.0 - px
            elif px > 91.0:
                crowd += px - 91.0
            if py > ph.FIELD_H - 9.0:
                crowd += py - (ph.FIELD_H - 9.0)
            # Being outside the goal-mouth x-corridor is what makes a position
            # unshootable; a blunt pull toward x=50 instead dragged the carrier
            # into the defender's lane, so tuning zeroed it and left no
            # x-gradient at all. The agent then parked in the attacking corner
            # at (18.9, 136.7) -- a spot from which *no* kick can cross the
            # line inside the mouth -- because every move scored identically
            # and STAY won the tie.
            mouth_miss = abs(px - 50.0) - params["goal_half"]
            if mouth_miss < 0.0:
                mouth_miss = 0.0
            lane = 0.0
            if oy > py:
                # Defender is between us and the goal: being in its x-lane means
                # dribbling into it, so prefer to carry around.
                gap = params["lane_width"] - abs(px - ox)
                if gap > 0.0:
                    lane = gap
            score = (w_shot * shot_term + w_prog * progress - w_risk * risk
                     - w_crowd * crowd - w_center * abs(px - 50.0)
                     - w_lane * lane - w_mouth * mouth_miss)
            if name == "STAY":
                # Standing still forfeits the move entirely and cannot tackle,
                # so it must win on merit rather than on a tie.
                score -= params["stay_penalty"]
            ranked.append((score, name))
        if not ranked:
            return {"move": "STAY"}
        ranked.sort(key=lambda item: (-item[0], item[1]))
        index = 0
        variants = int(params["restart_variation"])
        if variants > 1 and self._at_restart(view):
            # Deterministic, and only on the restart tick: changing the first move
            # sends the whole possession down a different line, which is all the
            # cycle needs to break.
            index = (view.score_opp % variants) % len(ranked)
        return {"move": ranked[index][1]}

    def _oscillation_ban(self) -> tuple:
        """Ban the last move when the player has been shuffling without progress.

        Compares distance *travelled* with distance *gained*. The previous test
        looked only at the net displacement across a fixed six-tick window,
        which is blind to a two-tick A-B-A cycle whenever the window spans an
        odd number of steps -- the endpoints then sit 4 units apart and the
        shuffle reads as healthy movement. A loose ball was lost to exactly that
        cycle on dev seed 10007.
        """
        if len(self.recent) < 5:
            return ()
        window = self.recent[-5:]
        travelled = 0.0
        ax, ay = window[0]
        for bx, by in window[1:]:
            travelled += hypot(bx - ax, by - ay)
            ax, ay = bx, by
        net = hypot(window[-1][0] - window[0][0], window[-1][1] - window[0][1])
        # Require genuinely no progress, not merely a curved path: a dribble
        # that turns covers ground inefficiently but still gets somewhere, and
        # banning its moves measurably cost goals scored.
        if (travelled > self.params["osc_travel_min"]
                and net < self.params["osc_net_limit"]
                and self.last_moves):
            return (self.last_moves[-1],)
        return ()

    # ------------------------------------------------------------------
    def _loose(self, view, started: float, deadline: float) -> dict[str, Any]:
        params = self._effective_params()
        plan = pl.intercept_plan(view, params)
        mx, my = view.me
        banned = self._oscillation_ban()

        if plan["we_win"]:
            tx, ty = plan["target"]
            return {"move": pl.move_toward(view, tx, ty, banned)}

        # They get there first. Hold the lane between the arrival point and our
        # goal rather than trailing the ball.
        ex, ey = plan["end"]
        radius = params["loose_defend_radius"]
        defend_params, pressing = self._defend_params(view)
        if view.iteration > 2 and (pressing or radius > 0.0):
            ox, oy = view.opp
            if pressing or hypot(ex - ox, ey - oy) <= radius or plan["opp_time"] <= 1.0:
                return pl.defend_choice(view, defend_params, self.cache, deadline)
        tx, ty = pl.shadow_point(ex, ey, params)
        # At a restart the carrier starts dead centre; owning the corridor is
        # what stops the straight power-3 opener.
        if view.iteration <= 2 or (abs(ex - 50.0) < 1.0 and ey > 90.0):
            tx = max(50.0 - params["corridor_half"],
                     min(50.0 + params["corridor_half"], tx))
        return {"move": pl.move_toward(view, tx, ty, banned)}
