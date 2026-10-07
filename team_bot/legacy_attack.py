"""Receding-horizon football policy, operating only on public observations."""
import math

from .geometry import Arena, DIRECTIONS, distance


class AttackingPolicy:
    def __init__(self):
        self.arena = None
        self.layout = None
        self.history = []

    def choose_action(self, observation):
        state = observation["state"]
        layout = (state["field"], state["obstacles"])
        if layout != self.layout or state["iteration"] == 0:
            self.arena = Arena(state)
            self.layout = layout
            self.history = []
        self.sign = 1 if observation["attack_direction"] == "UP" else -1
        me = state["players"][observation["player_id"]]
        other = state["players"][observation["opponent_id"]]
        me, other = (me["x"], me["y"]), (other["x"], other["y"])
        self.history = (self.history + [me])[-8:]
        ball = state["ball"]
        if ball["possession"] == observation["player_id"]:
            return self.attack(me, other, ball, observation["action_space"])
        if ball["status"] == "moving":
            path, _ = self.arena.trajectory(
                (ball["x"], ball["y"]),
                (ball["velocity"]["x"], ball["velocity"]["y"]),
                ball["remaining_kick_distance"])
            return {"move": self.intercept(me, path)}
        target = (ball["x"], ball["y"])
        if ball["possession"] == observation["opponent_id"]:
            # Goal-side pressure protects against an immediate straight shot.
            target = (other[0], other[1] - self.sign*5.)
        return {"move": self.navigate(me, target)}

    def navigate(self, me, target):
        arena = self.arena
        target = (min(max(target[0], arena.radius), arena.width-arena.radius),
                  min(max(target[1], arena.radius), arena.height-arena.radius))
        values = arena.route_distances(target)
        return min(arena.moves(me), key=lambda pair: (
            arena.route_length(pair[1], target, values)
            + .25*sum(distance(pair[1], p) < 1. for p in self.history[:-1]),
            pair[0] == "STAY"))[0]

    def intercept(self, me, path):
        arena = self.arena
        # Seek the earliest reachable point along the entire rebound trajectory.
        # Every turn's first/middle/last samples retain near-player crossings.
        candidates = []
        for i, (turn, x, y) in enumerate(path):
            if i % 4 and i != len(path)-1:
                continue
            point = (x, y)
            if distance(me, point) > turn*arena.speed+4.5:
                continue
            values = arena.route_distances(point)
            length = arena.route_length(me, point, values)
            if length <= turn*arena.speed+4.3:
                candidates.append((turn, length, point))
                if len(candidates) >= 4:
                    break
        if candidates:
            target = min(candidates)[2]
        else:
            target = path[-1][1:]
        # Catch a crossing this turn, even if its endpoint is already far away.
        first = [(x, y) for turn, x, y in path if turn == 1]
        catches = [(name, point) for name, point in arena.moves(me)
                   if any(distance(point, p) <= 4.5 for p in first)]
        if catches:
            return min(catches, key=lambda pair: distance(pair[1], target))[0]
        return self.navigate(me, target)

    def progress(self, point):
        return point[1] if self.sign > 0 else self.arena.height-point[1]

    def attack(self, me, other, ball, action_space):
        arena = self.arena
        age = ball["possession_steps"]
        best_score, best_action = -math.inf, {"move": "STAY"}
        # Navigate toward central scoring lanes, retaining width near a defender.
        target = (arena.width/2, arena.height-8 if self.sign > 0 else 8.)
        routes = arena.route_distances(target)
        for move, point in arena.moves(me):
            separation = distance(point, other)
            danger = max(0., 13.-separation) if age >= 2 else 0.
            score = (self.progress(point) - .45*abs(point[0]-arena.width/2)
                     - .3*arena.route_length(point, target, routes) - danger*5.)
            score -= .8*sum(distance(point, p) < 1. for p in self.history[:-1])
            if self.sign*(other[1]-point[1]) > -3 and separation < 30:
                score += 1.8*min(18., abs(point[0]-other[0]))
            if age >= 9:
                score -= 100.
            if score > best_score:
                best_score, best_action = score, {"move": move}
            # Joint movement and kick: kicks launch from the post-move position.
            for direction in action_space["kick"]["direction"]:
                dx, dy = DIRECTIONS[direction]
                origin = (point[0]+dx*4.55, point[1]+dy*4.55)
                path, goal = arena.trajectory(origin, (dx, dy), 96.)
                # Reject shots rebounding into our own body on the release turn.
                if any(t == 1 and distance(point, (x, y)) <= 4.5 for t, x, y in path):
                    continue
                margin = min(distance(other, (x, y))-arena.speed*t-4.5
                             for t, x, y in path)
                if goal == self.sign and margin >= 0:
                    # Positive margin means even a straight-line omniscient
                    # defender cannot intercept. Uncertain shots remain costly.
                    shot_score = 250. + min(20., margin)*9. - path[-1][0]
                    if shot_score > best_score:
                        best_score = shot_score
                        best_action = {"move": move, "kick": {"direction": direction, "power": 3}}
                # A short self-pass can reset control and beat the possession clock.
                short_path = [p for p in path if p[0] <= 4]
                if not short_path or (goal and path[-1][0] <= 4):
                    continue
                end = short_path[-1][1:]
                own_time = max(4., (distance(point, end)-4.5)/arena.speed)
                opp_time = max(0., (distance(other, end)-4.5)/arena.speed)
                pass_margin = min(distance(other, (x, y))-arena.speed*t-4.5
                                  for t, x, y in short_path)
                value = (self.progress(end) - .45*abs(end[0]-arena.width/2)
                         - .3*distance(end, target) - 30.
                         - max(0., own_time-opp_time+1.)*15.
                         - max(0., -pass_margin)*4.)
                if value > best_score:
                    best_score = value
                    best_action = {"move": move, "kick": {"direction": direction, "power": 1}}
        return best_action
