"""Attack with possession; contain threats and recover goal-side without it."""
import math

from .legacy_attack import AttackingPolicy
from .geometry import Arena, DIRECTIONS, distance
from .motion import positions


class Policy(AttackingPolicy):
    def __init__(self):
        super().__init__()
        self.mode = "recover"
        self.previous_other = None
        self.other_velocity = (0., 0.)

    def choose_action(self, observation):
        state = observation["state"]
        layout = (state["field"], state["obstacles"])
        if layout != self.layout or state["iteration"] == 0:
            self.arena = Arena(state)
            self.layout = layout
            self.history = []
            self.previous_other = None
        self.sign = 1 if observation["attack_direction"] == "UP" else -1
        self.player_id = observation["player_id"]
        self.opponent_id = observation["opponent_id"]
        me, other = [tuple(state["players"][p][axis] for axis in ("x", "y"))
                     for p in (self.player_id, self.opponent_id)]
        velocity = (0., 0.) if self.previous_other is None else (
            other[0]-self.previous_other[0], other[1]-self.previous_other[1])
        self.other_velocity = velocity if math.hypot(*velocity) <= self.arena.speed+0.1 else (0., 0.)
        self.previous_other = other
        self.history = (self.history+[me])[-8:]
        ball = state["ball"]
        if ball["possession"] == self.player_id:
            self.mode = "attack"
            return self.attack(me, other, ball, observation["action_space"])
        if ball["possession"] == self.opponent_id:
            return {"move": self.defend(me, other, ball)}
        if ball["status"] == "moving":
            path, goal = self.arena.trajectory((ball["x"], ball["y"]),
                (ball["velocity"]["x"], ball["velocity"]["y"]), ball["remaining_kick_distance"])
            self.mode = "intercept"
            if goal == -self.sign:
                self.mode = "save"
                return {"move": self.intercept(me, path)}
            ours = self.arrival(me, path)
            theirs = self.arrival(other, path)
            if ours <= theirs+.25:
                return {"move": self.intercept(me, path)}
            target = next(((x, y) for t, x, y in path if t >= theirs), path[-1][1:])
            return {"move": self.contain(me, target, other)}
        target = (ball["x"], ball["y"])
        ours = self.path_distance(me, target)
        theirs = self.path_distance(other, target)
        if ours <= theirs+1.:
            self.mode = "collect"
            return {"move": self.navigate(me, target)}
        return {"move": self.contain(me, target, other)}

    def path_distance(self, point, target):
        return self.arena.route_length(point, target, self.arena.route_distances(target))

    def arrival(self, point, path):
        for i, (turn, x, y) in enumerate(path):
            if i % 6 and i != len(path)-1:
                continue
            if distance(point, (x, y)) > turn*self.arena.speed+4.5:
                continue
            if self.path_distance(point, (x, y)) <= turn*self.arena.speed+4.3:
                return float(turn)
        return max(path[-1][0], (self.path_distance(point, path[-1][1:])-5.)/self.arena.speed)

    def goal_side_target(self, threat):
        arena = self.arena
        depth = self.progress(threat)
        gap = min(18., max(8., depth*.3))
        own_goal = (arena.width/2, 0. if self.sign > 0 else arena.height)
        length = distance(threat, own_goal) or 1.
        return (threat[0]+(own_goal[0]-threat[0])*gap/length,
                threat[1]+(own_goal[1]-threat[1])*gap/length)

    def contain(self, me, threat, other):
        self.mode = "recover" if self.progress(me) > self.progress(threat)-5. else "contain"
        target = self.goal_side_target(threat)
        arena = self.arena
        routes = arena.route_distances(target)
        def cost(pair):
            move, point = pair
            value = arena.route_length(point, target, routes)
            value += max(0., 9.-distance(point, other))*2.
            value += max(0., self.progress(point)-self.progress(threat)+3.)*.6
            return value
        return min(arena.moves(me), key=cost)[0]

    def defend(self, me, other, ball):
        # Control is protected for three steps. Otherwise tackle from goal-side.
        if (ball["possession_steps"] >= 3 and distance(me, other) <= 10.
                and self.progress(me) < self.progress(other)-1.):
            self.mode = "tackle"
            return self.navigate(me, other)
        return self.contain(me, other, other)

    def replies(self, me, other):
        """Plausible pursuit and continuation moves, with no opponent identity."""
        legal = self.arena.moves(other)
        pursue = min(legal, key=lambda pair: distance(pair[1], me))[0]
        continued = min(legal, key=lambda pair: distance(pair[1], (
            other[0]+self.other_velocity[0], other[1]+self.other_velocity[1])))[0]
        return list(dict.fromkeys([pursue, continued, "STAY"]))

    def attack(self, me, other, ball, action_space):
        action = super().attack(me, other, ball, action_space)
        if distance(me, other) > 16.:
            return action
        arena = self.arena
        old = {self.player_id: me, self.opponent_id: other}
        replies = self.replies(me, other)
        def outcomes(move):
            return [positions(arena, old, {self.player_id: move, self.opponent_id: reply}, me)
                    for reply in replies]

        def kick_value(move, direction, power):
            values = []
            dx, dy = DIRECTIONS[direction]
            for predicted in outcomes(move):
                own, opponent = predicted[self.player_id], predicted[self.opponent_id]
                path, goal = arena.trajectory((own[0]+dx*4.55, own[1]+dy*4.55),
                                              (dx, dy), arena.kick_distances[power-1])
                first = [(x, y) for t, x, y in path if t == 1]
                # A first-turn giveaway near our goal cannot be called a clearance.
                if any(distance(opponent, p) <= 4.5 for p in first):
                    values.append(-300.)
                    continue
                if any(distance(own, p) <= 4.5 for p in first):
                    values.append(-100.)
                    continue
                margin = min(distance(opponent, (x,y)) - max(0,t-1)*arena.speed-4.5
                             for t,x,y in path)
                if goal == -self.sign:
                    values.append(-400.)
                elif goal == self.sign and margin >= 0:
                    values.append(350.+margin)
                else:
                    end = path[-1][1:]
                    # Prefer a recoverable forward clearance over contact cycling.
                    race = (distance(opponent,end)-distance(own,end))/arena.speed
                    progress = self.progress(end)-self.progress(me)
                    values.append(progress + 6.*race + min(0.,margin)*3. - 10.)
            return min(values)*.65+sum(values)/len(values)*.35

        if "kick" in action:
            value = kick_value(action["move"], action["kick"]["direction"], action["kick"]["power"])
            if value > -25.:
                return action
        else:
            predicted = outcomes(action["move"])
            if all(distance(s[self.player_id], s[self.opponent_id]) > 6.15 for s in predicted):
                return action

        self.mode = "escape"
        best_score, best = -math.inf, action
        for move, point in arena.moves(me):
            predicted = outcomes(move)
            separation = min(distance(s[self.player_id], s[self.opponent_id]) for s in predicted)
            progress = min(self.progress(s[self.player_id])-self.progress(me) for s in predicted)
            # Preserve control before turning upfield. Protected possession allows
            # two steps to create space, but contact at age 3 is a turnover.
            score = progress + 3.*min(14.,separation)-24.
            if separation <= 6.15 and ball["possession_steps"] >= 3:
                score -= 180.
            if ball["possession_steps"] >= 9:
                score -= 150.
            if score > best_score:
                best_score, best = score, {"move":move}
            for direction in action_space["kick"]["direction"]:
                value = kick_value(move,direction,1)
                if value > best_score:
                    best_score,best = value,{"move":move,"kick":{"direction":direction,"power":1}}
        return best
