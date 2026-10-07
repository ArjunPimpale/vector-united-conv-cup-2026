"""Public-physics trajectory prediction and conservative obstacle navigation."""
import heapq
import math
from collections import namedtuple

Trace = namedtuple('Trace', 'path goal bounces')

DIRECTIONS = {
    "STAY": (0., 0.), "UP": (0., 1.),
    "UP_RIGHT": (math.sqrt(.5), math.sqrt(.5)), "RIGHT": (1., 0.),
    "DOWN_RIGHT": (math.sqrt(.5), -math.sqrt(.5)), "DOWN": (0., -1.),
    "DOWN_LEFT": (-math.sqrt(.5), -math.sqrt(.5)), "LEFT": (-1., 0.),
    "UP_LEFT": (-math.sqrt(.5), math.sqrt(.5)),
}


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def hits(point, radius, rect):
    x, y, w, h = rect
    dx = point[0] - min(max(point[0], x), x + w)
    dy = point[1] - min(max(point[1], y), y + h)
    return dx * dx + dy * dy < radius * radius


class Arena:
    def __init__(self, state):
        field = state["field"]
        self.width, self.height = field["width"], field["height"]
        self.radius, self.speed = field["player_radius"], field["player_speed"]
        self.goal_left = (self.width - field["goal_width"]) / 2
        self.goal_right = self.width - self.goal_left
        self.rects = [(o["x"], o["y"], o["width"], o["height"])
                      for o in state["obstacles"]]
        # Constants omitted from observations, fixed in official config/game.json.
        self.ball_radius, self.ball_speed = 1.5, 8.
        self.kick_distances = (32., 64., 96.)
        self.endpoint_links = {}
        self.nodes = []
        pad = self.radius + .2
        for x, y, w, h in self.rects:
            for point in ((x-pad, y-pad), (x-pad, y+h+pad),
                          (x+w+pad, y-pad), (x+w+pad, y+h+pad)):
                if self.valid(point):
                    self.nodes.append(point)
        self.links = [[] for _ in self.nodes]
        for i, a in enumerate(self.nodes):
            for j in range(i):
                b = self.nodes[j]
                if self.clear(a, b):
                    length = distance(a, b)
                    self.links[i].append((j, length))
                    self.links[j].append((i, length))

    def valid(self, point):
        x, y = point
        r = self.radius
        return (r <= x <= self.width-r and r <= y <= self.height-r
                and not any(hits(point, r, rect) for rect in self.rects))

    def clear(self, a, b):
        # Segment versus inflated boxes. Conservative at rounded corners.
        pad = self.radius + .1
        for x, y, w, h in self.rects:
            lo, hi = 0., 1.
            for start, delta, lower, upper in (
                (a[0], b[0]-a[0], x-pad, x+w+pad),
                (a[1], b[1]-a[1], y-pad, y+h+pad),
            ):
                if abs(delta) < 1e-10:
                    if not lower <= start <= upper:
                        lo, hi = 1., 0.
                        break
                else:
                    t1, t2 = (lower-start)/delta, (upper-start)/delta
                    lo, hi = max(lo, min(t1, t2)), min(hi, max(t1, t2))
            if lo <= hi:
                return False
        return True

    def moves(self, point):
        result = []
        for name, (dx, dy) in DIRECTIONS.items():
            dest = (point[0] + self.speed*dx, point[1] + self.speed*dy)
            if self.valid(dest):
                result.append((name, dest))
        return result or [("STAY", point)]

    def route_distances(self, target):
        values = [distance(p, target) if self.clear(p, target) else math.inf
                  for p in self.nodes]
        queue = [(value, i) for i, value in enumerate(values) if math.isfinite(value)]
        heapq.heapify(queue)
        while queue:
            value, i = heapq.heappop(queue)
            if value != values[i]:
                continue
            for j, length in self.links[i]:
                candidate = value + length
                if candidate < values[j]:
                    values[j] = candidate
                    heapq.heappush(queue, (candidate, j))
        return values

    def route_length(self, point, target, values):
        if self.clear(point, target):
            return distance(point, target)
        if point not in self.endpoint_links:
            self.endpoint_links[point]=[(i,distance(point,p)) for i,p in enumerate(self.nodes) if self.clear(point,p)]
        result = min((length+values[i] for i,length in self.endpoint_links[point]
                      if math.isfinite(values[i])),default=math.inf)
        # A point may be inside a conservative corner box but physically legal.
        return result if math.isfinite(result) else distance(point, target) + 30.

    def trajectory(self, origin, vector, remaining):
        trace = self.trace(origin, vector, remaining)
        return trace.path, trace.goal

    def trace(self, origin, vector, remaining):
        """Return substep positions (turn, x, y), and scored end (+1/-1/0).

        Players are deliberately excluded; callers compare their reachable sets.
        Bounce order, goal-mouth inclusivity and distance exhaustion match the
        documented engine. Local prediction does not access engine state.
        """
        x, y = origin
        vx, vy = vector
        norm = math.hypot(vx, vy)
        if remaining <= 1e-9 or norm == 0:
            return Trace([(0, x, y)], 0, [])
        vx, vy = vx/norm, vy/norm
        r = self.ball_radius
        path, turn, bounces = [], 0, []
        while remaining > 1e-9:
            turn += 1
            travel = min(self.ball_speed, remaining)
            steps = max(1, math.ceil(travel / max(.25, r*.45)))
            step = travel / steps
            for _ in range(steps):
                nx, ny = x + vx*step, y + vy*step
                if self.goal_left <= nx <= self.goal_right:
                    goal = 1 if ny+r >= self.height else -1 if ny-r <= 0 else 0
                    if goal:
                        path.append((turn, nx, ny))
                        return Trace(path, goal, bounces)
                if nx-r < 0 or nx+r > self.width:
                    bounces.append((turn, 'vertical_wall', nx, ny))
                    vx = -vx
                    nx = min(max(nx, r), self.width-r)
                if ny-r < 0 or ny+r > self.height:
                    bounces.append((turn, 'horizontal_wall', nx, ny))
                    vy = -vy
                    ny = min(max(ny, r), self.height-r)
                for rect in self.rects:
                    if not hits((nx, ny), r, rect):
                        continue
                    bounces.append((turn, 'obstacle', nx, ny))
                    ox, oy, w, h = rect
                    crossed_x = x <= ox-r or x >= ox+w+r
                    crossed_y = y <= oy-r or y >= oy+h+r
                    if crossed_x:
                        vx = -vx
                    if crossed_y:
                        vy = -vy
                    if not crossed_x and not crossed_y:
                        vx, vy = -vx, -vy
                    nx, ny = x + vx*.05, y + vy*.05
                    break
                x, y = nx, ny
                remaining = max(0., remaining-step)
                path.append((turn, x, y))
        return Trace(path, 0, bounces)
