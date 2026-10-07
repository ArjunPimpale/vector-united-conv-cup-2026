"""Simultaneous player movement forecast from public positions and actions."""
import math

from .geometry import DIRECTIONS, distance


def positions(arena, old, moves, ball):
    """Predict legal movement/contact without changing any observation."""
    proposed = {}
    for player in ("player_1", "player_2"):
        vx, vy = DIRECTIONS[moves[player]]
        point = (old[player][0]+vx*arena.speed, old[player][1]+vy*arena.speed)
        proposed[player] = point if arena.valid(point) else old[player]
    if distance(proposed["player_1"], proposed["player_2"]) >= 2*arena.radius:
        return proposed
    center = tuple((proposed["player_1"][i]+proposed["player_2"][i])/2 for i in (0, 1))
    dx, dy = (old["player_1"][i]-old["player_2"][i] for i in (0, 1))
    length = math.hypot(dx, dy)
    dx, dy = (dx/length, dy/length) if length else (1., 0.)
    resolved = {"player_1": (center[0]+dx*arena.radius, center[1]+dy*arena.radius),
                "player_2": (center[0]-dx*arena.radius, center[1]-dy*arena.radius)}
    if (sum(distance(old[p], resolved[p]) for p in old) > .1
            and all(arena.valid(p) for p in resolved.values())):
        return resolved
    solo = []
    for player in old:
        other = "player_2" if player == "player_1" else "player_1"
        if arena.valid(proposed[player]) and distance(proposed[player], old[other]) >= 2*arena.radius:
            solo.append((distance(old[player], ball)-distance(proposed[player], ball), player))
    if solo:
        _, player = max(solo)
        return {p: proposed[p] if p == player else old[p] for p in old}
    options = []
    for player in old:
        other = "player_2" if player == "player_1" else "player_1"
        for move, point in arena.moves(old[player]):
            if move != "STAY" and distance(point, old[other]) >= 2*arena.radius:
                options.append((distance(old[player], ball)-distance(point, ball), player, point))
    if options:
        _, player, point = max(options)
        return {p: point if p == player else old[p] for p in old}
    return dict(old)
