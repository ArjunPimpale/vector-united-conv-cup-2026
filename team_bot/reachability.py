"""Legal movement witnesses; unsuccessful bounded searches remain uncertain."""
import math
import time

from .geometry import distance


def turns(trace):
    result = {}
    path=trace.path[:-1] if trace.goal else trace.path
    for turn, x, y in path:
        result.setdefault(turn, []).append((x,y))
    return result


def samples(trace):
    return [(turn,p) for turn,points in turns(trace).items()
            for p in (points[0],points[len(points)//2],points[-1])]


def route_margin(arena, point, trace, launched=True):
    """Estimated interception margin; obstacle routing is not a proof."""
    distances = arena.route_distances(point)
    return min((arena.route_length(p,point,distances)-4.5
                -arena.speed*max(0,t-int(launched)) for t,p in samples(trace)),default=math.inf)


def interception_plan(arena, point, trace, launched=False, deadline=math.inf, width=20):
    """Return a legal solo movement witness, or status impossible/unknown/budget.

    With launched=True the first sample group is before any additional move.
    Contact with another moving player is handled at launch by the caller.
    Later witnesses describe solo reachability and are replanned each turn.
    """
    groups=turns(trace)
    if not groups:
        return {'status':'impossible','turn':None,'moves':[]}
    if all(distance(point,p)>4.5+arena.speed*max(0,t-int(launched))
           for t,points in groups.items() for p in points):
        return {'status':'impossible','turn':None,'moves':[]}
    frontier=[(point,[])]
    pruned=False
    for turn,points in groups.items():
        if time.perf_counter()>=deadline:
            return {'status':'budget','turn':None,'moves':[]}
        candidates={}
        for origin,moves in frontier:
            options=[('STAY',origin)] if launched and turn==1 else arena.moves(origin)
            for move,target in options:
                sequence=moves if launched and turn==1 else moves+[move]
                if any(distance(target,p)<=4.5 for p in points):
                    return {'status':'reachable','turn':turn,'moves':sequence,'point':target}
                candidates.setdefault(target,sequence)
        future=[(t,p) for t,ps in groups.items() if t>=turn for p in (ps[0],ps[-1])]
        ranked=sorted(candidates.items(),key=lambda item:min(
            distance(item[0],p)-arena.speed*(t-turn) for t,p in future))
        pruned |= len(ranked)>width
        frontier=ranked[:width]
    return {'status':'unknown' if pruned else 'impossible','turn':None,'moves':[]}
