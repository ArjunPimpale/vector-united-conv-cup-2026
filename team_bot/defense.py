"""Goal coverage, early recovery, and pressing only after checking shots."""
import math

from .attack import trace_for
from .geometry import DIRECTIONS, distance
from .motion import positions
from .reachability import samples, interception_plan


def coverage_target(policy,threat):
    arena=policy.arena
    depth=policy.progress(threat)
    # Move toward the goal early enough to cover diagonal and rebound routes.
    defend_depth=min(28.,max(8.,depth*.28))
    x=arena.width/2+(threat[0]-arena.width/2)*min(.6,defend_depth/max(1.,depth))
    y=defend_depth if policy.sign>0 else arena.height-defend_depth
    target=(max(arena.radius,min(arena.width-arena.radius,x)),y)
    if arena.valid(target):return target
    # Project an obstructed anchor onto a physically legal goal-covering point.
    candidates=[(px,depth if policy.sign>0 else arena.height-depth)
        for depth in (4.,6.,12.,20.,28.)
        for px in (arena.goal_left+arena.radius,arena.width/2,arena.goal_right-arena.radius)]
    candidates += [p for p in arena.nodes if policy.progress(p)<=36.]
    valid=[p for p in candidates if arena.valid(p)]
    return min(valid,key=lambda p:distance(p,target),default=(arena.width/2,
               arena.radius if policy.sign>0 else arena.height-arena.radius))


def baseline_defend(policy,me,other,ball):
    arena=policy.arena; old={policy.player_id:me,policy.opponent_id:other}
    target=coverage_target(policy,other)
    shots=[]
    for reply,enemy in arena.moves(other):
        for direction in DIRECTIONS:
            if direction=='STAY':continue
            if policy.expired():break
            trace=trace_for(policy,enemy,direction,3)
            if trace.goal!=-policy.sign:continue
            first=[(x,y) for t,x,y in trace.path if t==1]
            if any(distance(enemy,p)<=4.5 for p in first):continue
            shots.append((reply,direction,trace))
    scored=[]
    for move,point in arena.moves(me):
        risks=[]; worst_trace=None; worst=-math.inf
        for reply,direction,trace in shots:
            predicted=positions(arena,old,{policy.player_id:move,policy.opponent_id:reply},other)
            own,enemy=predicted[policy.player_id],predicted[policy.opponent_id]
            actual=trace if enemy==dict(arena.moves(other))[reply] else trace_for(policy,enemy,direction,3)
            if actual.goal!=-policy.sign:continue
            # Obstacle-aware estimated future coverage, with correct launch timing.
            danger=min((policy.route(own,p)-4.5-arena.speed*max(0,t-1)
                        for t,p in samples(actual)),default=1000.)
            # Immediate catches are concrete, not just a reachability estimate.
            catch_path=actual.path[:-1] if actual.goal else actual.path
            if any(t==1 and distance(own,(x,y))<=4.5 for t,x,y in catch_path):danger=-12.
            risks.append(danger)
            if danger>worst:worst,worst_trace=danger,actual
        costs=policy.route(point,target)*.32
        if risks:
            risks.sort(reverse=True)
            costs+=max(0.,risks[0])*8.+sum(max(0.,r) for r in risks[:6])*1.5
            costs+=risks[0]*.35
        # Do not tackle a kicking carrier. First measure the shot exposure.
        press=(ball['possession_steps']>=3 and distance(point,other)<distance(me,other)
               and policy.progress(point)<policy.progress(other))
        if press and (not risks or max(risks)<-2.):costs-=min(8.,distance(me,other)-distance(point,other))*2.
        costs+=max(0.,policy.progress(point)-policy.progress(other)+2.)*2.
        scored.append((costs,move,point,worst_trace))
    _,move,point,threat=min(scored,key=lambda item:item[0])
    policy.mode='RECOVER_GOAL' if policy.progress(me)>policy.progress(other)-5 else 'COVER_SHOTS'
    if (ball['possession_steps']>=3 and distance(point,other)<distance(me,other)
            and distance(point,other)<=8. and policy.progress(point)<policy.progress(other)):
        policy.mode='PRESS'
    policy.diagnostic.update({'intention':policy.mode,'target':list(target),'threat_count':len(shots)})
    if threat:
        policy.diagnostic['threat_path']=[[x,y] for t,x,y in threat.path[::6]]+[list(threat.path[-1][1:])]
        policy.diagnostic['turns_to_goal']=threat.path[-1][0]
    return move


def intercept(policy,me,path):
    from .geometry import Trace
    last_y=path[-1][2]
    goal=1 if last_y+1.5>=policy.arena.height else -1 if last_y-1.5<=0 else 0
    trace=Trace(path,goal,[])
    witness=interception_plan(policy.arena,me,trace,deadline=policy.deadline,width=28)
    if witness['status']=='reachable' and witness['moves']:
        policy.diagnostic.update({'intercept_turn':witness['turn'],'target':list(witness['point'])})
        return witness['moves'][0]
    return policy.legacy_intercept(me,path)


def defend(policy,me,other,ball):
    if not policy.coverage_enabled:return baseline_defend(policy,me,other,ball)
    arena=policy.arena;target=coverage_target(policy,other)
    old={policy.player_id:me,policy.opponent_id:other};scored=[]
    stale_cover=policy.memory_enabled and getattr(policy,'turns_without_control',0)>=48
    # Future carrier lanes include lateral and protected carries. No style
    # posterior removes a current or future direct/bank threat.
    for move,point in arena.moves(me):
        if policy.expired() and scored:break
        risks=[];worst_trace=None;worst=-math.inf;worst_enemy=None
        for reply in DIRECTIONS:
            moved=positions(arena,old,{policy.player_id:move,policy.opponent_id:reply},other)
            own,enemy=moved[policy.player_id],moved[policy.opponent_id]
            launches=[(1,enemy)]
            # Two more legal carrier moves, bounded by its forced-release clock.
            carrier=enemy
            for depth in (2,3):
                if ball['possession_steps']+depth>=10:break
                dx,dy=DIRECTIONS[reply]
                proposal=(carrier[0]+dx*arena.speed,carrier[1]+dy*arena.speed)
                carrier=proposal if arena.valid(proposal) else carrier
                # A future player collision is uncertain. Do not call an
                # obstacle-clear solo path a guaranteed attacking continuation.
                if distance(carrier,own)<2*arena.radius:break
                launches.append((depth,carrier))
            for depth,carrier in launches:
                for direction in DIRECTIONS:
                    if direction=='STAY':continue
                    trace=trace_for(policy,carrier,direction,3)
                    if trace.goal!=-policy.sign:continue
                    first=[(x,y) for t,x,y in trace.path[:-1] if t==1]
                    if any(distance(carrier,p)<=4.5 for p in first):continue
                    danger=min((policy.route(own,p)-4.5-arena.speed*max(0,t+depth-2)
                                for t,p in samples(trace)),default=1000.)
                    if depth==1 and any(distance(own,p)<=4.5 for p in first):danger=-12.
                    risks.append(danger)
                    if danger>worst:worst,worst_trace,worst_enemy=danger,trace,carrier
        # A long possession drought with no sampled scoring lane calls for an
        # obstacle-aware challenge. An anchor-distance penalty would eventually
        # outweigh a bounded closing reward and strand us far from the carrier.
        # Reassess every turn; any newly exposed lane restores coverage costs.
        cost=policy.route(point,other if stale_cover and not risks else target)*.32
        if risks:
            risks.sort(reverse=True)
            cost+=max(0.,risks[0])*8.+sum(max(0.,r) for r in risks[:6])*1.5
            # Extra slack is capped: it cannot keep rewarding departures from
            # the covering anchor just because a sampled lane looks easier.
            cost+=max(-2.,risks[0])*.35
        closing=policy.route(me,other)-policy.route(point,other)
        press=(ball['possession_steps']>=2 and closing>.05
               and policy.progress(point)<=policy.progress(other)+4.)
        if press and risks and max(risks)<-3. and distance(point,other)>24.:
            witness=interception_plan(arena,point,worst_trace,launched=True,deadline=policy.deadline-.02,width=12)
            press=witness['status']=='reachable'
        if press and (not risks or max(risks)<-3.):
            # A covered carrier outside the mouth must still be challenged.
            # Restrict pressure to a nearby, already covered lane; distant
            # advancing moves receive no reward for unused interception slack.
            cost-=min(4.,closing)*2.
        cost+=max(0.,policy.progress(point)-policy.progress(other)+2.)*2.
        scored.append([cost,move,point,worst_trace,worst, worst_enemy])
    scored.sort(key=lambda row:row[0])
    baseline_move=baseline_defend(policy,me,other,ball)
    baseline_row=next((row for row in scored if row[1]==baseline_move),None)
    # Preserve the successful v3 challenge when its current/future sampled lanes
    # retain a buffer. Override it when the approaching launch exposes coverage.
    if baseline_row is not None and baseline_row[4]<-2. and not stale_cover and baseline_row[3] is not None:
        scored.remove(baseline_row);scored.insert(0,baseline_row)
    # Witness diagnostics refine the most dangerous current launch. Future
    # launch witnesses remain estimates because carrier/defender may collide.
    best=scored[0]
    if best[3] and not policy.expired():
        witness=interception_plan(arena,best[2],best[3],launched=True,deadline=policy.deadline-.02,width=18)
        policy.diagnostic['cover_witness']={k:v for k,v in witness.items() if k!='moves'}
    _,move,point,threat,risk,_=best
    policy.mode='RECOVER_GOAL' if policy.progress(me)>policy.progress(other)-5 else 'COVER_SHOTS'
    if ball['possession_steps']>=3 and distance(point,other)<distance(me,other) and distance(point,other)<=8. and risk<-2.:policy.mode='PRESS'
    policy.diagnostic.update({'target':list(target),'immediate_danger':round(risk,3) if math.isfinite(risk) else None,
        'coverage_terms':[{'move':row[1],'cost':round(row[0],2),'worst_margin':round(row[4],2) if math.isfinite(row[4]) else None} for row in scored],
        'coverage_horizon':3,'coverage_uncertain':True,'stale_cover':stale_cover})
    if threat:
        policy.diagnostic['threat_path']=[[x,y] for t,x,y in threat.path[::6]]+[list(threat.path[-1][1:])]
        policy.diagnostic['turns_to_goal']=threat.path[-1][0]
    return move
