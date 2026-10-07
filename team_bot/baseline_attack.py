"""All-power shot selection and bounded dribble-to-shot planning."""
import math

from .geometry import DIRECTIONS, distance
from .motion import positions
from .reachability import interception_plan, samples


def positional(policy, me, other, age):
    arena=policy.arena
    goal=(arena.width/2,arena.height-8 if policy.sign>0 else 8.)
    value=policy.progress(me)-.3*policy.route(me,goal)-.3*abs(me[0]-arena.width/2)
    if age>=2:value-=max(0.,13.-distance(me,other))*5.
    value-=1.2*sum(distance(me,p)<1. for p in policy.history[:-1])
    if age>=9:value-=140.
    return value


def margin(policy, other, trace):
    return min((policy.route(other,p)-4.5-policy.arena.speed*max(0,t-1)
                for t,p in samples(trace)),default=math.inf)


def value(policy, me, other, trace, verify=False):
    arena=policy.arena
    goal_index=len(trace.path)-1 if trace.goal else -1
    first=[(x,y) for i,(t,x,y) in enumerate(trace.path) if t==1 and i!=goal_index]
    if any(distance(me,p)<=4.5 for p in first):return -250.
    if any(distance(other,p)<=4.5 for p in first):return -350.
    if trace.goal==-policy.sign:return -650.
    safety=margin(policy,other,trace)
    if trace.goal==policy.sign:
        score=390.+min(12.,safety)*4.-trace.path[-1][0]
        if safety<0:score-=max(0.,-safety)*12.
        # Legal movement witnesses refine the approximate obstacle route test.
        if verify:
            witness=interception_plan(arena,other,trace,launched=True,
                                      deadline=policy.deadline,width=18)
            if witness['status']=='reachable':score=-90.+8*witness['turn']
            elif witness['status']=='budget':return -math.inf
            elif witness['status']=='unknown':score-=45.
        return score
    end=trace.path[-1][1:]; flight=trace.path[-1][0]
    our_time=max(flight,(policy.route(me,end)-5.)/arena.speed)
    their_time=max(0.,(policy.route(other,end)-5.)/arena.speed)
    race=their_time-our_time
    base=positional(policy,end,other,0)
    # A pass must improve the position and be recoverable, including in flight.
    return (base-18.-2.*flight+min(6.,race)*5.
            -max(0.,1.-race)*18.-max(0.,-safety)*3.)


def trace_for(policy, point, direction, power):
    dx,dy=DIRECTIONS[direction]
    key=(point,direction,power)
    if key not in policy.shot_cache:
        policy.shot_cache[key]=policy.arena.trace((point[0]+dx*4.55,point[1]+dy*4.55),
                                                (dx,dy),32.*power)
    return policy.shot_cache[key]


def choose_attack(policy, me, other, ball, action_space):
    arena=policy.arena; age=ball['possession_steps']
    moves=sorted(arena.moves(me),key=lambda m:positional(policy,m[1],other,age),reverse=True)
    candidates=[]
    directions=sorted(action_space['kick']['direction'],
                      key=lambda d:DIRECTIONS[d][1]*policy.sign,reverse=True)
    for move,point in moves:
        candidates.append((positional(policy,point,other,age),{'move':move},None))
        for direction in directions:
            for power in action_space['kick']['power']:
                if policy.expired():break
                trace=trace_for(policy,point,direction,power)
                score=value(policy,point,other,trace)-.08*power
                action={'move':move,'kick':{'direction':direction,'power':power}}
                candidates.append((score,action,trace))
    candidates.sort(key=lambda c:c[0],reverse=True)
    # Never select an unexamined candidate when the shared deadline expires.
    best=(-math.inf,{'move':moves[0][0]},None)
    replies=policy.opponent_replies(me,other)
    old={policy.player_id:me,policy.opponent_id:other}
    examined=0
    shortlist=[c for c in candidates if 'kick' not in c[1]]
    shortlist += [c for c in candidates if 'kick' in c[1] and c[2].goal==policy.sign][:8]
    shortlist += [c for c in candidates if 'kick' in c[1] and c[2].goal!=policy.sign][:4]
    for estimate,action,trace in shortlist:
        if policy.expired():break
        values=[]; representative=None
        for reply in replies:
            predicted=positions(arena,old,{policy.player_id:action['move'],policy.opponent_id:reply},me)
            own,enemy=predicted[policy.player_id],predicted[policy.opponent_id]
            if 'kick' in action:
                kick=action['kick']; actual=trace_for(policy,own,kick['direction'],kick['power'])
                values.append(value(policy,own,enemy,actual,verify=actual.goal==policy.sign)-.08*kick['power'])
                representative=actual
            else:
                score=positional(policy,own,enemy,age)
                if age>=3 and distance(own,enemy)<=6.15:score-=250.
                values.append(score)
        if values:
            score=.55*min(values)+.45*sum(values)/len(values)
            if score>best[0]:best=(score,action,representative)
            examined+=1
    if best[0]==-math.inf:
        # Cheap root action avoids a forced release where possible.
        best=candidates[0]
    if policy.setup_enabled and 'kick' not in best[1] and not policy.expired():
        setup=setup_search(policy,best,action_space)
        if setup and setup[0]>best[0]:best=setup
    chosen=best[1]
    policy.mode='SETUP_SHOT' if getattr(policy,'setup_chosen',False) else 'ATTACK'
    if 'kick' in chosen:
        policy.mode='BANK_SHOT' if best[2] and best[2].goal==policy.sign and best[2].bounces else (
            'DIRECT_SHOT' if best[2] and best[2].goal==policy.sign else 'SELF_PASS')
    policy.diagnostic.update({'intention':policy.mode,'evaluated_actions':len(candidates),
        'refined_actions':examined,'kick_purpose':policy.mode,'selected_value':round(best[0],2)})
    if best[2]:
        policy.diagnostic['selected_path']=[[x,y] for t,x,y in best[2].path[::6]]+[list(best[2].path[-1][1:])]
        policy.diagnostic['bounces']=best[2].bounces
    return chosen


def setup_search(policy,best,action_space):
    """Three-turn movement beam; only execute the first move and replan."""
    frontier=[(policy.world,None,[],best[0])]; result=None
    for depth in range(1,4):
        expanded=[]
        for world,first,route,_ in frontier:
            if policy.expired():return result
            me=world.players[policy.player_id]; other=world.players[policy.opponent_id]
            reply=policy.opponent_replies(me,other)[0]
            for move,_ in policy.arena.moves(me):
                actions={policy.player_id:{'move':move},policy.opponent_id:{'move':reply}}
                predicted=world.predict(actions)
                if predicted.possession!=policy.player_id or predicted.done:continue
                own,enemy=predicted.players[policy.player_id],predicted.players[policy.opponent_id]
                score=positional(policy,own,enemy,predicted.possession_steps)
                expanded.append((predicted,first or {'move':move},route+[list(own)],score))
        # Retain distinct reachable states; avoid rounded collision boundaries.
        unique={}
        for item in sorted(expanded,key=lambda e:e[3],reverse=True):
            key=tuple(item[0].players.items())
            unique.setdefault(key,item)
        frontier=list(unique.values())[:12]
        for world,first,route,score in frontier[:6]:
            own=world.players[policy.player_id]; enemy=world.players[policy.opponent_id]
            for direction in action_space['kick']['direction']:
                if policy.expired():return result
                trace=trace_for(policy,own,direction,3)
                if trace.goal!=policy.sign:continue
                shot=value(policy,own,enemy,trace)
                planned=shot-32.*depth
                if planned>best[0]+10. and (result is None or planned>result[0]):
                    result=(planned,first,None)
                    policy.setup_chosen=True
                    policy.diagnostic['setup_route']=route
                    policy.diagnostic['future_shot_path']=[[x,y] for t,x,y in trace.path[::6]]
    return result
