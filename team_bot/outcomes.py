"""Physical root guard and contact-aware, bounded first-event ball races.

A future race is a scenario, not an exhaustive opponent strategy proof. Every
root guard tests all nine raw movement inputs, including blocked tackle inputs.
"""
import math
from .geometry import DIRECTIONS, Trace, distance
from .reachability import interception_plan


def event(world, initial_score, own):
    enemy='player_2' if own=='player_1' else 'player_1'
    if world.score[own]>initial_score[own]:return 'goal'
    if world.score[enemy]>initial_score[enemy]:return 'conceded'
    if world.possession==own:return 'our_claim'
    if world.possession==enemy:return 'opponent_claim'
    if world.done:return 'horizon'
    if world.ball_remaining_distance<=0:return 'stopped'
    return None


def guard(policy, action):
    outcomes=[]
    for reply in DIRECTIONS: # Deliberately do not filter using arena.moves().
        w=policy.world.predict({policy.player_id:action,policy.opponent_id:{'move':reply}})
        outcomes.append((reply,w,event(w,policy.world.score,policy.player_id)))
    return outcomes


def recovery_plan(policy, world, player):
    if world.ball_remaining_distance<=0:
        return {'status':'stationary','moves':[], 'point':world.ball_position}
    trace=policy.arena.trace(world.ball_position,world.ball_velocity,world.ball_remaining_distance)
    return interception_plan(policy.arena,world.players[player],trace,
                             deadline=policy.deadline-.02,width=5)


def chase(policy, world, player, target=None):
    point=world.players[player];target=target or world.ball_position
    routes=policy.arena.route_distances(target)
    return min(policy.arena.moves(point),key=lambda pair:policy.arena.route_length(pair[1],target,routes))[0]


def race(policy, start, enemy_mode='race', own_moves=None, horizon=14):
    """Exact catches/contact along specified future moves; cutoff stays uncertain."""
    own=policy.player_id;enemy=policy.opponent_id;world=start
    result=event(world,policy.world.score,own)
    if result and result!='stopped':return result_record(world,result,1,[],enemy_mode)
    plan=recovery_plan(policy,world,own) if own_moves is None else {'moves':own_moves,'status':'intent'}
    sequence=plan.get('moves',[]); played=[]
    for turn in range(2,horizon+1):
        if policy.expired():return result_record(world,'budget',turn-1,played,enemy_mode)
        move=sequence[turn-2] if turn-2<len(sequence) else chase(policy,world,own)
        if enemy_mode=='recover':
            goal=(policy.arena.width/2,policy.arena.height-12 if policy.sign>0 else 12.)
            reply=chase(policy,world,enemy,goal)
        elif enemy_mode=='hold':reply='STAY'
        else:reply=chase(policy,world,enemy)
        world=world.predict({own:{'move':move},enemy:{'move':reply}});played.append(move)
        result=event(world,policy.world.score,own)
        if result and result!='stopped':return result_record(world,result,turn,played,enemy_mode)
        # A stopped ball can be collected subsequently. It is not a hypothetical
        # unoccupied endpoint reward; simulate those collection moves as well.
    return result_record(world,'horizon',horizon,played,enemy_mode)


def result_record(world, result, turn, moves, assumption):
    return {'event':result,'turn':turn,'point':world.ball_position,
            'age':world.possession_steps,'moves':moves,'world':world,
            'assumption':assumption,'uncertain':turn>1 or result in ('horizon','budget')}


def loss_danger(policy, world):
    """Post-loss exposure: proximity of carrier to our goal and our recovery gap."""
    enemy=world.players[policy.opponent_id];own=world.players[policy.player_id]
    depth=policy.progress(enemy)
    goal=(policy.arena.width/2,8. if policy.sign>0 else policy.arena.height-8.)
    return max(0.,85.-depth)*.9+max(0.,policy.route(own,goal)-policy.route(enemy,goal))*.65


def physical_signature(world):
    return (tuple(world.players.items()),world.ball_position,world.ball_velocity,
            world.ball_remaining_distance,world.possession,world.possession_steps,
            tuple(world.score.items()))


def contact_race(policy, enemy_mode='race', horizon=12, width=8, start=None, seconds=.12):
    """Search legal catch routes through moving-body contact, with honest pruning.

    This is a witness against one observable continuation, not a proof that an
    adversarial opponent cannot win. An unsuccessful pruned search is unknown.
    """
    import time
    deadline=min(policy.deadline-.015,time.perf_counter()+seconds)
    frontier=[(policy.world if start is None else start,[])];pruned=False;own=policy.player_id;enemy=policy.opponent_id
    for turn in range(1,horizon+1):
        expanded={}
        for world,moves in frontier:
            if time.perf_counter()>=deadline:return {'event':'unknown','uncertain':True,'moves':[],'turn':None,'assumption':enemy_mode,'reason':'budget'}
            target=(policy.arena.width/2,policy.arena.height-12 if policy.sign>0 else 12.) if enemy_mode=='recover' else None
            reply=chase(policy,world,enemy,target)
            for move,point in policy.arena.moves(world.players[own]):
                w=world.predict({own:{'move':move},enemy:{'move':reply}});sequence=moves+[move]
                result=event(w,policy.world.score,own)
                if result=='our_claim':
                    record=result_record(w,result,turn,sequence,enemy_mode);record['uncertain']=True # Conditional opponent continuation, even with exact contact physics.
                    return record
                if result in ('goal','conceded','opponent_claim') or w.done:continue
                expanded.setdefault(physical_signature(w),(w,sequence))
        if not expanded:return {'event':'unknown' if pruned else 'opponent_claim','uncertain':pruned,'moves':[],'turn':turn,'assumption':enemy_mode,'reason':'frontier exhausted'}
        ranked=sorted(expanded.values(),key=lambda pair:policy.route(pair[0].players[own],pair[0].ball_position)-.25*distance(pair[0].players[enemy],pair[0].ball_position))
        pruned |= len(ranked)>width;frontier=ranked[:width]
    return {'event':'unknown','uncertain':True,'moves':[],'turn':horizon,'assumption':enemy_mode,'reason':'horizon'}
