"""All-power proposals, complete root guard, and first physical kick events."""
import math
from . import baseline_attack as baseline
from .baseline_attack import positional, trace_for, margin, value
from .geometry import distance
from .outcomes import guard, race, loss_danger, physical_signature, contact_race


def rank_event(policy, record):
    w=record['world'];event=record['event'];turn=record['turn']
    if event=='goal':return 430.-turn
    if event=='conceded':return -850.
    if event=='opponent_claim':return -200.-loss_danger(policy,w)-turn
    if event=='our_claim':
        return positional(policy,w.players[policy.player_id],w.players[policy.opponent_id],w.possession_steps)-5.-turn*1.5
    # No claim or goal within the horizon is uncertain, not a successful pass.
    return policy.progress(w.ball_position)-65.-turn*2.-loss_danger(policy,w)*.25


def choose_attack(policy, me, other, ball, action_space):
    if not policy.physical_enabled:return baseline.choose_attack(policy,me,other,ball,action_space)
    saved_deadline=policy.deadline
    import time
    policy.deadline=min(saved_deadline,time.perf_counter()+.12)
    baseline_action=baseline.choose_attack(policy,me,other,ball,action_space)
    policy.deadline=saved_deadline
    policy.diagnostic={}
    age=ball['possession_steps'];arena=policy.arena;candidates=[]
    for move,point in sorted(arena.moves(me),key=lambda m:positional(policy,m[1],other,age),reverse=True):
        candidates.append((positional(policy,point,other,age),{'move':move},None))
        for direction in action_space['kick']['direction']:
            for power in action_space['kick']['power']:
                trace=trace_for(policy,point,direction,power)
                estimate=value(policy,point,other,trace)-.08*power
                candidates.append((estimate,{'move':move,'kick':{'direction':direction,'power':power}},trace))
    candidates.sort(key=lambda c:c[0],reverse=True)
    shortlist=[c for c in candidates if not c[2]]
    shortlist += [c for c in candidates if c[2] and c[2].goal==policy.sign][:12]
    for power in (1,2,3):
        shortlist += [c for c in candidates if c[2] and c[2].goal!=policy.sign and c[1]['kick']['power']==power][:4]
    # Keep release-direction diversity even when endpoint estimates are misleading.
    for direction in action_space['kick']['direction']:
        shortlist += [c for c in candidates if c[2] and c[1]['kick']['direction']==direction][:1]
    shortlist += [c for c in candidates if c[1]==baseline_action]
    unique={str(c[1]):c for c in shortlist};shortlist=list(unique.values())
    guarded=[]
    policy.root_fallback=None
    for estimate,action,trace in shortlist:
        # Finish this bounded root batch even if optional planning has expired.
        # That guarantees the fallback compares dribbles with clearances rather
        # than freezing on the first unchecked or immediately losing proposal.
        cases=guard(policy,action)
        loss=any(e in ('opponent_claim','conceded') for _,w,e in cases)
        immediate_goal=all(e=='goal' for _,w,e in cases)
        physical_class=2 if immediate_goal else 0 if loss else 1
        preliminary=[]
        for reply,w,event in cases:
            own=w.players[policy.player_id];enemy=w.players[policy.opponent_id]
            if event=='goal':score=430.
            elif event=='conceded':score=-850.
            elif event=='opponent_claim':score=-300.-loss_danger(policy,w)
            elif event=='our_claim':score=positional(policy,own,enemy,w.possession_steps)-(5. if 'kick' in action else 0.)
            else:
                actual=trace_for(policy,own,action['kick']['direction'],action['kick']['power']) if 'kick' in action else trace_for(policy,own,'UP' if policy.sign>0 else 'DOWN',1)
                score=value(policy,own,enemy,actual) if actual.goal==policy.sign else policy.progress(w.ball_position)-160.-loss_danger(policy,w)*.25
            preliminary.append(score)
        score=.55*min(preliminary)+.45*sum(preliminary)/len(preliminary)
        if policy.root_fallback is None:policy.root_fallback=action
        guarded.append({'action':action,'trace':trace,'cases':cases,'class':physical_class,'score':score,'estimate':estimate,'event':None})
    # Resolve the most promising kick roots first; physical priority is retained.
    for item in sorted(guarded,key=lambda r:(r['action']==baseline_action,r['class'],r['estimate']),reverse=True):
        if policy.expired():break
        action=item['action'];cases=item['cases']
        if 'kick' not in action and all(e=='our_claim' for _,w,e in cases):continue
        values=[];records=[];seen={}
        replies=set(policy.opponent_replies(me,other))
        replies.add(min(cases,key=lambda c:distance(c[1].players[policy.player_id],c[1].players[policy.opponent_id]))[0])
        deep_cases=[c for c in cases if c[0] in replies]
        for reply,w,event in deep_cases:
            sig=physical_signature(w)
            if sig in seen:values.append(seen[sig]);continue
            own=w.players[policy.player_id];enemy=w.players[policy.opponent_id]
            if event:
                record={'world':w,'event':event,'turn':1,'moves':[],'point':w.ball_position,'age':w.possession_steps,'assumption':'all raw first moves','uncertain':False}
                score=rank_event(policy,record)
            else:
                kick=action.get('kick',{'direction':'UP' if policy.sign>0 else 'DOWN','power':1})
                actual=trace_for(policy,own,kick['direction'],kick['power'])
                if actual.goal==policy.sign:
                    score=value(policy,own,enemy,actual,verify=True)-.08*kick['power']
                    record={'world':w,'event':'projected_goal','turn':actual.path[-1][0],'moves':[], 'point':actual.path[-1][1:],'age':0,'assumption':'solo interception witness; contact replanned','uncertain':True}
                else:
                    # Multiple future responses share the exact first-turn guard.
                    modes=['race','recover']
                    weights=policy.observer.weights('free') if policy.adaptation_enabled else {'race':.5,'recover':.5}
                    scenarios=[]
                    for mode in modes:
                        result=race(policy,w,mode)
                        if result['event']=='opponent_claim' and not policy.expired():
                            witness=contact_race(policy,mode,start=w,width=6,seconds=.035)
                            if witness['event']=='our_claim':
                                witness['turn']+=1;result=witness
                        scenarios.append(result)
                    scores=[rank_event(policy,r) for r in scenarios]
                    total=sum(weights[m] for m in modes)
                    score=.55*min(scores)+.45*sum(weights[m]*s for m,s in zip(modes,scores))/total
                    record=scenarios[scores.index(min(scores))]
                if not math.isfinite(score):break
            seen[sig]=score;values.append(score);records.append(record)
            if policy.expired():break
        # Partial deeper searches never replace a fully validated fallback.
        if len(values)==len(deep_cases):
            item['score']=.55*min(values)+.45*sum(values)/len(values)
            immediate=[rank_event(policy,{'world':w,'event':e,'turn':1}) for _,w,e in cases if e]
            if immediate:item['score']=min(item['score'],min(immediate))
            item['deep_replies']=sorted(replies)
            if records:item['event']=min(records,key=lambda r:rank_event(policy,r))
    for item in guarded:
        if policy.memory_enabled and item['class']<2 and item['score']<300.:
            item['score']-=policy.memory.penalty(policy.world.state(),policy.player_id,item['action'])
    best=max(guarded,key=lambda r:(r['class'],r['score']))
    baseline_item=next((item for item in guarded if item['action']==baseline_action),None)
    reason='physical outcome and recurrence ranking'
    if baseline_item and baseline_item['class']==1 and best['class']<2:
        outcome=baseline_item['event']
        no_progress=(outcome and outcome['event']=='our_claim' and policy.progress(outcome['point'])<=policy.progress(me)+1.
                     and age<2 and distance(me,other)>16.)
        recurrence=policy.memory.penalty(policy.world.state(),policy.player_id,baseline_action) if policy.memory_enabled else 0.
        if not no_progress and recurrence<7. and best['score']<300.:
            best=baseline_item;reason='guarded v3 proposal retained under uncertain future scenarios'
        elif no_progress:reason='baseline pass returns control without progress or pressure relief'
    action=best['action'];trace=best['trace'];record=best['event']
    policy.mode='ATTACK'
    if 'kick' in action:
        policy.mode='BANK_SHOT' if trace.goal==policy.sign and trace.bounces else 'DIRECT_SHOT' if trace.goal==policy.sign else 'SELF_PASS'
        if best['class']==0:policy.mode='CLEARANCE'
    policy.diagnostic.update({'evaluated_actions':len(candidates),'guarded_actions':len(guarded),
        'opponent_inputs_per_guard':9,'deeper_opponent_inputs':best.get('deep_replies',[]),'selected_value':round(best['score'],2) if math.isfinite(best['score']) else None,
        'immediate_giveaway':best['class']==0,'kick_purpose':policy.mode,
        'fallback_reason':'deadline: complete physical root' if policy.expired() else None,
        'baseline_override_reason':reason})
    if trace:
        policy.diagnostic['selected_path']=[[x,y] for t,x,y in trace.path[::6]]+[list(trace.path[-1][1:])]
        policy.diagnostic['bounces']=trace.bounces
    if record is None and 'kick' in action:
        w=best['cases'][0][1]
        record={'event':'unresolved','turn':1,'point':w.ball_position,'age':w.possession_steps,'assumption':'complete first-turn guard; future search unfinished','uncertain':True,'moves':[]}
    if record:
        policy.diagnostic['first_outcome']={k:v for k,v in record.items() if k not in ('world','moves')}
        if record['event']=='our_claim' and record['moves']:
            # Store predicted ball positions to reject an intent as soon as its
            # assumed physical continuation disagrees with public observations.
            w=best['cases'][0][1];expected=[w.ball_position]
            for move in record['moves']:
                w=w.predict({policy.player_id:{'move':move},policy.opponent_id:{'move':'STAY'}})
                expected.append(w.ball_position)
                if w.possession or w.done:break
            policy.memory.intent={'expected':expected,'moves':list(record['moves']), 'target':record['point'],'purpose':policy.mode}
    return action
