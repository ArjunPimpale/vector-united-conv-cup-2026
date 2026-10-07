"""Deadline-bounded public-physics attack and goal protection."""
import time

from .legacy import Policy as LegacyPolicy
from .model import World
from .geometry import distance
from . import attack as attacks, defense
from .memory import Memory
from .opponent import Observer
from .outcomes import race, guard, contact_race


class Policy(LegacyPolicy):
    def __init__(self, attack_enabled=True, defense_enabled=True, setup_enabled=False, physical_enabled=True, coverage_enabled=True, memory_enabled=True, adaptation_enabled=False):
        super().__init__()
        self.attack_enabled=attack_enabled; self.defense_enabled=defense_enabled
        self.setup_enabled=setup_enabled; self.deadline=0.
        self.diagnostic={}; self.route_cache={}; self.route_values={}; self.shot_cache={}
        self.loose_best=None
        self.physical_enabled=physical_enabled; self.coverage_enabled=coverage_enabled
        self.memory_enabled=memory_enabled; self.adaptation_enabled=adaptation_enabled
        self.memory=Memory(); self.observer=Observer(); self.previous_iteration=None; self.turns_without_control=0

    def expired(self):
        return time.perf_counter()>=self.deadline

    def route(self,origin,target):
        key=(origin,target)
        if key not in self.route_values:
            if origin not in self.route_cache:self.route_cache[origin]=self.arena.route_distances(origin)
            self.route_values[key]=self.arena.route_length(target,origin,self.route_cache[origin])
        return self.route_values[key]

    def choose_action(self,observation):
        start=time.perf_counter(); self.deadline=start+.45
        self.route_cache={}; self.route_values={}; self.shot_cache={}; self.setup_chosen=False
        if self.arena is not None:self.arena.endpoint_links.clear()
        self.diagnostic={}; state=observation['state']; ball=state['ball']
        if state['iteration']==0 or ball['possession'] or ball['remaining_kick_distance']>0:
            self.loose_best=None
        else:
            closest=min(distance((p['x'],p['y']),(ball['x'],ball['y'])) for p in state['players'].values())
            if self.loose_best is None or closest<self.loose_best-.25:self.loose_best=closest
        self.world=World.from_state(state,self.loose_best)
        iteration=state['iteration']
        if self.previous_iteration is not None and iteration<=self.previous_iteration:
            self.memory=Memory(); self.observer=Observer(); self.previous_other=None; self.turns_without_control=0
        self.previous_iteration=iteration
        if self.memory.last_score is not None and state['score']!=self.memory.last_score:
            self.history=[];self.previous_other=None;self.other_velocity=(0.,0.);self.loose_best=None;self.turns_without_control=0
        self.turns_without_control=0 if ball['possession']==observation['player_id'] else self.turns_without_control+1
        self.memory.observe(state,observation['player_id'])
        self.observer.observe(observation)
        action=super().choose_action(observation)
        if self.physical_enabled and not ball['possession'] and not self.expired():
            action=self.loose_action(action)
        if self.memory_enabled:self.memory.remember(state,self.player_id,action)
        phase='ours' if ball['possession']==self.player_id else 'theirs' if ball['possession'] else 'free'
        self.diagnostic['opponent']=self.observer.summary(phase)
        self.diagnostic['adaptation_active']=self.adaptation_enabled
        self.diagnostic['turns_without_control']=self.turns_without_control
        self.diagnostic['cycle_count']=max(self.memory.visits.values(),default=0)
        self.diagnostic['observed_intent_outcome']=self.memory.last_outcome

        self.diagnostic.update({'intention':self.mode,'budget_exhausted':self.expired(),
                                'decision_ms':round((time.perf_counter()-start)*1000,2)})
        return action

    def opponent_replies(self,me,other):
        moves=self.arena.moves(other)
        pursue=min(moves,key=lambda m:distance(m[1],me))[0]
        goal=(self.arena.width/2,self.arena.height if self.sign>0 else 0.)
        retreat=min(moves,key=lambda m:distance(m[1],goal))[0]
        continued=min(moves,key=lambda m:distance(m[1],(other[0]+self.other_velocity[0],other[1]+self.other_velocity[1])))[0]
        return list(dict.fromkeys([pursue,retreat,continued,'STAY']))[:3]

    def attack(self,me,other,ball,action_space):
        if not self.attack_enabled:return super().attack(me,other,ball,action_space)
        return attacks.choose_attack(self,me,other,ball,action_space)

    def defend(self,me,other,ball):
        if not self.defense_enabled:return super().defend(me,other,ball)
        return defense.defend(self,me,other,ball)

    def contain(self,me,threat,other):
        if not self.defense_enabled:return super().contain(me,threat,other)
        self.mode='RECOVER_GOAL'
        target=defense.coverage_target(self,threat)
        self.diagnostic['target']=list(target)
        return self.navigate(me,target)

    def legacy_intercept(self,me,path):
        return super().intercept(me,path)

    def intercept(self,me,path):
        if not self.defense_enabled:return self.legacy_intercept(me,path)
        return defense.intercept(self,me,path)

    def loose_action(self,baseline):
        # Immediate collection is tested with simultaneous contact against every
        # raw opponent move. A solo witness alone is not enough.
        me=self.world.players[self.player_id]
        candidates=[]
        for move,point in self.arena.moves(me):
            action={'move':move};cases=guard(self,action)
            safe=sum(e=='our_claim' for _,w,e in cases)
            goals=sum(e=='conceded' for _,w,e in cases)
            if safe and not goals:candidates.append((safe,-self.route(point,self.world.ball_position),action))
        if candidates:
            # Prefer the existing useful collection witness when it also wins
            # control in a tested reply. Small changes in first-catch position
            # can destroy the following bank lane despite equal catch timing.
            baseline_candidate=next((item for item in candidates if item[2]==baseline),None)
            self.mode='COLLECT_NOW'
            return baseline if baseline_candidate else max(candidates,key=lambda x:x[:2])[2]
        intent=self.memory.intent
        if intent and intent['moves'] and self.mode!='save':
            move=intent['moves'][0]
            cases=guard(self,{'move':move})
            if not any(e in ('conceded','opponent_claim') for _,w,e in cases):
                intent['moves'].pop(0);self.mode='RECOVER_PASS'
                self.diagnostic['target']=list(intent['target']);return {'move':move}
        if self.world.ball_remaining_distance>0 and self.mode in ('intercept','collect','RECOVER_GOAL'):
            witness=contact_race(self)
            self.diagnostic['contact_race']={k:v for k,v in witness.items() if k not in ('world','moves')}
            if witness['event']=='our_claim' and witness['moves']:
                action={'move':witness['moves'][0]}
                if not any(e=='conceded' for _,w,e in guard(self,action)):
                    self.mode='CONTEST_BALL';return action
        # Compare the baseline witness against a physical race including contact.
        if self.mode in ('intercept','collect'):
            w=self.world.predict({self.player_id:baseline,self.opponent_id:{'move':self.opponent_replies(me,self.world.players[self.opponent_id])[0]}})
            result=race(self,w,'race')
            self.diagnostic['ball_race']={k:v for k,v in result.items() if k not in ('world','moves')}
            if result['event']=='opponent_claim' and self.progress(result['point'])<35. and self.progress(me)>self.progress(result['point'])+8.:
                self.mode='RECOVER_GOAL'
                return {'move':self.contain(me,result['point'],self.world.players[self.opponent_id])}
        return baseline
