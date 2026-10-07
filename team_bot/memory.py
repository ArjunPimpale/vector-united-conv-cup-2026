"""Bounded match recurrence memory; quantization never enters the physics."""
from collections import OrderedDict
from .geometry import distance


def action_key(action):
    kick=action.get('kick',{})
    return (action['move'],kick.get('direction'),kick.get('power'))


class Memory:
    def __init__(self):
        self.visits=OrderedDict();self.failures=OrderedDict();self.opening=[]
        self.previous=None;self.last_score=None;self.intent=None;self.last_outcome=None

    def context(self,state,own):
        enemy='player_2' if own=='player_1' else 'player_1'
        quant=lambda p:(round(p['x']/4),round(p['y']/4))
        ball=state['ball']
        return (quant(state['players'][own]),quant(state['players'][enemy]),
                quant(ball),ball['possession'],ball['possession_steps'],ball['status'],
                round(ball['velocity']['x']),round(ball['velocity']['y']))

    @staticmethod
    def bound(mapping,limit=256):
        while len(mapping)>limit:mapping.popitem(last=False)

    def observe(self,state,own):
        enemy='player_2' if own=='player_1' else 'player_1'
        if self.last_score is not None and state['score']!=self.last_score:
            if state['score'][enemy]>self.last_score[enemy]:
                for key in self.opening:
                    self.failures[key]=min(4,self.failures.get(key,0)+1)
                self.bound(self.failures)
            self.opening=[];self.intent=None;self.visits.clear()
        self.last_score=dict(state['score'])
        if self.intent:
            if state['ball']['possession'] or not self.intent['expected']:
                self.last_outcome=state['ball']['possession'] or 'flight';self.intent=None
            else:
                point=self.intent['expected'].pop(0)
                if distance((state['ball']['x'],state['ball']['y']),point)>1e-5:self.intent=None

    def penalty(self,state,own,action):
        key=(self.context(state,own),action_key(action))
        return min(50.,self.visits.get(key,0)*7.+self.failures.get(key,0)*12.)

    def remember(self,state,own,action):
        if state['ball']['possession']!=own:return
        key=(self.context(state,own),action_key(action))
        self.visits[key]=self.visits.get(key,0)+1;self.visits.move_to_end(key);self.bound(self.visits)
        if len(self.opening)<16:self.opening.append(key)
