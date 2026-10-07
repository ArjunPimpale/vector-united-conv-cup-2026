"""Phase-specific decaying tendencies inferred solely from public state pairs."""
import math
from .geometry import DIRECTIONS, distance, hits

CATEGORIES=('race','recover','hold')


class Observer:
    def __init__(self):
        self.counts={phase:{k:2. for k in CATEGORIES} for phase in ('ours','theirs','free')}
        self.samples={p:0 for p in self.counts};self.errors={p:.5 for p in self.counts}
        self.shooting={'early':2.,'late':2.};self.previous=None;self.masked=0
        self.style='balanced'

    def observe(self, observation):
        state=observation['state'];old=self.previous;self.previous=state
        if old is None or state['iteration']!=old['iteration']+1 or state['score']!=old['score']:
            return
        own=observation['player_id'];enemy=observation['opponent_id']
        pt=lambda s,p:(s['players'][p]['x'],s['players'][p]['y'])
        before=pt(old,enemy);after=pt(state,enemy)
        phase='ours' if old['ball']['possession']==own else 'theirs' if old['ball']['possession']==enemy else 'free'
        speed=state['field']['player_speed'];radius=state['field']['player_radius']
        # Contact and blocked movements make raw movement intent unidentifiable.
        if min(distance(pt(old,own),before),distance(pt(state,own),after))<2*radius+2*speed:
            self.masked+=1;return
        delta=(after[0]-before[0],after[1]-before[1])
        aligned=any(distance(delta,(dx*speed,dy*speed))<1e-6 for dx,dy in DIRECTIONS.values())
        if not aligned or (delta==(0.,0.) and before!=after):self.masked+=1;return
        # A stationary observation near an obstacle/boundary can be a blocked
        # move. Mask it rather than crediting deliberate holding.
        if delta==(0.,0.):
            field=state['field'];rects=[(r['x'],r['y'],r['width'],r['height']) for r in state['obstacles']]
            points=[(before[0]+dx*speed,before[1]+dy*speed) for dx,dy in DIRECTIONS.values()]
            if any(not (radius<=x<=field['width']-radius and radius<=y<=field['height']-radius)
                   or any(hits((x,y),radius,r) for r in rects) for x,y in points):
                self.masked+=1;return
        ball=old['ball'];target=pt(old,own) if phase=='ours' else (ball['x'],ball['y'])
        defend_top=enemy=='player_2';goal=(state['field']['width']/2,state['field']['height'] if defend_top else 0.)
        closing=distance(before,target)-distance(after,target)
        retreat=distance(before,goal)-distance(after,goal)
        category='hold' if delta==(0.,0.) else 'recover' if retreat>closing+1. else 'race' if closing>.5 else 'hold'
        counts=self.counts[phase];total=sum(counts.values());prob=counts[category]/total
        self.errors[phase]=.95*self.errors[phase]+.05*(1.-prob)
        for k in counts:counts[k]=2.+(counts[k]-2.)*.977
        counts[category]+=1.;self.samples[phase]+=1
        if phase=='theirs' and state['ball']['status']=='moving' and ball['possession_steps']<9:
            self.shooting['early' if ball['possession_steps']<3 else 'late']+=1.
        probabilities=self.weights(phase)
        leader=max(probabilities,key=probabilities.get)
        if self.confidence(phase)>.35 and probabilities[leader]>.5:self.style=leader
        elif probabilities.get(self.style,0.)<.38:self.style='balanced'

    def confidence(self,phase):
        return min(1.,self.samples[phase]/12.)*max(0.,1.-self.errors[phase])

    def weights(self,phase):
        counts=self.counts[phase];total=sum(counts.values());confidence=self.confidence(phase)
        return {k:(1.-confidence)/3.+confidence*counts[k]/total for k in CATEGORIES}

    def summary(self,phase):
        return {'phase':phase,'samples':self.samples[phase],'confidence':round(self.confidence(phase),3),
                'prediction_error':round(self.errors[phase],3),'weights':self.weights(phase),
                'style':self.style,'masked':self.masked}
