"""Causal association of frozen fusion tokens with accepted ACT chunks."""
import threading

import numpy as np


class FeatureMailbox:
    def __init__(self):
        self.lock = threading.RLock()
        self.pending = None
        self.accepted = {}
        self.selected = None

    def publish(self, tokens):
        value = np.asarray(tokens, dtype=np.float32)
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.shape != (25,512) or not np.isfinite(value).all():
            raise ValueError('frozen action-head input must be finite (25,512)')
        with self.lock:
            if self.pending is not None:
                raise RuntimeError('unconsumed ACT fusion result')
            self.pending = value.copy()

    def accept(self, *, issued_at, accepted, length):
        with self.lock:
            if accepted:
                if self.pending is None or len(self.pending) != length:
                    raise RuntimeError('accepted chunk has no matching frozen fusion tokens')
                self.accepted[float(issued_at)] = self.pending
                for key in sorted(self.accepted)[:-8]:
                    del self.accepted[key]
            self.pending = None

    def select(self, issued_at, index, now):
        self.select_many([(issued_at,index,1.)],now)

    def select_many(self, components, now):
        """Bind exactly the accepted native contributors, never pending tokens."""
        with self.lock:
            selected=[]
            for issued_at,index,weight in components:
                issued_at,weight=float(issued_at),float(weight)
                if (issued_at not in self.accepted or int(index)!=index
                        or not 0<=index<len(self.accepted[issued_at])):
                    raise RuntimeError('selected ACT action/fusion chunk mismatch')
                if now<issued_at: raise RuntimeError('future ACT feature leakage')
                if not np.isfinite(weight) or weight<=0.:
                    raise RuntimeError('invalid native ACT feature weight')
                selected.append((issued_at,int(index),weight))
            total=sum(v[2] for v in selected)
            if not selected or not np.isfinite(total):
                raise RuntimeError('empty or nonfinite native ACT feature mixture')
            self.selected=tuple((issued,index,weight/total) for issued,index,weight in selected)

    def reset(self):
        with self.lock:
            self.pending=None; self.accepted.clear(); self.selected=None

    def clear_selection(self):
        with self.lock:
            self.selected = None

    def current(self, now):
        with self.lock:
            if self.selected is None:
                return dict(feature=np.zeros(512,np.float32),available=False,index=0,age=0.,components=[])
            if any(now<issued for issued,_,_ in self.selected):
                raise RuntimeError('feature clock reversed')
            feature=sum((self.accepted[issued][index].astype(np.float64)*weight
                         for issued,index,weight in self.selected),start=np.zeros(512,np.float64))
            return dict(feature=feature.astype(np.float32),available=True,
                index=sum(index*weight for _,index,weight in self.selected),
                age=sum((float(now)-issued)*weight for issued,_,weight in self.selected),
                components=[dict(issued_at=issued,index=index,weight=weight)
                            for issued,index,weight in self.selected])


def observed_scheduler(base_scheduler, mailbox, *, on_created=None):
    """Observe native results without replacing actions, chunk state or weights.

    Ensemble features use the same contributing horizon tokens and exponential
    weights. Feature age/index are weighted summaries; their full provenance is
    retained separately. Critic context exposes the newest CONTRIBUTING block,
    never a pending or not-yet-valid block. No extra visual query is performed.
    """
    class ObservedScheduler(base_scheduler):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.observation_chunk=None; self.feature_components=[]
            if on_created is not None: on_created(self)

        def reset(self,now=0.):
            super().reset(now)
            mailbox.reset(); self.observation_chunk=None; self.feature_components=[]

        def accept(self,issued_at,actions,now):
            accepted=super().accept(issued_at,actions,now)
            mailbox.accept(issued_at=issued_at,accepted=accepted,length=len(actions))
            return accepted

        def action_for(self,now):
            result=super().action_for(now)  # Native action, including circular yaw, untouched.
            self.observation_chunk=None; self.feature_components=[]
            if result is None:
                mailbox.clear_selection()
                return result
            ensemble=self.temporal_ensemble
            chunks=ensemble.chunks if ensemble is not None else [self.chunk]
            components=[]; contributing=[]
            for chunk in chunks:
                if chunk is None: continue
                age=float(now)-chunk.issued_at
                if age<0. or age>=len(chunk.actions)*self.action_period or float(now)<chunk.valid_from:
                    continue
                index=int(np.floor(age/self.action_period+1e-6))
                if not 0<=index<len(chunk.actions): continue
                weight=float(np.exp(-ensemble.coeff*index)) if ensemble is not None else 1.
                components.append((chunk.issued_at,index,weight)); contributing.append(chunk)
            mailbox.select_many(components,now)
            self.observation_chunk=max(contributing,key=lambda c:c.issued_at)
            self.feature_components=mailbox.current(now)['components']
            return result
    return ObservedScheduler
