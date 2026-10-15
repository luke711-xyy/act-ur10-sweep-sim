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
        with self.lock:
            issued_at = float(issued_at)
            if issued_at not in self.accepted or not 0 <= index < len(self.accepted[issued_at]):
                raise RuntimeError('selected ACT action/fusion chunk mismatch')
            if now < issued_at:
                raise RuntimeError('future ACT feature leakage')
            self.selected = (issued_at,int(index))

    def clear_selection(self):
        with self.lock:
            self.selected = None

    def current(self, now):
        with self.lock:
            if self.selected is None:
                return dict(feature=np.zeros(512,np.float32),available=False,index=0,age=0.)
            issued,index = self.selected
            if now < issued:
                raise RuntimeError('feature clock reversed')
            return dict(feature=self.accepted[issued][index].copy(),available=True,
                        index=index,age=float(now)-issued)
