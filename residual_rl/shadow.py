"""Same-stream exact zero comparison. No policy, controller, or physics IO."""
import hashlib

import numpy as np

from .control import ResidualGovernor,ResidualReference


class ZeroResidualShadow:
    def __init__(self,bounds):
        self.composer=ResidualReference(ResidualGovernor(bounds))
        self.checked=0; self.active_checked=0; self.mismatches=0; self.max_error=0.
        self.native_digest=hashlib.sha256(); self.zero_digest=hashlib.sha256()
        self.last_audit=None

    def begin_frame(self,start):
        self.composer.begin_frame(start)

    def check(self,actual,raw,*,dt,active):
        actual=np.asarray(actual)
        candidate=self.composer.step(raw,np.zeros(3),dt=dt,active=active)
        exact=(actual.dtype==candidate.dtype and actual.shape==candidate.shape
               and actual.tobytes()==candidate.tobytes())
        error=float(np.max(np.abs(actual.astype(float)-candidate.astype(float))))
        self.checked+=1; self.active_checked+=int(active); self.mismatches+=int(not exact)
        self.max_error=max(self.max_error,error)
        self.native_digest.update(actual.tobytes()); self.zero_digest.update(candidate.tobytes())
        self.last_audit=dict(exact=exact,active=bool(active),max_abs_error=error,
                            native=actual.copy(),zero=candidate.copy())
        if not exact:
            raise RuntimeError('same-stream zero residual mismatch: '+repr(self.last_audit))
        return self.last_audit

    def summary(self):
        return dict(single_actual_ACT_stream=True,checked_steps=self.checked,
            active_checked_steps=self.active_checked,mismatches=self.mismatches,max_abs_error=self.max_error,
            native_sha256=self.native_digest.hexdigest(),zero_sha256=self.zero_digest.hexdigest(),
            extra_policy_queries=0,extra_physics_steps=0)
