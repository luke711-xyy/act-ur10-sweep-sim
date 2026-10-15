"""Wall-clock and resource protection, independent of policy reward."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .config import DEFAULT

STOP_RESERVE_SECONDS=60.
SPACE_RESERVE_BYTES=64*2**20
COMPACT_MAX_BYTES=32*2**20
EVIDENCE_RESERVE_BYTES=2**20


class BudgetStop(RuntimeError):
    pass


class StopRequest:
    def __init__(self): self.reason=None


@contextmanager
def worker_stop_signals():
    """Signals request cleanup at the next guard, never interrupt an atomic save."""
    request=StopRequest()
    def receive(number,_frame): request.reason='signal_'+signal.Signals(number).name
    previous={sig:signal.signal(sig,receive) for sig in (signal.SIGTERM,signal.SIGINT)}
    try: yield request
    finally:
        for sig,handler in previous.items(): signal.signal(sig,handler)


class BudgetGuard:
    def __init__(self,*,started_at,config=DEFAULT):
        self.started_at=float(started_at); self.config=config
        self.memory_since=None

    def check(self,*,now,residual_steps,disk_free,output_bytes,rss,dangerous_memory=False,
              reserve=True):
        cfg=self.config
        if disk_free<cfg.min_disk_bytes: return 'stop_disk'
        if output_bytes>=cfg.max_output_bytes: return 'stop_output'
        if rss>cfg.max_rss_bytes or dangerous_memory:
            if self.memory_since is None: self.memory_since=float(now)
            if now-self.memory_since>=cfg.pressure_dwell_seconds: return 'stop_memory'
        else: self.memory_since=None
        elapsed=float(now)-self.started_at
        if elapsed>=cfg.max_wall_seconds: return 'stop_total'
        if reserve:
            if (self.memory_since is not None
                    and now-self.memory_since>=max(0.,cfg.pressure_dwell_seconds-10.)):
                return 'stop_reserve_memory'
            if disk_free<=cfg.min_disk_bytes+SPACE_RESERVE_BYTES: return 'stop_reserve_disk'
            if output_bytes>=cfg.max_output_bytes-SPACE_RESERVE_BYTES: return 'stop_reserve_output'
            if elapsed>=cfg.max_wall_seconds-STOP_RESERVE_SECONDS: return 'stop_reserve_total'
        if elapsed>=cfg.stop_collection_seconds or residual_steps>=cfg.max_residual_steps:
            return 'stop_collection'
        return 'continue'


def resources(root,pid=None):
    root=Path(root)
    owned={pid or os.getpid()}
    listing=subprocess.check_output(['ps','-axo','pid=,ppid=,rss='],text=True,timeout=3)
    processes=[tuple(map(int,line.split())) for line in listing.splitlines()]
    while True:
        children={p for p,parent,_ in processes if parent in owned}
        if children<=owned: break
        owned.update(children)
    rss=sum(kib*1024 for p,_,kib in processes if p in owned)
    output_bytes=0
    for folder,_,files in os.walk(root,followlinks=False):
        for name in files:
            path=Path(folder)/name
            if not path.is_symlink():
                try: output_bytes+=path.stat().st_size
                except FileNotFoundError: pass
    if sys.platform=='darwin':
        pressure=subprocess.run(['sysctl','-n','kern.memorystatus_vm_pressure_level'],
            capture_output=True,text=True,timeout=3)
        if pressure.returncode: raise RuntimeError('cannot monitor macOS memory pressure')
        dangerous=int(pressure.stdout.strip())>=4
    else:
        fields={line.split(':')[0]:int(line.split()[1])*1024
                for line in Path('/proc/meminfo').read_text().splitlines()}
        dangerous=fields['MemAvailable']<.05*fields['MemTotal']
    return dict(disk_free=shutil.disk_usage(root).free,output_bytes=output_bytes,rss=rss,
                dangerous_memory=dangerous)


class LiveGuard:
    def __init__(self,root,started_at,config=DEFAULT,stop_request=None):
        self.root=Path(root); self.budget=BudgetGuard(started_at=started_at,config=config)
        self.residual_steps=0; self.last_sample=-float('inf'); self.sample=None
        self.phase='preflight'; self.status='continue'
        self.training_decision_stop=config.max_residual_steps
        self.stop_request=stop_request or StopRequest()

    def poll(self):
        if self.stop_request.reason: raise BudgetStop(self.stop_request.reason)
        request=self.root/'stop_request.json'
        if request.exists():
            self.stop_request.reason=json.loads(request.read_text())['reason']
            raise BudgetStop(self.stop_request.reason)
        now=time.monotonic()
        if now-self.last_sample>=1. or self.sample is None:
            self.sample=resources(self.root); self.last_sample=now
        self.status=self.budget.check(now=now,residual_steps=self.residual_steps,**self.sample)
        if (self.status=='continue' and self.phase=='train'
                and self.residual_steps>=self.training_decision_stop):
            self.status='stop_collection'
        if self.status.startswith('stop_') and (self.status!='stop_collection' or self.phase=='train'):
            raise BudgetStop(self.status)
        return self.status

    def checkpoint_allowance(self,*,compact):
        """Fresh hard-cap admission; work-stop requests do not authorize extra budget."""
        now=time.monotonic(); sample=resources(self.root); cfg=self.budget.config
        state=self.budget.check(now=now,residual_steps=self.residual_steps,**sample,reserve=False)
        if state not in ('continue','stop_collection'): raise BudgetStop(state)
        if sample['rss']>cfg.max_rss_bytes or sample['dangerous_memory']:
            raise BudgetStop('stop_memory_checkpoint_unsafe')
        remaining=cfg.max_wall_seconds-(now-self.budget.started_at)
        if remaining<=5.: raise BudgetStop('stop_total_checkpoint_reserve')
        margin=EVIDENCE_RESERVE_BYTES if compact else SPACE_RESERVE_BYTES
        allowed=min(sample['disk_free']-cfg.min_disk_bytes-margin,
                    cfg.max_output_bytes-sample['output_bytes']-margin)
        if compact: allowed=min(allowed,COMPACT_MAX_BYTES)
        if allowed<=0: raise BudgetStop('stop_space_checkpoint_reserve')
        # Leave five seconds for evidence/cleanup before the supervisor's hard stop.
        deadline=min(now+STOP_RESERVE_SECONDS-5.,self.budget.started_at+cfg.max_wall_seconds-5.)
        request=self.root/'stop_request.json'
        if request.exists(): deadline=min(deadline,float(json.loads(request.read_text())['cleanup_deadline'])-5.)
        if now>=deadline: raise BudgetStop('stop_cleanup_checkpoint_deadline')
        return int(allowed),deadline
