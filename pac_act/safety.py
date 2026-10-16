import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


class StopExperiment(RuntimeError):
    """Administrative stop, never an ordinary task failure."""


class Guard:
    def __init__(self,root,*,started_at,deadline,collect_deadline,free_gib,output_gib,
                 rss_gib=10.,rss_dwell=30.,clock=time.time):
        self.root=Path(root); self.clock=clock
        self.started_at,self.deadline,self.collect_deadline=started_at,deadline,collect_deadline
        self.free_gib,self.output_gib=free_gib,output_gib
        self.rss_gib,self.rss_dwell=rss_gib,rss_dwell
        self.high_since=None; self.pressure_since=None; self.last_poll=-float('inf')

    def identity(self):
        return {k:getattr(self,k) for k in ('started_at','deadline','collect_deadline','free_gib','output_gib','rss_gib','rss_dwell')}

    def poll(self,*,collecting=False,free_bytes=None,rss_bytes=None,output_bytes=None,pressure=None):
        now=self.clock()
        if now>=self.deadline: raise StopExperiment('total deadline reached')
        if collecting and now>=self.collect_deadline: raise StopExperiment('collection deadline reached')
        injected=any(v is not None for v in (free_bytes,rss_bytes,output_bytes,pressure))
        if not injected and now-self.last_poll<1.: return
        self.last_poll=now
        if free_bytes is None: free_bytes=shutil.disk_usage(self.root).free
        if rss_bytes is None:
            import psutil
            process=psutil.Process()
            children=process.children(recursive=True)
            rss_bytes=sum(p.memory_info().rss for p in [process,*children] if p.is_running())
        if output_bytes is None:
            output_bytes=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file() and not p.is_symlink())
        if pressure is None:
            if sys.platform=='darwin':
                result=subprocess.run(['sysctl','-n','kern.memorystatus_vm_pressure_level'],capture_output=True,text=True,timeout=2)
                pressure=result.returncode==0 and int(result.stdout.strip())>=4
            else:
                import psutil
                pressure=psutil.virtual_memory().available<512*2**20
        if free_bytes<self.free_gib*2**30: raise StopExperiment('disk reserve reached')
        if output_bytes>=self.output_gib*2**30: raise StopExperiment('experiment output limit reached')
        self.high_since=(now if self.high_since is None else self.high_since) if rss_bytes>self.rss_gib*2**30 else None
        self.pressure_since=(now if self.pressure_since is None else self.pressure_since) if pressure else None
        if any(since is not None and now-since>=self.rss_dwell for since in (self.high_since,self.pressure_since)):
            raise StopExperiment('sustained dangerous memory pressure')


def local_inference_available():
    """Detect active original training across worktrees, not just a new local lock."""
    import psutil
    for process in psutil.process_iter(['pid','cmdline']):
        if process.pid==os.getpid(): continue
        try: command=process.info['cmdline'] or []
        except (psutil.AccessDenied,psutil.NoSuchProcess): continue
        if any(Path(arg).name in ('train_residual_long.py','train_residual_sac.py') for arg in command):
            return False,dict(pid=process.pid,command=command)
    return True,None
