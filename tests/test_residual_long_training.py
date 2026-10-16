"""Long-run lifecycle checks. Never launch ACT, simulation or real training."""
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


def long_api():
    try:return importlib.import_module('residual_rl.long_training')
    except ModuleNotFoundError:return None


def test_reward_change_rejects_old_return_replay_and_checkpoint():
    from residual_rl.learning import EpisodicReplay,RecurrentSAC
    replay=EpisodicReplay();old=replay.state_dict()
    old['reward_contract']='contact_plus_terminal_v1'
    with pytest.raises(ValueError):replay.load_state_dict(old)
    learner=RecurrentSAC(4,6);old=copy.deepcopy(learner.checkpoint())
    old['config']['reward_contract']='contact_plus_terminal_v1'
    with pytest.raises(ValueError):learner.load_checkpoint(old)


def test_long_profile_is_explicit_fresh_process_only_and_budgets_apply():
    from residual_rl import config
    activate=getattr(config,'activate_long_training_profile',None)
    assert callable(activate),'long profile activation not implemented'
    # Reject importing it into a process whose runtime already bound pilot constants.
    import residual_rl.runtime
    with pytest.raises(RuntimeError):activate()
    code='''
from residual_rl.config import activate_long_training_profile
activate_long_training_profile()
from residual_rl.config import DEFAULT
from residual_rl.safety import BudgetGuard
g=BudgetGuard(started_at=0.)
sample=dict(residual_steps=999999,disk_free=30*2**30,output_bytes=0,rss=0)
assert g.check(now=21601.,**sample)=='continue'
assert g.check(now=259201.,**sample)=='stop_total'
assert g.check(now=1.,**dict(sample,residual_steps=1000000))=='stop_collection'
assert g.check(now=1.,**dict(sample,output_bytes=20*2**30))=='stop_output'
assert g.check(now=1.,**dict(sample,disk_free=7*2**30))=='stop_disk'
print(DEFAULT.to_dict())
'''
    p=subprocess.run([sys.executable,'-B','-c',code],cwd=Path(__file__).parents[1],capture_output=True,text=True,timeout=15)
    assert p.returncode==0,p.stderr


def test_long_schedule_is_lazy_balanced_and_training_only():
    api=long_api();assert api is not None,'long lifecycle missing'
    schedule=api.training_schedule()
    assert schedule['preflight']==schedule['validation']==[]
    assert len(json.dumps(schedule))<2000
    specs=api.iter_training_specs(schedule)
    first=[next(specs) for _ in range(12)]
    assert [r['target_count'] for r in first]==[1,2,3,4,5,6]*2
    assert all(r['phase']=='train' and 5000000<=r['seed']<6000000 for r in first)
    assert len({r['seed'] for r in first})==12


def test_long_checkpoint_milestones_are_50k_not_10k_and_use_real_count():
    api=long_api();assert api is not None,'long lifecycle missing'
    assert api.crossed_milestones(49999,50017)==[50000]
    assert api.crossed_milestones(0,49999)==[]
    assert api.crossed_milestones(950001,1000000)==[1000000]


def test_last_25hz_action_finishes_but_next_action_is_refused_without_off_by_one(tmp_path):
    api=long_api();assert callable(getattr(api,'new_guard',None)),'frame-boundary guard missing'
    code='''
from types import SimpleNamespace
from residual_rl.config import activate_long_training_profile
activate_long_training_profile()
from residual_rl.long_training import new_guard
from residual_rl.runtime import Collector
from residual_rl.safety import BudgetStop
import residual_rl.safety as safety
safety.resources=lambda root:dict(disk_free=30*2**30,output_bytes=0,rss=0,dangerous_memory=False)
g=new_guard('.',safety.time.monotonic());g.phase='train';g.residual_steps=1000000
for _ in range(4):g.poll()
c=object.__new__(Collector);c.guard=g;c.mode='train';c.env=SimpleNamespace(time=3.)
c.governor=SimpleNamespace(started_at=0.)
try:c.begin()
except BudgetStop as error:assert str(error)=='stop_step_budget'
else:raise AssertionError('extra RL action accepted')
assert g.residual_steps==1000000
'''
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=Path(__file__).parents[1],capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr


def test_long_entry_inspect_is_read_only_and_execute_requires_ack(tmp_path):
    entry=Path(__file__).parents[1]/'scripts/train_residual_long.py'
    assert entry.is_file(),'long training entry missing'
    result=subprocess.run([sys.executable,'-B',str(entry)],capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    value=json.loads(result.stdout)
    assert value['execution_started'] is False and value['configuration']['max_residual_steps']==1000000
    output=tmp_path/'not_created'
    result=subprocess.run([sys.executable,'-B',str(entry),'--mode','train','--out',str(output)],capture_output=True,text=True,timeout=15)
    assert result.returncode!=0 and not output.exists()


@pytest.mark.parametrize('stop_kind',['budget','unknown'])
def test_training_lifecycle_has_no_validation_and_counts_only_executed_masks(tmp_path,stop_kind):
    code=r'''
import json,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from residual_rl.config import activate_long_training_profile
activate_long_training_profile()
from residual_rl.config import DEFAULT
from residual_rl import runtime,safety
from residual_rl.storage import RunStore,write_json,sha256
from residual_rl.pilot import own_code_hashes
from residual_rl.learning import Episode
from residual_rl.long_training import execute,training_schedule
root=Path(sys.argv[1]);store=RunStore(root,create=True)
write_json(root/'configuration.json',DEFAULT.to_dict())
write_json(root/'seed_schedule.json',training_schedule())
write_json(root/'frozen_identity.json',runtime.identity())
write_json(root/'code_hashes.json',own_code_hashes())
write_json(root/'preflight_reference.json',dict(actor_dim=572,critic_dim=791))
write_json(root/'seal.json',{name:sha256(root/name) for name in
 ['configuration.json','seed_schedule.json','frozen_identity.json','code_hashes.json','preflight_reference.json']})
safety.resources=lambda *args,**kwargs:dict(disk_free=30*2**30,output_bytes=0,rss=0,dangerous_memory=False)
calls=[]
class NumericEngine:
 def __init__(self,store,guard):self.store=store;self.guard=guard
 def run(self,spec,mode,learner):
  assert mode=='train' and spec['phase']=='train' and learner is not None
  if len(calls)==2:
   if sys.argv[2]=='unknown':raise ValueError('diagnostic infrastructure failure')
   raise safety.BudgetStop('test_stop')
  calls.append(spec['target_count']);self.store.claim(spec)
  episode=Episode(spec['episode_id'],spec['target_count'],np.zeros((3,572),np.float32),
   np.zeros((3,791),np.float32),np.zeros((2,3),np.float32),np.zeros(2,np.float32),
   np.array([False,True]),np.array([False,True]))
  self.guard.residual_steps+=2 # One accepted mask plus a begun/unexecuted diagnostic context.
  row=dict(**spec,success=True,outcome='success',reason='synthetic lifecycle',
   final_collected=spec['target_count'],opportunities=1,planning_calls=1,physical_attempts=1,
   residual_frames=1,terminal_reward=0.,credited_contact_return=0.,rl_return=0.)
  row=self.store.commit_episode(row,episode,self.guard)
  return SimpleNamespace(row=row,episode=episode)
runtime.HybridRuntime=NumericEngine
try:execute(store,started_at=time.monotonic())
except ValueError as error:
 assert sys.argv[2]=='unknown' and str(error)=='diagnostic infrastructure failure'
summary=json.loads((root/'summary.json').read_text());progress=json.loads((root/'training_progress.json').read_text())
assert calls==[1,2]
assert summary['training_steps']==summary['residual_decisions']==progress['training_steps']==2
assert summary['updates']==summary['actor_updates']==0 and not summary['complete']
assert summary['by_phase']['train']['episodes']==2
assert progress['state']==('needs_attention' if sys.argv[2]=='unknown' else 'budget_stopped')
assert progress['active_episode'] is None
assert not (root/'videos').exists()
assert summary['save']['status']=='succeeded'
'''
    result=subprocess.run([sys.executable,'-B','-c',code,str(tmp_path/'run'),stop_kind],
        cwd=Path(__file__).parents[1],capture_output=True,text=True,timeout=40)
    assert result.returncode==0,result.stdout+result.stderr
