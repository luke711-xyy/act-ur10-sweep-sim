"""Recorded-frame audit tests: corrupt saved facts, never rerun physics."""
import copy
import gzip
import io
import json
from types import SimpleNamespace
import numpy as np
import pytest


def fixture(tmp_path,change=None,*,reverse=False,hold=False,all_change=None):
    from residual_rl.relative_action import RelativeActionGenerator,continuous_probe_action
    from residual_rl.storage import TraceWriter
    from sim.act.evaluate import _policy_reference_substep
    c=RelativeActionGenerator([[-1.,1.],[-1.,1.]])
    path=tmp_path/'trace.jsonl.gz';out=TraceWriter(path)
    previous=np.array([0.,0.,.001,0.],np.float32);count=0
    for frame in range(75):
        start=previous.copy();p=start[:2].copy();q=start+np.array([.002,0.,0.,.001],np.float32)
        t=frame*.04;raw=continuous_probe_action(t)
        target=c.generate(p,q,raw,dt=.04,active=True,allow_motion=not hold)
        if reverse:
            target[:2]=2*q[:2]-target[:2]
            c.last_audit['generated_target']=target.copy()
            c.last_audit['xy_correction']=target[:2]-q[:2]
        for substep in range(1,5):
            command=_policy_reference_substep(start,target,substep,4,.01,.08,.1,.35,True,.001)
            audit=copy.deepcopy(c.last_audit)
            audit.update(frame_started_at=t,residual_elapsed=t,substep=substep,substeps=4,
                control_dt=.01,execution_start=start.copy(),admittance_z=.001,
                contact_latched=True,control_reference=command.copy(),applied=command.copy(),filter_accepted=True)
            row=dict(t=t+substep*.01,control_index=count+1,command=command.copy(),
                command_before=previous.copy(),command_started_at=t+(substep-1)*.01,
                tcp_before=previous[:3].copy(),tcp=command[:3].copy(),
                tcp_yaw_before=float(previous[3]),tcp_yaw=float(command[3]),active=True,
                phase='ACT_residual',probe_stage='continuous_micro_adjust',control_audit=audit)
            if change and frame==30 and substep==1: change(row)
            if all_change:all_change(row)
            out.append(row);previous=command;count+=1
    out.close()
    return SimpleNamespace(trace_path=path,row=dict(outcome='failure',error=None,accepted_ACT_chunks=10,
        control_steps=count,action_contract='relative_xy80_yaw45_v1'))


def audit(result):
    from residual_rl.relative_audit import audit_relative_rollout
    return audit_relative_rollout(result)


def test_relative_audit_checks_every_executed_control_and_all_25hz_frames(tmp_path):
    report=audit(fixture(tmp_path))
    assert report['passed'] and report['audited_control_steps']==300
    assert report['action_frames']==75 and report['changing_action_frames']>65
    assert report['max_relative_xy_correction']<=.8+1e-10
    assert report['Z_changes']==0 and report['continuous_contact_verified'] is False


@pytest.mark.parametrize('field', ['basis','ratio','clock','z','raw','acceptance'])
def test_relative_audit_rejects_corrupt_physical_or_action_evidence(tmp_path,field):
    def corrupt(row):
        a=row['control_audit']
        if field=='basis':a['actual_xy'][0]+=1.
        elif field=='ratio':a['generated_target'][0]+=1.
        elif field=='clock':a['frame_started_at']+=.01
        elif field=='z':a['admittance_z']+=.002
        elif field=='raw':a['raw_action'][0]=.8
        else:a['filter_accepted']=False
    with pytest.raises(RuntimeError):audit(fixture(tmp_path,corrupt))


def test_relative_audit_never_calls_a_safety_failure_a_pass(tmp_path):
    value=fixture(tmp_path);value.row['outcome']='safety'
    with pytest.raises(RuntimeError):audit(value)


def test_relative_audit_rejects_consistently_executed_reversed_correction(tmp_path):
    with pytest.raises(RuntimeError):audit(fixture(tmp_path,reverse=True))


def test_relative_audit_rejects_a_continuous_arm_that_only_holds(tmp_path):
    with pytest.raises(RuntimeError):audit(fixture(tmp_path,hold=True))


def test_relative_audit_cannot_skip_alternating_active_substeps(tmp_path):
    def skip(row):
        if row['control_index']%2==0:
            row['active']=False;row['phase']='ACT_handoff';row['control_audit']=None
    with pytest.raises(RuntimeError):audit(fixture(tmp_path,all_change=skip))


@pytest.mark.parametrize('bounds', [[[float('nan'),float('nan')]]*2,[[1.,-1.],[-1.,1.]]])
def test_relative_audit_rejects_nonfinite_or_reversed_workspace_evidence(tmp_path,bounds,monkeypatch):
    value=fixture(tmp_path)
    with gzip.open(value.trace_path,'rt') as stream:rows=[json.loads(line) for line in stream]
    for row in rows:row['control_audit']['bounds']=bounds
    # Exercise an untrusted reader independently of the strict normal JSON writer.
    import residual_rl.relative_audit as module
    serialized='\n'.join(json.dumps(row) for row in rows)
    monkeypatch.setattr(module.gzip,'open',lambda *args,**kwargs:io.StringIO(serialized))
    with pytest.raises(RuntimeError):audit(value)
