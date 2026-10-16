import json
from pathlib import Path
import subprocess

import pytest

from pac_act.storage import RunStore,sha256,write_json,append_json


def completed_batch(root,*,zero_planning=False):
    store=RunStore(root);plan=[]
    for i in range(120):
        goal=i%6+1;seed=2120000+i;key=f'five_view_100k:n{goal}:l{i//6:02d}'
        fingerprint=f'{i:064x}'
        spec=dict(role='comparison',arm='new_schedule_zero',seed=seed,target_count=goal,
            paired_original_key=key,layout_fingerprint=fingerprint)
        folder=store.claim(spec)
        store.commit(dict(**spec,opportunities=1,planning_calls=0 if zero_planning and i==0 else 1,
            physical_attempts=0 if zero_planning and i==0 else 1,
            success=False,final_collected=0,outcome='failure'))
        video=folder/'replay.mp4';video.write_bytes(b'verified fixture video')
        write_json(folder/'media.json',dict(file='replay.mp4',bytes=video.stat().st_size,sha256=sha256(video),
            verification=dict(frames=5,fps=25.,size=[960,640],bytes=video.stat().st_size,sha256=sha256(video))))
        plan.append(dict(seed=seed,target_count=goal,key=key,layout_fingerprint=fingerprint))
    write_json(root/'episode_plan.json',plan)
    write_json(root/'metadata.json',dict(label='new_schedule_no_noise_100k',training=False,noise=False,
        checkpoint_step=100000,plan_sha256=sha256(root/'episode_plan.json')))
    write_json(root/'summary.json',dict(complete=True,episodes=120,media_pending=False,verified_videos=120,
        training_updates=0,strict_successes=0))
    write_json(root/'status.json',dict(state='complete',completed=120,training_started=False))
    return store


def test_dependency_waits_for_runner_even_after_complete_status(tmp_path):
    from pac_act.preflight_queue import dependency_ready
    root=tmp_path/'batch';completed_batch(root)
    assert dependency_ready(root,active_pids=[123]) is None
    assert dependency_ready(root,active_pids=[])['episodes']==120


def test_dependency_preserves_failure_before_planning_without_requiring_a_retry(tmp_path):
    from pac_act.preflight_queue import dependency_ready
    root=tmp_path/'batch';completed_batch(root,zero_planning=True)
    assert dependency_ready(root,active_pids=[])['episodes']==120


@pytest.mark.parametrize('hazard',['duplicate','missing_media','changed_video','orphan','pairing','unverified'])
def test_dependency_refuses_incomplete_or_changed_results(tmp_path,hazard):
    from pac_act.preflight_queue import dependency_ready
    root=tmp_path/'batch';store=completed_batch(root);row=store.rows()[0];folder=root/'episodes'/store.key(row)
    if hazard=='duplicate':append_json(root/'episodes.jsonl',row)
    elif hazard=='missing_media':(folder/'media.json').unlink()
    elif hazard=='changed_video':(folder/'replay.mp4').write_bytes(b'bad')
    elif hazard=='orphan':write_json(root/'attempts/orphan.json',{})
    elif hazard=='pairing':
        plan=json.loads((root/'episode_plan.json').read_text());plan[0]['layout_fingerprint']='wrong'
        write_json(root/'episode_plan.json',plan)
    else:write_json(root/'summary.json',dict(complete=True,episodes=120,media_pending=True))
    with pytest.raises((RuntimeError,ValueError,OSError)):dependency_ready(root,active_pids=[])


def test_dependency_abnormal_exit_stops_without_retry(tmp_path):
    from pac_act.preflight_queue import dependency_ready
    root=tmp_path/'batch';root.mkdir()
    write_json(root/'status.json',dict(state='needs_attention',error='original error'))
    with pytest.raises(RuntimeError,match='needs_attention'):dependency_ready(root,active_pids=[])
    write_json(root/'status.json',dict(state='running'))
    with pytest.raises(RuntimeError,match='exited'):dependency_ready(root,active_pids=[])


def test_queue_launches_only_preflight_once_after_actual_completion(tmp_path,monkeypatch):
    import pac_act.preflight_queue as module
    root=tmp_path/'batch';completed_batch(root)
    bundle=tmp_path/'bundle';(bundle/'source/scripts').mkdir(parents=True)
    write_json(bundle/'manifest.json',dict(identity='fixture'))
    (bundle/'source/scripts/train_pac_act.py').write_text('# frozen fixture\n')
    queue=tmp_path/'queue';out=tmp_path/'new_preflight';calls=[]
    def execute(command,**kwargs):
        calls.append(command);out.mkdir()
        write_json(out/'acceptance.json',dict(engineering_passed=True,training_started=False,
            GPU_training_started=False))
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(module,'dependency_pids',lambda root:[])
    monkeypatch.setattr(module,'local_slot_available',lambda:True)
    monkeypatch.setattr(module.subprocess,'run',execute)
    result=module.run_queue(bundle,root,out,queue,start_index=120,device='cpu')
    assert result['state']=='preflight_complete' and len(calls)==1
    command=calls[0]
    assert command[command.index('--mode')+1]=='preflight'
    assert command[command.index('--preflight-start-index')+1]=='120'
    assert '--acknowledge-gpu-training' not in command
    assert (queue/'preflight_started.json').is_file()
    with pytest.raises(FileExistsError):module.run_queue(bundle,root,out,queue,start_index=120)
    assert len(calls)==1


def test_queue_failure_is_terminal_not_an_automatic_retry(tmp_path,monkeypatch):
    import pac_act.preflight_queue as module
    root=tmp_path/'batch';completed_batch(root)
    bundle=tmp_path/'bundle';(bundle/'source/scripts').mkdir(parents=True)
    write_json(bundle/'manifest.json',{});(bundle/'source/scripts/train_pac_act.py').write_text('# frozen fixture\n')
    queue=tmp_path/'queue';calls=[]
    monkeypatch.setattr(module,'dependency_pids',lambda root:[])
    monkeypatch.setattr(module,'local_slot_available',lambda:True)
    def execute(command,**kwargs):calls.append(command);return subprocess.CompletedProcess(command,2)
    monkeypatch.setattr(module.subprocess,'run',execute)
    with pytest.raises(RuntimeError,match='preflight'):module.run_queue(bundle,root,tmp_path/'out',queue,start_index=120)
    assert len(calls)==1
    assert json.loads((queue/'status.json').read_text())['state']=='needs_attention'
    with pytest.raises(FileExistsError):module.run_queue(bundle,root,tmp_path/'out',queue,start_index=120)
    assert len(calls)==1


@pytest.mark.parametrize('hazard',['dependency_failure','media_drift','new_encoder'])
def test_queue_rechecks_dependency_after_waiting_for_a_busy_slot(tmp_path,monkeypatch,hazard):
    import pac_act.preflight_queue as module
    root=tmp_path/'batch';store=completed_batch(root)
    bundle=tmp_path/'bundle';(bundle/'source/scripts').mkdir(parents=True)
    write_json(bundle/'manifest.json',{});(bundle/'source/scripts/train_pac_act.py').write_text('# fixture\n')
    queue=tmp_path/'queue';out=tmp_path/'out';slot_calls=[];sleeps=[];launched=[];encoder=[]
    def slot():
        slot_calls.append(True)
        if len(slot_calls)==1:
            if hazard=='dependency_failure':write_json(root/'status.json',dict(state='needs_attention'))
            elif hazard=='media_drift':
                row=store.rows()[0];(root/'episodes'/store.key(row)/'replay.mp4').write_bytes(b'changed')
            else:encoder.append(456)
            return False
        return True
    def sleep(seconds):
        sleeps.append(seconds)
        if hazard=='new_encoder' and len(sleeps)>1:raise RuntimeError('test ends while encoder active')
    monkeypatch.setattr(module,'dependency_pids',lambda root:list(encoder))
    monkeypatch.setattr(module,'local_slot_available',slot)
    monkeypatch.setattr(module.time,'sleep',sleep)
    def execute(command,**kwargs):
        launched.append(command);out.mkdir()
        write_json(out/'acceptance.json',dict(engineering_passed=True,training_started=False,GPU_training_started=False))
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(module.subprocess,'run',execute)
    with pytest.raises(RuntimeError):module.run_queue(bundle,root,out,queue,start_index=120)
    assert not launched and not (queue/'preflight_started.json').exists()


@pytest.mark.parametrize('device',['mps','cuda'])
def test_queue_rejects_non_CPU_before_writing_any_output(tmp_path,device):
    from pac_act.preflight_queue import run_queue
    with pytest.raises(ValueError,match='CPU'):run_queue(tmp_path/'bundle',tmp_path/'batch',tmp_path/'out',tmp_path/'queue',device=device)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('case',['out_equals_bundle','out_inside_bundle','out_contains_bundle',
    'queue_equals_bundle','queue_inside_bundle','queue_contains_bundle','same_destinations',
    'queue_inside_out','out_inside_queue','bundle_symlink','dependency_symlink'])
def test_queue_destinations_are_disjoint_from_inputs_and_each_other(tmp_path,case):
    from pac_act.preflight_queue import run_queue
    bundle=tmp_path/'sealed/bundle';bundle.mkdir(parents=True)
    batch=tmp_path/'evaluations/batch';batch.mkdir(parents=True)
    out=tmp_path/'out';queue=tmp_path/'queue'
    if case=='out_equals_bundle':out=bundle
    elif case=='out_inside_bundle':out=bundle/'run'
    elif case=='out_contains_bundle':out=bundle.parent
    elif case=='queue_equals_bundle':queue=bundle
    elif case=='queue_inside_bundle':queue=bundle/'queue'
    elif case=='queue_contains_bundle':queue=bundle.parent
    elif case=='same_destinations':queue=out
    elif case=='queue_inside_out':queue=out/'queue'
    elif case=='out_inside_queue':out=queue/'out'
    elif case=='bundle_symlink':
        alias=tmp_path/'alias';alias.symlink_to(bundle,target_is_directory=True);queue=alias/'queue'
    else:
        alias=tmp_path/'alias';alias.symlink_to(batch,target_is_directory=True);out=alias/'run'
    before=set(tmp_path.rglob('*'))
    with pytest.raises(ValueError,match='overlap'):run_queue(bundle,batch,out,queue)
    assert set(tmp_path.rglob('*'))==before


@pytest.mark.parametrize('failure',['missing_input','authorization_write'])
def test_queue_initialization_failure_keeps_original_error_and_terminal_status(tmp_path,monkeypatch,failure):
    import pac_act.preflight_queue as module
    batch=tmp_path/'batch';completed_batch(batch)
    bundle=tmp_path/'bundle';bundle.mkdir();queue=tmp_path/'queue'
    if failure=='authorization_write':
        write_json(bundle/'manifest.json',{})
        original=module.write_json
        def broken(path,value):
            if Path(path).name=='authorization.json':raise OSError('original authorization write failed')
            return original(path,value)
        monkeypatch.setattr(module,'write_json',broken)
    with pytest.raises(OSError) as caught:module.run_queue(bundle,batch,tmp_path/'out',queue)
    status=json.loads((queue/'status.json').read_text())
    assert status['state']=='needs_attention' and str(caught.value) in status['error']
    assert not (queue/'preflight_started.json').exists()
