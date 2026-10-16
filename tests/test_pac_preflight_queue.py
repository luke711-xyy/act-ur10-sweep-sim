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
