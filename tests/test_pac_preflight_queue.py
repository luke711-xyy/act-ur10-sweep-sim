import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from pac_act.storage import RunStore,sha256,write_json,append_json


DINO_SCHEDULE = dict(prediction_steps=25, execution_steps=5, action_hz=25.0,
    query_hz=5.0, temporal_ensemble=False, inference_advances_simulation=False,
    deterministic=True)


def _fixture_video(path):
    """Make a tiny real MP4 so DINO dependency tests exercise actual decoding."""
    import imageio_ffmpeg
    path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner',
        '-loglevel', 'error', '-f', 'lavfi', '-i', 'color=c=red:s=16x16:r=1:d=1',
        '-frames:v', '1', '-an', '-y', str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return path


def completed_dinov3_batch(root):
    label='hybrid_dinov3_200k'; checkpoint=root/'checkpoint'/'step_200000'
    checkpoint.mkdir(parents=True);model_path=str(checkpoint)
    (checkpoint/'model.safetensors').write_bytes(b'DINO checkpoint fixture')
    weight_digest=sha256(checkpoint/'model.safetensors')
    checkpoint_files={'model.safetensors':[ (checkpoint/'model.safetensors').stat().st_size,
        (checkpoint/'model.safetensors').stat().st_mtime_ns]}
    checkpoint_hashes={'model.safetensors':weight_digest}
    video_root=root/'videos'; video_root.mkdir(parents=True)
    prototype=_fixture_video(root/'fixture.mp4')
    profile={'admittance_configuration':'act.hybrid_admittance','ACT_predicted_Z_executed':False}
    plan=[]; rows=[]
    for i in range(120):
        target=i//20+1; seed=2120000+i; fingerprint=f'{i+1:064x}'
        key=f'{label}:n{target}:l{i:02d}'
        plan.append(dict(layout_index=i,seed=seed,layout_fingerprint=fingerprint,
            target_count=target,model_label=label,model_path=model_path,
            checkpoint_step=200000,key=key))
        video=video_root/f'{key.replace(":","_")}.mp4'; shutil.copyfile(prototype,video)
        rows.append(dict(layout_index=i,seed=seed,layout_fingerprint=fingerprint,
            target_count=target,model_label=label,model_path=model_path,model_checkpoint=model_path,
            checkpoint_step=200000,key=key,success=False,reason='fixture failure',
            collected=0,peak_collected=0,final_collected=0,planning_calls=1,physical_attempts=1,
            inference_strategy='hybrid_astar_act_trial4_dinov3_frozen',
            act_chunk_schedule=dict(DINO_SCHEDULE),hybrid_control_profile=profile,
            video_path=str(video),video_error='',video_deferred=True,replay_episode_id=f'replay-{i}',
            replay_error='',frame_images_deleted=True,image_cleanup_error=''))
    plan_path=root/'episode_plan.json'; write_json(plan_path,plan)
    rows_path=root/'episodes.jsonl'; rows_path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    config_path=root/'evaluation_config.yaml';config_path.write_text('fixture: frozen sync5 evaluation\n')
    metadata=dict(planned=120,layouts=120,targets=[1,2,3,4,5,6],
        layout_seed_mode='independent_per_episode',episodes_per_target_per_model=20,
        episode_plan_sha256=sha256(plan_path),config_sha256=sha256(config_path),
        checkpoint_files_sha256=checkpoint_hashes,schedule_adapter=dict(DINO_SCHEDULE),
        schedule_adapter_sha256='a'*64,preview=False,
        models=[dict(label=label,step=200000,path=model_path,weights_sha256=weight_digest,
            checkpoint_files=checkpoint_files)],
        hybrid_control_profile=profile)
    write_json(root/'metadata.json',metadata)
    successes=sum(row['success'] is True for row in rows)
    peak=[[0 for _ in range(6)] for _ in range(7)]
    final=[[0 for _ in range(6)] for _ in range(7)]
    by_target={}
    for target in range(1,7):
        selected=[row for row in rows if row['target_count']==target]
        for row in selected:
            peak[row['peak_collected']][target-1]+=1
            final[row['final_collected']][target-1]+=1
        by_target[str(target)]={'episodes':20,'successes':0,'success_rate':0.0}
    write_json(root/'summary.json',dict(complete=True,completed=120,planned=120,
        models={label:dict(episodes=120,successes=successes,exact_peak_matches=0,
            peak_count_matrix=peak,final_count_matrix=final,by_target=by_target)}))
    write_json(root/'status.json',dict(state='complete',completed=120,planned=120,current=None))
    return rows


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


def test_dinov3_batch_requires_explicit_profile_and_emits_video_evidence(tmp_path):
    from pac_act.preflight_queue import dependency_ready
    root=tmp_path/'dino-batch';completed_dinov3_batch(root)
    with pytest.raises(RuntimeError,match='wrong dependency protocol'):
        dependency_ready(root,active_pids=[])
    certificate=dependency_ready(root,active_pids=[],profile='dinov3_sync5_200k')
    assert certificate['episodes']==120
    assert certificate['strict_successes']==0
    assert len(certificate['videos'])==120
    assert all(video['frames']>=1 and video['bytes']>0 and len(video['sha256'])==64
        for video in certificate['videos'])


def test_dinov3_queue_passes_explicit_profile_and_preflight_index_once(tmp_path,monkeypatch):
    import pac_act.preflight_queue as module
    root=tmp_path/'dino-batch';completed_dinov3_batch(root)
    bundle=tmp_path/'bundle';(bundle/'source/scripts').mkdir(parents=True)
    write_json(bundle/'manifest.json',dict(identity='fixture'))
    (bundle/'source/scripts/train_pac_act.py').write_text('# fixture\n')
    queue=tmp_path/'queue';out=tmp_path/'preflight';calls=[]
    monkeypatch.setattr(module,'dependency_pids',lambda root:[])
    monkeypatch.setattr(module,'local_slot_available',lambda:True)
    def execute(command,**kwargs):
        calls.append(command);out.mkdir()
        write_json(out/'acceptance.json',dict(engineering_passed=True,training_started=False,
            GPU_training_started=False))
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(module.subprocess,'run',execute)
    result=module.run_queue(bundle,root,out,queue,start_index=120,
        dependency_profile='dinov3_sync5_200k')
    assert result['state']=='preflight_complete' and len(calls)==1
    command=calls[0]
    assert command[command.index('--preflight-start-index')+1]=='120'
    authorization=json.loads((queue/'authorization.json').read_text())
    certificate=json.loads((queue/'dependency_verified.json').read_text())
    assert authorization['dependency_profile']=='dinov3_sync5_200k'
    assert certificate['profile']=='dinov3_sync5_200k' and certificate['verified_videos']==120


@pytest.mark.parametrize('hazard',['incomplete_status','wrong_checkpoint','wrong_schedule',
    'wrong_metadata_schedule','checkpoint_file_changed','plan_digest','duplicate_seed','retry',
    'pending_media','missing_video','invalid_video','outside_video','symlink_video'])
def test_dinov3_profile_refuses_protocol_or_media_drift(tmp_path,hazard):
    from pac_act.preflight_queue import dependency_ready
    root=tmp_path/'dino-batch';rows=completed_dinov3_batch(root)
    if hazard=='incomplete_status':
        write_json(root/'status.json',dict(state='complete',completed=119,planned=120,current=None))
    elif hazard=='wrong_checkpoint':
        rows[0]['checkpoint_step']=100000
        (root/'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    elif hazard=='wrong_schedule':
        rows[0]['act_chunk_schedule']['execution_steps']=25
        (root/'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    elif hazard=='wrong_metadata_schedule':
        metadata=json.loads((root/'metadata.json').read_text())
        metadata['schedule_adapter']['execution_steps']=25
        write_json(root/'metadata.json',metadata)
    elif hazard=='checkpoint_file_changed':
        model= json.loads((root/'metadata.json').read_text())['models'][0]
        (Path(model['path'])/'model.safetensors').write_bytes(b'changed checkpoint')
    elif hazard=='plan_digest':
        metadata=json.loads((root/'metadata.json').read_text())
        metadata['episode_plan_sha256']='c'*64
        write_json(root/'metadata.json',metadata)
    elif hazard=='duplicate_seed':
        rows[1]['seed']=rows[0]['seed']
        (root/'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    elif hazard=='retry':
        rows[0]['physical_attempts']=2
        (root/'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    elif hazard=='pending_media':
        status=json.loads((root/'status.json').read_text())
        status['media_pending']=rows[0]['key']
        write_json(root/'status.json',status)
    elif hazard=='missing_video':
        Path(rows[0]['video_path']).unlink()
    elif hazard=='invalid_video':
        Path(rows[0]['video_path']).write_bytes(b'not an mp4')
    elif hazard=='symlink_video':
        video=Path(rows[0]['video_path']);real=video.with_name('real.mp4')
        video.rename(real);video.symlink_to(real.name)
    else:
        outside=tmp_path/'outside.mp4';shutil.copyfile(rows[0]['video_path'],outside)
        rows[0]['video_path']=str(outside)
        (root/'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises((RuntimeError,ValueError,OSError)):
        dependency_ready(root,active_pids=[],profile='dinov3_sync5_200k')


def test_dinov3_dependency_waits_for_recovery_launcher(tmp_path):
    import pac_act.preflight_queue as module
    root=tmp_path/'dino-batch';completed_dinov3_batch(root)
    assert module.dependency_ready(root,active_pids=[321],profile='dinov3_sync5_200k') is None
    class Process:
        pid=321
        info={'cmdline':['python','/opt/batch/recovery_launcher.py','--batch-root',str(root)]}
    fake_psutil=SimpleNamespace(process_iter=lambda attrs:[Process()],NoSuchProcess=RuntimeError,
        AccessDenied=PermissionError)
    monkeypatch=pytest.MonkeyPatch()
    try:
        monkeypatch.setitem(sys.modules,'psutil',fake_psutil)
        assert module.dependency_pids(root)==[321]
    finally:
        monkeypatch.undo()


@pytest.mark.parametrize('command',['python {root}/recovery_episode_entrypoint.py --replay-root {root}/replays',
    'python -u -m sim.act.evaluate --replay-root {root}/replays',
    'ffmpeg -i pipe:0 {root}/videos/out.mp4'])
def test_dinov3_dependency_detects_episode_and_encoder_processes(tmp_path,command):
    import pac_act.preflight_queue as module
    root=tmp_path/'dino-batch';root.mkdir()
    class Process:
        pid=654
        info={'cmdline':command.format(root=root).split()}
    fake_psutil=SimpleNamespace(process_iter=lambda attrs:[Process()],NoSuchProcess=RuntimeError,
        AccessDenied=PermissionError)
    monkeypatch=pytest.MonkeyPatch()
    try:
        monkeypatch.setitem(sys.modules,'psutil',fake_psutil)
        assert module.dependency_pids(root)==[654]
    finally:
        monkeypatch.undo()


def test_queue_cli_passes_explicit_dependency_profile(tmp_path,monkeypatch):
    import pac_act.preflight_queue as module
    from scripts import queue_pac_preflight
    captured={}
    def run_queue(*args,**kwargs):captured.update(args=args,kwargs=kwargs)
    monkeypatch.setattr(module,'run_queue',run_queue)
    monkeypatch.setattr(sys,'argv',['queue_pac_preflight.py','--bundle',str(tmp_path/'bundle'),
        '--after-batch',str(tmp_path/'batch'),'--out',str(tmp_path/'out'),
        '--queue-state',str(tmp_path/'queue'),'--preflight-start-index','120',
        '--dependency-profile','dinov3_sync5_200k'])
    queue_pac_preflight.main()
    assert captured['kwargs']==dict(start_index=120,device='cpu',dependency_profile='dinov3_sync5_200k')


def test_dinov3_queue_detects_ledger_change_after_waiting_for_slot(tmp_path,monkeypatch):
    import pac_act.preflight_queue as module
    root=tmp_path/'dino-batch';completed_dinov3_batch(root)
    bundle=tmp_path/'bundle';(bundle/'source/scripts').mkdir(parents=True)
    write_json(bundle/'manifest.json',dict(identity='fixture'))
    (bundle/'source/scripts/train_pac_act.py').write_text('# fixture\n')
    slot_calls=[];launches=[]
    def slot():
        slot_calls.append(True)
        return len(slot_calls)>1
    def sleep(_):
        ledger=root/'episodes.jsonl'
        ledger.write_bytes(ledger.read_bytes()+b'\n')
    monkeypatch.setattr(module,'dependency_pids',lambda root:[])
    monkeypatch.setattr(module,'local_slot_available',slot)
    monkeypatch.setattr(module.time,'sleep',sleep)
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:launches.append(a))
    queue=tmp_path/'queue';out=tmp_path/'out'
    with pytest.raises(RuntimeError,match='changed while waiting|changed after'):
        module.run_queue(bundle,root,out,queue,start_index=120,dependency_profile='dinov3_sync5_200k')
    assert not launches and not (queue/'preflight_started.json').exists()
