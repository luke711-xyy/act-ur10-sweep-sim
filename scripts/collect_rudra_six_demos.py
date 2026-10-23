"""Collect one native-scene preview per goal using the existing expert rules."""
import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sim.act.collect_dataset import EpisodeSpec, training_episode_plan, _expert_metadata
from sim.act.dataset import ActDatasetWriter
from sim.act.naming import next_demo_name
from sim.act.rollout import run_expert_episode
from sim.config import save_config
from sim.controllers.rudra_contact import demonstration_config
from sim.environments.rudra_sweep_env import collection_config


def checked_configuration(path):
    report=json.loads(Path(path).read_text())
    native=report.get('reports',{}).get('native',{})
    if not native.get('passed') or native.get('acquisition_phase')!='maintain':
        raise ValueError('a passing native contact test with stable force acquisition is required')
    cfg=demonstration_config(collection_config())
    if report.get('configuration')!=cfg.to_dict():
        raise ValueError('contact test configuration does not match collection configuration')
    return cfg


def six_demo_plan():
    return sorted(training_episode_plan(seed_base=410000,episodes_per_target=1),key=lambda s:s.target_count)


def write_json(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False))
    tmp.replace(path)


@contextmanager
def terminal_failure_status(out):
    """Publish a terminal error without retrying unknown execution/IO failures."""
    try:
        yield
    except Exception as exc:
        path=out/'progress.json'
        status=json.loads(path.read_text()) if path.exists() else dict(completed=0,total=6)
        status.update(state='failed',error_type=type(exc).__name__,reason=str(exc))
        try:
            write_json(path,status)
        except OSError:
            # Keep the original error if the storage itself is unavailable.
            pass
        print(json.dumps(dict(event='collection_stopped',**status)),flush=True)
        raise


def run_candidate(cfg,seed):
    try:
        return run_expert_episode(cfg,seed=seed,collect_observations=True),None
    except RuntimeError as exc:
        if str(exc)!='expert candidate search returned no executable plan':
            raise
        # This known planning outcome may use the original seed window.
        # No measured quantity is invented when no executable plan exists.
        return None,dict(reason=str(exc),collected=None,frames=None)


def collect(cfg,out,gate_report):
    save_config(cfg,str(out/'collection-config.yaml'))
    plan=six_demo_plan()
    write_json(out/'plan.json',[vars(s) for s in plan])
    writer=ActDatasetWriter(str(out),image_format='webp-lossless')
    completed=[]
    failures=[]
    for planned in plan:
        local=cfg.copy();local.set_path('task.target_count',planned.target_count)
        for attempt in range(6):
            seed=planned.seed+attempt
            status=dict(state='running',completed=len(completed),total=6,target=planned.target_count,
                        seed=seed,generation_attempt=attempt+1)
            write_json(out/'progress.json',status)
            print(json.dumps(dict(event='demo_started',**status)),flush=True)
            result,planning_failure=run_candidate(local,seed)
            if result is None or not result.success:
                outcome=(planning_failure if result is None else
                         dict(reason=result.failure_reason,collected=result.collected,
                              frames=result.sampled_frames))
                failure=dict(target_count=planned.target_count,seed=seed,**outcome)
                failures.append(failure);write_json(out/'failed-attempts.json',failures)
                print(json.dumps(dict(event='demo_candidate_failed',**failure)),flush=True)
                del result;gc.collect()
                continue
            spec=EpisodeSpec(split='pilot',target_count=planned.target_count,seed=seed,
                             layout_id=planned.layout_id,layout_kind=planned.layout_kind)
            metadata=_expert_metadata(spec,result,planned.seed,attempt+1,local)
            metadata.update(episode_kind='expert_preview',split='pilot',
                            dataset_kind='rudra_native_six_preview',scene_backend='rudra_native',
                            scene_profile=str(cfg.scene.profile),camera_views=['top','wrist'],
                            coordinate_frame='original fixed sweep task frame',
                            task_to_native_world_translation_m=[.005,-.685,-.002],
                            contact_profile='trial4_1n',brush_sole_friction=[.1,.0001,.0001],
                            admittance_seek_MBK=[10.,400.,10.],admittance_maintain_MBK=[10.,800.,10.],
                            contact_gate_report=str(Path(gate_report).resolve()))
            name=next_demo_name([out],planned.target_count)
            writer.add_episode(name,result.observations,result.actions,True,metadata,control_trace=result.trace)
            record=dict(episode_id=name,target_count=planned.target_count,seed=seed,
                        collected=result.collected,frames=result.sampled_frames,
                        peak_force=result.peak_force,peak_raw_force=result.peak_raw_force)
            completed.append(record)
            write_json(out/'progress.json',dict(state='running',completed=len(completed),total=6,episodes=completed))
            print(json.dumps(dict(event='demo_saved',**record)),flush=True)
            del result;gc.collect()
            break
        else:
            write_json(out/'progress.json',dict(state='failed',completed=len(completed),total=6,
                                               episodes=completed,reason='expert seed retry window exhausted'))
            return 2
    summary=dict(state='complete',completed=6,total=6,episodes=completed,
                 camera_views=['top','wrist'],image_size=[320,320],schema_version=5,
                 dataset_kind='six preview demonstrations; not a full training dataset')
    write_json(out/'progress.json',summary)
    write_json(out/'summary.json',summary)
    print(json.dumps(dict(event='six_demos_complete',**summary)),flush=True)
    return 0


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gate-report',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    cfg=checked_configuration(args.gate_report)
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('six-demo capture requires a new empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    # Enter only after claiming a new empty directory: a rejected second
    # invocation must never relabel a previous run's progress.
    with terminal_failure_status(out):
        return collect(cfg,out,args.gate_report)


if __name__=='__main__':
    raise SystemExit(main())
