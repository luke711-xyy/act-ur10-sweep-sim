"""One-shot paired, deterministic 100K ACT under the new chunk schedule.

Imports only a previously sealed bundle; never imports a mutable worktree's
policy/control modules. This entry cannot resume, train or load a PPO Actor.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

LABEL='new_schedule_no_noise_100k'
ARM='new_schedule_zero'


def validate_plan(rows):
    if len(rows)!=120 or {r['seed'] for r in rows}!=set(range(2120000,2120120)):
        raise ValueError('requires all 120 distinct original paired seeds')
    if Counter(r['target_count'] for r in rows)!=Counter({n:20 for n in range(1,7)}):
        raise ValueError('requires twenty layouts per target')
    fingerprints=[]
    for row in rows:
        poses=np.asarray(row['initial_object_poses'],dtype='<f4')
        if poses.shape!=(6,7) or not np.isfinite(poses).all():raise ValueError('invalid initial poses')
        digest=hashlib.sha256(poses.tobytes()).hexdigest()
        if digest!=row['layout_fingerprint']:raise ValueError('planned initial pose fingerprint differs')
        fingerprints.append(digest)
    if len(set(fingerprints))!=120:raise ValueError('layouts are not unique')
    return rows


class PairedRegistry:
    """Explicit evaluation-only exception for the user-authorized held-out plan."""
    def __init__(self,plan):
        self.plan={r['seed']:r for r in validate_plan(plan)};self.used=set()
    def register(self,spec,fingerprint):
        expected=self.plan.get(spec['seed'])
        if (spec['role']!='comparison' or spec['arm']!=ARM or expected is None
                or spec['target_count']!=expected['target_count']
                or fingerprint!=expected['layout_fingerprint'] or spec['seed'] in self.used):
            raise ValueError('unpaired, changed, repeated or non-evaluation layout')
        self.used.add(spec['seed'])


def summarize(rows):
    counts=np.zeros((7,6),int);by={}
    for n in range(1,7):
        group=[r for r in rows if r['target_count']==n]
        for r in group:
            value=r.get('final_collected')
            if value is not None:counts[int(value),n-1]+=1
        by[str(n)]=dict(episodes=len(group),strict_successes=sum(r.get('success') is True for r in group),
            strict_success_rate=sum(r.get('success') is True for r in group)/len(group) if group else None)
    complete=(len(rows)==120 and len({r.get('seed') for r in rows})==120
        and all(g['episodes']==20 for g in by.values()) and all(r.get('outcome')!='interrupted' for r in rows))
    known=[r for r in rows if r.get('final_collected') is not None]
    return dict(model_label=LABEL,episodes=len(rows),complete=complete,
        strict_successes=sum(r.get('success') is True for r in rows),
        exact_count_matches=sum(r['final_collected']==r['target_count'] for r in known),
        unknown_counts=len(rows)-len(known),
        mean_absolute_count_error=sum(abs(r['final_collected']-r['target_count']) for r in known)/len(known) if known else None,
        count_matrix=counts.tolist(),by_target=by,training_updates=0,exploration_noise=False,
        count_match_is_not_strict_success=True)


def render_density(summary,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    counts=np.asarray(summary['count_matrix']);totals=np.array([summary['by_target'][str(n)]['episodes'] for n in range(1,7)])
    density=counts/np.maximum(totals,1)
    fig,ax=plt.subplots(figsize=(10,8),layout='constrained')
    im=ax.imshow(density,origin='lower',aspect='auto',vmin=0,vmax=1,cmap='magma')
    for i in range(7):
        for j in range(6):
            text=f'{100*density[i,j]:.0f}%\n{counts[i,j]}/{totals[j]}' if totals[j] else '—'
            ax.text(j,i,text,ha='center',va='center',color='black' if density[i,j]>.65 else 'white')
    for j in range(6):ax.add_patch(Rectangle((j-.5,j+.5),1,1,fill=False,edgecolor='#d9eef5',linewidth=2))
    ax.set(xticks=range(6),xticklabels=range(1,7),yticks=range(7),xlabel='Target count',ylabel='Final fully collected count',
        title=f'Hybrid A* to ACT | new schedule | no noise | {summary["episodes"]}/120\nStrict successes {summary["strict_successes"]}; exact final count {summary["exact_count_matches"]}')
    fig.colorbar(im,ax=ax,label='Frequency within tested target')
    for suffix in ('png','svg'):
        temp=out/f'.count_density.tmp.{suffix}';fig.savefig(temp,dpi=160);temp.replace(out/f'count_density.{suffix}')
    plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=('mps','cpu'),default='mps')
    args=p.parse_args(argv)
    bundle=args.bundle.resolve();source=bundle/'source'
    sys.path.insert(0,str(source))
    from pac_act.identity import verify_bundle,watch_bundle,FileIdentityWatch
    from pac_act.lifecycle import require_runtime,require_local_slot,experiment_lock,Collection,cfg_for
    from pac_act.storage import RunStore,write_json,sha256
    from pac_act.safety import Guard
    require_runtime();require_local_slot();frozen=verify_bundle(bundle)
    if sha256(args.plan)!=args.plan_sha256:raise ValueError('original paired plan hash differs')
    plan=validate_plan(json.loads(args.plan.read_text()))
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    with experiment_lock(out,local=True):
        (out/'episode_plan.json').write_bytes(args.plan.read_bytes())
        metadata=dict(label=LABEL,bundle_identity=frozen['identity'],runner_sha256=sha256(Path(__file__)),
            plan_sha256=args.plan_sha256,device=args.device,checkpoint_step=100000,noise=False,training=False,
            frequency_hz=25,predict_steps=25,execute_steps=5,temporal_ensemble=False,
            max_opportunities=120,max_physical_attempts_per_layout=1,frame_budget_per_episode=2000,
            video_retention='retain through the whole batch; no per-episode deletion',
            heldout_plan_authorized_for_evaluation_only=True,started_at=time.time())
        write_json(out/'metadata.json',metadata)
        guard=Guard(out,started_at=metadata['started_at'],deadline=metadata['started_at']+24*3600,
            collect_deadline=metadata['started_at']+24*3600,free_gib=8,output_gib=20)
        bundle_watch=watch_bundle(bundle,frozen)
        files_watch=FileIdentityWatch({Path(__file__).resolve():metadata['runner_sha256'],
            out/'episode_plan.json':metadata['plan_sha256'],out/'metadata.json':sha256(out/'metadata.json')})
        def identities():bundle_watch();files_watch()
        guard.identity_check=identities;guard.poll()
        store=RunStore(out);registry=PairedRegistry(plan);world=None
        try:
            import torch
            from pac_act.policy import ChunkPolicy
            from pac_act.hybrid import HybridEpisode
            from pac_act.media import render_episode
            if args.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no automatic device fallback')
            torch.manual_seed(42)
            policy=ChunkPolicy(frozen['model'],args.device,build_critic=False)
            collection=Collection(bundle,policy,store,registry,guard,{})
            summary=summarize([]);write_json(out/'summary.json',summary);render_density(summary,out)
            for row in plan:
                guard.poll();spec=dict(role='comparison',arm=ARM,seed=row['seed'],target_count=row['target_count'],
                    model_label=LABEL,checkpoint_step=100000,paired_original_key=row['key'])
                write_json(out/'status.json',dict(state='running',seed=row['seed'],completed=len(store.rows()),training_started=False))
                world=HybridEpisode(cfg_for(bundle,row['target_count']),policy,spec,store,registry,guard)
                while not world.done:collection.block(world,deterministic=True,learning=False)
                world.finish();world=None
                summary=summarize(store.rows());write_json(out/'summary.json',dict(summary,complete=False,media_pending=True));render_density(summary,out)
                folder=out/'episodes'/store.key(spec)
                render_episode(folder,guard)
                print(json.dumps(dict(seed=row['seed'],completed=len(store.rows()),strict_successes=summary['strict_successes'])),flush=True)
            rows=store.rows();summary=summarize(rows)
            if not summary['complete'] or len(list((out/'attempts').glob('*.json')))!=120:raise RuntimeError('incomplete/duplicate opportunity ledger')
            for row in rows:
                folder=out/'episodes'/store.key(row);media=json.loads((folder/'media.json').read_text())
                video=folder/media['file']
                if video.stat().st_size!=media['bytes'] or sha256(video)!=media['sha256']:raise RuntimeError('verified media changed')
            verify_bundle(bundle);write_json(out/'summary.json',dict(summary,media_pending=False,verified_videos=120))
            write_json(out/'status.json',dict(state='complete',completed=120,training_started=False))
        except BaseException as error:
            if world is not None and not world.closed:
                try:world.interrupt(error)
                except BaseException as cleanup:error.add_note('cleanup: '+repr(cleanup))
            write_json(out/'status.json',dict(state='needs_attention',error=repr(error),automatic_retry=False,training_started=False))
            raise


if __name__=='__main__':main()
