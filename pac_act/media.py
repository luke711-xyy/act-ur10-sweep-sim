"""Reconstruct media from stored integration states, never rerun ACT or IK."""
import gzip
import json
from pathlib import Path

import numpy as np

from .storage import sha256,write_json


def render_episode(folder,guard):
    import imageio.v2 as imageio
    import mujoco
    from sim.config import Config
    from sim.environments.sweep_env import SweepEnv
    from sim.act.media_retention import verify_inference_video
    # sim.Config is loaded from the saved effective file, not a Mac path.
    from sim.config import load_config
    import yaml
    folder=Path(folder); result=json.loads((folder/'result.json').read_text())
    if (folder/'media.json').exists(): raise FileExistsError('media opportunity already recorded')
    config=json.loads((folder/'configuration.json').read_text())
    cfg=Config(config)
    env=SweepEnv(cfg,seed=result['seed']); env.reset(seed=result['seed'])
    initial=json.loads((folder/'layout.json').read_text())
    np.testing.assert_array_equal(np.c_[env.component_positions(),env.component_quats()].astype('<f4'),np.asarray(initial['poses'],dtype='<f4'))
    path=folder/'replay.mp4'; frames=0; times=[]; forces=[]; errors=[]; physical_rows=0
    try:
        with imageio.get_writer(path,fps=25,codec='libx264',macro_block_size=16) as writer:
            with gzip.open(folder/'physical_trace.jsonl.gz','rt') as stream:
                pending=None
                for i,line in enumerate(stream):
                    guard.poll()
                    row=json.loads(line); physical_rows+=1
                    if 'replay_state' not in row: raise ValueError('missing recorded state; command replay forbidden')
                    times.append(row['t']); forces.append(row['normal_force'])
                    errors.append(1000*abs(row['tcp'][2]-row['command'][2]))
                    pending=row
                    if i%4!=3: continue
                    _render_state(env,row,mujoco)
                    writer.append_data(env.render_rgb('overhead_cam',size=(960,640))); frames+=1; pending=None
                if pending is not None:
                    _render_state(env,pending,mujoco)
                    writer.append_data(env.render_rgb('overhead_cam',size=(960,640))); frames+=1
                if not frames:
                    writer.append_data(env.render_rgb('overhead_cam',size=(960,640))); frames=1
    finally: env.close()
    verification=verify_inference_video(path,frames=frames,fps=25.)
    _curves(folder,times,forces,errors)
    record=dict(file=path.name,sha256=sha256(path),bytes=path.stat().st_size,
        frames=frames,physical_rows=physical_rows,verification=verification,
        rendering='saved_integration_state_only',policy_reexecuted=False,IK_reexecuted=False)
    write_json(folder/'media.json',record)
    return record


def _render_state(env,row,mujoco):
    saved=row['replay_state']; spec=mujoco.mjtState.mjSTATE_INTEGRATION
    values=np.asarray(saved['values'],np.float64)
    if saved['spec']!=int(spec) or values.shape!=(mujoco.mj_stateSize(env.model,spec),) or not np.isfinite(values).all():
        raise ValueError('invalid replay integration state')
    mujoco.mj_setState(env.model,env.data,values,spec)
    mujoco.mj_forward(env.model,env.data); mujoco.mj_rnePostConstraint(env.model,env.data)


def _curves(folder,times,forces,errors):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,5),sharex=True,layout='constrained')
    axes[0].plot(times,forces,color='#138879',linewidth=1); axes[0].axhline(1,color='#8b667b',linestyle='--')
    axes[0].set_ylabel('Filtered wrist force / N'); axes[1].plot(times,errors,color='#396fa8',linewidth=1)
    axes[1].set_ylabel('|Actual Z - command Z| / mm'); axes[1].set_xlabel('Simulation time / s')
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(folder/'force_Z.png',dpi=160); plt.close(fig)


def heatmap(rows,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    counts=np.zeros((7,6),int); totals=np.zeros(6,int)
    for row in rows:
        n=row['target_count']-1; totals[n]+=1
        if row.get('final_collected') is not None: counts[row['final_collected'],n]+=1
    density=counts/np.maximum(1,totals)[None,:]
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
    im=ax.imshow(density,origin='lower',vmin=0,vmax=1,cmap='magma',aspect='auto')
    for i in range(7):
        for j in range(6): ax.text(j,i,f'{counts[i,j]}/{totals[j]}',ha='center',va='center',color='black' if density[i,j]>.65 else 'white')
    ax.set_xticks(range(6),range(1,7)); ax.set_yticks(range(7)); ax.set_xlabel('Target count'); ax.set_ylabel('Measured final count')
    fig.colorbar(im,ax=ax,label='Fraction within target'); fig.savefig(path,dpi=160); plt.close(fig)
