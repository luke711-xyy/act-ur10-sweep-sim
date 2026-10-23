"""Bounded old/new physical contact check before collecting native-scene demos."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np

from sim.act.rollout import _force_loop, _overforce_relief_loop, _contact_z_command
from sim.controllers.hybrid import Command
from sim.environments.rudra_sweep_env import RudraSweepEnv, collection_config
from sim.environments.sweep_env import SweepEnv


def contact_report(rows,target_force=.1):
    steady = [r for r in rows if r['phase']=='contact']
    if len(steady)<100:
        return dict(passed=False,reason='insufficient sustained-contact samples')
    force = np.array([r['force'] for r in steady])
    z = np.array([r['z'] for r in steady])
    report = dict(contact_fraction=float(np.mean([r['contact'] for r in steady])),
                  endpoint_contact_fraction=float(np.mean([r.get('endpoint_contact',r['contact']) for r in steady])),
                  physics_contact_fraction=(float(np.mean([r['physics_contact_fraction'] for r in steady]))
                                            if all('physics_contact_fraction' in r for r in steady) else None),
                  contact_gate_basis='any touch within each 10ms control interval',
                  force_mean=float(force.mean()),force_p95=float(np.quantile(force,.95)),
                  force_max=float(max(r['force'] for r in rows)),
                  raw_peak=float(max(r['raw_peak'] for r in rows)),
                  z_std_mm=float(z.std()*1000),
                  xy_error_p95_mm=float(np.quantile([r['xy_error'] for r in steady],.95)*1000),
                  yaw_error_p95_deg=float(np.degrees(np.quantile([r['yaw_error'] for r in steady],.95))),
                  tilt_max_deg=float(np.degrees(max(r['tilt'] for r in steady))))
    report['passed'] = bool(report['contact_fraction']>=.95 and .3*target_force<=report['force_mean']<=3*target_force
                            and report['force_p95']<=max(1.,2*target_force) and report['force_max']<=5.
                            and report['raw_peak']<=20. and report['z_std_mm']<=.5
                            # Original joint servos have kv/kp=0.1 s lag;
                            # allow 10 mm at <=80 mm/s instead of requiring
                            # 2 mm from a deliberately lagged physical servo.
                            and report['xy_error_p95_mm']<=10.
                            and report['yaw_error_p95_deg']<=5. and report['tilt_max_deg']<=1.)
    return report


def run_check(cfg,backend,out):
    env = (RudraSweepEnv if backend=='native' else SweepEnv)(cfg,seed=410000)
    rows = []
    try:
        env.reset()
        admittance = (_force_loop(cfg,env) if cfg.get_path('scene.contact_profile',None) else _force_loop(cfg))
        relief = _overforce_relief_loop(cfg)
        start = env.time
        latch_time = None
        nominal = None
        z_ref = .1
        motion_start = None
        for tick in range(1300):
            t = env.time-start
            x,y,yaw = .42,0.,0.
            if t<.5:
                phase = 'air'
            elif latch_time is None:
                phase = 'landing'
                speed = float(cfg.controller.z_speed if z_ref>.008 else cfg.controller.expert_contact_z_speed)
                z_ref = max(float(cfg.workspace.z_search_min),z_ref-speed*env.control_dt)
            else:
                age = t-latch_time
                ready = (not hasattr(admittance,'loop') or admittance.loop.ready)
                if age>=2. and ready and motion_start is None:
                    motion_start = t
                phase = 'build' if motion_start is None else 'contact'
                if motion_start is not None:
                    u = t-motion_start
                    x = .42-.04*(1-np.cos(2*np.pi*u/4))
                    y = .02*np.sin(2*np.pi*u/4)
                    yaw = .20*np.sin(2*np.pi*u/4)  # max0.314 < configured0.35rad/s
                z_ref = _contact_z_command(cfg,admittance,relief,nominal,env.normal_force())
            env.step_control(Command(x,y,z_ref,yaw))
            tcp = env.tcp()
            if latch_time is None and tcp[2]<=float(cfg.controller.expert_contact_geometry_tolerance):
                latch_time = env.time-start
                nominal = float(tcp[2])
                admittance.reset()
                relief.reset()
            yaw_error = float(abs(np.arctan2(np.sin(env.ee.tcp_yaw()-yaw),np.cos(env.ee.tcp_yaw()-yaw))))
            normal = env.data.site_xmat[env.ee.tcp_site_id].reshape(3,3)[:,2]
            tilt = float(np.arccos(np.clip(normal[2],-1,1)))
            rows.append(dict(t=float(env.time-start),phase=phase,
                             contact=bool(env.last_brush_contact_any),
                             endpoint_contact=bool(env.brush_table_contact()),
                             physics_contact_fraction=env.last_contact_samples/env.decimation,
                             force=env.normal_force(),raw_peak=float(env.last_raw_force_peak or 0),
                             compensated=env.compensated_normal_force_raw(),z=float(tcp[2]),
                             command_z=z_ref,xy_error=float(np.linalg.norm(tcp[:2]-[x,y])),
                             yaw_error=yaw_error,tilt=tilt))
            if tick%200==0:
                print(json.dumps(dict(event='contact_check',backend=backend,**rows[-1])),flush=True)
            if not np.isfinite(env.data.qpos).all():
                raise RuntimeError('nonfinite native simulation state')
        report = contact_report(rows,float(cfg.controller.desired_force))
        if hasattr(admittance,'loop'):
            report['acquisition_phase'] = admittance.loop.phase
            report['acquisition_transitions'] = admittance.loop.transitions
        report['mujoco_warnings'] = int(sum(env.data.warning.number))
        report['passed'] = report['passed'] and report['mujoco_warnings']==0
        with (out/f'{backend}-contact.csv').open('w') as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
        print(json.dumps(dict(event='contact_result',backend=backend,**report)),flush=True)
        return report,rows
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    parser.add_argument('--profile',choices=['legacy','trial4-low-sole'],default='legacy')
    parser.add_argument('--native-only',action='store_true')
    args = parser.parse_args()
    out = Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cfg = collection_config()
    if args.profile=='trial4-low-sole':
        from sim.controllers.rudra_contact import demonstration_config
        cfg = demonstration_config(cfg)
    reports,series = {},{}
    for backend in (('native',) if args.native_only else ('original','native')):
        reports[backend],series[backend] = run_check(cfg,backend,out)
        (out/'admittance-report.json').write_text(json.dumps(dict(reports=reports,controller=cfg.controller.to_dict(),configuration=cfg.to_dict(),contact_measurement='any brush/table contact during each 10ms control interval; endpoint and 1ms physics-step occupancy are reported separately'),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,1,figsize=(11,6),sharex=True)
    for backend,rows in series.items():
        t = [r['t'] for r in rows]
        axes[0].plot(t,[r['force'] for r in rows],label=backend)
        axes[1].plot(t,[1000*r['z'] for r in rows],label=backend)
    axes[0].axhline(float(cfg.controller.desired_force),color='black',linestyle='--',label=f'target {cfg.controller.desired_force} N')
    axes[0].set(ylabel='Filtered normal force (N)',ylim=(-.02,max(1.,max(r.get('force_max',0) for r in reports.values())*1.1)))
    axes[1].set(ylabel='TCP height (mm)',xlabel='Simulation time (s)',ylim=(-1.5,2))
    for axis in axes:axis.grid(alpha=.25);axis.legend()
    fig.tight_layout();fig.savefig(out/'admittance-comparison.png',dpi=160);plt.close(fig)
    return 0 if reports['native']['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
