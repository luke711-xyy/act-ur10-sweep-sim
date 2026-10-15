"""Reviewed standalone probe harness. Original bench/records stay immutable.

Only test-harness safety/evidence handling changes; force control and physics
still use the unmodified project's implementation and approved config copies.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import bench as base


def precommand_finite(filtered,signed,state):
    return bool(np.isfinite(np.r_[filtered,signed,np.asarray(state).ravel()]).all())


def audit_advanced_sample(gate,filtered,signed,state):
    return gate.observe(filtered,signed,state)


def validate_probe(x,y):
    if not math.isfinite(x) or not math.isfinite(y) or x != .42 or y != 0.:
        raise ValueError('only the approved clear desktop point (0.42,0.0) is allowed')


def sensor_audit(rows):
    channels=('filtered_force','signed_compensated_force','raw_tared_force','raw_peak',
              'table_normal_force','tcp','tcp_z','command_z','delta_z',
              'wrist_ft_world','compensated_wrench_world','wrench')
    if not rows:
        return {'all_recorded_sensor_values_finite':None,'sensor_evidence_rows':0}
    finite=all(np.isfinite(np.asarray(row[key],dtype=float)).all()
               for row in rows for key in channels if key in row)
    return {'all_recorded_sensor_values_finite':bool(finite),'sensor_evidence_rows':len(rows),
            'complete_recorded_sensor_schema':all(all(k in row for k in channels) for row in rows)}


def exception_evidence(error,pending,state,tb):
    before=(pending or {}).get('time_before')
    after=state.get('time')
    advanced=(before is not None and after is not None and float(after)>float(before)+1e-10)
    last=state.get('last_recorded_time')
    not_in_trace=bool(advanced and (last is None or float(after)>float(last)+1e-10))
    return {'error':repr(error),'pending':pending,'state':state,'traceback':tb,
            'physical_step_advanced':bool(advanced),'advanced_step_missing_trace_row':not_in_trace}


def snapshot(env,last_recorded_time=None):
    out={'time':None,'last_recorded_time':last_recorded_time}
    if env is None:return out
    try:
        out.update(time=float(env.time),qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy(),
                   sensordata=env.data.sensordata.copy(),wrist_ft_world=env.ee.wrist_ft_world(),
                   compensated_wrench_world=env.ee.compensated_wrench_world(),tcp=env.tcp())
    except Exception as error:
        out['snapshot_error']=repr(error)
    return out


table_measurement=base.table_measurement


def run_case(label,values,*,x=.42,y=0.,hold_seconds=45.):
    from sim.config import load_config
    from sim.environments.sweep_env import SweepEnv
    from sim.controllers.hybrid import Command
    from sim.act.rollout import _force_loop,_overforce_relief_loop,_contact_z_command
    from sim.act.inference_lock import inference_lease

    validate_probe(x,y)
    if not label or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in label):
        raise ValueError('invalid artifact label')
    if not 10<=hold_seconds<=45:raise ValueError('invalid short-probe duration')
    cfg=base.make_config(load_config(str(base.BATCH/'evaluation_config.yaml')),values)
    folder=ROOT/label;folder.mkdir()
    before=base.protected_snapshot()
    fingerprints={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(base.__file__))}
    base.write(folder/'before.json',{'protected_inputs':before,'test_harness_sha256':fingerprints})
    base.write(folder/'configuration.json',cfg.to_dict())
    rows=[];env=None;failure=None;exception=None;invalid=None;nominal=None;pending=None;begin=None
    identity=None;started=time.monotonic()
    print(json.dumps({'event':'waiting_for_slot','case':label,'parameters':values,'reviewed_harness':True}),flush=True)
    try:
        with inference_lease():
            base.verify_protected(before)
            env=SweepEnv(cfg,seed=2120018);env.record_raw_force_peaks=True;env.reset(seed=2120018)
            poses=np.c_[env.component_positions(),env.component_quats()].astype('<f4')
            base.write(folder/'initial_layout.json',{'poses':poses,'sha256':hashlib.sha256(poses.tobytes()).hexdigest()})
            if np.min(np.abs(env.component_positions()[:,0]-x))<.08:raise RuntimeError('probe not clear of parts')
            dt=float(env.control_dt);begin=float(env.time)
            loop=_force_loop(cfg);relief=_overforce_relief_loop(cfg)
            gate=base.SafetyGate(cfg.controller.safe_max_force,cfg.controller.safe_force_dwell_steps)
            phase='approach';command_z=float(cfg.end_effector.z_home);hold_start=None
            print(json.dumps({'event':'fixed_xy_test_started','case':label}),flush=True)
            for _ in range(round((hold_seconds+15.)/dt)):
                signed=float(env.ee.wrist_sign*env.ee.compensated_wrench_world()[2])
                state=snapshot(env,begin+rows[-1]['t'] if rows else None)
                underlying=np.r_[env.data.qpos,env.data.qvel,env.data.sensordata,
                                 env.ee.wrist_ft_world(),env.ee.compensated_wrench_world()]
                if not precommand_finite(float(env.normal_force()),signed,underlying):
                    failure='nonfinite sensor or state';invalid=state;break
                if phase=='approach' and float(env.tcp()[2])<=env.table_top_z+.003:
                    phase='admittance';nominal=float(env.tcp()[2]);hold_start=float(env.time)
                    loop.reset();relief.reset()
                if phase=='approach':command_z=max(env.table_top_z+.002,command_z-.01*dt)
                else:command_z=_contact_z_command(cfg,loop,relief,nominal,float(env.normal_force()))
                pending={'phase':phase,'command':[x,y,command_z,0.],'time_before':float(env.time)}
                env.step_control(Command(x,y,command_z,0.))
                force,geom,parts=table_measurement(env)
                signed=float(env.ee.wrist_sign*env.ee.compensated_wrench_world()[2])
                row={'t':float(env.time)-begin,'advanced':True,'phase':phase,
                     'filtered_force':float(env.normal_force()),'signed_compensated_force':signed,
                     'raw_tared_force':float(env.raw_normal_force()),'raw_peak':float(env.last_raw_force_peak),
                     'table_normal_force':force,'table_geom_contact':geom,'part_contacts':parts,
                     'tcp':env.tcp().tolist(),'tcp_z':float(env.tcp()[2]),'command_z':command_z,
                     'delta_z':0. if nominal is None else command_z-nominal,
                     'wrist_ft_world':env.ee.wrist_ft_world().tolist(),
                     'compensated_wrench_world':env.ee.compensated_wrench_world().tolist(),
                     'wrench':env.wrench().tolist()}
                rows.append(row)
                underlying=np.r_[env.data.qpos,env.data.qvel,env.data.sensordata,
                                 row['wrist_ft_world'],row['compensated_wrench_world'],row['wrench']]
                reason=audit_advanced_sample(gate,row['filtered_force'],signed,underlying)
                if reason:
                    failure=reason;invalid=snapshot(env,begin+rows[-1]['t']);break
                if parts:failure='component contact invalidates isolated bench';break
                if env.unsafe_robot_collision():failure='unsafe robot collision';break
                if hold_start is not None and float(env.time)-hold_start>=hold_seconds:break
            if nominal is None and failure is None:failure='3 mm geometry handoff not reached'
            if hold_start is not None and float(env.time)-hold_start<hold_seconds-.02 and failure is None:
                failure='requested contact observation duration not reached'
    except Exception as error:
        failure=repr(error)
        exception=exception_evidence(error,pending,snapshot(env,begin+rows[-1]['t'] if rows else None),traceback.format_exc())
    finally:
        if env is not None:env.close()
        try:
            identity=base.verify_protected(before)
            current={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(base.__file__))}
            if current!=fingerprints:raise RuntimeError('test harness changed during this attempt')
        except Exception as error:
            identity={'verification_failure':repr(error)};failure=f'{failure}; identity failure: {error}'
        with gzip.open(folder/'trace.json.gz','xt') as f:json.dump(base.json_safe(rows),f,allow_nan=False)
        statistics=base.metrics(rows,dt=1/float(cfg.sim.control_hz))
        statistics.update(sensor_audit(rows),trace_rows=len(rows))
        if exception and exception['advanced_step_missing_trace_row']:statistics['actual_steps']+=1
        result={'case':label,'parameters':values,'fixed_xy':[x,y],'handoff_height_m':.003,
                'hold_seconds_requested':hold_seconds,'latched_nominal_z_m':nominal,'failure':failure,
                'invalid_sample':invalid,'exception':exception,'wall_seconds':time.monotonic()-started,
                'no_act_model_or_task_rollout':True,'no_rgb_or_video':True,
                'identity_after':identity,'statistics':statistics}
        base.write(folder/'summary.json',result)
        print(json.dumps(base.json_safe({k:v for k,v in result.items() if k!='identity_after'})),flush=True)
    if 'verification_failure' in identity:raise RuntimeError('protected inputs changed; stop probes')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label',required=True)
    for key in base.FIELDS:parser.add_argument(f'--{key}',type=float)
    parser.add_argument('--hold-seconds',type=float,default=45.)
    args=parser.parse_args()
    run_case(args.label,{k:getattr(args,k) for k in base.FIELDS if getattr(args,k) is not None},hold_seconds=args.hold_seconds)
