"""Derived human-readable reports must not replace count error by success."""
import importlib
from types import SimpleNamespace

import pytest


def call(name,*args,**kwargs):
    try: module=importlib.import_module('residual_rl.reporting')
    except ModuleNotFoundError: module=SimpleNamespace()
    function=getattr(module,name,None)
    assert callable(function),name+' missing'
    return function(*args,**kwargs)


def test_report_distinguishes_native_success_from_quantity_match_and_admin_stop():
    rows=[dict(phase='validation_base',target_count=1,success=False,final_collected=1,
        outcome='failure',residual_frames=0,mean_wrist_N=2.,max_Z_error_m=.001,
        credited_contact_return=60.,terminal_reward=-20.,rl_return=40.),
        dict(phase='validation_base',target_count=6,success=True,final_collected=6,
        outcome='success',residual_frames=100,mean_wrist_N=1.,max_Z_error_m=.0005,
        credited_contact_return=10.,terminal_reward=60.,rl_return=70.),
        dict(phase='validation_base',target_count=1,success=None,final_collected=0,
        outcome='budget_truncated',residual_frames=0)]
    report=call('compute_report',rows)
    phase=report['validation_base']
    assert phase['strict_success_rate']==.5 and phase['quantity_match_rate']==1.
    assert phase['administrative_or_unknown']==1 and phase['mean_absolute_error']==0
    assert phase['mean_wrist_N']==1.5 and phase['mean_rl_return']==55.
    assert phase['by_target']['1']['strict_success_rate']==0.


def test_report_does_not_fabricate_unknown_force_or_count_values(tmp_path):
    rows=[dict(phase='train',target_count=1,success=False,final_collected=None,
               outcome='failure',residual_frames=0,mean_wrist_N=None)]
    report=call('compute_report',rows)['train']
    assert report['known_quantity_episodes']==0 and report['mean_absolute_error'] is None
    assert report['mean_wrist_N'] is None
    result=call('write_report',tmp_path,rows)
    assert result.is_file() and (tmp_path/'metrics_report.json').is_file()
