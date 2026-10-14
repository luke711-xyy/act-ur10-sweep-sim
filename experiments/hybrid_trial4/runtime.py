"""Import the preserved trial-4 runtime without private export directories.

Importing constructs no policy, environment, rollout or training process.
The historical vendor scripts are dependencies, not supported standalone CLIs.
"""
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[2]
VENDOR=Path(__file__).resolve().parent/'vendor'
CANDIDATE=VENDOR/'admittance_2mm_target_gain_search_goal1_20261014'


@lru_cache(maxsize=1)
def load_runtime():
    def load(path,name):
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        spec.loader.exec_module(module);return module
    sys.path.insert(0,str(PROJECT))
    previous=sys.modules.get('phases')
    try:
        phase=load(CANDIDATE/'phases.py','portable_hybrid_trial4_phases')
        sys.modules['phases']=phase
        runtime=load(CANDIDATE/'runner.py','portable_hybrid_trial4_runner')
    finally:
        if previous is None:sys.modules.pop('phases',None)
        else:sys.modules['phases']=previous
    runtime.TRIAL_SPEC=phase.AdmittanceSpec(1.,(10.,400.,10.),(10.,800.,10.))
    return runtime
