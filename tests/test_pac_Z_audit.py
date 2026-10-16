import gzip
import json
import pytest


@pytest.mark.parametrize('nominal,command,wanted',[(None,.001,False),(.001,.0029,True),(.001,.0031,False)])
def test_acceptance_never_passes_Z_bound_without_a_recorded_nominal(tmp_path,nominal,command,wanted):
    from pac_act.acceptance import _audit_saved_Z
    folder=tmp_path/'episodes'/'one';folder.mkdir(parents=True)
    row=dict(phase='PAC_ACT',z_owner='admittance',command=[0,0,command,0])
    if nominal is not None:row['pac_nominal_z_m']=nominal
    with gzip.open(folder/'physical_trace.jsonl.gz','wt') as f:f.write(json.dumps(row)+'\n')
    checks={};_audit_saved_Z(tmp_path,checks)
    assert checks['Z_total_2mm_bound'] is wanted
