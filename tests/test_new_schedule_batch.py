import copy
import hashlib

import numpy as np
import pytest


def paired_plan():
    rows=[]
    for i in range(120):
        poses=np.zeros((6,7),dtype='<f4'); poses[0,0]=i/1000
        rows.append(dict(seed=2120000+i,target_count=i%6+1,key=f'old:{i}',
            initial_object_poses=poses.tolist(),layout_fingerprint=hashlib.sha256(poses.tobytes()).hexdigest()))
    return rows


def test_paired_registry_accepts_only_authorized_heldout_layouts_once():
    from scripts.evaluate_new_schedule_batch import PairedRegistry
    registry=PairedRegistry(paired_plan()); spec=dict(role='comparison',arm='new_schedule_zero',seed=2120000,target_count=1)
    registry.register(spec,paired_plan()[0]['layout_fingerprint'])
    with pytest.raises(ValueError): registry.register(spec,paired_plan()[0]['layout_fingerprint'])
    with pytest.raises(ValueError): registry.register(dict(spec,role='train'),paired_plan()[0]['layout_fingerprint'])
    with pytest.raises(ValueError): registry.register(dict(spec,seed=2120001,target_count=2),'wrong')


@pytest.mark.parametrize('change',['duplicate','pose','target'])
def test_plan_rejects_nonpaired_or_corrupt_layouts(change):
    from scripts.evaluate_new_schedule_batch import validate_plan
    rows=paired_plan()
    if change=='duplicate':rows[1]=copy.deepcopy(rows[0])
    elif change=='pose':rows[0]['initial_object_poses'][0][0]=.9
    else:rows[0]['target_count']=6
    with pytest.raises(ValueError):validate_plan(rows)


def test_summary_keeps_unknowns_failures_and_strict_success_separate():
    from scripts.evaluate_new_schedule_batch import summarize
    data=summarize([dict(target_count=1,final_collected=1,success=False,outcome='failure'),
                    dict(target_count=1,final_collected=1,success=True,outcome='success'),
                    dict(target_count=2,final_collected=None,success=None,outcome='interrupted')])
    assert data['episodes']==3 and not data['complete']
    assert data['strict_successes']==1 and data['exact_count_matches']==2
    assert data['unknown_counts']==1 and data['count_matrix'][0][1]==0
    assert data['count_matrix'][1][0]==2
    assert data['by_target']['1']['strict_success_rate']==.5
