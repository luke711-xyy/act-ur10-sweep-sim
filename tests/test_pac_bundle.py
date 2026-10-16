import json
from pathlib import Path
import subprocess
import sys

import pytest


def test_exclusion_import_reads_only_manifests_not_images(tmp_path):
    from pac_act.exclusions import extract
    source=tmp_path/'manifest.jsonl'
    source.write_text(json.dumps(dict(seed=4200000,layout_fingerprint='abc',images={'cam':['/does/not/exist.png']}))+'\n')
    result=extract([source])
    assert result['seeds']==[4200000] and result['fingerprints']==['abc']
    assert result['evidence'][0]['bytes']>0


def test_exclusions_include_nested_prior_and_read_complete_active_prefix(tmp_path):
    from pac_act.exclusions import extract
    a=tmp_path/'schedule.json'; a.write_text(json.dumps({'prior_consumed_layouts':[{'seed':3150000,'layout_fingerprint':'x'}]}))
    b=tmp_path/'active.jsonl'; b.write_text('{"seed":5000000,"layout_fingerprint":"y"}\n{"seed":')
    result=extract([a,b])
    assert result['seeds']==[3150000,5000000]
    assert result['fingerprints']==['x','y']
    assert result['evidence'][1]['ignored_incomplete_tail'] is True


def test_copy_of_bundle_source_contains_every_identity_file(tmp_path):
    from pac_act.identity import copy_source_tree,source_files,PROJECT
    destination=tmp_path/'source'
    copy_source_tree(PROJECT,destination)
    assert source_files(destination)==source_files(PROJECT)
    assert (destination/'sim/model/assets').is_dir()
    # Control loader must run with the copied code root, not this checkout.
    p=subprocess.run([sys.executable,'-B','-c','from pac_act.hybrid import load_trial4; f,m=load_trial4(); print(m.PROJECT)'],
        cwd=destination,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert str(destination) in p.stdout
