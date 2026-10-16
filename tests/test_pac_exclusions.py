import json
from pathlib import Path
import subprocess
import sys


def test_exclusion_generator_preserves_explicit_unavailable_history_status(tmp_path):
    old_plan=tmp_path/'old-plan.json'
    old_plan.write_text(json.dumps([
        {'seed':2120000+i,'layout_fingerprint':f'{i+1:064x}'} for i in range(120)
    ]))
    dataset=tmp_path/'manifest.jsonl'
    dataset.write_text(json.dumps({'seed':4200000,'layout_fingerprint':'dataset-layout'})+'\n')
    schedule=tmp_path/'residual-seed-schedule.json'
    schedule.write_text(json.dumps({'training':[{'seed':3300000}],
        'validation':[{'seed':3140000}],'preflight':[{'seed':3152300}]}))
    ledger=tmp_path/'residual-history-inventory.json'
    ledger.write_text(json.dumps({
        'record_type':'history_inventory',
        'historical_status':'legacy_attempt_ledger_not_found_on_host',
        'historical_attempts_reconstructed':False,
        'missing_legacy_export_root':'/Users/legacy/exports',
        'searched_roots':['/project/runs','/project/worktrees'],
    }))
    output=tmp_path/'exclusions.json'
    script=Path(__file__).parents[1]/'scripts/prepare_pac_act_exclusions.py'
    result=subprocess.run([sys.executable,'-B',str(script),'--old-plan',str(old_plan),
        '--dataset-manifest',str(dataset),'--residual-schedule',str(schedule),
        '--residual-ledger',str(ledger),'--out',str(output)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    exclusions=json.loads(output.read_text())
    assert exclusions['complete_known_inputs'] is True
    assert exclusions['seeds']==sorted([2120000+i for i in range(120)]
        +[4200000,3300000,3140000,3152300])
    assert exclusions['residual_history']['historical_status']=='legacy_attempt_ledger_not_found_on_host'
    assert exclusions['residual_history']['historical_attempts_reconstructed'] is False
