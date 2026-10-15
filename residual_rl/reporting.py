"""Derived metrics only: never infer success from rewards or quantity matches."""
from pathlib import Path
import math

from .storage import write_json


def _mean(rows,key):
    values=[float(row[key]) for row in rows if row.get(key) is not None]
    if any(not math.isfinite(value) for value in values):
        raise ValueError('nonfinite derived report input: '+key)
    return sum(values)/len(values) if values else None


def _aggregate(attempts):
    rows=[row for row in attempts if type(row.get('success')) is bool]
    known=[row for row in rows if row.get('final_collected') is not None]
    return dict(attempts=len(attempts),task_episodes=len(rows),
        administrative_or_unknown=len(attempts)-len(rows),known_quantity_episodes=len(known),
        strict_successes=sum(row['success'] for row in rows),
        strict_success_rate=sum(row['success'] for row in rows)/len(rows) if rows else None,
        quantity_match_rate=sum(row['final_collected']==row['target_count'] for row in known)/len(known)
                            if known else None,
        mean_absolute_error=sum(abs(row['final_collected']-row['target_count']) for row in known)/len(known)
                            if known else None,
        safety_terminations=sum(row.get('outcome')=='safety' for row in rows),
        prefix_only_episodes=sum(row.get('residual_frames',0)==0 for row in rows),
        mean_rl_return=_mean(rows,'rl_return'),
        mean_contact_return=_mean(rows,'credited_contact_return'),
        mean_terminal_reward=_mean(rows,'terminal_reward'),
        mean_wrist_N=_mean(rows,'mean_wrist_N'),
        mean_peak_wrist_N=_mean(rows,'peak_wrist_N'),
        mean_Z_error_m=_mean(rows,'mean_Z_error_m'),
        mean_max_Z_error_m=_mean(rows,'max_Z_error_m'),
        wrist_below_0_1N_fraction=_mean(rows,'wrist_below_0_1N_fraction'),
        mean_XY_residual_m_s=_mean(rows,'mean_XY_residual_m_s'),
        nonzero_residual_fraction=_mean(rows,'nonzero_residual_fraction'))


def compute_report(rows):
    result={}
    for phase in sorted({row['phase'] for row in rows}):
        selected=[row for row in rows if row['phase']==phase]
        result[phase]=_aggregate(selected)
        result[phase]['by_target']={str(goal):_aggregate([r for r in selected if r['target_count']==goal])
                                  for goal in range(1,7)}
    return result


def write_report(root,rows):
    root=Path(root); report=compute_report(rows)
    write_json(root/'metrics_report.json',report)
    lines=['# Residual SAC pilot metrics','',
        'Task failures before residual activation remain in the success denominator. '
        'Administrative interruptions are separate; missing measurements are not zero.','',
        '| Phase | Task episodes | Strict success | Count match | Mean absolute count error | Mean return |',
        '|---|---:|---:|---:|---:|---:|']
    def value(x): return 'not measured' if x is None else f'{x:.4g}'
    for phase,stats in report.items():
        lines.append('| '+' | '.join([phase,str(stats['task_episodes']),
            value(stats['strict_success_rate']),value(stats['quantity_match_rate']),
            value(stats['mean_absolute_error']),value(stats['mean_rl_return'])])+' |')
    for phase,stats in report.items():
        lines.extend(['',f'## {phase}','',
            f'- Credited contact / terminal reward means: {value(stats["mean_contact_return"])} / '
                f'{value(stats["mean_terminal_reward"])}',
            f'- Wrist mean / mean episode peak (N): {value(stats["mean_wrist_N"])} / '
                f'{value(stats["mean_peak_wrist_N"])}',
            f'- Z mean / mean episode maximum error (m): {value(stats["mean_Z_error_m"])} / '
                f'{value(stats["mean_max_Z_error_m"])}',
            f'- Wrist below 0.1 N fraction (contact proxy only): {value(stats["wrist_below_0_1N_fraction"])}',
            f'- XY residual mean (m/s) / nonzero residual fraction: '
                f'{value(stats["mean_XY_residual_m_s"])} / {value(stats["nonzero_residual_fraction"])}',
            f'- Safety / prefix-only task episodes / administrative or unknown attempts: '
                f'{stats["safety_terminations"]} / {stats["prefix_only_episodes"]} / '
                f'{stats["administrative_or_unknown"]}',
            '', '| Target | Task episodes | Strict success | Count match | Count error |',
            '|---|---:|---:|---:|---:|'])
        for goal,target in stats['by_target'].items():
            lines.append('| '+' | '.join([goal,str(target['task_episodes']),
                value(target['strict_success_rate']),value(target['quantity_match_rate']),
                value(target['mean_absolute_error'])])+' |')
    lines.extend(['','Wrist reaction is not table-only pressure; low wrist force does not prove continuous contact.',
        'Force, Z and residual summaries are equal-weight means of the available per-episode metrics; raw series are retained.',
        'Twenty-ms contact debounce ignores shorter touches. Uncapped extra-contact penalties can outrank recovery success.',
        'Six validation layouts do not establish convergence, generalization, causality or statistical significance.',''])
    path=root/'report.md'; path.write_text('\n'.join(lines),encoding='utf-8')
    return path
