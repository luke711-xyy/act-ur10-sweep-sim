"""Read-only PAC-ACT observer; never imports a learner or loads model weights."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import parse_qs, unquote, urlparse

from .storage import read_rows, last_row, active_updates

READ_LIMIT = 16 * 2**20
_process_cache = {}


def _safe_file(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('invalid relative source')
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError('symlink source rejected')
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('source outside run')
    return path


def _document(root, name, warnings):
    try:
        path = _safe_file(root, name)
        if not path.exists(): return {}
        if not path.is_file() or path.stat().st_size > READ_LIMIT:
            raise ValueError('not a bounded regular JSON file')
        value = json.loads(path.read_text())
        if not isinstance(value, dict): raise ValueError('expected JSON object')
        return value
    except (OSError, ValueError) as error:
        warnings.append(f'{name}: {error}')
        return {}


class _Rows(list):
    complete = True


def _rows(root, name, warnings):
    result = _Rows()
    try:
        path = _safe_file(root, name)
        if not path.exists(): return result
        if path.exists() and (not path.is_file() or path.stat().st_size > READ_LIMIT):
            raise ValueError('log exceeds bounded observer read limit')
        with path.open('rb') as stream:
            for number, line in enumerate(stream, 1):
                if not line.endswith(b'\n'): break  # a live, not-yet-committed append
                if not line.strip(): continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict): raise ValueError('expected JSON object')
                    result.append(row)
                except (ValueError, UnicodeError) as error:
                    result.complete = False
                    warnings.append(f'{name}:{number}: {error}; 仅显示可解析记录，总量未知。')
    except (OSError, ValueError) as error:
        result.complete = False
        warnings.append(f'{name}: {error}')
    return result


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _clean(value, warnings):
    if isinstance(value, float) and not math.isfinite(value):
        warnings.append('日志存在非有限值；显示为缺失，不补零。')
        return None
    if isinstance(value, dict): return {k: _clean(v, warnings) for k, v in value.items()}
    if isinstance(value, list): return [_clean(v, warnings) for v in value]
    return value


def _summary(rows):
    tasks = [r for r in rows if isinstance(r.get('success'), bool) and r.get('outcome') != 'interrupted']
    known = [r for r in tasks if _finite(r.get('final_collected')) and _finite(r.get('target_count'))]
    rewards = [r['environment_return'] for r in tasks if _finite(r.get('environment_return'))]
    return dict(attempts=len(rows), task_episodes=len(tasks),
        interrupted=sum(r.get('outcome') == 'interrupted' for r in rows),
        strict_successes=sum(r['success'] for r in tasks),
        strict_success_rate=sum(r['success'] for r in tasks) / len(tasks) if tasks else None,
        mean_absolute_count_error=sum(abs(r['target_count']-r['final_collected']) for r in known) / len(known) if known else None,
        mean_environment_return=sum(rewards) / len(rewards) if rewards else None,
        known_counts=len(known), unknown_counts=len(tasks)-len(known))


def _curves(rows):
    tasks = [r for r in rows if isinstance(r.get('success'), bool) and r.get('outcome') != 'interrupted']
    curves = []
    for i, row in enumerate(tasks):
        summary = _summary(tasks[max(0, i-29):i+1])
        curves.append(dict(update=i+1, episode_return=row.get('environment_return'),
            strict_success_percent=summary['strict_success_rate']*100,
            mean_absolute_count_error=summary['mean_absolute_count_error']))
    return curves


def _episode(root, row, warnings):
    result = dict(row, video_url=None, force_url=None)
    if not all(k in row for k in ('role', 'seed')): return result
    try:
        from .storage import RunStore
        folder = 'episodes/' + RunStore.key(row)
        manifest = _document(root, folder+'/media.json', warnings)
        video = _safe_file(root, folder+'/'+manifest.get('file', 'replay.mp4'))
        if video.suffix == '.mp4' and video.is_file() and manifest.get('bytes') == video.stat().st_size:
            result['video_url'] = '/' + str(video.relative_to(root))
        force = _safe_file(root, folder+'/force_Z.png')
        if force.is_file(): result['force_url'] = '/' + str(force.relative_to(root))
    except (OSError, ValueError, TypeError):
        warnings.append('某条回合媒体路径无效，未提供链接。')
    return result


def _checkpoint_records(root, index, warnings):
    records = []; entries = []
    for field in ('milestones', 'slots'):
        values = index.get(field, [])
        if not isinstance(values, list):
            warnings.append('checkpoint 索引 '+field+' 不是列表；未将其视为已保存。')
        else: entries.extend(values)
    for row in entries:
        if not isinstance(row, dict):
            warnings.append('checkpoint 索引含无效记录；已跳过。'); continue
        result = dict(row, verification='incomplete')
        try:
            path = _safe_file(root, 'checkpoints/'+row['file'])
            if (path.is_file() and path.stat().st_size == row['bytes'] and row['bytes'] > 0
                    and re.fullmatch('[0-9a-f]{64}', row.get('sha256', ''))):
                result['verification'] = 'metadata_and_file_size_only'
        except (OSError, ValueError, KeyError, TypeError): pass
        records.append(result)
    return records


def _process(root):
    """Small cached process metadata only; no signals and no child control."""
    key = str(root.resolve()); cached = _process_cache.get(key)
    if cached and time.monotonic()-cached[0] < 10: return cached[1]
    result = dict(alive=None)
    try:
        import psutil
        result = dict(alive=False)
        for process in psutil.process_iter(['pid', 'cmdline', 'memory_info', 'cpu_times']):
            try:
                info = process.info; command = info['cmdline'] or []
                direct_entry=any(Path(a).name in (
                    'train_pac_act.py', 'resume_no_video_skip_acceptance_baseline.py')
                    for a in command)
                adapter_entry=('--mode' in command and command[command.index('--mode')+1]=='train')
                if not (direct_entry or adapter_entry) or '--out' not in command: continue
                if str(Path(command[command.index('--out')+1]).resolve()) != key: continue
                result = dict(alive=True, pid=info['pid'], rss_bytes=info['memory_info'].rss,
                    cpu_seconds=info['cpu_times'].user+info['cpu_times'].system)
                break
            except (psutil.Error, IndexError, OSError): continue
    except (ImportError, OSError): pass
    _process_cache[key] = (time.monotonic(), result)
    if len(_process_cache) > 32: _process_cache.pop(next(iter(_process_cache)))
    return result


def snapshot(root, page=1, page_size=25, role='all', arm='all'):
    root = Path(root).resolve(); warnings = []
    page_size = max(1, min(100, int(page_size)))
    rows = _rows(root, 'episodes.jsonl', warnings)
    updates_complete = True
    try:
        source = _safe_file(root, 'updates.jsonl')
        if source.exists() and source.stat().st_size > READ_LIMIT:
            raise ValueError('updates log exceeds observer limit')
        updates = [r for r in active_updates(root) if r.get('accepted', True)]
    except (OSError, ValueError) as error:
        updates_complete = False
        warnings.append('更新谱系不可读取：'+str(error)); updates = []
    status = _document(root, 'status.json', warnings)
    design = _document(root, 'preflight_design.json', warnings)
    acceptance = _document(root, 'acceptance.json', warnings)
    run_doc = _document(root, 'run.json', warnings)
    is_preflight = bool(design) or str(status.get('state', '')).startswith('preflight')
    phase = 'preflight' if is_preflight else 'train' if run_doc or status.get('state') in ('train', 'training', 'collecting') else 'unknown'
    counts = _document(root, 'progress.json', warnings)
    for filename, field in [('action_journal.jsonl', 'sampled_actions'), ('block_journal.jsonl', 'sampled_blocks')]:
        journal = last_row(root/filename)
        if journal and _finite(journal.get(field)): counts[field] = max(counts.get(field, 0), journal[field])
    if is_preflight:
        sampled = []; used = []; learning_complete = True
        revisions = {u['update']-1 for u in updates if _finite(u.get('update'))}
        # Only the registered actor opportunities, never frames or physical traces.
        for path in (root/'episodes').glob('preflight_*_actor/learning_blocks.jsonl'):
            blocks = _rows(root, str(path.relative_to(root)), warnings)
            learning_complete = learning_complete and blocks.complete
            for block in blocks:
                if _finite(block.get('length')) and 0 < block['length'] <= 5:
                    sampled.append(block)
                    if block.get('policy_revision') in revisions: used.append(block)
        counts.update(sampled_actions=sum(r['length'] for r in sampled) if learning_complete else None,
            sampled_blocks=len(sampled) if learning_complete else None,
            used_actions=sum(r['length'] for r in used) if learning_complete and updates_complete else None,
            used_blocks=len(used) if learning_complete and updates_complete else None,
            updates=len(updates) if updates_complete else None,
            counter_source='preflight actor learning_blocks + accepted update revisions')
    elif updates and 'updates' not in counts: counts['updates'] = len(updates)
    if 'sampled_actions' in counts:
        needed = [counts.get('sampled_actions'), counts.get('used_actions', 0), counts.get('discarded_actions', 0)]
        counts['pending_actions'] = max(0, needed[0]-needed[1]-needed[2]) if all(_finite(v) for v in needed) else None
    groups = {}
    for row in rows:
        group = row.get('role', 'unknown')+':'+row.get('arm', 'actor')
        groups.setdefault(group, []).append(row)
    effect_groups = {key: dict(summary=_summary(values), metrics=_curves(values)) for key, values in groups.items()}
    selected = [r for r in rows if (role == 'all' or r.get('role') == role) and (arm == 'all' or r.get('arm', 'actor') == arm)]
    pages = max(1, math.ceil(len(selected)/page_size)); page = max(1, min(pages, int(page)))
    checkpoints = _document(root, 'checkpoints/index.json', warnings)
    result = dict(status=status, progress=counts, run=dict(path=str(root), name=root.name, phase=phase,
            training_started=status.get('training_started'),cpu_acceptance=run_doc.get('cpu_acceptance'),
            baseline_validation=run_doc.get('baseline_validation'),
            budget=design.get('budget', run_doc.get('budget', {}))),
        process=_process(root), observed_at=time.time(),
        episodes=[_episode(root, row, warnings) for row in selected[(page-1)*page_size:page*page_size]],
        total_episodes=len(rows) if rows.complete else None, parsed_episodes=len(rows), data_complete=rows.complete,
        filtered_episodes=len(selected) if rows.complete else None, parsed_filtered_episodes=len(selected), overview=_summary(selected),
        page=page, pages=pages, page_size=page_size, updates=updates,
        effect_groups=effect_groups, training_metrics=_curves([r for r in rows if r.get('role') == 'train']),
        checkpoints=checkpoints, checkpoint_records=_checkpoint_records(root, checkpoints, warnings),
        validations=_rows(root, 'validations.jsonl', warnings),
        preflight=dict(completed_rounds=sum('acceptance_round' in u for u in updates), expected_rounds=2,
            opportunity_limit=24, engineering_passed=acceptance.get('engineering_passed'), checks=acceptance.get('checks', {})))
    if status.get('state') == 'preflight_complete' and _finite(acceptance.get('elapsed_s')):
        result['run']['elapsed_seconds'] = acceptance['elapsed_s']
    if not rows.complete:
        for summary in [result['overview'], *[g['summary'] for g in effect_groups.values()]]:
            for field in summary: summary[field] = None
    result = _clean(result, warnings)
    result['warnings'] = list(dict.fromkeys(warnings))
    return result


def create_server(root, host='127.0.0.1', port=8897):
    root = Path(root).resolve()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            try:
                if parsed.path == '/api':
                    query = parse_qs(parsed.query)
                    value = snapshot(root, query.get('page', [1])[0], query.get('size', [25])[0],
                        query.get('role', ['all'])[0], query.get('arm', ['all'])[0])
                    body = json.dumps(value, allow_nan=False).encode()
                    self._headers(200, 'application/json', len(body)); self.wfile.write(body)
                elif parsed.path == '/':
                    body = Path(__file__).with_name('dashboard.html').read_bytes()
                    self._headers(200, 'text/html; charset=utf-8', len(body)); self.wfile.write(body)
                else:
                    candidate = _safe_file(root, unquote(parsed.path).removeprefix('/'))
                    if candidate.suffix not in ('.png', '.svg', '.mp4') or not candidate.is_file():
                        self.send_error(404); return
                    self._media(candidate)
            except (BrokenPipeError, ConnectionResetError): pass
            except (ValueError, OSError, json.JSONDecodeError) as error:
                self.send_error(404 if parsed.path != '/api' else 503, str(error))

        def _headers(self, code, content, size, extra=()):
            self.send_response(code); self.send_header('Content-Type', content)
            self.send_header('Cache-Control', 'no-store'); self.send_header('Content-Length', str(size))
            self.send_header('X-Content-Type-Options', 'nosniff')
            for key, value in extra: self.send_header(key, value)
            self.end_headers()

        def _media(self, path):
            size = path.stat().st_size; start, end = 0, size-1
            extra = [('Accept-Ranges', 'bytes')]; code = 200
            if self.headers.get('Range'):
                match = re.fullmatch(r'bytes=(\d*)-(\d*)', self.headers['Range'])
                if not match or not any(match.groups()):
                    self.send_error(416); return
                first, last = match.groups()
                start = int(first) if first else max(0, size-int(last))
                end = min(size-1, int(last)) if first and last else size-1
                if start > end or start >= size:
                    self._headers(416, 'text/plain', 0, [('Content-Range', f'bytes */{size}')]); return
                code = 206; extra.append(('Content-Range', f'bytes {start}-{end}/{size}'))
            content = {'.png': 'image/png', '.svg': 'image/svg+xml', '.mp4': 'video/mp4'}[path.suffix]
            self._headers(code, content, end-start+1, extra)
            with path.open('rb') as stream:
                stream.seek(start); remaining = end-start+1
                while remaining > 0:
                    block = stream.read(min(65536, remaining))
                    if not block: break
                    self.wfile.write(block); remaining -= len(block)

        def log_message(self, *args): pass
    return ThreadingHTTPServer((host, port), Handler)


def serve(root, host='127.0.0.1', port=8897):
    create_server(root, host, port).serve_forever()
