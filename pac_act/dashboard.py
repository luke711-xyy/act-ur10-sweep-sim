"""Read-only local workbench. No command endpoint and no training ownership."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import math
from pathlib import Path
from urllib.parse import parse_qs,urlparse

from .storage import read_rows,last_row


def snapshot(root,page=1,page_size=25):
    root=Path(root); page_size=max(1,min(100,int(page_size)))
    rows=read_rows(root/'episodes.jsonl'); updates=read_rows(root/'updates.jsonl')
    pages=max(1,math.ceil(len(rows)/page_size)); page=max(1,min(pages,int(page)))
    def document(path): return json.loads(path.read_text()) if path.exists() else {}
    counts=document(root/'progress.json')
    journal=last_row(root/'action_journal.jsonl')
    if journal: counts['sampled_actions']=max(counts.get('sampled_actions',0),journal['sampled_actions'])
    blocks=last_row(root/'block_journal.jsonl')
    if blocks: counts['sampled_blocks']=blocks['sampled_blocks']
    training=[]; train_rows=[r for r in rows if r.get('role')=='train']
    for i,row in enumerate(train_rows):
        window=train_rows[max(0,i-29):i+1]
        known=[r for r in window if r.get('final_collected') is not None]
        training.append(dict(update=i+1,episode_return=row.get('environment_return'),
            strict_success_percent=100*sum(r.get('success') is True for r in window)/len(window),
            mean_absolute_count_error=sum(abs(r['target_count']-r['final_collected']) for r in known)/len(known) if known else None))
    return dict(status=document(root/'status.json'),progress=counts,episodes=rows[(page-1)*page_size:page*page_size],
        updates=updates,total_episodes=len(rows),pages=pages,page=page,page_size=page_size,
        training_metrics=training,checkpoints=document(root/'checkpoints/index.json'),validations=read_rows(root/'validations.jsonl'))


def serve(root,host='127.0.0.1',port=8897):
    root=Path(root).resolve()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed=urlparse(self.path)
            try:
                if parsed.path=='/api':
                    query=parse_qs(parsed.query)
                    body=json.dumps(snapshot(root,query.get('page',[1])[0],query.get('size',[25])[0]),allow_nan=False).encode()
                    content='application/json'
                elif parsed.path=='/':
                    body=Path(__file__).with_name('dashboard.html').read_bytes(); content='text/html; charset=utf-8'
                else:
                    candidate=(root/parsed.path.removeprefix('/')).resolve()
                    if not candidate.is_relative_to(root) or candidate.suffix not in ('.png','.svg','.mp4') or not candidate.is_file():
                        self.send_error(404); return
                    body=candidate.read_bytes(); content={'.png':'image/png','.svg':'image/svg+xml','.mp4':'video/mp4'}[candidate.suffix]
                self.send_response(200); self.send_header('Content-Type',content)
                self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(body)))
                self.end_headers(); self.wfile.write(body)
            except (ValueError,OSError,json.JSONDecodeError) as error: self.send_error(503,str(error))
        def log_message(self,*args): pass
    ThreadingHTTPServer((host,port),Handler).serve_forever()
