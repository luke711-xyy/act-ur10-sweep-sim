"""Local MuJoCo robot-learning workbench."""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
from threading import RLock


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ACT MuJoCo Sweep Workbench</title>
<style>
:root{--bg:#10151b;--panel:#18222b;--line:#30434f;--text:#eaf1f4;--muted:#94aab4;--accent:#62d6b0}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% 0,#20373b 0,#10151b 42%);color:var(--text);font:14px/1.45 Inter,system-ui,sans-serif}
main{max-width:1500px;margin:auto;padding:28px}header{display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:22px}h1{margin:0;font-size:27px;letter-spacing:-.03em}h2{font-size:15px;margin:0 0 14px;color:#fff}.muted,.hint{color:var(--muted)}.eyebrow{color:var(--accent);text-transform:uppercase;font-size:11px;letter-spacing:.14em}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}.panel{background:linear-gradient(145deg,rgba(30,43,53,.96),rgba(20,29,36,.96));border:1px solid var(--line);border-radius:13px;padding:16px}.simulation{grid-column:span 8}.demonstrations{grid-column:span 4}.training,.inference,.parameters{grid-column:span 4}.wide,.telemetry{grid-column:1/-1}
.views{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.view{margin:0;background:#0e1419;border:1px solid #263742;border-radius:9px;overflow:hidden}.view img{display:block;width:100%;aspect-ratio:1.15;object-fit:cover;background:#0a0f12}.view figcaption{padding:8px 10px;color:var(--muted);font-size:12px}.tag{float:right;color:var(--accent);font-size:10px;text-transform:uppercase}
.signal-card{margin-top:12px;background:#0e1419;border:1px solid #263742;border-radius:9px;padding:10px}.signal-header{display:flex;justify-content:space-between;gap:10px;align-items:baseline;color:#cfe1e5;font-size:12px}.signal-header span:last-child{color:var(--muted);font-size:11px}.fz-chart{display:block;width:100%;height:auto;margin-top:7px;overflow:visible}.fz-grid{stroke:#2a3d46;stroke-width:1}.fz-path{fill:none;stroke:var(--accent);stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round}.fz-cursor{stroke:#f3c969;stroke-width:1.5;stroke-dasharray:4 3}.signal-empty{fill:var(--muted);font-size:12px}.axis{display:flex;justify-content:space-between;color:#718b94;font-size:10px}
button,select,input{font:inherit;color:var(--text);background:#23343d;border:1px solid #3d5863;border-radius:7px;padding:8px 10px}button{cursor:pointer}button:hover{border-color:var(--accent);color:#fff}button.primary{background:#237c6b;border-color:#4fc19e}button.danger{border-color:#915454;color:#ffb2b2}input,select{width:100%;margin:4px 0 9px}label{display:block;color:var(--muted);font-size:12px}.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.row>*{flex:0 0 auto}.meta{background:#0e1419;border-radius:8px;padding:10px;min-height:74px;color:#a9c0c9;white-space:pre-wrap}.job{background:#0e1419;border-radius:8px;padding:9px;margin-top:10px;font-size:12px}.job strong{color:var(--accent)}.job pre{max-height:100px;overflow:auto;margin:7px 0 0;color:#b9c8ce}.status{margin-top:13px;display:flex;gap:9px;flex-wrap:wrap}.pill{border:1px solid var(--line);border-radius:99px;padding:5px 9px;color:var(--muted);font-size:12px}.ok{color:var(--accent)}.bad{color:#f08f8f}a{color:var(--accent)}
.episode-filters{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:8px 0 10px}.episode-filters select{margin-bottom:0}.episode-list{max-height:220px;overflow:auto;margin:8px 0 12px;border:1px solid #263742;border-radius:9px;background:#0e1419}.episode-row{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;width:100%;border:0;border-bottom:1px solid #22323b;border-radius:0;background:transparent;text-align:left;padding:9px 10px}.episode-row:last-child{border-bottom:0}.episode-row.selected{background:#203a3a;color:#fff}.episode-row strong{display:block;overflow:hidden;text-overflow:ellipsis}.episode-row small{color:var(--muted)}.episode-row .result{align-self:center;font-size:11px;color:var(--accent)}.episode-row .result.failure{color:#f0a0a0}.generation-panel{border-top:1px solid #263742;padding-top:10px}.generation-grid{display:grid;grid-template-columns:1.1fr 1.1fr .7fr;gap:8px;align-items:end}.generation-grid select,.generation-grid input{margin-bottom:0}.generation-grid label:last-child{grid-column:1/-1}.control-grid{display:grid;grid-template-columns:1fr;gap:8px;align-items:end}.telemetry-head{display:flex;align-items:end;justify-content:space-between;gap:14px}.telemetry-head label{width:210px}.telemetry-layout{display:grid;grid-template-columns:minmax(0,2fr) minmax(280px,1fr);gap:14px}.telemetry-chart{background:#0e1419;border:1px solid #263742;border-radius:9px;padding:10px}.telemetry-chart svg{display:block;width:100%;height:auto}.telemetry-path{fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}.telemetry-legend{display:flex;gap:12px;flex-wrap:wrap;margin:4px 0 0;color:var(--muted);font-size:11px}.legend-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}.frame-table{width:100%;border-collapse:collapse;font-size:12px}.frame-table th,.frame-table td{padding:7px 8px;border-bottom:1px solid #263742;text-align:left;vertical-align:top}.frame-table th{width:34%;color:var(--muted);font-weight:500}.frame-table td{font-variant-numeric:tabular-nums;word-break:break-word}.frame-table tr:last-child th,.frame-table tr:last-child td{border-bottom:0}.frame-detail{background:#0e1419;border:1px solid #263742;border-radius:9px;overflow:hidden}.phase-badge{display:inline-block;border:1px solid #3d5863;border-radius:99px;padding:2px 7px;color:#cfe1e5}.sr-only{position:absolute!important;width:1px!important;height:1px!important;padding:0!important;margin:-1px!important;overflow:hidden!important;clip:rect(0,0,0,0)!important;white-space:nowrap!important;border:0!important}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:2px}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;transition:none!important}}
@media(max-width:1000px){.simulation,.demonstrations,.training,.inference,.parameters{grid-column:1/-1}.telemetry-layout{grid-template-columns:1fr}}@media(max-width:650px){.views{grid-template-columns:1fr}.control-grid,.episode-filters,.generation-grid{grid-template-columns:1fr}.telemetry-head{align-items:stretch;flex-direction:column}.telemetry-head label{width:100%}}
</style></head><body><main>
<header><div><div class="eyebrow">local embodied-learning lab</div><h1>ACT · UR10 MuJoCo Sweep Workbench</h1><div class="muted">Observe the simulator, inspect expert trajectories, then launch reproducible local ACT jobs.</div></div><div class="row"><button onclick="resetScene()">Reset scene</button><button onclick="refreshAll()">Refresh</button></div></header>
<div class="grid">
<section class="panel simulation"><h2>Simulation <span class="muted">· live state / playback canvas</span></h2><div class="views">
<figure class="view"><img id="overheadImage"><figcaption>Perception / overhead</figcaption></figure>
<figure class="view"><img id="wristImage"><figcaption>ACT input / wrist</figcaption></figure>
<figure class="view"><img id="inspectionImage"><figcaption>斜前上方观察 <span class="tag">inspection-only</span></figcaption></figure>
</div><div id="healthPills" class="status"></div><div class="signal-card"><div class="signal-header"><span>Fz · full episode trace</span><span id="fzMeta">Select an episode</span></div><svg id="fzChart" class="fz-chart" viewBox="0 0 640 170" role="img" aria-label="Full episode Fz force trace"><line class="fz-grid" x1="24" y1="152" x2="632" y2="152"></line><line class="fz-grid" x1="24" y1="86" x2="632" y2="86"></line><path id="fzPath" class="fz-path" d=""></path><line id="fzCursor" class="fz-cursor" x1="24" y1="18" x2="24" y2="152"></line><text id="fzEmpty" class="signal-empty" x="320" y="84" text-anchor="middle">No force trace saved</text></svg><div class="axis"><span>0 s</span><span id="fzEndTime">—</span><span>force / N</span></div></div></section>
<section class="panel demonstrations"><h2>Demonstrations <span class="muted">· episode playback</span></h2>
<div class="episode-filters"><label>目标数量筛选<select id="episodeTargetFilter" onchange="applyEpisodeFilters()"><option value="all">all</option><option value="1">1</option><option value="2">2</option><option value="3">3</option><option value="4">4</option><option value="5">5</option><option value="6">6</option></select></label><label>结果筛选<select id="episodeStatusFilter" onchange="applyEpisodeFilters()"><option value="all">all</option><option value="success">success</option><option value="failed">failed</option></select></label></div>
<select id="episodeSelect" class="sr-only" onchange="selectEpisode(this.value)" aria-label="Selected demonstration"></select><div id="episodeList" class="episode-list" aria-label="Demonstration episode list"></div>
<div class="generation-panel"><div class="hint" style="margin-bottom:8px">生成示教数据 · 成功数据使用物理质量门；失败数据使用可解释的故障模式。目标 6 固定只有 6 个零件，因此错误数量只生成少扫。</div><div class="generation-grid"><label>目标数量<select id="previewTarget"><option>1</option><option>2</option><option selected>3</option><option>4</option><option>5</option><option>6</option></select></label><label>生成结果<select id="previewOutcome" onchange="toggleFailureMode()"><option value="success">success · 成功示教</option><option value="failed">failed · 失败示教</option></select></label><label>生成条数<input id="previewCount" type="number" min="1" max="100" value="1"></label><label id="previewFailureModeWrap">失败类型<select id="previewFailureMode"><option value="">自动轮换</option><option value="stall_outside_tray">stall_outside_tray · 卡在收集区左右壁外</option><option value="wrong_count">wrong_count · 错误数量（自动多扫/少扫）</option><option value="misroute">misroute · 路径偏离</option></select></label></div><button class="primary" onclick="makePreview()">生成示教数据</button></div>
<div class="row"><button onclick="togglePlay()" id="playButton">Play</button><button onclick="stepFrame(-1)" aria-label="Previous frame">‹</button><button onclick="stepFrame(1)" aria-label="Next frame">›</button></div>
<div class="hint" style="margin-top:10px">Play drives the three views above frame by frame; the inspection view remains observation-only.</div>
<input id="frameSlider" type="range" min="0" max="0" value="0" oninput="seekFrame(this.value)"><div id="episodeMeta" class="meta">No episode loaded.</div>
</section>
<section class="panel telemetry"><div class="telemetry-head"><div><h2>Frame inspector <span class="muted">· synchronized observation and action traces</span></h2><div class="hint">拖动上方帧滑块，图线游标、三路画面和右侧数值会同步更新。</div></div><label>Signal group<select id="signalGroup" onchange="renderTelemetry(Number($('frameSlider').value||0))"><option value="tcp">TCP pose</option><option value="action">ACT label</option><option value="zcontrol">Policy / applied Z</option><option value="wrench">6D wrench</option><option value="joints">UR10 joints</option><option value="counts">Counts & contact</option></select></label></div><div class="telemetry-layout"><div class="telemetry-chart"><svg viewBox="0 0 900 260" role="img" aria-label="Selected episode signals"><g id="telemetryGrid"></g><g id="telemetryPaths"></g><line id="telemetryCursor" class="fz-cursor" x1="42" y1="18" x2="42" y2="224"></line><text id="telemetryEmpty" class="signal-empty" x="450" y="126" text-anchor="middle">Select an episode with detailed signals</text></svg><div id="telemetryLegend" class="telemetry-legend"></div><div class="axis"><span id="telemetryStart">0 s</span><span id="telemetryRange">—</span><span id="telemetryEnd">—</span></div></div><div class="frame-detail"><table class="frame-table"><tbody id="frameData"><tr><th>Frame</th><td>—</td></tr></tbody></table></div></div></section>
<section class="panel training"><h2>Training <span class="muted">· local ACT</span></h2>
<label>Dataset directory<input id="trainDataset" value="runs/act_dataset"></label><label>Output directory<input id="trainOut" value="runs/act_model"></label><label>Steps (optional)<input id="trainSteps" type="number" min="1" placeholder="use config default"></label>
<div class="row"><button class="primary" onclick="startTraining()">Start training job</button><button class="danger" onclick="stopLatest('train')">Stop</button></div><div id="trainJob" class="job">No training job.</div>
</section>
<section class="panel inference"><h2>Inference <span class="muted">· MuJoCo rollout</span></h2>
<label>Model checkpoint<input id="inferModel" value="runs/act_model"></label><label>Exact target count (of 6)<select id="inferTarget"><option>1</option><option>2</option><option selected>3</option><option>4</option><option>5</option><option>6</option></select></label><label>Seed<input id="inferSeed" type="number" value="0"></label><label class="check"><input id="inferRandomize" type="checkbox" checked> Randomize layout seed on each run</label><label class="check"><input id="inferPreview" type="checkbox" checked> Simulation preview · allow late inference</label>
<div class="row"><button class="primary" onclick="startInference()">Run inference</button><button class="danger" onclick="stopLatest('inference')">Stop</button></div><div id="inferJob" class="job">No inference job.</div>
</section>
<section class="panel parameters"><h2>Parameters <span class="muted">· explicit, editable knobs</span></h2>
<div class="grid"><div style="grid-column:span 6"><label>Desired force (N)<input id="pForce" type="number" step=".05" min="0" max="5"></label><label>Sweep speed (m/s)<input id="pSpeed" type="number" step=".01" min=".01" max="1"></label><label>Target count N (six parts fixed)<input id="pTarget" type="number" min="1" max="6"></label></div><div style="grid-column:span 6"><label>Safe max force (N)<input id="pSafe" type="number" step=".05" min=".1" max="20"></label><label>Episode max time (s)<input id="pTime" type="number" step="1" min="1" max="600"></label><label>ACT batch size<input id="pBatch" type="number" min="1" max="256"></label></div></div>
<button class="primary" onclick="applyParameters()">Apply and reset scene</button><div id="parameterStatus" class="hint"></div>
</section>
<section class="panel wide"><h2>Dataset tooling <span class="muted">· LeRobot-compatible export</span></h2><div class="row"><a href="https://huggingface.co/spaces/lerobot/visualize_dataset" target="_blank">Open official LeRobot Dataset Visualizer ↗</a><span class="hint">Rerun shows synchronized cameras, state and action; Foxglove adds a seekable play/pause timeline. Hub Dataset Viewer remains useful for row filtering and Parquet inspection.</span></div><div id="globalStatus" class="hint" style="margin-top:10px"></div></section>
</div></main><script>
const $=id=>document.getElementById(id);let episodes=[];let current=null;let selectedEpisodeId=null;let lastInferenceReplayId=null;let previewJobId=null;let previewJobHandled=null;let timer=null;let playbackBusy=false;let signalSeries=null;let currentFrameData=null;let seekSerial=0;let seekAbort=null;let signalSerial=0;let signalAbort=null;let selectionSerial=0;
function image(id,b64){$(id).src=b64?'data:image/png;base64,'+b64:''}
async function json(url,opts){const r=await fetch(url,opts);const body=await r.text();let data=null;try{data=body?JSON.parse(body):null}catch(_){if(!r.ok)throw new Error(`HTTP ${r.status}: ${body.slice(0,180)||r.statusText}`);throw new Error(`Invalid JSON from ${url}`)}if(!r.ok)throw new Error(data?.detail||data?.message||`HTTP ${r.status}: ${r.statusText}`);return data}
function esc(value){return String(value??'').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]))}
async function refreshAll(){try{const [h,f]=await Promise.all([json('/api/health'),json('/api/frames')]);if(!timer&&!current){image('overheadImage',f.overhead);image('wristImage',f.wrist);image('inspectionImage',f.inspection)}$('healthPills').innerHTML=`<span class="pill ok">sim ${h.time.toFixed(2)} s</span><span class="pill">full ${h.collected}/${h.components} · goal ${h.target_count}</span><span class="pill">Fz ${h.normal_force.toFixed(2)} N</span><span class="pill">${h.model}</span>`;await loadEpisodes();await refreshJobs()}catch(e){$('globalStatus').textContent='Dashboard error: '+e.message;$('globalStatus').className='bad'}}
function filteredEpisodes(){const target=$('episodeTargetFilter')?.value||'all';const status=$('episodeStatusFilter')?.value||'all';return episodes.filter(e=>(target==='all'||String(e.target_count)===target)&&(status==='all'||(status==='success'?Boolean(e.success):!Boolean(e.success))))}
function renderEpisodeList(){const visible=filteredEpisodes().slice().reverse();$('episodeList').innerHTML=visible.length?visible.map(e=>`<button class="episode-row ${e.episode_id===selectedEpisodeId?'selected':''}" data-episode-id="${esc(e.episode_id)}" onclick="selectEpisode('${esc(e.episode_id)}')" oncontextmenu="deleteEpisode(event,'${esc(e.episode_id)}');return false;" title="右键删除这条记录"><span><strong>${esc(e.episode_id)}</strong><small>${esc(e.split)} · exact ${e.collected??'?'} / ${e.target_count??'?'} · ${e.success?'success':'failed'}${e.failure_mode?' · '+esc(e.failure_mode):''} · ${e.length} frames · ${esc(e.planner_strategy||'legacy')} / ${esc(e.planner_status||'unknown')}</small></span><span class="result ${e.success?'':'failure'}">${e.success?'SUCCESS':'FAIL'}</span></button>`).join(''):'<div class="hint" style="padding:12px">当前筛选没有示教数据。</div>'}
async function deleteEpisode(event,id){event.preventDefault();if(!confirm(`删除示教记录 ${id}？此操作不可撤销。`))return;try{if(timer){clearInterval(timer);timer=null;$('playButton').textContent='Play'}if(selectedEpisodeId===id){selectedEpisodeId=null;current=null}await json(`/api/episodes/${encodeURIComponent(id)}`,{method:'DELETE'});await loadEpisodes();$('globalStatus').textContent=`已删除示教记录 · ${id}`}catch(e){$('globalStatus').textContent='删除失败: '+e.message;$('globalStatus').className='bad'}}
function applyEpisodeFilters(){renderEpisodeList();const visible=filteredEpisodes();if(visible.length&&!visible.some(e=>e.episode_id===selectedEpisodeId))selectEpisode(visible[visible.length-1].episode_id)}
async function loadEpisodes(){const data=await json('/api/episodes');episodes=data;const select=$('episodeSelect');const old=select.value;select.innerHTML=data.length?data.map(e=>`<option value="${esc(e.episode_id)}">${e.preview?'preview':'dataset'} · ${esc(e.episode_id)} · ${e.success?'success':'failure'}</option>`).join(''):'<option value="">No episodes yet</option>';renderEpisodeList();const visible=filteredEpisodes();if(visible.length){const preferred=selectedEpisodeId||old;const id=visible.some(e=>e.episode_id===preferred)?preferred:visible[visible.length-1].episode_id;select.value=id;if(id!==current?.episode_id)await selectEpisode(id)}}
async function loadSignals(id){const requestId=++signalSerial;if(signalAbort)signalAbort.abort();const controller=new AbortController();signalAbort=controller;signalSeries=null;try{const next=await json(`/api/episodes/${encodeURIComponent(id)}/signals`,{signal:controller.signal});if(requestId!==signalSerial||current?.episode_id!==id||controller.signal.aborted)return;signalSeries=next}catch(e){if(e.name==='AbortError')return;if(requestId!==signalSerial||current?.episode_id!==id)return;signalSeries=null}finally{if(signalAbort===controller)signalAbort=null}if(requestId===signalSerial&&current?.episode_id===id&&!controller.signal.aborted){renderFz(0);renderTelemetry(0)}}
function renderFz(frameIndex=0){const path=$('fzPath'),cursor=$('fzCursor'),empty=$('fzEmpty'),meta=$('fzMeta'),end=$('fzEndTime');const t=signalSeries?.t||[],values=(signalSeries?.fz||[]).map(Number);const n=Math.min(t.length,values.length);if(!n){path.setAttribute('d','');cursor.setAttribute('x1','24');cursor.setAttribute('x2','24');empty.style.display='block';meta.textContent='No force trace saved';end.textContent='—';return}empty.style.display='none';const width=608,height=134,padX=24,padY=18;const lo=Math.min(0,...values.slice(0,n));const hi=Math.max(1,...values.slice(0,n));const span=Math.max(hi-lo,1);const d=values.slice(0,n).map((value,i)=>{const x=padX+i/(Math.max(1,n-1))*width;const y=padY+(hi-value)/span*height;return `${i?'L':'M'}${x.toFixed(2)},${y.toFixed(2)}`}).join(' ');path.setAttribute('d',d);const frameCount=Math.max(1,(current?.length||1)-1);const cursorIndex=Math.round(Math.max(0,Math.min(Number(frameIndex),frameCount))/frameCount*(n-1));const x=padX+cursorIndex/Math.max(1,n-1)*width;cursor.setAttribute('x1',x.toFixed(2));cursor.setAttribute('x2',x.toFixed(2));meta.textContent=`${n} samples · peak ${Math.max(...values.slice(0,n)).toFixed(2)} N`;end.textContent=`${Number(t[n-1]||0).toFixed(2)} s`}
const palette=['#62d6b0','#f3c969','#7fb5ff','#ef8ba3','#b39cff','#8bd1e8'];
function signalGroup(){const kind=$('signalGroup').value;const rows=signalSeries||{};if(kind==='tcp')return {labels:['x','y','z','yaw'],values:rows.tcp||[],unit:'m / rad'};if(kind==='action')return {labels:['dx','dy','dz','dyaw'],values:rows.action||[],unit:'m / rad per 40 ms'};if(kind==='zcontrol')return {labels:['policy z','applied z'],values:(rows.policy_z||[]).map((value,i)=>[Number(value),Number(rows.applied_z?.[i]??value)]),unit:'m'};if(kind==='wrench')return {labels:['Fx','Fy','Fz','Tx','Ty','Tz'],values:rows.wrench||[],unit:'N / Nm'};if(kind==='joints')return {labels:['q1','q2','q3','q4','q5','q6'],values:rows.joint_position||[],unit:'rad'};const env=rows.environment_state||[];return {labels:['total','target','collected','contact'],values:env.map((row,i)=>[Number(row[0]||0),Number(row[1]||0),Number(row[2]||0),rows.contact_latched?.[i]?1:0]),unit:'count / boolean'}}
function renderTelemetry(frameIndex=0){const paths=$('telemetryPaths'),grid=$('telemetryGrid'),empty=$('telemetryEmpty'),legend=$('telemetryLegend'),cursor=$('telemetryCursor');paths.innerHTML='';grid.innerHTML='';legend.innerHTML='';const group=signalGroup(),rows=group.values||[],n=rows.length;if(!n){empty.style.display='block';$('telemetryRange').textContent='No detailed signals';$('telemetryEnd').textContent='—';return}empty.style.display='none';const width=824,height=206,x0=42,y0=18;for(let i=0;i<5;i++){const y=y0+i*height/4;const line=document.createElementNS('http://www.w3.org/2000/svg','line');line.setAttribute('x1',x0);line.setAttribute('x2',x0+width);line.setAttribute('y1',y);line.setAttribute('y2',y);line.setAttribute('class','fz-grid');grid.appendChild(line)}const flat=rows.flatMap(row=>row.map(Number)).filter(Number.isFinite);let lo=Math.min(...flat),hi=Math.max(...flat);if(lo===hi){lo-=1;hi+=1}const margin=(hi-lo)*.08;lo-=margin;hi+=margin;group.labels.forEach((label,j)=>{const values=rows.map(row=>Number(row[j]||0));const d=values.map((value,i)=>{const x=x0+i/Math.max(1,n-1)*width;const y=y0+(hi-value)/(hi-lo)*height;return `${i?'L':'M'}${x.toFixed(2)},${y.toFixed(2)}`}).join(' ');const path=document.createElementNS('http://www.w3.org/2000/svg','path');path.setAttribute('d',d);path.setAttribute('class','telemetry-path');path.setAttribute('stroke',palette[j%palette.length]);paths.appendChild(path);legend.insertAdjacentHTML('beforeend',`<span><i class="legend-dot" style="background:${palette[j%palette.length]}"></i>${label}</span>`)});const index=Math.max(0,Math.min(Number(frameIndex),n-1));const x=x0+index/Math.max(1,n-1)*width;cursor.setAttribute('x1',x);cursor.setAttribute('x2',x);const t=signalSeries?.t||[];$('telemetryStart').textContent=`${Number(t[0]||0).toFixed(2)} s`;$('telemetryEnd').textContent=`${Number(t[n-1]||0).toFixed(2)} s`;$('telemetryRange').textContent=`${lo.toFixed(3)} … ${hi.toFixed(3)} ${group.unit}`}
function fmt(values,digits=4){return (values||[]).map(v=>Number(v).toFixed(digits)).join(', ')}
function renderFrameData(data){currentFrameData=data;if(!data){$('frameData').innerHTML='<tr><th>Frame</th><td>Legacy episode: no per-frame record</td></tr>';return}const env=data.environment_state||[];const objects=(data.objects||[]).map(o=>`#${o.index+1} [${fmt(o.position,3)}]${o.collected?' ✓':''}`).join('<br>')||'legacy episode: not logged';$('frameData').innerHTML=`<tr><th>Frame / time</th><td>${data.frame+1}/${current.length} · ${Number(data.t).toFixed(3)} s</td></tr><tr><th>Phase</th><td><span class="phase-badge">${esc(data.phase)}</span> · ${data.policy_mask?'ACT label valid':'stability / unload masked'}</td></tr><tr><th>Contact latch</th><td>${data.contact_latched?'latched':'open'} · Fz ${Number(data.normal_force).toFixed(3)} N</td></tr><tr><th>Count</th><td>${data.fully_collected}/${data.total_count} fully inside · exact goal ${data.target_count}</td></tr><tr><th>TCP [x,y,z,yaw]</th><td>${fmt(data.tcp_pose)}</td></tr><tr><th>ACT [dx,dy,dz,dyaw]</th><td>${fmt(data.action,5)}</td></tr><tr><th>Policy target [x,y,z,yaw]</th><td>${fmt(data.policy_reference)}</td></tr><tr><th>Applied [x,y,z,yaw]</th><td>${fmt(data.reference)}</td></tr><tr><th>Z ownership</th><td>${esc(data.z_owner)} · policy ${Number(data.policy_z).toFixed(4)} m · applied ${Number(data.applied_z).toFixed(4)} m</td></tr><tr><th>Wrench [F,T]</th><td>${fmt(data.wrench,3)}</td></tr><tr><th>Joints q1…q6</th><td>${fmt(data.joint_position,3)}</td></tr><tr><th>Objects [x,y,z]</th><td>${objects}</td></tr><tr><th>Environment</th><td>${fmt(env,0)}</td></tr>`}
async function selectEpisode(id){if(!id)return;const selectionId=++selectionSerial;if(timer){clearInterval(timer);timer=null;$('playButton').textContent='Play'}if(seekAbort)seekAbort.abort();seekSerial++;selectedEpisodeId=id;current=episodes.find(e=>e.episode_id===id)||null;if(!current)return;$('episodeSelect').value=id;renderEpisodeList();$('frameSlider').max=Math.max(0,current.length-1);$('frameSlider').value=0;await loadSignals(id);if(selectionId!==selectionSerial||current?.episode_id!==id)return;await seekFrame(0)}
async function seekFrame(index){if(!current)return;const episodeId=current.episode_id;const requestId=++seekSerial;if(seekAbort)seekAbort.abort();const controller=new AbortController();seekAbort=controller;index=Math.max(0,Math.min(Number(index),current.length-1));if(requestId!==seekSerial||current?.episode_id!==episodeId)return;$('frameSlider').value=index;try{const base=`/api/episodes/${encodeURIComponent(episodeId)}/frame/${index}`;const [f,d]=await Promise.all([json(base,{signal:controller.signal}),json(base+'/data',{signal:controller.signal}).catch(e=>{if(e.name==='AbortError')throw e;return null})]);if(requestId!==seekSerial||current?.episode_id!==episodeId||controller.signal.aborted)return;image('overheadImage',f.overhead);image('wristImage',f.wrist);image('inspectionImage',f.inspection);renderFz(index);renderTelemetry(index);renderFrameData(d);const firstContact=Array.isArray(current.first_contact_position)?`\nfirst contact [x,y,z]: ${fmt(current.first_contact_position,4)}`:'';$('episodeMeta').textContent=`${episodeId}\nframe ${index+1}/${current.length} · ${Number(d?.t??index/(current.fps||25)).toFixed(2)} s · goal ${current.target_count??'?'} / 6\nseed ${current.seed??'?'} · ${current.success?'SUCCESS':'FAILURE'} · planner ${current.planner_strategy||'legacy'} / ${current.planner_status||'unknown'}${firstContact}${current.failure_reason?'\nreason: '+current.failure_reason:''}`}catch(e){if(e.name==='AbortError')return;if(requestId!==seekSerial||current?.episode_id!==episodeId)return;$('episodeMeta').textContent=e.message}finally{if(seekAbort===controller)seekAbort=null}}
function stepFrame(delta){seekFrame(Number($('frameSlider').value)+delta)}
function togglePlay(){if(timer){clearInterval(timer);timer=null;$('playButton').textContent='Play';return}if(!current)return;$('playButton').textContent='Pause';const tick=async()=>{if(playbackBusy||!timer)return;const n=Number($('frameSlider').value)+1;if(n>=current.length){clearInterval(timer);timer=null;$('playButton').textContent='Play';return}playbackBusy=true;try{await seekFrame(n)}finally{playbackBusy=false}};timer=setInterval(tick,1000/Math.max(1,current.fps||25));tick()}
function toggleFailureMode(){const failed=$('previewOutcome').value==='failed';$('previewFailureModeWrap').style.display=failed?'block':'none'}
async function makePreview(){try{const target=Number($('previewTarget').value),outcome=$('previewOutcome').value,count=Math.max(1,Math.min(100,Number($('previewCount').value)||1)),failure_mode=$('previewFailureMode').value;$('globalStatus').textContent=`Generating ${count} ${outcome} demonstration${count===1?'':'s'} in MuJoCo…`;const response=await json('/api/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({seed:Date.now()%100000,target_count:target,outcome,count,failure_mode})});if(response.job_id){previewJobId=response.job_id;previewJobHandled=null;$('globalStatus').textContent=`后台生成已启动 · 0/${count} · 工作台仍可操作`;await refreshJobs();return}const records=response.records||[response];const latest=records[records.length-1];selectedEpisodeId=latest.episode_id;await loadEpisodes();$('episodeSelect').value=latest.episode_id;await selectEpisode(latest.episode_id);$('globalStatus').textContent=`Generated ${records.length} ${outcome} demonstration${records.length===1?'':'s'} · latest ${latest.episode_id} · goal ${target}/6.`}catch(e){$('globalStatus').textContent='Generation failed: '+e.message}}
async function resetScene(){if(timer){clearInterval(timer);timer=null;$('playButton').textContent='Play'}await json('/api/reset',{method:'POST'});await refreshAll()}
function fillParameters(c){$('pForce').value=c.controller.desired_force;$('pSpeed').value=c.controller.sweep_speed;$('pSafe').value=c.controller.safe_max_force;$('pTime').value=c.episode.max_time;$('pTarget').value=c.task.target_count;$('previewTarget').value=c.task.target_count;$('inferTarget').value=c.task.target_count;$('pBatch').value=c.act.batch_size}
async function initParameters(){try{fillParameters(await json('/api/config'))}catch(e){$('parameterStatus').textContent=e.message}}
async function applyParameters(){const values={"controller.desired_force":Number($('pForce').value),"controller.sweep_speed":Number($('pSpeed').value),"controller.safe_max_force":Number($('pSafe').value),"episode.max_time":Number($('pTime').value),"task.target_count":Number($('pTarget').value),"act.batch_size":Number($('pBatch').value)};try{await json('/api/config/apply',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(values)});$('parameterStatus').textContent='Applied and saved to runs/workbench/config.yaml.';await refreshAll()}catch(e){$('parameterStatus').textContent='Rejected: '+e.message}}
async function startTraining(){try{const body={dataset:$('trainDataset').value,out:$('trainOut').value};if($('trainSteps').value)body.steps=Number($('trainSteps').value);await json('/api/jobs/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});await refreshJobs()}catch(e){$('trainJob').textContent=e.message}}
async function startInference(){try{let seed=Number($('inferSeed').value);if($('inferRandomize').checked){seed=Math.floor(Math.random()*1000000000);$('inferSeed').value=seed}await json('/api/jobs/inference',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:$('inferModel').value,seed,target_count:Number($('inferTarget').value),preview:$('inferPreview').checked})});await refreshJobs()}catch(e){$('inferJob').textContent=e.message}}
function renderJob(jobNode,j){const outputNode=jobNode.querySelector('pre');const scrollTop=outputNode?.scrollTop||0;const wasAtBottom=outputNode?outputNode.scrollTop+outputNode.clientHeight>=outputNode.scrollHeight-8:true;jobNode.innerHTML=`<strong>${j.state}</strong> · ${j.job_id}<pre>${(j.output||[]).join('\n')}</pre>`;const nextOutputNode=jobNode.querySelector('pre');if(nextOutputNode)nextOutputNode.scrollTop=wasAtBottom?nextOutputNode.scrollHeight:scrollTop}
async function refreshJobs(){const jobs=await json('/api/jobs');for(const kind of ['train','inference']){const j=jobs.find(x=>x.kind===kind);if(j){const jobNode=$(kind==='train'?'trainJob':'inferJob');renderJob(jobNode,j)}}const preview=previewJobId?jobs.find(x=>x.kind==='preview'&&x.job_id===previewJobId):jobs.find(x=>x.kind==='preview'&&['starting','running','stopping'].includes(x.state));if(preview&&!previewJobId)previewJobId=preview.job_id;if(!preview)return;const p=preview.progress||{};if(preview.state==='stopping'){$('globalStatus').textContent=`正在停止示教生成… · ${p.completed||0}/${p.total||'?'} · 请等待当前回合退出`;}else if(['starting','running'].includes(preview.state)){$('globalStatus').textContent=`后台生成示教中 · ${p.completed||0}/${p.total||'?'} · 工作台仍可操作`;}else if(previewJobHandled!==preview.job_id){previewJobHandled=preview.job_id;await loadEpisodes();if(p.latest_episode_id)await selectEpisode(p.latest_episode_id);$('globalStatus').textContent=`示教生成${preview.state==='done'?'完成':'结束'} · ${p.completed||0}/${p.total||'?'} 条已保存`}}
async function refreshInferenceReplay(){try{const jobs=await json('/api/jobs');const j=jobs.find(x=>x.kind==='inference');if(!j||!['done','failed'].includes(j.state))return;const lines=j.output||[];for(let i=lines.length-1;i>=0;i--){try{const summary=JSON.parse(lines[i]);if(summary.replay_episode_id&&summary.replay_episode_id!==lastInferenceReplayId){lastInferenceReplayId=summary.replay_episode_id;await loadEpisodes();await selectEpisode(summary.replay_episode_id);$('globalStatus').textContent='Inference replay loaded · '+summary.replay_episode_id;}break}catch(e){}}}catch(e){}}
async function stopLatest(kind){const jobs=await json('/api/jobs');const j=jobs.find(x=>x.kind===kind&&['starting','running','stopping'].includes(x.state));if(j){await json(`/api/jobs/${j.job_id}/stop`,{method:'POST'});await refreshJobs()}}
async function refreshPreviewProgress(){try{const jobs=await json('/api/jobs');const preview=previewJobId?jobs.find(x=>x.kind==='preview'&&x.job_id===previewJobId):jobs.find(x=>x.kind==='preview'&&['starting','running','stopping'].includes(x.state));if(!preview)return;const p=preview.progress||{};const retry=p.retry?` · attempt ${p.attempted||0} · retry ${p.retry}`:'';const error=p.last_error?` · ${p.last_error}`:'';if(['starting','running'].includes(preview.state))$('globalStatus').textContent=`后台生成示教中 · ${p.completed||0}/${p.total||'?'}${retry}${error}`;else if(preview.state==='stopping')$('globalStatus').textContent=`正在停止示教生成… · ${p.completed||0}/${p.total||'?'}${retry}`;}catch(e){}}
toggleFailureMode();initParameters();refreshAll();setInterval(refreshAll,3000);setInterval(refreshInferenceReplay,3000);
setInterval(refreshPreviewProgress,1000);
</script></body></html>"""


def _png_bytes(image):
    from PIL import Image

    if image is None:
        return None
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def create_app(cfg):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Dashboard dependencies are missing: uv pip install -e '.[web]'") from exc
    from ..config import save_config
    from ..environments.sweep_env import SweepEnv
    from .jobs import JobRegistry, build_inference_argv, build_train_argv
    from .workbench import EpisodeNotFound, FrameNotFound, WorkbenchState

    project_root = Path(__file__).resolve().parents[2]
    app = FastAPI(title="ACT MuJoCo Sweep Workbench")
    app.state.cfg = cfg
    app.state.env = None
    app.state.env_lock = RLock()
    dataset_root = Path(str(cfg.act.dataset_dir))
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    app.state.workbench = WorkbenchState(
        cfg, dataset_root=dataset_root,
        preview_root=project_root / "runs" / "workbench_previews")
    app.state.jobs = JobRegistry(project_root)

    def env():
        if app.state.env is None:
            current = app.state.cfg
            app.state.env = SweepEnv(current, seed=int(current.seed))
            app.state.env.reset(seed=int(current.seed))
        return app.state.env

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML

    @app.get("/api/config")
    def config():
        return {**app.state.cfg.to_dict(), "_workbench_allowed": [
            "controller.desired_force", "controller.sweep_speed",
            "controller.safe_max_force", "episode.max_time",
            "task.target_count", "act.batch_size"]}

    @app.get("/api/health")
    def health():
        with app.state.env_lock:
            e = env()
            current = app.state.cfg
            return {"time": e.time, "components": len(e.layout),
                    "target_count": int(current.task.target_count),
                    "collected": int(e.collected_mask().sum()),
                    "tcp": e.tcp().tolist(), "normal_force": e.normal_force(),
                    "model": str(current.end_effector.type),
                    "camera_roles": {"overhead": "perception", "wrist": "act_input",
                                     "inspection": "inspection_only"}}

    @app.post("/api/reset")
    def reset():
        with app.state.env_lock:
            if app.state.env is not None:
                app.state.env.close()
            app.state.env = None
            env()
            return {"ok": True}

    @app.get("/api/frames")
    async def frames():
        with app.state.env_lock:
            e = env()
            size = tuple(int(v) for v in app.state.cfg.act.image_size)
            return {"overhead": _png_bytes(e.render_rgb("overhead_cam", size=size)),
                    "wrist": _png_bytes(e.render_wrist_rgb(size=size)),
                    "inspection": _png_bytes(e.render_rgb("inspection_cam", size=size)),
                    "camera_roles": {"overhead": "perception", "wrist": "act_input",
                                     "inspection": "inspection_only"}}

    @app.get("/api/episodes")
    def episodes():
        return app.state.workbench.list_episodes()

    @app.get("/api/episodes/{episode_id}")
    def episode(episode_id: str):
        try:
            return app.state.workbench.episode_metadata(episode_id)
        except (EpisodeNotFound, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/api/episodes/{episode_id}")
    def delete_episode(episode_id: str):
        try:
            return app.state.workbench.delete_episode(episode_id)
        except (EpisodeNotFound, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/episodes/{episode_id}/frame/{frame_index}")
    def episode_frame(episode_id: str, frame_index: int):
        try:
            return {key: _png_bytes(value) for key, value in
                    app.state.workbench.load_episode_frame(episode_id, frame_index).items()}
        except (EpisodeNotFound, FrameNotFound, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/episodes/{episode_id}/signals")
    def episode_signals(episode_id: str):
        try:
            return app.state.workbench.load_episode_signals(episode_id)
        except (EpisodeNotFound, FileNotFoundError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/episodes/{episode_id}/frame/{frame_index}/data")
    def episode_frame_data(episode_id: str, frame_index: int):
        try:
            return app.state.workbench.load_episode_frame_data(episode_id, frame_index)
        except (EpisodeNotFound, FrameNotFound, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/preview/{preview_id}")
    def preview_metadata(preview_id: str):
        try:
            metadata = app.state.workbench.episode_metadata(preview_id)
            if not metadata.get("preview"):
                raise EpisodeNotFound(preview_id)
            return metadata
        except (EpisodeNotFound, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/preview/{preview_id}/frame/{frame_index}")
    def preview_frame(preview_id: str, frame_index: int):
        try:
            metadata = app.state.workbench.episode_metadata(preview_id)
            if not metadata.get("preview"):
                raise EpisodeNotFound(preview_id)
            return {key: _png_bytes(value) for key, value in
                    app.state.workbench.load_episode_frame(preview_id, frame_index).items()}
        except (EpisodeNotFound, FrameNotFound, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/preview")
    def preview(payload: dict):
        seed = int(payload.get("seed", app.state.cfg.seed))
        target_count = payload.get("target_count", app.state.cfg.task.target_count)
        if not 1 <= int(target_count) <= 6:
            raise HTTPException(status_code=422, detail="target_count must be between 1 and 6")
        outcome = str(payload.get("outcome", "success")).lower()
        if outcome not in {"success", "failed"}:
            raise HTTPException(status_code=422, detail="outcome must be success or failed")
        count = int(payload.get("count", 1))
        if not 1 <= count <= 100:
            raise HTTPException(status_code=422, detail="count must be between 1 and 100")
        failure_mode = str(payload.get("failure_mode", ""))
        allowed_failure_modes = {
            "stall_outside_tray", "wrong_count", "wrong_count_over",
            "wrong_count_under", "misroute",
        }
        if failure_mode and failure_mode not in allowed_failure_modes:
            raise HTTPException(status_code=422, detail="unknown failure_mode")
        if count == 1:
            try:
                return app.state.workbench.build_preview(
                    seed, int(target_count), outcome=outcome, failure_mode=failure_mode)
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        try:
            def run_preview_batch(job):
                def report(progress):
                    completed = int(progress.get("completed", 0))
                    total = int(progress.get("total", count))
                    attempted = int(progress.get("attempted", 0))
                    retry = int(progress.get("retry", 0))
                    last_error = str(progress.get("last_error", ""))
                    detail = (
                        f" · attempt {attempted} · retry {retry}"
                        if attempted else ""
                    )
                    if last_error:
                        detail += f" · last: {last_error[:160]}"
                    app.state.jobs.update(
                        job.job_id,
                        message=f"preview {completed}/{total}{detail}",
                        progress=progress,
                    )

                app.state.workbench.build_previews(
                    seed, int(target_count), outcome=outcome, count=count,
                    failure_mode=failure_mode, progress_callback=report,
                    cancel_event=job.cancel_event)

            return app.state.jobs.start_callable("preview", run_preview_batch)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/preview/fill")
    def fill_previews(payload: dict):
        """Fill every requested target bucket to exact outcome counts."""
        targets = payload.get("target_counts", list(range(1, 7)))
        try:
            targets = [int(value) for value in targets]
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="target_counts must be integers") from exc
        if not targets or any(target < 1 or target > 6 for target in targets):
            raise HTTPException(status_code=422, detail="target_counts must be in 1..6")
        success_count = int(payload.get("success_count", 8))
        failure_count = int(payload.get("failure_count", 2))
        seed = int(payload.get("seed", app.state.cfg.seed))
        if success_count < 0 or failure_count < 0:
            raise HTTPException(status_code=422, detail="outcome counts must be non-negative")
        output = {}
        try:
            for index, target in enumerate(targets):
                result = app.state.workbench.fill_target_counts(
                    seed=seed + index * 1000003 * 4096,
                    target_count=target,
                    success_count=success_count,
                    failure_count=failure_count,
                )
                output[str(target)] = {
                    "target_count": result["target_count"],
                    "success": result["success"],
                    "failed": result["failed"],
                    "generated": [
                        {key: record.get(key) for key in (
                            "episode_id", "success", "target_count", "failure_mode")}
                        for record in result["generated"]
                    ],
                }
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"targets": output, "total": sum(
            item["success"] + item["failed"] for item in output.values())}

    @app.post("/api/preview/fill-failure-mix")
    def fill_failure_mix(payload: dict):
        """Fill each requested target with the persistent failure families."""
        targets = [int(value) for value in payload.get("target_counts", range(1, 7))]
        if not targets or any(target < 1 or target > 6 for target in targets):
            raise HTTPException(status_code=422, detail="target_counts must be in 1..6")
        seed = int(payload.get("seed", app.state.cfg.seed))
        explicit_wrong_over = "wrong_over_count" in payload
        counts = {
            "stall_count": int(payload.get("stall_count", 2)),
            "wrong_over_count": int(payload.get("wrong_over_count", 2)),
            "wrong_under_count": int(payload.get("wrong_under_count", 2)),
            "misroute_count": int(payload.get("misroute_count", 2)),
        }
        if any(value < 0 for value in counts.values()):
            raise HTTPException(status_code=422, detail="failure counts must be non-negative")
        output = {}
        try:
            for index, target in enumerate(targets):
                target_counts = dict(counts)
                # The default all-target request should remain usable with the
                # fixed six-part scene.  An explicitly requested target-6
                # over-count still raises, so the physical boundary cannot be
                # hidden by an implicit downgrade.
                if target == 6 and not explicit_wrong_over:
                    target_counts["wrong_over_count"] = 0
                result = app.state.workbench.fill_failure_mix(
                    seed=seed + index * 1000003 * 4096,
                    target_count=target, **target_counts)
                output[str(target)] = {
                    "target_count": result["target_count"],
                    "counts": result["counts"],
                    "generated": [
                        {key: record.get(key) for key in (
                            "episode_id", "success", "target_count",
                            "failure_mode", "failure_actual_count")}
                        for record in result["generated"]
                    ],
                }
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"targets": output}

    @app.get("/api/jobs")
    def jobs():
        return app.state.jobs.list()

    @app.post("/api/jobs/train")
    def train(payload: dict):
        argv = build_train_argv(project_root, payload.get("config", str(app.state.cfg.get_path("_source_config"))),
                                payload.get("dataset", str(app.state.cfg.act.dataset_dir)),
                                payload.get("out", str(app.state.cfg.act.model_dir)), payload.get("steps"),
                                preview_training=bool(payload.get("preview_training", False)),
                                resume=payload.get("resume"))
        try:
            return app.state.jobs.start("train", argv)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/jobs/inference")
    def inference(payload: dict):
        try:
            target_count = int(payload.get(
                "target_count", app.state.cfg.get_path("task.target_count", 3)
            ))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422,
                                detail="target_count must be an integer from 1 to 6") from exc
        if not 1 <= target_count <= 6:
            raise HTTPException(status_code=422,
                                detail="target_count must be between 1 and 6")
        preview = bool(payload.get("preview", True))
        argv = build_inference_argv(project_root,
                                    payload.get("config", str(app.state.cfg.get_path("_source_config"))),
                                    payload.get("seed", app.state.cfg.seed), payload.get("model"),
                                    target_count, preview=preview)
        try:
            return app.state.jobs.start("inference", argv)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/jobs/{job_id}/stop")
    def stop_job(job_id: str):
        try:
            return app.state.jobs.stop(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    allowed = {
        "controller.desired_force": (0.0, 5.0),
        "controller.sweep_speed": (0.01, 1.0),
        "controller.safe_max_force": (0.1, 20.0),
        "episode.max_time": (1.0, 600.0),
        "task.target_count": (1, 6),
        "act.batch_size": (1, 256),
    }

    @app.post("/api/config/apply")
    def apply_config(payload: dict):
        unknown = sorted(set(payload) - set(allowed))
        if unknown:
            raise HTTPException(status_code=422, detail=f"unsupported parameter(s): {unknown}")
        updated = app.state.cfg.copy()
        for path, (low, high) in allowed.items():
            if path not in payload:
                continue
            try:
                value = float(payload[path])
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=f"{path} must be numeric") from exc
            if not math.isfinite(value) or value < low or value > high or (path == "task.target_count" and value % 1):
                raise HTTPException(status_code=422, detail=f"{path} outside [{low}, {high}]")
            updated.set_path(path, int(value) if path in {"task.target_count", "act.batch_size"} else value)
        target = project_root / "runs" / "workbench" / "config.yaml"
        updated.set_path("_source_config", str(target))
        save_config(updated, str(target))
        app.state.cfg = updated
        app.state.workbench.cfg = updated
        reset()
        return {"ok": True, "config_path": str(target), "config": updated.to_dict()}

    return app
