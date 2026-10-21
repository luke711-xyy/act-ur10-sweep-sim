/* Read-only live training monitor; no chart dependencies or policy runtime. */
(function (root) {
  'use strict';
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const fmt = (value, digits = 4) => finite(value) ? value.toFixed(digits) : '—';
  const count = value => finite(value) ? value.toLocaleString('en-US') : '—';

  function geometry(points) {
    const values = points.filter(p => finite(p.step) && finite(p.loss));
    if (!values.length) return {rawPath: '', smoothPath: '', markers: [], ticks: []};
    const firstStep = values[0].step, lastStep = values.at(-1).step;
    let low = Math.min(...values.map(p => p.loss), ...values.map(p => p.smooth).filter(finite));
    let high = Math.max(...values.map(p => p.loss), ...values.map(p => p.smooth).filter(finite));
    const margin = Math.max((high - low) * 0.06, Math.abs(high) * 0.03, 0.001);
    low = Math.max(0, low - margin); high += margin;
    const x = step => 62 + (step - firstStep) / Math.max(1, lastStep - firstStep) * 634;
    const y = loss => 20 + (high - loss) / Math.max(high - low, 1e-9) * 214;
    function path(key) {
      let open = false;
      return points.map(p => {
        if (!finite(p.step) || !finite(p[key])) { open = false; return ''; }
        const command = open ? 'L' : 'M'; open = true;
        return `${command}${x(p.step).toFixed(2)},${y(p[key]).toFixed(2)}`;
      }).join(' ');
    }
    return {firstStep, lastStep, rawPath: path('loss'), smoothPath: path('smooth'),
      markers: values.map(p => ({x: x(p.step), y: y(p.loss)})),
      ticks: Array.from({length: 5}, (_, i) => ({y: 20 + i * 214 / 4,
        label: fmt(high - i * (high - low) / 4, high < 0.1 ? 4 : 3)}))};
  }

  function view(data) {
    const states = {starting: '准备中', running: '训练中', stopping: '正在停止',
      done: '作业已结束', failed: '作业失败', stopped: '已停止', exited: '训练进程已退出',
      idle: '暂无训练作业'};
    let status = states[data.state] || '状态未知';
    if (data.training_state === 'training_complete') status = '训练已结束';
    else if (data.training_state === 'failed') status = '训练失败';
    else if (data.freshness === 'stale') status += ' · 超过30秒未更新，可能正在保存检查点';
    else if (data.freshness === 'waiting') status += ' · 等待首条训练指标';
    else if (data.freshness === 'live') status += ' · 实时';
    const stages = {full: '全量训练', pilot: '小样本检查', train: '训练'};
    const policy = data.policy_type === 'diffusion' ? 'Diffusion Policy' : 'ACT';
    return {status, steps: `${count(data.step)} / ${count(data.max_steps)}`,
      loss: fmt(data.loss), mean: fmt(data.mean_loss),
      seconds: finite(data.seconds_per_update) ? `${fmt(data.seconds_per_update, 2)} s` : '—',
      rate: finite(data.updates_per_minute) ? `${fmt(data.updates_per_minute, 1)} / min` : '—',
      identity: data.job_id ? `${policy} · ${stages[data.stage] || data.stage || ''} · ${data.job_id}`
        : '启动训练后自动接入；刷新页面不会丢失进度。',
      age: finite(data.age_seconds) ? `${Math.floor(data.age_seconds)}秒前更新` : '尚无指标'};
  }

  function render(data) {
    const $ = id => document.getElementById(id);
    const text = view(data);
    for (const [key, id] of Object.entries({status: 'trainingStatus', steps: 'trainingSteps',
      loss: 'trainingLoss', mean: 'trainingMean', seconds: 'trainingSeconds', rate: 'trainingRate',
      identity: 'trainingIdentity', age: 'trainingAge'})) $(id).textContent = text[key];
    $('trainingStatus').className = ['stale', 'stopped'].includes(data.freshness) ? 'training-warning' : 'hint';
    $('trainingSource').textContent = data.out_dir || '只读训练日志 · 不加载模型';
    $('trainingNotice').textContent = data.invalid_rows
      ? `${data.invalid_rows}条无效指标未绘制；请检查作业日志。` :
      `最近${data.mean_samples || 0}条指标均值；速度按最近${data.rate_updates || 0}次更新计时。曲线不是验证集指标。`;
    const chart = geometry(data.points || []);
    $('trainingRaw').setAttribute('d', chart.rawPath);
    $('trainingSmooth').setAttribute('d', chart.smoothPath);
    $('trainingEmpty').style.display = chart.markers.length ? 'none' : 'block';
    const ns = 'http://www.w3.org/2000/svg';
    const grid = $('trainingGrid'); grid.replaceChildren();
    for (const tick of chart.ticks) {
      const line = document.createElementNS(ns, 'line');
      for (const [k, value] of Object.entries({x1: 62, x2: 696, y1: tick.y, y2: tick.y})) line.setAttribute(k, value);
      line.setAttribute('class', 'fz-grid'); grid.appendChild(line);
      const label = document.createElementNS(ns, 'text');
      label.setAttribute('x', 54); label.setAttribute('y', tick.y + 4);
      label.setAttribute('text-anchor', 'end'); label.textContent = tick.label; grid.appendChild(label);
    }
    $('trainingFirst').textContent = count(chart.firstStep);
    $('trainingLast').textContent = count(chart.lastStep);
    const marker = $('trainingMarker');
    marker.style.display = chart.markers.length === 1 ? 'block' : 'none';
    if (chart.markers.length === 1) {
      marker.setAttribute('cx', chart.markers[0].x); marker.setAttribute('cy', chart.markers[0].y);
    }
  }

  function start() {
    if (!document.getElementById('trainingMonitor')) return;
    let busy = false;
    async function refresh() {
      if (busy || document.visibilityState === 'hidden') return;
      busy = true;
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 8000);
      try {
        const response = await fetch('/api/training/progress', {cache: 'no-store', signal: controller.signal});
        const body = await response.json();
        if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
        render(body);
      } catch (error) {
        const node = document.getElementById('trainingStatus');
        node.className = 'training-warning';
        node.textContent = '进度连接中断，显示上次数据；正在自动重试。';
        document.getElementById('trainingSeconds').textContent = '—';
        document.getElementById('trainingRate').textContent = '—';
      } finally {
        clearTimeout(timeout); busy = false;
      }
    }
    refresh();
    setInterval(refresh, 3000);
    document.addEventListener('visibilitychange', refresh);
  }

  root.WorkbenchTraining = {geometry, view};
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})(globalThis);
