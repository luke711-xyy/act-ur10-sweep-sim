import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

function monitor() {
  const file = new URL('../sim/web/training_monitor.js', import.meta.url);
  const context = {document: {readyState: 'loading', addEventListener() {}}};
  vm.runInNewContext(fs.existsSync(file) ? fs.readFileSync(file, 'utf8') : '', context);
  assert.ok(context.WorkbenchTraining, 'training chart implementation is missing');
  return context.WorkbenchTraining;
}

test('chart uses real optimizer steps, retains zero loss and draws a single sample', () => {
  const api = monitor();
  const single = api.geometry([{step: 5, loss: 0, smooth: 0}]);
  assert.equal(single.markers.length, 1);
  assert.ok(Number.isFinite(single.markers[0].x));
  assert.ok(Number.isFinite(single.markers[0].y));
  const series = api.geometry([{step: 10, loss: 1, smooth: 1},
    {step: 20, loss: 2, smooth: 1.5}, {step: 110, loss: 1, smooth: 1}]);
  assert.equal(series.firstStep, 10);
  assert.equal(series.lastStep, 110);
  assert.ok(Math.abs((series.markers[1].x-series.markers[0].x) /
    (series.markers[2].x-series.markers[0].x) - 0.1) < 1e-8);
});

test('missing values break a curve rather than fabricate zero or NaN coordinates', () => {
  const result = monitor().geometry([{step: 1, loss: 2, smooth: 2},
    {step: 2, loss: null, smooth: null}, {step: 3, loss: 1, smooth: 1}]);
  assert.equal((result.rawPath.match(/M/g) || []).length, 2);
  assert.ok(!result.rawPath.includes('NaN'));
  assert.equal(result.markers.length, 2);
});

test('metric labels preserve real zero but never turn missing speed into zero', () => {
  const view = monitor().view({state: 'running', freshness: 'live', policy_type: 'diffusion',
    stage: 'full', step: 1234, max_steps: 100000, loss: 0, mean_loss: 0,
    seconds_per_update: 2, updates_per_minute: 30, age_seconds: 1});
  assert.equal(view.loss, '0.0000');
  assert.equal(view.seconds, '2.00 s');
  assert.equal(view.rate, '30.0 / min');
  assert.match(view.steps, /1,234/);
  const stale = monitor().view({state: 'running', freshness: 'stale', loss: 1,
    seconds_per_update: null, updates_per_minute: null, age_seconds: 90});
  assert.equal(stale.seconds, '—');
  assert.match(stale.status, /未更新/);
});

test('finished, exited and idle jobs cannot claim to be training live', () => {
  const api = monitor();
  assert.match(api.view({state: 'exited', freshness: 'stopped'}).status, /退出/);
  assert.match(api.view({state: 'done', freshness: 'stopped'}).status, /结束/);
  assert.match(api.view({state: 'idle', freshness: 'empty'}).status, /暂无/);
});

test('monitor identifies Transformer diffusion separately from legacy and ACT', () => {
  const api = monitor();
  assert.match(api.view({policy_type: 'diffusion', backbone: 'transformer',
    stage: 'full', job_id: 'new'}).identity, /Diffusion Policy.*Transformer.*new/);
  assert.doesNotMatch(api.view({policy_type: 'diffusion', stage: 'full', job_id: 'old'}).identity,
    /Transformer/);
  assert.match(api.view({policy_type: 'act', job_id: 'act'}).identity, /^ACT/);
});
