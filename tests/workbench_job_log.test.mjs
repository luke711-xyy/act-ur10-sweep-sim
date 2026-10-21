import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

function renderJob() {
  const source = fs.readFileSync(new URL('../sim/web/app.py', import.meta.url), 'utf8');
  const implementation = source.split('\n').find(line => line.startsWith('function renderJob('));
  const context = {renderInferenceVideo() {}};
  vm.runInNewContext(implementation, context);
  return context.renderJob;
}

// Replacing innerHTML creates a fresh <pre>, resetting both scroll axes in a browser.
function panel(scrollLeft, scrollTop) {
  function log(left, top, height) {
    let text = 'previous output';
    return {
      scrollLeft: left, scrollTop: top, clientHeight: 100, scrollHeight: height,
      get textContent() { return text; },
      set textContent(value) { text = value; this.scrollHeight = 1200; },
    };
  }
  let output = log(scrollLeft, scrollTop, 1000);
  const state = {textContent: 'running'};
  const identity = {textContent: ' · existing'};
  return {
    querySelector(selector) {
      if (selector === 'strong') return state;
      if (selector === '.job-id') return identity;
      assert.equal(selector, 'pre');
      return output;
    },
    set innerHTML(value) {
      output = log(0, 0, 1200);
    },
  };
}

for (const kind of ['train', 'inference']) {
  test(`${kind} refresh reuses the log node so an active scroll is not interrupted`, () => {
    const node = panel(620, 210);
    const original = node.querySelector('pre');
    renderJob()(node, {kind, state: 'running', job_id: kind, output: ['new line']});
    assert.equal(node.querySelector('pre'), original);
    assert.equal(original.textContent, 'new line');
  });

  test(`${kind} log keeps its horizontal reading position across refreshes`, () => {
    const render = renderJob();
    const node = panel(620, 210);
    for (let update = 0; update < 3; update++) {
      render(node, {kind, state: 'running', job_id: kind, output: [`update ${update}`]});
      assert.equal(node.querySelector('pre').scrollLeft, 620);
      assert.equal(node.querySelector('pre').scrollTop, 210);
    }
  });

  test(`${kind} log follows new lines at the bottom without jumping left`, () => {
    const render = renderJob();
    const node = panel(340, 900);
    render(node, {kind, state: 'running', job_id: kind, output: ['new line']});
    assert.equal(node.querySelector('pre').scrollLeft, 340);
    assert.equal(node.querySelector('pre').scrollTop, 1200);
  });
}

test('the first job log starts at the left and follows the latest output', () => {
  const node = panel(0, 0);
  const originalQuery = node.querySelector.bind(node);
  let firstQuery = true;
  node.querySelector = selector => {
    if (firstQuery) {
      firstQuery = false;
      return null;
    }
    return originalQuery(selector);
  };
  renderJob()(node, {kind: 'train', state: 'running', job_id: 'new', output: ['first line']});
  assert.equal(node.querySelector('pre').scrollLeft, 0);
  assert.equal(node.querySelector('pre').scrollTop, 1200);
});
