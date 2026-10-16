import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

function charts() {
  const html=fs.readFileSync(new URL('../pac_act/dashboard.html',import.meta.url),'utf8');
  const source=html.match(/<script>([\s\S]*?)<\/script>/)[1];
  const context={document:{getElementById:()=>({value:.6})},setInterval:()=>{},
    fetch:()=>new Promise(()=>{})}; vm.runInNewContext(source,context);
  assert.ok(context.PacCharts,'testable chart series API is missing');
  return context.PacCharts;
}
test('EMA retains real zero, resets at missing values and never fabricates metrics',()=>{
  const points=charts().series([{update:1,y:0},{update:2,y:10},{update:3,y:null},{update:4,y:20}], 'y',.5);
  assert.deepEqual(JSON.parse(JSON.stringify(points)),[
    {x:1,raw:0,smooth:0},{x:2,raw:10,smooth:5},null,{x:4,raw:20,smooth:20}]);
});
test('single updates produce a visible point rather than an empty curve',()=>{
  const result=charts().geometry([{x:1,raw:.021,smooth:.021}],480,220);
  assert.equal(result.points.length,1);
  assert.ok(Number.isFinite(result.points[0].x));
  assert.ok(Number.isFinite(result.points[0].rawY));
});
test('missing intervals break both curves',()=>{
  const result=charts().geometry([{x:1,raw:1,smooth:1},null,{x:3,raw:3,smooth:3}],480,220);
  assert.equal((result.rawPath.match(/M/g)||[]).length,2);
  assert.equal((result.smoothPath.match(/M/g)||[]).length,2);
});
test('unit conversions never turn null into zero',()=>{
  assert.equal(charts().scaled(null,100),null);
  assert.equal(charts().scaled(undefined,1/2**20),null);
  assert.equal(charts().scaled(0,100),0);
  assert.equal(charts().scaled(.5,100),50);
});
test('isolated curve segments remain visible beyond forty records',()=>{
  const values=[{x:1,raw:1,smooth:1},null,
    ...Array.from({length:40},(_,i)=>({x:i+3,raw:2,smooth:2}))];
  const points=charts().markers(charts().geometry(values));
  assert.equal(points.length,1);
  assert.equal(points[0].xValue,1);
});
