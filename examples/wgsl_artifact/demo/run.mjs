#!/usr/bin/env node
import http from 'node:http';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '../../..');
let chromium;
try { ({ chromium } = await import('playwright')); }
catch { ({ chromium } = await import(path.join(root, '../..', 'node_modules/playwright/index.js'))); }
const manifest = JSON.parse(fs.readFileSync(path.join(here, '../manifest.json'), 'utf8'));
const wgsl = fs.readFileSync(path.join(here, '../evaluator.wgsl'), 'utf8');
if (createHash('sha256').update(wgsl).digest('hex') !== manifest.wgsl_sha256) {
  throw Error('WGSL does not match the sealed manifest');
}
const server = http.createServer((req, res) => {
  if (req.url === '/evaluator.wgsl') {
    res.writeHead(200, {'Content-Type': 'text/plain'}); res.end(wgsl);
  } else { res.writeHead(200, {'Content-Type': 'text/html'}); res.end('<!doctype html><title>WGSL artifact demo</title>'); }
});
await new Promise(resolve => server.listen(0, 'localhost', resolve));
let browser;
try {
  browser = await chromium.launch({channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu']});
  const page = await browser.newPage();
  await page.goto(`http://localhost:${server.address().port}/`);
  const result = await page.evaluate(async ({manifest}) => {
    const assert = (v, message) => { if (!v) throw Error(message); };
    assert(navigator.gpu, 'WebGPU unavailable');
    const adapter = await navigator.gpu.requestAdapter();
    assert(adapter, 'WebGPU adapter unavailable');
    const device = await adapter.requestDevice();
    const rawInfo = adapter.info || {};
    const info = {vendor:rawInfo.vendor || '', architecture:rawInfo.architecture || '',
      device:rawInfo.device || '', description:rawInfo.description || '',
      isFallbackAdapter:adapter.isFallbackAdapter ?? null};
    const source = await (await fetch('/evaluator.wgsl')).text();
    const evaluator = device.createShaderModule({code: source});
    const diagnostics = await evaluator.getCompilationInfo();
    assert(!diagnostics.messages.some(m => m.type === 'error'),
      diagnostics.messages.map(m => `${m.type}:${m.lineNum}:${m.linePos}:${m.message}`).join('\n'));
    const producer = device.createShaderModule({code: `
      struct Params { count:u32, generation:u32, pad0:u32, pad1:u32 };
      @group(0) @binding(0) var<storage, read_write> samples: array<f32>;
      @group(0) @binding(1) var<storage, read> params: Params;
      @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i=gid.x; if(i>=9u){return;}
        if(i==8u){samples[i]=bitcast<f32>(params.pad0);return;}
        let points = array<f32,8>(-0.2,-0.15,-0.1,0.0,0.1,0.15,0.2,0.3);
        samples[i]=points[i];
      }`});
    const consumer = device.createShaderModule({code: `
      struct Result { value:f32, first:f32, second:f32, in_envelope:u32, status:u32, generation:u32, pad0:u32, pad1:u32 };
      struct Evidence { value:f32, first:f32, second:f32, flags:u32 };
      @group(0) @binding(0) var<storage, read> results:array<Result>;
      @group(0) @binding(1) var<storage, read_write> evidence:array<Evidence>;
      @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
        let i=gid.x; if(i>=9u){return;}
        let r=results[i];
        evidence[i]=Evidence(r.value,r.first,r.second,r.in_envelope | (r.status<<1u) | (r.generation<<8u));
      }`});
    for (const module of [producer, consumer]) {
      const details = await module.getCompilationInfo();
      assert(!details.messages.some(m => m.type === 'error'),
        details.messages.map(m => `${m.type}:${m.lineNum}:${m.linePos}:${m.message}`).join('\n'));
    }
    const pipeline = async module => device.createComputePipelineAsync({layout:'auto', compute:{module,entryPoint:'main'}});
    const [p0,p1,p2] = await Promise.all([pipeline(producer),pipeline(evaluator),pipeline(consumer)]);
    const U = GPUBufferUsage;
    const samples = device.createBuffer({size:36,usage:U.STORAGE});
    const results = device.createBuffer({size:288,usage:U.STORAGE});
    const params = device.createBuffer({size:16,usage:U.STORAGE|U.COPY_DST});
    const evidence = device.createBuffer({size:144,usage:U.STORAGE|U.COPY_SRC});
    const readback = device.createBuffer({size:144,usage:U.COPY_DST|U.MAP_READ});
    device.queue.writeBuffer(params,0,new Uint32Array([9,manifest.generation,0x7fc00000,0]));
    const bg0=device.createBindGroup({layout:p0.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:samples}},{binding:1,resource:{buffer:params}}]});
    const bg1=device.createBindGroup({layout:p1.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:params}},{binding:1,resource:{buffer:samples}},{binding:2,resource:{buffer:results}}]});
    const bg2=device.createBindGroup({layout:p2.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:results}},{binding:1,resource:{buffer:evidence}}]});
    const encode = (withReadback) => {
      const encoder=device.createCommandEncoder();
      const pass=encoder.beginComputePass();
      for(const [p,b] of [[p0,bg0],[p1,bg1],[p2,bg2]]) {pass.setPipeline(p);pass.setBindGroup(0,b);pass.dispatchWorkgroups(1);}
      pass.end();
      if(withReadback) encoder.copyBufferToBuffer(evidence,0,readback,0,144);
      return encoder.finish();
    };
    const times=[];
    for(let j=0;j<6;j++) {const t=performance.now();device.queue.submit([encode(false)]);await device.queue.onSubmittedWorkDone();if(j>0)times.push(performance.now()-t);}
    const evalTimes=[];
    for(let j=0;j<6;j++) {
      const encoder=device.createCommandEncoder();const pass=encoder.beginComputePass();
      pass.setPipeline(p1);pass.setBindGroup(0,bg1);pass.dispatchWorkgroups(1);pass.end();
      const t=performance.now();device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
      if(j>0)evalTimes.push(performance.now()-t);
    }
    device.queue.submit([encode(true)]); // one submission for the demonstrated chain
    await readback.mapAsync(GPUMapMode.READ);
    const bytes=readback.getMappedRange();
    const view=new DataView(bytes); const rows=[];
    for(let i=0;i<9;i++) rows.push({value:view.getFloat32(i*16,true),first:view.getFloat32(i*16+4,true),second:view.getFloat32(i*16+8,true),flags:view.getUint32(i*16+12,true)});
    readback.unmap();
    const points=[-0.2,-0.15,-0.1,0,0.1,0.15,0.2,0.3,NaN];
    const max=[0,0,0], rounding=[0,0,0];
    const f32Horner=(coeffs,x)=>{
      const t=Math.fround(x);let acc=Math.fround(coeffs[coeffs.length-1]);
      for(let k=coeffs.length-2;k>=0;k--) acc=Math.fround(Math.fround(acc*t)+Math.fround(coeffs[k]));
      return acc;
    };
    rows.forEach((r,i)=>{
      assert((r.flags>>8)===manifest.generation, `generation ${i}`);
      if(i>=7) {
        assert((r.flags&7)===2, 'outside envelope or nonfinite status');
        assert(r.value===0 && r.first===0 && r.second===0, 'invalid sample has numeric output');
        return;
      }
      assert((r.flags&7)===1, `valid status ${i}: ${r.flags}`);
      const e=points[i], z=Math.exp(3*e);
      const exact=[(2/3)*(z-1-3*e),2*(z-1),6*z];
      const arrays=[manifest.model.coefficients,manifest.model.first_coefficients,manifest.model.second_coefficients];
      [r.value,r.first,r.second].forEach((v,k)=>{
        max[k]=Math.max(max[k],Math.abs(v-exact[k]));
        rounding[k]=Math.max(rounding[k],Math.abs(v-f32Horner(arrays[k],e)));
      });
      assert(r.value>=-1e-6, 'negative strain energy');
      assert(r.second>0, 'lost convexity');
    });
    const tolerance=manifest.errors.acceptance_absolute_tolerance;
    assert(max[0]<=tolerance.value && max[1]<=tolerance.first && max[2]<=tolerance.second,
      `tolerance failed: ${max}`);
    times.sort((a,b)=>a-b);evalTimes.sort((a,b)=>a-b);
    return {adapter:info, precision:'f32', sample_count:9, submission_count:1,
      final_readback_bytes:144, intermediate_readback_bytes:0,
      max_absolute_error:{value:max[0],first:max[1],second:max[2]},
      max_gpu_vs_separate_f32_horner:{value:rounding[0],first:rounding[1],second:rounding[2]},
      warm_chain_submit_and_wait_ms_median:times[2],
      warm_evaluator_submit_and_wait_ms_median:evalTimes[2], rows, chrome:navigator.userAgent};
  }, {manifest});
  fs.writeFileSync(path.join(here,'evidence.json'),JSON.stringify(result,null,2)+'\n');
  console.log(`Chrome WGSL PASS: ${JSON.stringify({adapter:result.adapter,precision:result.precision,errors:result.max_absolute_error,warm_evaluator_ms:result.warm_evaluator_submit_and_wait_ms_median,readback_bytes:result.final_readback_bytes})}`);
} finally { if(browser) await browser.close(); server.close(); }
