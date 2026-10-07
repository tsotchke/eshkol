import assert from 'node:assert/strict';
import { webgpuLaunchOptions, validateHardwareAdapter, requireHardwareWebGpu, loadWebGpuPlaywright } from '../../scripts/lib/webgpu_test_browser.mjs';

assert.throws(() => webgpuLaunchOptions({ platform: 'linux', env: {} }), /X11 display/);
const linux = webgpuLaunchOptions({ platform: 'linux', env: { DISPLAY: ':99' } });
assert.equal(linux.headless, false);
assert(linux.args.includes('--use-angle=vulkan'));
assert(linux.args.includes('--ozone-platform=x11'));
assert(linux.ignoreDefaultArgs.includes('--enable-unsafe-swiftshader'));
assert(!linux.args.some(arg => /swiftshader|lavapipe/.test(arg)));
for (const platform of ['darwin', 'win32']) {
    assert.equal(webgpuLaunchOptions({ platform, env: {} }).headless, true);
    assert.equal(webgpuLaunchOptions({ platform, env: {}, headless: false }).headless, false);
}
const hardware = { adapter: true, info: { vendor: 'nvidia', architecture: 'ampere' }, fallback: false, readback: 42 };
assert.deepEqual(validateHardwareAdapter(hardware), hardware.info);
for (const bad of [null, { adapter: false }, { ...hardware, fallback: true },
                   { ...hardware, readback: 0 }, { ...hardware, info: {} },
                   ...['swiftshader', 'llvmpipe', 'lavapipe', 'warp'].map(architecture => ({ ...hardware, info: { vendor: 'test', architecture } }))]) {
    assert.throws(() => validateHardwareAdapter(bad));
}
for (const observed of [hardware, { adapter: false }]) {
    let closed = false;
    const browser = { version: () => 'fixture', newPage: async () => ({
        goto: async () => {}, evaluate: async () => observed, close: async () => { closed = true; }
    }) };
    if (observed.adapter) await requireHardwareWebGpu(browser, 'http://localhost/', { report: () => {} });
    else await assert.rejects(requireHardwareWebGpu(browser, 'http://localhost/', { report: () => {} }), /unavailable/);
    assert(closed, 'admission page must close after both success and refusal');
}
console.log('PASS: browser environment hardware admission controls');

const api = { chromium: { launch() {} } };
assert.equal(await loadWebGpuPlaywright(async () => api), api);
assert.equal(await loadWebGpuPlaywright(async () => ({ default: api })), api);
const imports = [];
assert.equal(await loadWebGpuPlaywright(async specifier => {
    imports.push(specifier);
    if (specifier === 'playwright') throw new Error('package absent');
    return api;
}), api);
assert.equal(imports[0], 'playwright');
assert(imports[1].endsWith('/.scratch/node_modules/playwright/index.mjs'));
await assert.rejects(loadWebGpuPlaywright(async () => ({})), /Chromium export/);
console.log('PASS: Playwright package and ESM fallback loading controls');
