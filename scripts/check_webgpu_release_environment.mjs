#!/usr/bin/env node
// Fail before the long release build if the real browser GPU lane is unavailable.
import http from 'node:http';
import { webgpuLaunchOptions, requireHardwareWebGpu, loadWebGpuPlaywright } from './lib/webgpu_test_browser.mjs';
const { chromium } = await loadWebGpuPlaywright();
const server = http.createServer((_request, response) => {
    response.writeHead(200, { 'Content-Type': 'text/html' });
    response.end('<!doctype html><title>WebGPU release preflight</title>');
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
const timeout = setTimeout(async () => {
    console.error('WebGPU release preflight timed out');
    try { await browser?.close(); } finally { server.close(); process.exit(2); }
}, 30000);
try {
    browser = await chromium.launch(webgpuLaunchOptions());
    await requireHardwareWebGpu(browser, `http://127.0.0.1:${server.address().port}/`);
    console.log('PASS: real WebGPU browser environment');
} finally {
    clearTimeout(timeout);
    await browser?.close();
    await new Promise(resolve => server.close(resolve));
}
