// Browser test environment, shared by the live and differential WebGPU gates.
// Linux needs an X11 display (xvfb-run is sufficient) and native Vulkan.
// https://github.com/gpuweb/gpuweb/wiki/Implementation-Status
export async function loadWebGpuPlaywright(importModule = specifier => import(specifier)) {
    let loaded;
    try { loaded = await importModule('playwright'); }
    catch { loaded = await importModule(new URL('../../.scratch/node_modules/playwright/index.mjs', import.meta.url).href); }
    const api = loaded.chromium ? loaded : loaded.default;
    if (typeof api?.chromium?.launch !== 'function') throw new Error('Playwright Chromium export is unavailable');
    return api;
}

export function webgpuLaunchOptions({ platform = process.platform, env = process.env,
                                     headless = true } = {}) {
    const args = ['--enable-unsafe-webgpu', '--enable-gpu'];
    if (platform === 'linux') {
        if (!env.DISPLAY) throw new Error('Linux WebGPU tests require an X11 display; run with xvfb-run -a');
        args.push('--use-angle=vulkan',
                  '--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan',
                  '--ozone-platform=x11');
        headless = false;
    }
    return { channel: 'chrome', headless, args,
             ignoreDefaultArgs: ['--enable-unsafe-swiftshader'] };
}

export function validateHardwareAdapter(observed) {
    if (!observed?.adapter) throw new Error('A real WebGPU adapter is unavailable');
    const info = observed.info;
    if (observed.fallback || /swiftshader|llvmpipe|lavapipe|swrast|\bwarp\b|basic render driver|software/i.test(JSON.stringify(info)))
        throw new Error('Software/fallback WebGPU adapters cannot certify the hardware gate');
    if (!info || typeof info.vendor !== 'string' || !info.vendor.trim() ||
        typeof info.architecture !== 'string' || !info.architecture.trim())
        throw new Error('WebGPU hardware adapter identity is unavailable');
    if (observed.readback !== 42) throw new Error('WebGPU device readback failed');
    return info;
}

export async function requireHardwareWebGpu(browser, url, { report = console.log } = {}) {
    const page = await browser.newPage();
    try {
        await page.goto(url, { timeout: 15000 });
        const observed = await page.evaluate(async () => {
            const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
            if (!adapter) return { adapter: false };
            const raw = adapter.info;
            const info = { vendor: raw?.vendor, architecture: raw?.architecture,
                           device: raw?.device, description: raw?.description };
            const fallback = Boolean(adapter.isFallbackAdapter || raw?.isFallbackAdapter);
            // Reject software before requesting a device or allocating a buffer.
            if (fallback || /swiftshader|llvmpipe|lavapipe|swrast|\bwarp\b|basic render driver|software/i.test(JSON.stringify(info)))
                return { adapter: true, info, fallback: true };
            const device = await adapter.requestDevice();
            let buffer;
            try {
                buffer = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
                device.queue.writeBuffer(buffer, 0, new Uint32Array([42]));
                await buffer.mapAsync(GPUMapMode.READ);
                const readback = new Uint32Array(buffer.getMappedRange())[0];
                buffer.unmap();
                return { adapter: true, info, fallback, readback };
            } finally {
                buffer?.destroy();
                device.destroy();
            }
        });
        const info = validateHardwareAdapter(observed);
        report('WebGPU hardware: ' + JSON.stringify({ chrome: browser.version(), ...info }));
        return info;
    } finally {
        await page.close();
    }
}
