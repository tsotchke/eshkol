/*
 * gpu_vulkan.c — Vulkan compute backend for the Eshkol GPU dispatch.
 *
 * See gpu_vulkan.h for the contract. Structure:
 *   loader     dlopen the Vulkan loader, resolve entry points
 *   device     enumerate, choose, create device + queue, pipelines
 *   memory     host-visible staging buffer per EshkolGPUBuffer, plus a
 *              device-local twin when the device has separate VRAM
 *   kernels    elementwise / reduce / matmul command recording
 *   admission  per-device bitwise probe of every kernel at init
 *
 * Only the Vulkan 1.0 core API is used. The kernels are the GLSL sources in
 * lib/backend/gpu/vulkan/, embedded as SPIR-V by scripts/gen_vulkan_spirv.sh.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#include "gpu_vulkan.h"

#if defined(__has_include)
#  if __has_include(<vulkan/vulkan.h>) && !defined(ESHKOL_VULKAN_USE_ABI_HEADER)
#    define VK_NO_PROTOTYPES
#    include <vulkan/vulkan.h>
#  else
#    include "vulkan_abi.h"
#  endif
#else
#  include "vulkan_abi.h"
#endif
#include "vulkan_spirv.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#  include <pthread.h>
#endif

/* ===================================================================== */
/* Locking and loader                                                     */
/* ===================================================================== */

#if defined(_WIN32)
static INIT_ONCE g_vk_once = INIT_ONCE_STATIC_INIT;
static CRITICAL_SECTION g_vk_cs;
static BOOL CALLBACK vk_cs_init(PINIT_ONCE o, PVOID p, PVOID* c) {
    (void)o; (void)p; (void)c; InitializeCriticalSection(&g_vk_cs); return TRUE;
}
static void vk_lock(void) { InitOnceExecuteOnce(&g_vk_once, vk_cs_init, NULL, NULL); EnterCriticalSection(&g_vk_cs); }
static void vk_unlock(void) { LeaveCriticalSection(&g_vk_cs); }
#else
static pthread_mutex_t g_vk_mutex = PTHREAD_MUTEX_INITIALIZER;
static void vk_lock(void) { pthread_mutex_lock(&g_vk_mutex); }
static void vk_unlock(void) { pthread_mutex_unlock(&g_vk_mutex); }
#endif

#define VK_FN(name) PFN_##name name

typedef struct {
    /* loader / instance level */
    VK_FN(vkGetInstanceProcAddr);
    VK_FN(vkCreateInstance);
    VK_FN(vkDestroyInstance);
    VK_FN(vkEnumeratePhysicalDevices);
    VK_FN(vkGetPhysicalDeviceProperties);
    VK_FN(vkGetPhysicalDeviceFeatures);
    VK_FN(vkGetPhysicalDeviceQueueFamilyProperties);
    VK_FN(vkGetPhysicalDeviceMemoryProperties);
    VK_FN(vkCreateDevice);
    VK_FN(vkGetDeviceProcAddr);
    /* device level */
    VK_FN(vkDestroyDevice);
    VK_FN(vkGetDeviceQueue);
    VK_FN(vkDeviceWaitIdle);
    VK_FN(vkCreateBuffer);
    VK_FN(vkDestroyBuffer);
    VK_FN(vkGetBufferMemoryRequirements);
    VK_FN(vkAllocateMemory);
    VK_FN(vkFreeMemory);
    VK_FN(vkBindBufferMemory);
    VK_FN(vkMapMemory);
    VK_FN(vkUnmapMemory);
    VK_FN(vkCreateShaderModule);
    VK_FN(vkDestroyShaderModule);
    VK_FN(vkCreateDescriptorSetLayout);
    VK_FN(vkDestroyDescriptorSetLayout);
    VK_FN(vkCreatePipelineLayout);
    VK_FN(vkDestroyPipelineLayout);
    VK_FN(vkCreateComputePipelines);
    VK_FN(vkDestroyPipeline);
    VK_FN(vkCreateDescriptorPool);
    VK_FN(vkDestroyDescriptorPool);
    VK_FN(vkResetDescriptorPool);
    VK_FN(vkAllocateDescriptorSets);
    VK_FN(vkUpdateDescriptorSets);
    VK_FN(vkCreateCommandPool);
    VK_FN(vkDestroyCommandPool);
    VK_FN(vkAllocateCommandBuffers);
    VK_FN(vkResetCommandBuffer);
    VK_FN(vkBeginCommandBuffer);
    VK_FN(vkEndCommandBuffer);
    VK_FN(vkCmdBindPipeline);
    VK_FN(vkCmdBindDescriptorSets);
    VK_FN(vkCmdPushConstants);
    VK_FN(vkCmdDispatch);
    VK_FN(vkCmdCopyBuffer);
    VK_FN(vkCmdPipelineBarrier);
    VK_FN(vkQueueSubmit);
    VK_FN(vkCreateFence);
    VK_FN(vkDestroyFence);
    VK_FN(vkWaitForFences);
    VK_FN(vkResetFences);
} VkFns;

enum { PIPE_ELEMENTWISE = 0, PIPE_REDUCE, PIPE_MATMUL_F64, PIPE_MATMUL_F32, PIPE_COUNT };

typedef struct {
    void* lib;
    VkFns f;
    VkInstance instance;
    int instance_ready;

    int device_ready;
    int device_index;
    VkPhysicalDevice phys;
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceMemoryProperties mem_props;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family;
    uint32_t host_mem_type;     /* HOST_VISIBLE | HOST_COHERENT */
    int host_mem_device_local;  /* host type is also DEVICE_LOCAL: no twin */
    uint32_t dev_mem_type;      /* DEVICE_LOCAL, used when !host_mem_device_local */

    VkDescriptorSetLayout set_layout;
    VkPipelineLayout pipe_layout;
    VkShaderModule modules[PIPE_COUNT];
    VkPipeline pipes[PIPE_COUNT];
    VkDescriptorPool desc_pool;
    VkCommandPool cmd_pool;
    VkCommandBuffer cmd;
    VkFence fence;

    uint32_t elementwise_admitted;  /* bit per EshkolElementwiseOp */
    uint32_t reduce_admitted;       /* bit per EshkolReduceOp */
    int matmul_f64_admitted;
    int matmul_f32_admitted;

    char status[256];
} VkState;

static VkState g;

static void vk_status(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g.status, sizeof(g.status), fmt, ap);
    va_end(ap);
}

static void* vk_dlopen(const char* path) {
#if defined(_WIN32)
    return (void*)LoadLibraryA(path);
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

static void* vk_dlsym(void* lib, const char* name) {
#if defined(_WIN32)
    return (void*)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

static int vk_open_loader(void) {
    static const char* const names[] = {
#if defined(_WIN32)
        "vulkan-1.dll",
#elif defined(__APPLE__)
        "libvulkan.1.dylib", "libvulkan.dylib",
#else
        "libvulkan.so.1", "libvulkan.so",
#endif
        NULL
    };
    const char* override_path = getenv("ESHKOL_VULKAN_LIBRARY");
    if (g.lib) return 1;
    if (override_path && *override_path) {
        g.lib = vk_dlopen(override_path);
    } else {
        for (int i = 0; names[i] && !g.lib; i++) g.lib = vk_dlopen(names[i]);
    }
    if (!g.lib) {
        vk_status("Vulkan loader not found");
        return 0;
    }
    {
        /* dlsym returns an object pointer; copy its bits into the function pointer. */
        void* sym = vk_dlsym(g.lib, "vkGetInstanceProcAddr");
        memcpy(&g.f.vkGetInstanceProcAddr, &sym, sizeof(sym));
    }
    if (!g.f.vkGetInstanceProcAddr) {
        vk_status("Vulkan loader has no vkGetInstanceProcAddr");
        return 0;
    }
    return 1;
}

#define LOAD_GLOBAL(name) \
    (g.f.name = (PFN_##name)g.f.vkGetInstanceProcAddr(NULL, #name))
#define LOAD_INSTANCE(name) \
    (g.f.name = (PFN_##name)g.f.vkGetInstanceProcAddr(g.instance, #name))
#define LOAD_DEVICE(name) \
    (g.f.name = (PFN_##name)g.f.vkGetDeviceProcAddr(g.device, #name))

static int vk_open_instance(void) {
    if (g.instance_ready) return 1;
    if (!vk_open_loader()) return 0;
    if (!LOAD_GLOBAL(vkCreateInstance)) {
        vk_status("vkCreateInstance unavailable");
        return 0;
    }
    VkApplicationInfo app;
    memset(&app, 0, sizeof(app));
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "eshkol";
    app.pEngineName = "eshkol-gpu";
    app.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;
    VkResult r = g.f.vkCreateInstance(&ci, NULL, &g.instance);
    if (r != VK_SUCCESS) {
        vk_status("vkCreateInstance failed (%d)", (int)r);
        return 0;
    }
    if (!LOAD_INSTANCE(vkDestroyInstance) || !LOAD_INSTANCE(vkEnumeratePhysicalDevices) ||
        !LOAD_INSTANCE(vkGetPhysicalDeviceProperties) || !LOAD_INSTANCE(vkGetPhysicalDeviceFeatures) ||
        !LOAD_INSTANCE(vkGetPhysicalDeviceQueueFamilyProperties) ||
        !LOAD_INSTANCE(vkGetPhysicalDeviceMemoryProperties) || !LOAD_INSTANCE(vkCreateDevice) ||
        !LOAD_INSTANCE(vkGetDeviceProcAddr)) {
        vk_status("Vulkan instance entry points missing");
        if (g.f.vkDestroyInstance) g.f.vkDestroyInstance(g.instance, NULL);
        g.instance = NULL;
        return 0;
    }
    g.instance_ready = 1;
    return 1;
}

/* Enumerate physical devices into `out` (caller frees). */
static uint32_t vk_list_devices(VkPhysicalDevice** out) {
    uint32_t count = 0;
    *out = NULL;
    if (!vk_open_instance()) return 0;
    if (g.f.vkEnumeratePhysicalDevices(g.instance, &count, NULL) != VK_SUCCESS || count == 0)
        return 0;
    VkPhysicalDevice* list = (VkPhysicalDevice*)calloc(count, sizeof(VkPhysicalDevice));
    if (!list) return 0;
    VkResult r = g.f.vkEnumeratePhysicalDevices(g.instance, &count, list);
    if (r != VK_SUCCESS && r != VK_INCOMPLETE) { free(list); return 0; }
    *out = list;
    return count;
}

static int vk_compute_family(VkPhysicalDevice pd, uint32_t* family) {
    uint32_t n = 0;
    g.f.vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, NULL);
    if (n == 0) return 0;
    VkQueueFamilyProperties* q = (VkQueueFamilyProperties*)calloc(n, sizeof(*q));
    if (!q) return 0;
    g.f.vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, q);
    int found = 0;
    /* Prefer a compute-only family (async compute), else any compute family. */
    for (uint32_t i = 0; i < n && !found; i++)
        if ((q[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(q[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
            q[i].queueCount > 0) { *family = i; found = 1; }
    for (uint32_t i = 0; i < n && !found; i++)
        if ((q[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && q[i].queueCount > 0) { *family = i; found = 1; }
    free(q);
    return found;
}

static int vk_find_memory(uint32_t type_bits, VkMemoryPropertyFlags want,
                          VkMemoryPropertyFlags prefer, uint32_t* index) {
    int best = -1;
    for (uint32_t i = 0; i < g.mem_props.memoryTypeCount; i++) {
        if (!(type_bits & (1u << i))) continue;
        VkMemoryPropertyFlags fl = g.mem_props.memoryTypes[i].propertyFlags;
        if ((fl & want) != want) continue;
        if ((fl & prefer) == prefer) { *index = i; return 1; }
        if (best < 0) best = (int)i;
    }
    if (best < 0) return 0;
    *index = (uint32_t)best;
    return 1;
}

/* ===================================================================== */
/* Device lifetime                                                        */
/* ===================================================================== */

static void vk_destroy_device_locked(void) {
    if (!g.device) { g.device_ready = 0; return; }
    if (g.f.vkDeviceWaitIdle) g.f.vkDeviceWaitIdle(g.device);
    if (g.fence) g.f.vkDestroyFence(g.device, g.fence, NULL);
    if (g.cmd_pool) g.f.vkDestroyCommandPool(g.device, g.cmd_pool, NULL);
    if (g.desc_pool) g.f.vkDestroyDescriptorPool(g.device, g.desc_pool, NULL);
    for (int i = 0; i < PIPE_COUNT; i++) {
        if (g.pipes[i]) g.f.vkDestroyPipeline(g.device, g.pipes[i], NULL);
        if (g.modules[i]) g.f.vkDestroyShaderModule(g.device, g.modules[i], NULL);
        g.pipes[i] = VK_NULL_HANDLE;
        g.modules[i] = VK_NULL_HANDLE;
    }
    if (g.pipe_layout) g.f.vkDestroyPipelineLayout(g.device, g.pipe_layout, NULL);
    if (g.set_layout) g.f.vkDestroyDescriptorSetLayout(g.device, g.set_layout, NULL);
    g.f.vkDestroyDevice(g.device, NULL);
    g.device = NULL;
    g.fence = VK_NULL_HANDLE;
    g.cmd_pool = VK_NULL_HANDLE;
    g.cmd = NULL;
    g.desc_pool = VK_NULL_HANDLE;
    g.pipe_layout = VK_NULL_HANDLE;
    g.set_layout = VK_NULL_HANDLE;
    g.device_ready = 0;
    g.device_index = -1;
    g.elementwise_admitted = 0;
    g.reduce_admitted = 0;
    g.matmul_f64_admitted = 0;
    g.matmul_f32_admitted = 0;
}

static int vk_load_device_fns(void) {
    return LOAD_DEVICE(vkDestroyDevice) && LOAD_DEVICE(vkGetDeviceQueue) &&
           LOAD_DEVICE(vkDeviceWaitIdle) && LOAD_DEVICE(vkCreateBuffer) &&
           LOAD_DEVICE(vkDestroyBuffer) && LOAD_DEVICE(vkGetBufferMemoryRequirements) &&
           LOAD_DEVICE(vkAllocateMemory) && LOAD_DEVICE(vkFreeMemory) &&
           LOAD_DEVICE(vkBindBufferMemory) && LOAD_DEVICE(vkMapMemory) &&
           LOAD_DEVICE(vkUnmapMemory) && LOAD_DEVICE(vkCreateShaderModule) &&
           LOAD_DEVICE(vkDestroyShaderModule) && LOAD_DEVICE(vkCreateDescriptorSetLayout) &&
           LOAD_DEVICE(vkDestroyDescriptorSetLayout) && LOAD_DEVICE(vkCreatePipelineLayout) &&
           LOAD_DEVICE(vkDestroyPipelineLayout) && LOAD_DEVICE(vkCreateComputePipelines) &&
           LOAD_DEVICE(vkDestroyPipeline) && LOAD_DEVICE(vkCreateDescriptorPool) &&
           LOAD_DEVICE(vkDestroyDescriptorPool) && LOAD_DEVICE(vkResetDescriptorPool) &&
           LOAD_DEVICE(vkAllocateDescriptorSets) && LOAD_DEVICE(vkUpdateDescriptorSets) &&
           LOAD_DEVICE(vkCreateCommandPool) && LOAD_DEVICE(vkDestroyCommandPool) &&
           LOAD_DEVICE(vkAllocateCommandBuffers) && LOAD_DEVICE(vkResetCommandBuffer) &&
           LOAD_DEVICE(vkBeginCommandBuffer) && LOAD_DEVICE(vkEndCommandBuffer) &&
           LOAD_DEVICE(vkCmdBindPipeline) && LOAD_DEVICE(vkCmdBindDescriptorSets) &&
           LOAD_DEVICE(vkCmdPushConstants) && LOAD_DEVICE(vkCmdDispatch) &&
           LOAD_DEVICE(vkCmdCopyBuffer) && LOAD_DEVICE(vkCmdPipelineBarrier) &&
           LOAD_DEVICE(vkQueueSubmit) && LOAD_DEVICE(vkCreateFence) &&
           LOAD_DEVICE(vkDestroyFence) && LOAD_DEVICE(vkWaitForFences) &&
           LOAD_DEVICE(vkResetFences);
}

static int vk_create_pipeline(int which, const uint32_t* code, size_t size) {
    VkShaderModuleCreateInfo smci;
    memset(&smci, 0, sizeof(smci));
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = size;
    smci.pCode = code;
    if (g.f.vkCreateShaderModule(g.device, &smci, NULL, &g.modules[which]) != VK_SUCCESS)
        return 0;
    VkComputePipelineCreateInfo pci;
    memset(&pci, 0, sizeof(pci));
    pci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pci.stage.module = g.modules[which];
    pci.stage.pName = "main";
    pci.layout = g.pipe_layout;
    pci.basePipelineIndex = -1;
    return g.f.vkCreateComputePipelines(g.device, VK_NULL_HANDLE, 1, &pci, NULL,
                                        &g.pipes[which]) == VK_SUCCESS;
}

#define PUSH_CONSTANT_BYTES 24u
#define MAX_SETS_PER_OP 16u

static int vk_create_device_locked(uint32_t index, VkPhysicalDevice pd) {
    VkPhysicalDeviceFeatures feats;
    uint32_t family = 0;
    memset(&feats, 0, sizeof(feats));
    g.f.vkGetPhysicalDeviceProperties(pd, &g.props);
    g.f.vkGetPhysicalDeviceFeatures(pd, &feats);
    if (!feats.shaderFloat64) {
        vk_status("device %u (%s) has no shaderFloat64", index, g.props.deviceName);
        return 0;
    }
    if (!vk_compute_family(pd, &family)) {
        vk_status("device %u (%s) has no compute queue", index, g.props.deviceName);
        return 0;
    }
    if (g.props.limits.maxPushConstantsSize < PUSH_CONSTANT_BYTES) {
        vk_status("device %u push-constant space too small", index);
        return 0;
    }
    g.f.vkGetPhysicalDeviceMemoryProperties(pd, &g.mem_props);

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci;
    memset(&qci, 0, sizeof(qci));
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkPhysicalDeviceFeatures enable;
    memset(&enable, 0, sizeof(enable));
    enable.shaderFloat64 = VK_TRUE;
    VkDeviceCreateInfo dci;
    memset(&dci, 0, sizeof(dci));
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.pEnabledFeatures = &enable;
    VkResult r = g.f.vkCreateDevice(pd, &dci, NULL, &g.device);
    if (r != VK_SUCCESS) {
        vk_status("vkCreateDevice failed on device %u (%d)", index, (int)r);
        g.device = NULL;
        return 0;
    }
    if (!vk_load_device_fns()) {
        vk_status("device entry points missing on device %u", index);
        if (g.f.vkDestroyDevice) g.f.vkDestroyDevice(g.device, NULL);
        g.device = NULL;
        return 0;
    }
    g.phys = pd;
    g.queue_family = family;
    g.f.vkGetDeviceQueue(g.device, family, 0, &g.queue);

    /* Memory types are resolved against a probe buffer's requirements. */
    {
        VkBufferCreateInfo bci;
        VkBuffer probe = VK_NULL_HANDLE;
        VkMemoryRequirements req;
        memset(&bci, 0, sizeof(bci));
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = 256;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (g.f.vkCreateBuffer(g.device, &bci, NULL, &probe) != VK_SUCCESS) {
            vk_status("buffer creation failed on device %u", index);
            goto fail;
        }
        g.f.vkGetBufferMemoryRequirements(g.device, probe, &req);
        g.f.vkDestroyBuffer(g.device, probe, NULL);
        if (!vk_find_memory(req.memoryTypeBits,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            VK_MEMORY_PROPERTY_HOST_CACHED_BIT, &g.host_mem_type)) {
            vk_status("no host-visible coherent memory on device %u", index);
            goto fail;
        }
        g.host_mem_device_local =
            (g.mem_props.memoryTypes[g.host_mem_type].propertyFlags &
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        if (!g.host_mem_device_local) {
            uint32_t dl = 0;
            if (vk_find_memory(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, &dl)) {
                g.dev_mem_type = dl;
            } else {
                g.host_mem_device_local = 1;  /* no VRAM type: run on host memory */
            }
        }
    }

    {
        VkDescriptorSetLayoutBinding b[3];
        memset(b, 0, sizeof(b));
        for (uint32_t i = 0; i < 3; i++) {
            b[i].binding = i;
            b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo lci;
        memset(&lci, 0, sizeof(lci));
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 3;
        lci.pBindings = b;
        if (g.f.vkCreateDescriptorSetLayout(g.device, &lci, NULL, &g.set_layout) != VK_SUCCESS)
            goto fail_status;
        VkPushConstantRange pcr;
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = PUSH_CONSTANT_BYTES;
        VkPipelineLayoutCreateInfo plci;
        memset(&plci, 0, sizeof(plci));
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &g.set_layout;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        if (g.f.vkCreatePipelineLayout(g.device, &plci, NULL, &g.pipe_layout) != VK_SUCCESS)
            goto fail_status;
    }
    if (!vk_create_pipeline(PIPE_ELEMENTWISE, eshkol_vk_spirv_elementwise_f64,
                            eshkol_vk_spirv_elementwise_f64_size) ||
        !vk_create_pipeline(PIPE_REDUCE, eshkol_vk_spirv_reduce_f64,
                            eshkol_vk_spirv_reduce_f64_size) ||
        !vk_create_pipeline(PIPE_MATMUL_F64, eshkol_vk_spirv_matmul_f64,
                            eshkol_vk_spirv_matmul_f64_size) ||
        !vk_create_pipeline(PIPE_MATMUL_F32, eshkol_vk_spirv_matmul_f32,
                            eshkol_vk_spirv_matmul_f32_size)) {
        vk_status("compute pipeline creation failed on device %u (%s)", index, g.props.deviceName);
        goto fail;
    }
    {
        VkDescriptorPoolSize ps;
        ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps.descriptorCount = 3 * MAX_SETS_PER_OP;
        VkDescriptorPoolCreateInfo dpci;
        memset(&dpci, 0, sizeof(dpci));
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = MAX_SETS_PER_OP;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &ps;
        if (g.f.vkCreateDescriptorPool(g.device, &dpci, NULL, &g.desc_pool) != VK_SUCCESS)
            goto fail_status;
        VkCommandPoolCreateInfo cpci;
        memset(&cpci, 0, sizeof(cpci));
        cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpci.queueFamilyIndex = family;
        if (g.f.vkCreateCommandPool(g.device, &cpci, NULL, &g.cmd_pool) != VK_SUCCESS)
            goto fail_status;
        VkCommandBufferAllocateInfo cbai;
        memset(&cbai, 0, sizeof(cbai));
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = g.cmd_pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        if (g.f.vkAllocateCommandBuffers(g.device, &cbai, &g.cmd) != VK_SUCCESS)
            goto fail_status;
        VkFenceCreateInfo fci;
        memset(&fci, 0, sizeof(fci));
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (g.f.vkCreateFence(g.device, &fci, NULL, &g.fence) != VK_SUCCESS)
            goto fail_status;
    }
    g.device_index = (int)index;
    g.device_ready = 1;
    return 1;

fail_status:
    vk_status("Vulkan object creation failed on device %u", index);
fail:
    vk_destroy_device_locked();
    return 0;
}

/* ===================================================================== */
/* Memory                                                                 */
/* ===================================================================== */

typedef struct {
    VkBuffer host_buf;
    VkDeviceMemory host_mem;
    void* mapped;
    VkBuffer dev_buf;            /* VK_NULL_HANDLE when the host buffer is used directly */
    VkDeviceMemory dev_mem;
    VkDeviceSize size;
    int wrapped;                 /* host_ptr belongs to the caller */
} VkBufRec;

static int vk_make_buffer(VkDeviceSize size, uint32_t mem_type, VkBuffer* buf,
                          VkDeviceMemory* mem) {
    VkBufferCreateInfo bci;
    VkMemoryRequirements req;
    memset(&bci, 0, sizeof(bci));
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    *buf = VK_NULL_HANDLE;
    *mem = VK_NULL_HANDLE;
    if (g.f.vkCreateBuffer(g.device, &bci, NULL, buf) != VK_SUCCESS) return 0;
    g.f.vkGetBufferMemoryRequirements(g.device, *buf, &req);
    if (!(req.memoryTypeBits & (1u << mem_type))) {
        g.f.vkDestroyBuffer(g.device, *buf, NULL);
        *buf = VK_NULL_HANDLE;
        return 0;
    }
    VkMemoryAllocateInfo mai;
    memset(&mai, 0, sizeof(mai));
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = mem_type;
    if (g.f.vkAllocateMemory(g.device, &mai, NULL, mem) != VK_SUCCESS ||
        g.f.vkBindBufferMemory(g.device, *buf, *mem, 0) != VK_SUCCESS) {
        if (*mem) g.f.vkFreeMemory(g.device, *mem, NULL);
        g.f.vkDestroyBuffer(g.device, *buf, NULL);
        *buf = VK_NULL_HANDLE;
        *mem = VK_NULL_HANDLE;
        return 0;
    }
    return 1;
}

static void vk_rec_destroy(VkBufRec* rec) {
    if (!rec) return;
    if (g.device) {
        if (rec->mapped) g.f.vkUnmapMemory(g.device, rec->host_mem);
        if (rec->host_buf) g.f.vkDestroyBuffer(g.device, rec->host_buf, NULL);
        if (rec->host_mem) g.f.vkFreeMemory(g.device, rec->host_mem, NULL);
        if (rec->dev_buf) g.f.vkDestroyBuffer(g.device, rec->dev_buf, NULL);
        if (rec->dev_mem) g.f.vkFreeMemory(g.device, rec->dev_mem, NULL);
    }
    free(rec);
}

/* Device-only temporary (reduction partials). */
static VkBufRec* vk_rec_device_only(VkDeviceSize size) {
    VkBufRec* rec = (VkBufRec*)calloc(1, sizeof(*rec));
    if (!rec) return NULL;
    rec->size = size;
    uint32_t type = g.host_mem_device_local ? g.host_mem_type : g.dev_mem_type;
    VkBuffer* b = g.host_mem_device_local ? &rec->host_buf : &rec->dev_buf;
    VkDeviceMemory* m = g.host_mem_device_local ? &rec->host_mem : &rec->dev_mem;
    if (!vk_make_buffer(size, type, b, m)) { vk_rec_destroy(rec); return NULL; }
    return rec;
}

static VkBufRec* vk_rec_create(VkDeviceSize size) {
    VkBufRec* rec = (VkBufRec*)calloc(1, sizeof(*rec));
    if (!rec) return NULL;
    rec->size = size;
    if (!vk_make_buffer(size, g.host_mem_type, &rec->host_buf, &rec->host_mem) ||
        g.f.vkMapMemory(g.device, rec->host_mem, 0, VK_WHOLE_SIZE, 0, &rec->mapped) != VK_SUCCESS) {
        rec->mapped = NULL;
        vk_rec_destroy(rec);
        return NULL;
    }
    if (!g.host_mem_device_local &&
        !vk_make_buffer(size, g.dev_mem_type, &rec->dev_buf, &rec->dev_mem)) {
        vk_rec_destroy(rec);
        return NULL;
    }
    return rec;
}

/* The buffer a kernel binds. */
static VkBuffer vk_rec_kernel_buffer(const VkBufRec* rec) {
    return rec->dev_buf ? rec->dev_buf : rec->host_buf;
}

static VkBufRec* vk_rec_of(const EshkolGPUBuffer* b) {
    if (!b || b->backend != ESHKOL_GPU_VULKAN || !b->backend_data) return NULL;
    return (VkBufRec*)b->backend_data;
}

/* ===================================================================== */
/* Command recording helpers (caller holds the lock)                      */
/* ===================================================================== */

static int vk_begin(void) {
    VkCommandBufferBeginInfo bi;
    if (g.f.vkResetDescriptorPool(g.device, g.desc_pool, 0) != VK_SUCCESS) return 0;
    if (g.f.vkResetCommandBuffer(g.cmd, 0) != VK_SUCCESS) return 0;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    return g.f.vkBeginCommandBuffer(g.cmd, &bi) == VK_SUCCESS;
}

static void vk_barrier(VkPipelineStageFlags src_stage, VkAccessFlags src_access,
                       VkPipelineStageFlags dst_stage, VkAccessFlags dst_access) {
    VkMemoryBarrier mb;
    memset(&mb, 0, sizeof(mb));
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = src_access;
    mb.dstAccessMask = dst_access;
    g.f.vkCmdPipelineBarrier(g.cmd, src_stage, dst_stage, 0, 1, &mb, 0, NULL, 0, NULL);
}

static void vk_copy(VkBuffer src, VkBuffer dst, VkDeviceSize bytes) {
    VkBufferCopy region;
    region.srcOffset = 0;
    region.dstOffset = 0;
    region.size = bytes;
    g.f.vkCmdCopyBuffer(g.cmd, src, dst, 1, &region);
}

static int vk_bind_and_dispatch(int pipe, VkBuffer b0, VkDeviceSize r0, VkBuffer b1,
                                VkDeviceSize r1, VkBuffer b2, VkDeviceSize r2,
                                const void* push, uint32_t gx, uint32_t gy) {
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo ai;
    memset(&ai, 0, sizeof(ai));
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = g.desc_pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &g.set_layout;
    if (g.f.vkAllocateDescriptorSets(g.device, &ai, &set) != VK_SUCCESS) return 0;
    VkDescriptorBufferInfo infos[3];
    VkWriteDescriptorSet w[3];
    VkBuffer bufs[3] = { b0, b1, b2 };
    VkDeviceSize ranges[3] = { r0, r1, r2 };
    memset(w, 0, sizeof(w));
    for (uint32_t i = 0; i < 3; i++) {
        infos[i].buffer = bufs[i];
        infos[i].offset = 0;
        infos[i].range = ranges[i];
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = set;
        w[i].dstBinding = i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &infos[i];
    }
    g.f.vkUpdateDescriptorSets(g.device, 3, w, 0, NULL);
    g.f.vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipes[pipe]);
    g.f.vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipe_layout, 0, 1,
                                &set, 0, NULL);
    g.f.vkCmdPushConstants(g.cmd, g.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           PUSH_CONSTANT_BYTES, push);
    g.f.vkCmdDispatch(g.cmd, gx, gy, 1);
    return 1;
}

static int vk_submit_and_wait(void) {
    VkSubmitInfo si;
    if (g.f.vkEndCommandBuffer(g.cmd) != VK_SUCCESS) return 0;
    memset(&si, 0, sizeof(si));
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g.cmd;
    if (g.f.vkQueueSubmit(g.queue, 1, &si, g.fence) != VK_SUCCESS) return 0;
    VkResult r = g.f.vkWaitForFences(g.device, 1, &g.fence, VK_TRUE, UINT64_MAX);
    g.f.vkResetFences(g.device, 1, &g.fence);
    return r == VK_SUCCESS;
}

/* Split `groups` workgroups over x/y within the device limits. */
static int vk_grid(uint64_t groups, uint32_t* gx, uint32_t* gy) {
    uint64_t maxx = g.props.limits.maxComputeWorkGroupCount[0];
    uint64_t maxy = g.props.limits.maxComputeWorkGroupCount[1];
    if (groups == 0) groups = 1;
    uint64_t x = groups < maxx ? groups : maxx;
    uint64_t y = (groups + x - 1) / x;
    if (y > maxy) return 0;
    *gx = (uint32_t)x;
    *gy = (uint32_t)y;
    return 1;
}

static int vk_range_ok(uint64_t bytes) {
    return bytes > 0 && bytes <= (uint64_t)g.props.limits.maxStorageBufferRange;
}

/* Stage an input: caller memory -> mapped (wrapped buffers), mapped -> VRAM. */
static void vk_stage_in(const EshkolGPUBuffer* b, VkBufRec* rec, uint64_t bytes, int* copies) {
    if (rec->wrapped && b->host_ptr != rec->mapped) memcpy(rec->mapped, b->host_ptr, (size_t)bytes);
    if (rec->dev_buf) { vk_copy(rec->host_buf, rec->dev_buf, bytes); (*copies)++; }
}

/* After the last dispatch: VRAM -> mapped, then make the writes host-visible. */
static void vk_stage_out_record(VkBufRec* rec, uint64_t bytes) {
    if (rec->dev_buf) {
        vk_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        vk_copy(rec->dev_buf, rec->host_buf, bytes);
        vk_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
    } else {
        vk_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                   VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
    }
}

static void vk_stage_out_copy(EshkolGPUBuffer* b, VkBufRec* rec, uint64_t bytes) {
    if (rec->wrapped && b->host_ptr != rec->mapped) memcpy(b->host_ptr, rec->mapped, (size_t)bytes);
}

static void vk_inputs_ready(int copies) {
    if (copies)
        vk_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
}

/* ===================================================================== */
/* Kernels (caller holds the lock; admission not checked here)            */
/* ===================================================================== */

typedef struct { double ident; uint32_t n, op, has_b, pad; } PushElementwise;
typedef struct { double init; uint32_t n, op, chunk, pad; } PushReduce;
typedef struct { uint32_t M, K, N, pad[3]; } PushMatmul;

static int vk_elementwise_locked(EshkolGPUBuffer* a, EshkolGPUBuffer* b, EshkolGPUBuffer* out,
                                 uint64_t n, EshkolElementwiseOp op) {
    VkBufRec* ra = vk_rec_of(a);
    VkBufRec* rb = b ? vk_rec_of(b) : NULL;
    VkBufRec* ro = vk_rec_of(out);
    uint64_t bytes = n * sizeof(double);
    uint32_t gx, gy;
    int copies = 0;
    if (!ra || !ro || (b && b->host_ptr && !rb)) return ESHKOL_VK_UNSUPPORTED;
    if (n > 0xFFFFFFFFull || !vk_range_ok(bytes) || bytes > ra->size || bytes > ro->size ||
        (rb && bytes > rb->size))
        return ESHKOL_VK_UNSUPPORTED;
    if (!vk_grid((n + 255) / 256, &gx, &gy)) return ESHKOL_VK_UNSUPPORTED;
    PushElementwise pc;
    memset(&pc, 0, sizeof(pc));
    pc.ident = (op == ESHKOL_ELEMWISE_MUL || op == ESHKOL_ELEMWISE_DIV) ? 1.0 : 0.0;
    pc.n = (uint32_t)n;
    pc.op = (uint32_t)op;
    pc.has_b = rb ? 1u : 0u;
    if (!vk_begin()) return ESHKOL_VK_DEVICE_ERROR;
    vk_stage_in(a, ra, bytes, &copies);
    if (rb && rb != ra) vk_stage_in(b, rb, bytes, &copies);
    vk_inputs_ready(copies);
    VkBuffer kb = rb ? vk_rec_kernel_buffer(rb) : vk_rec_kernel_buffer(ra);
    if (!vk_bind_and_dispatch(PIPE_ELEMENTWISE, vk_rec_kernel_buffer(ra), bytes, kb, bytes,
                              vk_rec_kernel_buffer(ro), bytes, &pc, gx, gy))
        return ESHKOL_VK_DEVICE_ERROR;
    vk_stage_out_record(ro, bytes);
    if (!vk_submit_and_wait()) return ESHKOL_VK_DEVICE_ERROR;
    vk_stage_out_copy(out, ro, bytes);
    return ESHKOL_VK_OK;
}

static double vk_reduce_identity(EshkolReduceOp op) {
    switch (op) {
        case ESHKOL_REDUCE_PROD: return 1.0;
        case ESHKOL_REDUCE_MIN: return INFINITY;
        case ESHKOL_REDUCE_MAX: return -INFINITY;
        case ESHKOL_REDUCE_SUM:
        case ESHKOL_REDUCE_MEAN:
        default: return 0.0;
    }
}

static int vk_reduce_locked(EshkolGPUBuffer* in, EshkolGPUBuffer* out, uint64_t n,
                            EshkolReduceOp op) {
    VkBufRec* ri = vk_rec_of(in);
    VkBufRec* ro = vk_rec_of(out);
    VkBufRec* tmp[2] = { NULL, NULL };
    uint64_t bytes = n * sizeof(double);
    int copies = 0;
    int rc = ESHKOL_VK_DEVICE_ERROR;
    if (!ri || !ro) return ESHKOL_VK_UNSUPPORTED;
    if (n > 0xFFFFFFFFull || !vk_range_ok(bytes) || bytes > ri->size || ro->size < sizeof(double))
        return ESHKOL_VK_UNSUPPORTED;
    uint64_t first_partials = (n + ESHKOL_VK_REDUCE_CHUNK - 1) / ESHKOL_VK_REDUCE_CHUNK;
    if (first_partials > 1) {
        tmp[0] = vk_rec_device_only(first_partials * sizeof(double));
        uint64_t second = (first_partials + ESHKOL_VK_REDUCE_CHUNK - 1) / ESHKOL_VK_REDUCE_CHUNK;
        if (!tmp[0]) goto done;
        if (second > 1) {
            tmp[1] = vk_rec_device_only(second * sizeof(double));
            if (!tmp[1]) goto done;
        }
    }
    if (!vk_begin()) goto done;
    vk_stage_in(in, ri, bytes, &copies);
    vk_inputs_ready(copies);
    {
        VkBuffer src = vk_rec_kernel_buffer(ri);
        VkDeviceSize src_bytes = bytes;
        uint64_t count = n;
        int pass = 0;
        for (;;) {
            uint64_t partials = (count + ESHKOL_VK_REDUCE_CHUNK - 1) / ESHKOL_VK_REDUCE_CHUNK;
            uint32_t gx, gy;
            VkBuffer dst;
            VkDeviceSize dst_bytes = partials * sizeof(double);
            if (partials == 1) {
                dst = vk_rec_kernel_buffer(ro);
                dst_bytes = sizeof(double);
            } else {
                dst = vk_rec_kernel_buffer(tmp[pass & 1]);
            }
            if (!vk_grid((partials + 63) / 64, &gx, &gy)) { rc = ESHKOL_VK_UNSUPPORTED; goto done; }
            PushReduce pc;
            memset(&pc, 0, sizeof(pc));
            pc.init = vk_reduce_identity(op);
            pc.n = (uint32_t)count;
            pc.op = (uint32_t)op;
            pc.chunk = ESHKOL_VK_REDUCE_CHUNK;
            if (pass > 0)
                vk_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
            if (pass >= (int)MAX_SETS_PER_OP ||
                !vk_bind_and_dispatch(PIPE_REDUCE, src, src_bytes, src, src_bytes, dst, dst_bytes,
                                      &pc, gx, gy))
                goto done;
            if (partials == 1) break;
            src = dst;
            src_bytes = dst_bytes;
            count = partials;
            pass++;
        }
    }
    vk_stage_out_record(ro, sizeof(double));
    if (!vk_submit_and_wait()) goto done;
    {
        double* res = (double*)ro->mapped;
        if (op == ESHKOL_REDUCE_MEAN) res[0] = res[0] / (double)n;
        vk_stage_out_copy(out, ro, sizeof(double));
    }
    rc = ESHKOL_VK_OK;
done:
    vk_rec_destroy(tmp[0]);
    vk_rec_destroy(tmp[1]);
    return rc;
}

static int vk_matmul_locked(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                            uint64_t M, uint64_t K, uint64_t N, size_t elem, int pipe) {
    VkBufRec* ra = vk_rec_of(A);
    VkBufRec* rb = vk_rec_of(B);
    VkBufRec* rc_ = vk_rec_of(C);
    uint32_t gx, gy;
    int copies = 0;
    if (!ra || !rb || !rc_) return ESHKOL_VK_UNSUPPORTED;
    if (M == 0 || N == 0 || M * K > 0xFFFFFFFFull || K * N > 0xFFFFFFFFull ||
        M * N > 0xFFFFFFFFull)
        return ESHKOL_VK_UNSUPPORTED;
    uint64_t ab = M * K * elem, bb = K * N * elem, cb = M * N * elem;
    if (K == 0 || !vk_range_ok(ab) || !vk_range_ok(bb) || !vk_range_ok(cb) ||
        ab > ra->size || bb > rb->size || cb > rc_->size)
        return ESHKOL_VK_UNSUPPORTED;
    gx = (uint32_t)((N + 15) / 16);
    gy = (uint32_t)((M + 15) / 16);
    if (gx > g.props.limits.maxComputeWorkGroupCount[0] ||
        gy > g.props.limits.maxComputeWorkGroupCount[1])
        return ESHKOL_VK_UNSUPPORTED;
    PushMatmul pc;
    memset(&pc, 0, sizeof(pc));
    pc.M = (uint32_t)M;
    pc.K = (uint32_t)K;
    pc.N = (uint32_t)N;
    if (!vk_begin()) return ESHKOL_VK_DEVICE_ERROR;
    vk_stage_in(A, ra, ab, &copies);
    if (rb != ra) vk_stage_in(B, rb, bb, &copies);
    vk_inputs_ready(copies);
    if (!vk_bind_and_dispatch(pipe, vk_rec_kernel_buffer(ra), ab, vk_rec_kernel_buffer(rb), bb,
                              vk_rec_kernel_buffer(rc_), cb, &pc, gx, gy))
        return ESHKOL_VK_DEVICE_ERROR;
    vk_stage_out_record(rc_, cb);
    if (!vk_submit_and_wait()) return ESHKOL_VK_DEVICE_ERROR;
    vk_stage_out_copy(C, rc_, cb);
    return ESHKOL_VK_OK;
}

/* ===================================================================== */
/* Admission probe                                                        */
/* ===================================================================== */

/* Host references, written so the compiler cannot fuse or reorder them. */
static double ref_elementwise(double x, double y, int has_b, EshkolElementwiseOp op) {
    volatile double r;
    switch (op) {
        case ESHKOL_ELEMWISE_ADD: r = x + (has_b ? y : 0.0); break;
        case ESHKOL_ELEMWISE_SUB: r = x - (has_b ? y : 0.0); break;
        case ESHKOL_ELEMWISE_MUL: r = x * (has_b ? y : 1.0); break;
        case ESHKOL_ELEMWISE_DIV: r = x / (has_b ? y : 1.0); break;
        case ESHKOL_ELEMWISE_NEG: r = -x; break;
        case ESHKOL_ELEMWISE_ABS: r = x < 0 ? -x : x; break;
        case ESHKOL_ELEMWISE_RELU: r = x > 0 ? x : 0; break;
        case ESHKOL_ELEMWISE_SQRT: r = sqrt(x); break;
        case ESHKOL_ELEMWISE_RECIPROCAL: r = 1.0 / x; break;
        default: r = x; break;
    }
    return r;
}

static int same_bits(double a, double b) {
    if (isnan(a) && isnan(b)) return 1;
    return memcmp(&a, &b, sizeof(double)) == 0;
}

static uint64_t probe_rng(uint64_t* s) {
    *s = *s * 6364136223846793005ull + 1442695040888963407ull;
    uint64_t z = *s;
    z ^= z >> 33; z *= 0xff51afd7ed558ccdull; z ^= z >> 33;
    return z;
}

static double probe_value(uint64_t* s) {
    uint64_t r = probe_rng(s);
    if ((r & 3) == 0) {  /* raw bit pattern: subnormals, huge, tiny, NaN, inf */
        double d;
        uint64_t bits = probe_rng(s);
        memcpy(&d, &bits, sizeof(d));
        return d;
    }
    return ((double)(int64_t)(probe_rng(s) >> 11) / 9007199254740992.0) *
           ldexp(1.0, (int)(r >> 58) - 16);
}

static const double k_probe_edges[] = {
    0.0, -0.0, 1.0, -1.0, 0.5, 3.0, 0.1, 1.0 / 3.0, 3.141592653589793,
    4.9406564584124654e-324, -2.5e-320, 2.2250738585072014e-308,
    1.7976931348623157e308, -1.7976931348623157e308, 1e300, 1e-300,
    9007199254740993.0, INFINITY, -INFINITY, NAN
};
#define PROBE_EDGES (sizeof(k_probe_edges) / sizeof(k_probe_edges[0]))
#define PROBE_RANDOM 4096u
#define PROBE_REDUCE_N 70000u  /* three passes: 70000 -> 274 -> 2 -> 1 */

static void vk_probe_locked(void) {
    const uint64_t n = PROBE_EDGES * PROBE_EDGES + PROBE_RANDOM;
    const uint64_t cap = PROBE_REDUCE_N;  /* >= n; also holds the reduction input */
    EshkolGPUBuffer a, b, o, s;
    double* ha = (double*)malloc((size_t)cap * sizeof(double));
    double* hb = (double*)malloc((size_t)cap * sizeof(double));
    double* ho = (double*)malloc((size_t)cap * sizeof(double));
    uint64_t seed = 0x5eed0fe5c0ffee01ull;
    memset(&a, 0, sizeof(a)); memset(&b, 0, sizeof(b));
    memset(&o, 0, sizeof(o)); memset(&s, 0, sizeof(s));
    g.elementwise_admitted = 0;
    g.reduce_admitted = 0;
    g.matmul_f64_admitted = 0;
    g.matmul_f32_admitted = 0;
    if (!ha || !hb || !ho) goto out;
    for (uint64_t i = 0; i < PROBE_EDGES * PROBE_EDGES; i++) {
        ha[i] = k_probe_edges[i / PROBE_EDGES];
        hb[i] = k_probe_edges[i % PROBE_EDGES];
    }
    for (uint64_t i = PROBE_EDGES * PROBE_EDGES; i < n; i++) {
        ha[i] = probe_value(&seed);
        hb[i] = probe_value(&seed);
    }
    {
        VkBufRec* r1 = vk_rec_create(cap * sizeof(double));
        VkBufRec* r2 = vk_rec_create(cap * sizeof(double));
        VkBufRec* r3 = vk_rec_create(cap * sizeof(double));
        if (!r1 || !r2 || !r3) {
            vk_rec_destroy(r1); vk_rec_destroy(r2); vk_rec_destroy(r3);
            goto out;
        }
        r1->wrapped = r2->wrapped = r3->wrapped = 1;
        a.host_ptr = ha; a.backend = ESHKOL_GPU_VULKAN; a.backend_data = r1; a.size_bytes = cap * 8;
        b.host_ptr = hb; b.backend = ESHKOL_GPU_VULKAN; b.backend_data = r2; b.size_bytes = cap * 8;
        o.host_ptr = ho; o.backend = ESHKOL_GPU_VULKAN; o.backend_data = r3; o.size_bytes = cap * 8;
    }
    {
        static const EshkolElementwiseOp ops[] = {
            ESHKOL_ELEMWISE_ADD, ESHKOL_ELEMWISE_SUB, ESHKOL_ELEMWISE_MUL, ESHKOL_ELEMWISE_DIV,
            ESHKOL_ELEMWISE_NEG, ESHKOL_ELEMWISE_ABS, ESHKOL_ELEMWISE_RELU, ESHKOL_ELEMWISE_SQRT,
            ESHKOL_ELEMWISE_RECIPROCAL
        };
        for (size_t k = 0; k < sizeof(ops) / sizeof(ops[0]); k++) {
            EshkolElementwiseOp op = ops[k];
            int binary = op <= ESHKOL_ELEMWISE_DIV;
            int ok = 1;
            /* Binary ops are probed with and without the b operand. */
            for (int with_b = binary; with_b >= 0 && ok; with_b--) {
                if (vk_elementwise_locked(&a, with_b ? &b : NULL, &o, n, op) != ESHKOL_VK_OK) {
                    ok = 0;
                    break;
                }
                for (uint64_t i = 0; i < n && ok; i++)
                    ok = same_bits(ho[i], ref_elementwise(ha[i], hb[i], with_b, op));
            }
            if (ok) g.elementwise_admitted |= 1u << (unsigned)op;
        }
    }
    {
        /* Reductions: the fixed chunk tree, mirrored on the host. Inputs avoid
         * NaN so min/max probe ordering, not NaN propagation. */
        const uint64_t rn = PROBE_REDUCE_N;
        double res = 0.0;
        VkBufRec* r4 = vk_rec_create(sizeof(double));
        double* tree = (double*)malloc((size_t)rn * sizeof(double));
        if (r4 && tree) {
            r4->wrapped = 1;
            s.host_ptr = &res; s.backend = ESHKOL_GPU_VULKAN; s.backend_data = r4; s.size_bytes = 8;
            for (uint64_t i = 0; i < rn; i++) {
                double v = probe_value(&seed);
                if (isnan(v) || isinf(v) || fabs(v) > 1e30) v = (double)(i % 977) * 0.001 - 0.3;
                ha[i] = v;
            }
            for (int op = ESHKOL_REDUCE_SUM; op <= ESHKOL_REDUCE_MEAN; op++) {
                uint64_t count = rn;
                memcpy(tree, ha, (size_t)rn * sizeof(double));
                while (1) {
                    uint64_t parts = (count + ESHKOL_VK_REDUCE_CHUNK - 1) / ESHKOL_VK_REDUCE_CHUNK;
                    for (uint64_t p = 0; p < parts; p++) {
                        volatile double r = vk_reduce_identity((EshkolReduceOp)op);
                        uint64_t end = (p + 1) * ESHKOL_VK_REDUCE_CHUNK;
                        if (end > count) end = count;
                        for (uint64_t i = p * ESHKOL_VK_REDUCE_CHUNK; i < end; i++) {
                            double x = tree[i];
                            if (op == ESHKOL_REDUCE_SUM || op == ESHKOL_REDUCE_MEAN) r = r + x;
                            else if (op == ESHKOL_REDUCE_PROD) r = r * x;
                            else if (op == ESHKOL_REDUCE_MIN) r = (x < r) ? x : r;
                            else r = (x > r) ? x : r;
                        }
                        tree[p] = r;
                    }
                    count = parts;
                    if (parts == 1) break;
                }
                double expect = tree[0];
                if (op == ESHKOL_REDUCE_MEAN) expect = expect / (double)rn;
                if (vk_reduce_locked(&a, &s, rn, (EshkolReduceOp)op) == ESHKOL_VK_OK &&
                    same_bits(res, expect))
                    g.reduce_admitted |= 1u << (unsigned)op;
            }
        }
        free(tree);
        vk_rec_destroy(r4);
        s.backend_data = NULL;
    }
    {
        /* Matmul: odd shapes crossing the 16x16 tile edge. */
        const uint64_t M = 19, K = 23, N = 17;
        for (uint64_t i = 0; i < M * K; i++) ha[i] = probe_value(&seed);
        for (uint64_t i = 0; i < K * N; i++) hb[i] = probe_value(&seed);
        for (uint64_t i = 0; i < M * K; i++) if (isnan(ha[i]) || isinf(ha[i])) ha[i] = 0.25;
        for (uint64_t i = 0; i < K * N; i++) if (isnan(hb[i]) || isinf(hb[i])) hb[i] = -0.75;
        if (vk_matmul_locked(&a, &b, &o, M, K, N, sizeof(double), PIPE_MATMUL_F64) == ESHKOL_VK_OK) {
            int ok = 1;
            for (uint64_t i = 0; i < M && ok; i++)
                for (uint64_t j = 0; j < N && ok; j++) {
                    volatile double acc = 0.0;
                    for (uint64_t k = 0; k < K; k++) {
                        volatile double prod = ha[i * K + k] * hb[k * N + j];
                        acc = acc + prod;
                    }
                    ok = same_bits(ho[i * N + j], acc);
                }
            g.matmul_f64_admitted = ok;
        }
        float* fa = (float*)ha;
        float* fb = (float*)hb;
        float* fo = (float*)ho;
        for (uint64_t i = 0; i < M * K; i++) fa[i] = (float)((double)(i % 37) * 0.37 - 5.0);
        for (uint64_t i = 0; i < K * N; i++) fb[i] = (float)((double)(i % 29) * -0.29 + 3.0);
        if (vk_matmul_locked(&a, &b, &o, M, K, N, sizeof(float), PIPE_MATMUL_F32) == ESHKOL_VK_OK) {
            int ok = 1;
            for (uint64_t i = 0; i < M && ok; i++)
                for (uint64_t j = 0; j < N && ok; j++) {
                    volatile float acc = 0.0f;
                    for (uint64_t k = 0; k < K; k++) {
                        volatile float prod = fa[i * K + k] * fb[k * N + j];
                        acc = acc + prod;
                    }
                    float got = fo[i * N + j], want = acc;
                    ok = memcmp(&got, &want, sizeof(float)) == 0;
                }
            g.matmul_f32_admitted = ok;
        }
    }
out:
    vk_rec_destroy((VkBufRec*)a.backend_data);
    vk_rec_destroy((VkBufRec*)b.backend_data);
    vk_rec_destroy((VkBufRec*)o.backend_data);
    free(ha);
    free(hb);
    free(ho);
}

/* ===================================================================== */
/* Public API                                                             */
/* ===================================================================== */

static int vk_open_index_locked(uint32_t index, VkPhysicalDevice pd) {
    if (!vk_create_device_locked(index, pd)) return 0;
    vk_probe_locked();
    if (!g.elementwise_admitted && !g.reduce_admitted && !g.matmul_f64_admitted) {
        vk_status("device %u (%s): no kernel passed the admission probe", index,
                  g.props.deviceName);
        vk_destroy_device_locked();
        return 0;
    }
    vk_status("device %u: %s; memory %s; admitted elementwise 0x%x reduce 0x%x matmul f64 %d f32 %d",
              index, g.props.deviceName,
              g.host_mem_device_local ? "host-visible (shared)" : "device-local with host staging",
              g.elementwise_admitted, g.reduce_admitted, g.matmul_f64_admitted,
              g.matmul_f32_admitted);
    return 1;
}

int eshkol_vk_init_device(int index) {
    VkPhysicalDevice* list = NULL;
    int ok = 0;
    vk_lock();
    if (g.device_ready) {
        if (g.device_index == index) { vk_unlock(); return 1; }
        vk_destroy_device_locked();
    }
    uint32_t count = vk_list_devices(&list);
    if (index < 0 || (uint32_t)index >= count) {
        vk_status("Vulkan device %d not present (%u devices)", index, count);
    } else {
        ok = vk_open_index_locked((uint32_t)index, list[index]);
    }
    free(list);
    vk_unlock();
    return ok;
}

static int vk_type_rank(VkPhysicalDeviceType t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 4;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 3;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 2;
        case VK_PHYSICAL_DEVICE_TYPE_OTHER: return 1;
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return 0;
        default: return 0;
    }
}

int eshkol_vk_init(unsigned flags) {
    const char* env = getenv("ESHKOL_VULKAN_DEVICE");
    if (env && *env) return eshkol_vk_init_device(atoi(env));

    VkPhysicalDevice* list = NULL;
    int ok = 0;
    vk_lock();
    if (g.device_ready) { vk_unlock(); return 1; }
    uint32_t count = vk_list_devices(&list);
    if (count == 0) {
        if (g.instance_ready) vk_status("no Vulkan physical devices");
        free(list);
        vk_unlock();
        return 0;
    }
    /* Try candidates best-first: device type rank, then enumeration order. */
    int attempted = 0;
    for (int rank = 4; rank >= 0 && !ok; rank--) {
        if (rank == 0 && !(flags & ESHKOL_VK_ALLOW_CPU_DEVICE)) break;
        for (uint32_t i = 0; i < count && !ok; i++) {
            VkPhysicalDeviceProperties p;
            g.f.vkGetPhysicalDeviceProperties(list[i], &p);
            if (vk_type_rank(p.deviceType) != rank) continue;
            attempted = 1;
            ok = vk_open_index_locked(i, list[i]);
        }
    }
    if (!ok && !attempted)
        vk_status("%u Vulkan device(s), none GPU-type; CPU-type devices need ESHKOL_GPU_BACKEND=vulkan",
                  count);
    free(list);
    vk_unlock();
    return ok;
}

void eshkol_vk_shutdown(void) {
    vk_lock();
    vk_destroy_device_locked();
    if (g.instance_ready && g.f.vkDestroyInstance) g.f.vkDestroyInstance(g.instance, NULL);
    g.instance = NULL;
    g.instance_ready = 0;
    /* The loader stays mapped: unloading ICDs mid-process is not safe on every driver. */
    vk_unlock();
}

int eshkol_vk_available(void) { return g.device_ready; }
const char* eshkol_vk_status(void) { return g.status[0] ? g.status : "Vulkan not initialised"; }
const char* eshkol_vk_device_name(void) { return g.device_ready ? g.props.deviceName : ""; }
int eshkol_vk_device_type(void) { return g.device_ready ? (int)g.props.deviceType : -1; }
int eshkol_vk_device_index(void) { return g.device_ready ? g.device_index : -1; }
int eshkol_vk_supports_f64(void) { return g.device_ready; }

int eshkol_vk_device_count(void) {
    VkPhysicalDevice* list = NULL;
    vk_lock();
    uint32_t count = vk_list_devices(&list);
    free(list);
    vk_unlock();
    return (int)count;
}

int eshkol_vk_device_describe(int index, char* name, size_t name_cap, int* device_type,
                              int* has_f64) {
    VkPhysicalDevice* list = NULL;
    int ok = 0;
    vk_lock();
    uint32_t count = vk_list_devices(&list);
    if (index >= 0 && (uint32_t)index < count) {
        VkPhysicalDeviceProperties p;
        VkPhysicalDeviceFeatures f;
        memset(&f, 0, sizeof(f));
        g.f.vkGetPhysicalDeviceProperties(list[index], &p);
        g.f.vkGetPhysicalDeviceFeatures(list[index], &f);
        if (name && name_cap) snprintf(name, name_cap, "%s", p.deviceName);
        if (device_type) *device_type = (int)p.deviceType;
        if (has_f64) *has_f64 = f.shaderFloat64 ? 1 : 0;
        ok = 1;
    }
    free(list);
    vk_unlock();
    return ok;
}

int eshkol_vk_elementwise_admitted(EshkolElementwiseOp op) {
    return g.device_ready && (unsigned)op < 32u && (g.elementwise_admitted >> (unsigned)op) & 1u;
}
int eshkol_vk_reduce_admitted(EshkolReduceOp op) {
    return g.device_ready && (unsigned)op < 32u && (g.reduce_admitted >> (unsigned)op) & 1u;
}
int eshkol_vk_matmul_admitted(int single_precision) {
    return g.device_ready && (single_precision ? g.matmul_f32_admitted : g.matmul_f64_admitted);
}

static int vk_new_buffer(void* host_ptr, size_t size_bytes, EshkolGPUBuffer* out, int wrapped) {
    if (!out) return ESHKOL_VK_INVALID;
    memset(out, 0, sizeof(*out));
    if (size_bytes == 0 || (wrapped && !host_ptr)) return ESHKOL_VK_INVALID;
    vk_lock();
    if (!g.device_ready) { vk_unlock(); return ESHKOL_VK_UNSUPPORTED; }
    VkBufRec* rec = vk_rec_create(size_bytes);
    vk_unlock();
    if (!rec) return ESHKOL_VK_DEVICE_ERROR;
    rec->wrapped = wrapped;
    out->host_ptr = wrapped ? host_ptr : rec->mapped;
    out->device_ptr = rec->mapped;
    out->size_bytes = size_bytes;
    out->mem_type = wrapped ? ESHKOL_MEM_HOST : ESHKOL_MEM_UNIFIED;
    out->backend = ESHKOL_GPU_VULKAN;
    out->flags = wrapped ? 1u : 0u;  /* bit 0: non-owning wrap, as in the other backends */
    out->backend_data = rec;
    return ESHKOL_VK_OK;
}

int eshkol_vk_alloc(size_t size_bytes, EshkolGPUBuffer* out_buffer) {
    return vk_new_buffer(NULL, size_bytes, out_buffer, 0);
}

int eshkol_vk_wrap_host(void* host_ptr, size_t size_bytes, EshkolGPUBuffer* out_buffer) {
    return vk_new_buffer(host_ptr, size_bytes, out_buffer, 1);
}

void eshkol_vk_free(EshkolGPUBuffer* buffer) {
    VkBufRec* rec = vk_rec_of(buffer);
    if (rec) {
        vk_lock();
        vk_rec_destroy(rec);
        vk_unlock();
    }
    if (buffer) memset(buffer, 0, sizeof(*buffer));
}

int eshkol_vk_sync(EshkolGPUBuffer* buffer, EshkolSyncDirection direction) {
    VkBufRec* rec = vk_rec_of(buffer);
    int rc = ESHKOL_VK_OK;
    if (!rec) return ESHKOL_VK_INVALID;
    vk_lock();
    if (!g.device_ready) { vk_unlock(); return ESHKOL_VK_UNSUPPORTED; }
    if (direction == ESHKOL_SYNC_HOST_TO_DEVICE || direction == ESHKOL_SYNC_BIDIRECTIONAL) {
        if (rec->wrapped) memcpy(rec->mapped, buffer->host_ptr, (size_t)rec->size);
        if (rec->dev_buf) {
            if (!vk_begin()) rc = ESHKOL_VK_DEVICE_ERROR;
            else {
                vk_copy(rec->host_buf, rec->dev_buf, rec->size);
                if (!vk_submit_and_wait()) rc = ESHKOL_VK_DEVICE_ERROR;
            }
        }
    }
    if (rc == ESHKOL_VK_OK &&
        (direction == ESHKOL_SYNC_DEVICE_TO_HOST || direction == ESHKOL_SYNC_BIDIRECTIONAL)) {
        if (rec->dev_buf) {
            if (!vk_begin()) rc = ESHKOL_VK_DEVICE_ERROR;
            else {
                vk_copy(rec->dev_buf, rec->host_buf, rec->size);
                vk_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                           VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
                if (!vk_submit_and_wait()) rc = ESHKOL_VK_DEVICE_ERROR;
            }
        }
        if (rc == ESHKOL_VK_OK && rec->wrapped)
            memcpy(buffer->host_ptr, rec->mapped, (size_t)rec->size);
    }
    vk_unlock();
    return rc;
}

int eshkol_vk_elementwise_f64(EshkolGPUBuffer* a, EshkolGPUBuffer* b, EshkolGPUBuffer* out,
                              uint64_t n, EshkolElementwiseOp op) {
    int rc;
    if (!a || !out || n == 0) return ESHKOL_VK_INVALID;
    vk_lock();
    if (!g.device_ready || (unsigned)op >= 32u || !((g.elementwise_admitted >> (unsigned)op) & 1u))
        rc = ESHKOL_VK_UNSUPPORTED;
    else
        rc = vk_elementwise_locked(a, b, out, n, op);
    vk_unlock();
    return rc;
}

int eshkol_vk_reduce_f64(EshkolGPUBuffer* in, EshkolGPUBuffer* out, uint64_t n,
                         EshkolReduceOp op) {
    int rc;
    if (!in || !out || n == 0) return ESHKOL_VK_INVALID;
    vk_lock();
    if (!g.device_ready || (unsigned)op >= 32u || !((g.reduce_admitted >> (unsigned)op) & 1u))
        rc = ESHKOL_VK_UNSUPPORTED;
    else
        rc = vk_reduce_locked(in, out, n, op);
    vk_unlock();
    return rc;
}

int eshkol_vk_matmul_f64(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                         uint64_t M, uint64_t K, uint64_t N) {
    int rc;
    if (!A || !B || !C) return ESHKOL_VK_INVALID;
    vk_lock();
    rc = (g.device_ready && g.matmul_f64_admitted)
             ? vk_matmul_locked(A, B, C, M, K, N, sizeof(double), PIPE_MATMUL_F64)
             : ESHKOL_VK_UNSUPPORTED;
    vk_unlock();
    return rc;
}

int eshkol_vk_matmul_f32(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                         uint64_t M, uint64_t K, uint64_t N) {
    int rc;
    if (!A || !B || !C) return ESHKOL_VK_INVALID;
    vk_lock();
    rc = (g.device_ready && g.matmul_f32_admitted)
             ? vk_matmul_locked(A, B, C, M, K, N, sizeof(float), PIPE_MATMUL_F32)
             : ESHKOL_VK_UNSUPPORTED;
    vk_unlock();
    return rc;
}
