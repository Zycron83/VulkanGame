#pragma once

#include <GLFW/glfw3.h>
#include <mutex>
#include <queue>
#include <vulkan/vulkan.hpp>

#include "Buffer.h"

struct VulkanContext {
    vk::Instance instance;
    vk::SurfaceKHR surface;
    vk::PhysicalDevice physicalDevice;
    vk::Device device;
    struct Queue : vk::Queue {
        uint32_t index;
    };
    Queue graphicsQueue, presentQueue, transferQueue;
    vk::CommandPool transferCommandPool;
    struct Transfer {
        vk::CommandBuffer cmdBuf;
        vk::Fence fence;
        AllocBuffer stagingBuffer;
    };
    std::queue<Transfer> transfers;
    vk::CommandPool commandPool;
    vk::detail::DispatchLoaderDynamic dldid;
    VmaAllocator allocator;
    constexpr static size_t FRAME_COUNT = 2;
    size_t currentFrame = 0;
    size_t frameIndex() { return currentFrame % FRAME_COUNT; };
    struct Frame {
		vk::CommandBuffer commandBuffer;
		vk::Semaphore imageAvailableSemaphore;
		vk::Fence inFlightFence;
		AllocBuffer uniformBuffer;
	};
	std::array<Frame, FRAME_COUNT> frames;

    void init(GLFWwindow *);
    void deinit();

    struct Variant;
    std::queue<std::pair<size_t, Variant>> destruction_queue;

    struct Upload { 
        AllocBuffer dst;  
        const void* src; 
    };
    vk::CommandBuffer batchTransferCommandBuffer = nullptr;
    void beginTransferBatch();
    void endTransferBatch();
    std::mutex uploadMtx;
    std::mutex submitMtx;
    void uploadBuffers(vk::ArrayProxy<Upload>);
    void waitForTransfers();
    void pruneDestructionQueue();
    void queueDestroy(Variant);
};

struct VulkanContext::Variant {
    enum Type {
        eAllocBuffer,
        eMallocPtr
    } type;
    union {
        AllocBuffer allocBuffer;
        void *mallocPtr;
    } value;
    Variant(AllocBuffer buf) : type{eAllocBuffer}, value{ .allocBuffer = buf } {}
    Variant(void *ptr) : type{eMallocPtr}, value{ .mallocPtr = ptr } {}
};