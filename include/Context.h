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
        bool operator==(Queue lhs) {
            return (vk::Queue)lhs == (vk::Queue)*this;
        }
    };
    // ... add queue mutexes
    Queue graphicsQueue, presentQueue, transferQueue;
    std::mutex queueMtx1, queueMtx2, queueMtx3;
    std::mutex &graphicsQueueMtx() {
        return queueMtx1;
    }
    std::mutex &presentQueueMtx() {
        if (graphicsQueue == presentQueue) return queueMtx1;
        return queueMtx2;
    }
    std::mutex &transferQueueMtx() {
        if (graphicsQueue == transferQueue) return queueMtx1;
        if (presentQueue == transferQueue) return queueMtx2;
        return queueMtx3;
    }
    std::mutex createMtx;
    vk::CommandPool transferCommandPool;
    std::mutex transferCommandPoolMtx; // ... create multiple pools for each thread instead.
    struct Transfer {
        vk::CommandBuffer cmdBuf;
        vk::Fence fence;
        AllocBuffer stagingBuffer;
    };
    std::queue<Transfer> transfers;
    vk::CommandPool commandPool;
    std::mutex commandPoolMtx;
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

    std::queue<std::pair<size_t, AllocBuffer>> destruction_queue;

    struct Upload { 
        AllocBuffer dst;  
        const void* src; 
    };
    void beginTransferBatch();
    void endTransferBatch();
    std::mutex uploadMtx;
    void uploadBuffers(vk::ArrayProxy<const Upload>);
    void waitForTransfers();
    void pruneDestructionQueue();
    void queueDestroy(AllocBuffer);
};