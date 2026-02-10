#pragma once

#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>

struct VulkanContext;

struct AllocBuffer {
    vk::Buffer buffer;
    VmaAllocation alloc;
    VmaAllocationInfo info;
    size_t size;
    vk::DeviceAddress address;
    std::string name;

    void initHost(std::string name, VulkanContext &vkc, size_t size, vk::BufferUsageFlags usage);
    void initDevice(std::string name, VulkanContext &vkc, size_t size, vk::BufferUsageFlags usage);
    void init(
        std::string name,
        VulkanContext &vkc, 
        size_t size, 
        vk::BufferUsageFlags usage, 
        VmaMemoryUsage memUsage = VMA_MEMORY_USAGE_AUTO,
        VmaAllocationCreateFlags allocFlags = 0
    );

    void deinit(VmaAllocator allocator) {
        vmaDestroyBuffer(allocator, buffer, alloc);
        // std::println("Deinit Buffer {}", name);
        buffer = nullptr;
        size = 0;
    }
};