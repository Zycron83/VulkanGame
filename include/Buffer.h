#pragma once

#include <print>

#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.hpp>

struct VulkanContext;

struct AllocBuffer {
    vk::Buffer buffer;
    vma::Allocation alloc;
    vma::AllocationInfo info;
    size_t size, capacity;
    vk::DeviceAddress address;
    std::string name;

    void initHost(std::string name, VulkanContext &vkc, size_t size, vk::BufferUsageFlags usage);
    void initDevice(std::string name, VulkanContext &vkc, size_t size, vk::BufferUsageFlags usage);
    void init(
        std::string name,
        VulkanContext &vkc, 
        size_t size, 
        vk::BufferUsageFlags usage, 
        vma::MemoryUsage memUsage = vma::MemoryUsage::eAuto,
        vma::AllocationCreateFlags allocFlags = {}
    );

    void deinit(vma::Allocator allocator) {
        allocator.destroyBuffer(buffer, alloc);
        // std::println("Deinit Buffer {}", name);
        buffer = nullptr;
        address = 0;
        size = 0;
    }
};