#pragma once

#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.hpp>

struct VulkanContext;

struct AllocBuffer {
    vk::Buffer buffer = nullptr;
    vma::Allocation alloc;
    vma::AllocationInfo info;
    size_t size = 0, capacity = 0;
    vk::DeviceAddress address = 0;
    std::string name;

    void initHost(std::string name, VulkanContext &vkc, size_t size, vk::BufferUsageFlags usage);
    void initDevice(std::string name, VulkanContext &vkc, size_t size, vk::BufferUsageFlags usage);
    void init(
        std::string name,
        VulkanContext &vkc, 
        size_t size, 
        size_t capacity,
        vk::BufferUsageFlags usage, 
        vma::MemoryUsage memUsage = vma::MemoryUsage::eAuto,
        vma::AllocationCreateFlags allocFlags = {}
    );
    void resize(size_t newSize);

    void deinit(VulkanContext &vkc);
};