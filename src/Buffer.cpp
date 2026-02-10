#include "Buffer.h"
#include "Context.h"
#include "Util.hpp"

#include <mutex>
#include <print>

void AllocBuffer::initHost(std::string name, VulkanContext &vkc, size_t size, vk::BufferUsageFlags usage) {
    this->init(name, vkc, size, usage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, 
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT
    );
}

void AllocBuffer::initDevice(std::string name, VulkanContext &vkc, size_t size, vk::BufferUsageFlags usage) {
    this->init(name, vkc, size, usage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
}

void AllocBuffer::init(
    std::string name,
    VulkanContext &vkc, 
    size_t size, 
    vk::BufferUsageFlags usage, 
    VmaMemoryUsage memUsage,
    VmaAllocationCreateFlags allocFlags
) {
    this->name = std::move(name);
    
    if (buffer != NULL) { 
        this->deinit(vkc.allocator);
    }
    this->size = size;
    vk::BufferCreateInfo bufferInfo{{}, this->size, usage /* | vk::BufferUsageFlagBits::eShaderDeviceAddress */ };

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = memUsage;
    allocInfo.flags = allocFlags;
    
    std::scoped_lock lock{vkc.createMtx};
    auto result = vmaCreateBuffer(vkc.allocator, 
        reinterpret_cast<VkBufferCreateInfo *>(&bufferInfo), 
        &allocInfo, 
        reinterpret_cast<VkBuffer *>(&this->buffer), 
        &this->alloc, 
        &this->info
    );
    if (result != VkResult::VK_SUCCESS) {
        std::println("vmaCreateBuffer FAILED for {}", name);
    }
    vk::DebugUtilsObjectNameInfoEXT nameInfo{
        vk::ObjectType::eBuffer, 
        (uint64_t)(VkBuffer)buffer, 
        name.c_str()
    };
    Unwrap(vkc.device.setDebugUtilsObjectNameEXT(&nameInfo, vkc.dldid), "Buffer debug naming failed");
    // std::println("Init Buffer {}", this->name); // -R

}