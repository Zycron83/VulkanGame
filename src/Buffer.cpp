#include "Buffer.h"
#include "Context.h"
#include "Debug.h"

extern DebugNameState g_DebugNameState;

void AllocBuffer::initHost(const VulkanContext &vkc, size_t size, vk::BufferUsageFlags usage) {
    this->init(vkc, size, usage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, 
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT
    );
}

void AllocBuffer::initDevice(const VulkanContext &vkc, size_t size, vk::BufferUsageFlags usage) {
    this->init(vkc, size, usage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
}

void AllocBuffer::init(
    const VulkanContext &vkc, 
    size_t size, 
    vk::BufferUsageFlags usage, 
    VmaMemoryUsage memUsage,
    VmaAllocationCreateFlags allocFlags
) {
    
    if (buffer != NULL) { 
        this->deinit(vkc.allocator);
    }
    this->size = size;
    vk::BufferCreateInfo bufferInfo{{}, this->size, usage /* | vk::BufferUsageFlagBits::eShaderDeviceAddress */ };

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = memUsage;
    allocInfo.flags = allocFlags;
    
    vmaCreateBuffer(vkc.allocator, 
        reinterpret_cast<VkBufferCreateInfo *>(&bufferInfo), 
        &allocInfo, 
        reinterpret_cast<VkBuffer *>(&this->buffer), 
        &this->alloc, 
        &this->info
    );
    g_DebugNameState.NameBuffer(buffer);

}