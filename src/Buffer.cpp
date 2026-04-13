#include "Buffer.h"
#include "Context.h"
#include "Util.hpp"
#include "vk_mem_alloc_enums.hpp"
#include "vulkan/vulkan.hpp"

#include <mutex>
#include <print>

void AllocBuffer::initHost(std::string name, VulkanContext &vkc, size_t size, vk::BufferUsageFlags usage) {
    this->init(name, vkc, size, size, usage, vma::MemoryUsage::eAutoPreferHost, 
        vma::AllocationCreateFlagBits::eHostAccessSequentialWrite | vma::AllocationCreateFlagBits::eMapped
    );
}

void AllocBuffer::initDevice(std::string name, VulkanContext &vkc, size_t size, vk::BufferUsageFlags usage) {
    this->init(name, vkc, size, size * 1.1, usage, vma::MemoryUsage::eAutoPreferDevice);
}

void AllocBuffer::init(
    std::string name,
    VulkanContext &vkc, 
    size_t size, 
    size_t capacity,
    vk::BufferUsageFlags usage, 
    vma::MemoryUsage memUsage,
    vma::AllocationCreateFlags allocFlags
) {
    Unwrap(buffer == nullptr, std::format("Tried to init already existing buffer. Old: {} -> New: {}", this->name, name).c_str());

    this->name = std::move(name);
    this->size = size;
    this->capacity = capacity;
    vk::BufferCreateInfo bufferInfo{{}, capacity, usage | vk::BufferUsageFlagBits::eShaderDeviceAddress};

    vma::AllocationCreateInfo allocInfo{};
    allocInfo.usage = memUsage;
    allocInfo.flags = allocFlags;
    
    std::scoped_lock lock{vkc.createMtx};
    auto result = vkc.allocator.createBuffer(
        &bufferInfo, &allocInfo, 
        &this->buffer,
        &this->alloc,
        &this->info
    );
    this->address = vkc.device.getBufferAddress(vk::BufferDeviceAddressInfo{this->buffer});

    if (result != vk::Result::eSuccess) {
        std::println("vma::createBuffer FAILED for {}", name);
    }
    vk::DebugUtilsObjectNameInfoEXT nameInfo{
        vk::ObjectType::eBuffer, 
        (uint64_t)(VkBuffer)buffer, 
        name.c_str()
    };
    Unwrap(vkc.device.setDebugUtilsObjectNameEXT(&nameInfo, vkc.dldid), "Buffer debug naming failed");
    // std::println("Init Buffer {}", this->name); // -R

}

void AllocBuffer::deinit(VulkanContext &vkc) {
    vkc.queueDestroy(std::move(*this));
    // std::println("Deinit Buffer {}", name);
    buffer = nullptr;
    address = 0;
    size = 0;
}