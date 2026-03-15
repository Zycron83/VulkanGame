#include "Buffer.h"
#include "Context.h"
#include "Util.hpp"
#include "vk_mem_alloc_enums.hpp"

#include <mutex>
#include <print>
#include "cpptrace/cpptrace.hpp"

void AllocBuffer::initHost(std::string name, VulkanContext &vkc, size_t size, vk::BufferUsageFlags usage) {
    this->init(name, vkc, size, usage, vma::MemoryUsage::eAutoPreferHost, 
        vma::AllocationCreateFlagBits::eHostAccessSequentialWrite | vma::AllocationCreateFlagBits::eMapped
    );
}

void AllocBuffer::initDevice(std::string name, VulkanContext &vkc, size_t size, vk::BufferUsageFlags usage) {
    this->init(name, vkc, size, usage, vma::MemoryUsage::eAutoPreferDevice);
}

void AllocBuffer::init(
    std::string name,
    VulkanContext &vkc, 
    size_t size, 
    vk::BufferUsageFlags usage, 
    vma::MemoryUsage memUsage,
    vma::AllocationCreateFlags allocFlags
) {
    if (buffer != NULL) {
        // cpptrace::generate_trace().print();
        // assert(false);
        this->deinit(vkc.allocator);
    }
    this->name = std::move(name);
    this->size = size;
    vk::BufferCreateInfo bufferInfo{{}, this->size, usage /* | vk::BufferUsageFlagBits::eShaderDeviceAddress */ };

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