#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#include <vk_mem_alloc.h>

#include <cstdint>
#include <mutex>
#include <set>
#include <stdexcept>
#include <algorithm>
    using std::ranges::any_of, std::ranges::all_of, std::ranges::contains;
#include <print>
#include <array>

#include "Context.h"
#include "Util.hpp"
#include "Debug.h"
extern DebugNameState g_DebugNameState;

static const std::array validationLayers = {
	"VK_LAYER_KHRONOS_validation",
};
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

static const std::array requiredExtensions = {
	vk::KHRSwapchainExtensionName,
	// vk::EXTShaderObjectExtensionName,
	#ifdef __APPLE__
		vk::KHRPortabilitySubsetExtensionName,
	#endif
};

void VulkanContext::init(GLFWwindow *window) {
    
    // INSTANCE
    
    auto availableLayers = vk::enumerateInstanceLayerProperties();

	std::println("Checking Layers");
	if (enableValidationLayers) {
		bool layersAvailable = std::ranges::all_of(validationLayers, [&](const char * required) {
			return std::ranges::any_of(availableLayers, [=](auto available) {
				return strcmp(available.layerName, required) == 0;
			});
		});
		if (not layersAvailable) {
			throw std::runtime_error("Requested layers not all available");
		}
	}

	std::println("Getting required extensions from GLFW");
	uint32_t extCount;
	auto extNames = Unwrap(glfwGetRequiredInstanceExtensions(&extCount), "Couldn't get vulkan instance extensions: ???");
	std::vector instanceExtensions(extNames, extNames + extCount);
	instanceExtensions.push_back(vk::EXTDebugUtilsExtensionName);

	std::println("Checking Extensions");
	bool extensionsAvailable = std::ranges::all_of(instanceExtensions, [&](const char * required) {
		return std::ranges::any_of(vk::enumerateInstanceExtensionProperties(), [=](auto available) {
			return strcmp(available.extensionName, required);
		});
	});
	if (not extensionsAvailable) {
		throw std::runtime_error("Requested instance extensions not all available");
	}

	// uint32_t true32[] = { vk::True };
	// std::vector<vk::LayerSettingEXT> settings = {
	// 	vk::LayerSettingEXT{"VK_LAYER_KHRONOS_validation", "validate_best_practices", vk::LayerSettingTypeEXT::eBool32, 1, &true32}
	// };
	// vk::LayerSettingsCreateInfoEXT settingsInfo{1, settings.data()};
	auto appInfo = vk::ApplicationInfo{}
		.setApiVersion(vk::ApiVersion14)
		.setPApplicationName("VulkanApp");
	auto ici = vk::InstanceCreateInfo{}
		.setPApplicationInfo(&appInfo)
		.setEnabledExtensionCount(instanceExtensions.size())
		.setPpEnabledExtensionNames(instanceExtensions.data());
	if (enableValidationLayers) {
		ici.setPEnabledLayerNames(validationLayers);
	}

	uint32_t v = vk::enumerateInstanceVersion();
	std::println("Instance is {}.{}", vk::apiVersionMajor(v), vk::apiVersionMinor(v));
	this->instance = vk::createInstance(ici);

    // SURFACE

    glfwCreateWindowSurface(this->instance, window, nullptr, (VkSurfaceKHR *)&this->surface);

    // PHYSICAL DEVICE

    auto physicalDevices = instance.enumeratePhysicalDevices();
    if (physicalDevices.empty()) throw std::runtime_error("No physical devices");
    for (auto pd : physicalDevices) {
        auto props = pd.getProperties();
        std::println("PD: {}", (char *)props.deviceName);
    }

    for (auto pd : physicalDevices) {
        auto props = pd.getProperties();
        
        auto availableExtensions = pd.enumerateDeviceExtensionProperties();
        
        bool hasExtensions = all_of(requiredExtensions, [&](const char * required) {
            bool ret = any_of(availableExtensions, [=](auto available) {
                return strcmp(available.extensionName, required) == 0;
            });
            if (!ret) {
                std::println("{} is missing {}", props.deviceName.data(), required);
            }
            return ret;
        });
        bool hasFormat = contains(pd.getSurfaceFormatsKHR(surface), { 
            vk::Format::eB8G8R8A8Srgb, 
            vk::ColorSpaceKHR::eSrgbNonlinear 
        });
        bool hasPresentMode = contains(pd.getSurfacePresentModesKHR(surface), vk::PresentModeKHR::eMailbox);
        bool hasCapabilities = true; // TODO: actually check for swapchain support

        if (hasExtensions && hasPresentMode) {
            std::println("Chose {} reporting vulkan {}.{}", props.deviceName.data(), vk::apiVersionMajor(props.apiVersion), vk::apiVersionMinor(props.apiVersion));
            this->physicalDevice = pd;
        }
        
    }
    if (!this->physicalDevice) throw std::runtime_error("No valid physical devices");

    // DEVICE

    std::optional<uint32_t> gi, pi, ti;
    int i = 0;

    for (auto family : physicalDevice.getQueueFamilyProperties()) {
        if (!gi && family.queueFlags & vk::QueueFlagBits::eGraphics) {
            gi = i;
        }
        if (!ti && family.queueFlags & vk::QueueFlagBits::eTransfer) {
            ti = i;
        }
        if (!pi && physicalDevice.getSurfaceSupportKHR(i, surface)) {
            pi = i;
        }
        i++;
    }

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    std::set indices = {
        Unwrap(gi, "No suitable graphics queue"), 
        Unwrap(pi, "No suitable present queue"), 
        Unwrap(ti, "No suitable transfer queue")
    };

    float priority = 1.0f;
    for (auto fam : indices) {
        vk::DeviceQueueCreateInfo ci{{}, fam, 1, &priority};
        queueCreateInfos.push_back(ci);
    }

    auto dci = vk::DeviceCreateInfo{}
        .setQueueCreateInfos(queueCreateInfos)
        .setPEnabledExtensionNames(requiredExtensions)
    ;
    
    vk::StructureChain featureChain = {
        dci,
        vk::PhysicalDeviceVulkan11Features{}
            .setShaderDrawParameters(true),
        vk::PhysicalDeviceVulkan12Features{}
            .setScalarBlockLayout(true),
            // .setTimelineSemaphore(true),
        vk::PhysicalDeviceVulkan13Features{}
            .setDynamicRendering(true)
            .setSynchronization2(true),
    };

    this->device = physicalDevice.createDevice(featureChain.get());

    // QUEUES

    this->graphicsQueue = Queue(this->device.getQueue(gi.value(), 0));
    this->graphicsQueue.index = gi.value();
    this->presentQueue = Queue(this->device.getQueue(pi.value(), 0));
    this->presentQueue.index = pi.value();
    this->transferQueue = Queue(this->device.getQueue(ti.value(), 0));
    this->transferQueue.index = ti.value();

    // COMMAND POOLS

    vk::CommandPoolCreateInfo transferPoolInfo{
        vk::CommandPoolCreateFlagBits::eTransient,
        this->transferQueue.index,
    };

    this->transferCommandPool = this->device.createCommandPool(transferPoolInfo);

    vk::CommandPoolCreateInfo poolInfo{
        vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        this->graphicsQueue.index
    };

    this->commandPool = this->device.createCommandPool(poolInfo);

    vk::CommandBufferAllocateInfo allocInfo{
        this->transferCommandPool,
        vk::CommandBufferLevel::ePrimary,
        1
    };

    // DLDID

    this->dldid = vk::detail::DispatchLoaderDynamic(this->instance, vkGetInstanceProcAddr, this->device);

    // ALLOCATOR

    VmaAllocatorCreateInfo allocatorInfo = {
        // .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = this->physicalDevice,
        .device = this->device,
        .instance = this->instance,
        .vulkanApiVersion = vk::ApiVersion14,
    };
    vmaCreateAllocator(&allocatorInfo, &this->allocator);

}

void VulkanContext::deinit() {
    this->pruneDestructionQueue();
    vmaDestroyAllocator(this->allocator);
    this->device.destroyCommandPool(this->commandPool);
    this->device.destroyCommandPool(this->transferCommandPool);
    this->device.destroy();
    this->instance.destroySurfaceKHR(this->surface);
    this->instance.destroy();
}

// void VulkanContext::beginTransferBatch() {
//     g_DebugNameState.Name("Batch Transfer Command Buffer");
//     batchTransferCommandBuffer = device.allocateCommandBuffers(vk::CommandBufferAllocateInfo{
//         this->transferCommandPool,
//         vk::CommandBufferLevel::ePrimary,
//         1
//     })[0];
//     g_DebugNameState.NameCommandBuffer(batchTransferCommandBuffer);
//     batchTransferCommandBuffer.begin(vk::CommandBufferBeginInfo{
//         vk::CommandBufferUsageFlagBits::eOneTimeSubmit
//     });
// }

// void VulkanContext::endTransferBatch() {
//     batchTransferCommandBuffer.end();
//     Transfer transfer;
//     transfer.stagingBuffer.buffer = nullptr;

//     vk::Fence uploadFence = this->device.createFence(vk::FenceCreateInfo{});
//     // Submit
//     vk::SubmitInfo submitInfo{};
//     submitInfo.setCommandBuffers(batchTransferCommandBuffer);
//     this->transferQueue.submit(submitInfo, uploadFence);

//     // Store the transfer info for later synchronization
//     transfer.cmdBuf = batchTransferCommandBuffer;
//     transfer.fence = uploadFence;
//     this->transfers.push(transfer);
//     batchTransferCommandBuffer = nullptr;
// }

// Upload one or more buffers via one staging buffer and command buffer
void VulkanContext::uploadBuffers(vk::ArrayProxy<Upload> uploads) {
    std::lock_guard lock(uploadMtx);
    bool createCmdBuf = batchTransferCommandBuffer == nullptr;
    vk::CommandBuffer cmdBuf = batchTransferCommandBuffer;
    // Staging buffer
    Transfer transfer;
    size_t stagingSize = 0;
    for (const auto &upload : uploads) {
        stagingSize += upload.dst.size;
    }

    transfer.stagingBuffer.initHost(*this, 
        stagingSize, 
        vk::BufferUsageFlagBits::eTransferSrc
    );
    assert(transfer.stagingBuffer.info.pMappedData != nullptr);
    if (!createCmdBuf) this->queueDestroy(transfer.stagingBuffer);

    size_t offset = 0;
    for (const auto &upload : uploads) {
        memcpy(static_cast<char *>(transfer.stagingBuffer.info.pMappedData) + offset, upload.src, upload.dst.size);
        offset += upload.dst.size;
    }
    
    // Command buffer
    if (createCmdBuf) {
        cmdBuf = device.allocateCommandBuffers(vk::CommandBufferAllocateInfo{
            this->transferCommandPool,
            vk::CommandBufferLevel::ePrimary,
            1
        })[0];
        cmdBuf.begin(vk::CommandBufferBeginInfo{
            vk::CommandBufferUsageFlagBits::eOneTimeSubmit
        });
    }

    // Copy
    offset = 0;
    for (const auto &upload : uploads) {
        vk::BufferCopy copyRegion{
            offset,
            0,
            upload.dst.size
        };
        cmdBuf.copyBuffer(transfer.stagingBuffer.buffer, upload.dst.buffer, copyRegion);
        offset += upload.dst.size;
    }

    if (createCmdBuf) {
        cmdBuf.end();

        vk::Fence uploadFence = this->device.createFence(vk::FenceCreateInfo{});
        // Submit
        {
            std::lock_guard lock(submitMtx);
            vk::SubmitInfo submitInfo{};
            submitInfo.setCommandBuffers(cmdBuf);
            this->transferQueue.submit(submitInfo, uploadFence);
        }

        // Store the transfer info for later synchronization
        transfer.cmdBuf = cmdBuf;
        transfer.fence = uploadFence;
        this->transfers.push(transfer);
    }
    uploadMtx.unlock();
}

void VulkanContext::waitForTransfers() {
    // while (!transfers.empty()) {
    //     auto transfer = transfers.front();
    //     std::ignore = this->device.waitForFences(transfer.fence, true, UINT64_MAX);
    //     this->device.freeCommandBuffers(this->transferCommandPool, transfer.cmdBuf);
    //     if (transfer.stagingBuffer.buffer) transfer.stagingBuffer.deinit(this->allocator);
    //     this->device.destroyFence(transfer.fence);
    //     transfers.pop();
    // }
    std::lock_guard lock(uploadMtx);

    while (!transfers.empty() && this->device.getFenceStatus(transfers.front().fence) == vk::Result::eSuccess) {
        auto transfer = transfers.front();
        this->device.freeCommandBuffers(this->transferCommandPool, transfer.cmdBuf);
        if (transfer.stagingBuffer.buffer) transfer.stagingBuffer.deinit(this->allocator);
        this->device.destroyFence(transfer.fence);
        transfers.pop();
    }

    // ... Reset command pool
}

void VulkanContext::pruneDestructionQueue() {
    while (not destruction_queue.empty()) {
        auto [frame, item] = destruction_queue.front();
        if (currentFrame - frame <= FRAME_COUNT) return;
        using enum Variant::Type;
        switch (item.type) {
            case eAllocBuffer: 
                item.value.allocBuffer.deinit(allocator); 
                break;
            case eMallocPtr: 
                std::println("Placeholder: Freeing eMallocPtr from Destruction Queue on Frame: {}", currentFrame);
                break;
        }   
        destruction_queue.pop();
    }
}

void VulkanContext::queueDestroy(Variant var) { destruction_queue.push({currentFrame, var}); };