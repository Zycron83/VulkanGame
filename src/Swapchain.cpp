#include "Context.h"
#include "Swapchain.h"

void Swapchain::init(VulkanContext &vkc, vk::Extent2D extent, vk::SwapchainKHR oldSwapchain) {
    auto colorFormat = vk::Format::eB8G8R8A8Srgb;
    auto depthFormat = vk::Format::eD32Sfloat;

    // std::array<uint32_t, 2> queueIndices{ device.graphicsQueue.index, device.presentQueue.index };
    auto caps = vkc.physicalDevice.getSurfaceCapabilitiesKHR(vkc.surface);

    auto ci = vk::SwapchainCreateInfoKHR{}
        .setSurface(vkc.surface)
        .setImageFormat(colorFormat)
        .setImageExtent(extent)
        .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
        .setImageArrayLayers(1)
        .setPresentMode(vk::PresentModeKHR::eMailbox)
        .setMinImageCount(caps.minImageCount)
        .setImageSharingMode(vk::SharingMode::eExclusive)
        .setClipped(true)
    ;
    if (oldSwapchain) {
        ci.setOldSwapchain(oldSwapchain);
    }

    *this = Swapchain(vkc.device.createSwapchainKHR(ci));
    this->format = colorFormat;
    this->depthBufferFormat = depthFormat;
    this->extent = ci.imageExtent;
    this->images = vkc.device.getSwapchainImagesKHR(*this);
    
    auto vci = vk::ImageViewCreateInfo{}
        .setViewType(vk::ImageViewType::e2D)
        .setFormat(format)
        .setSubresourceRange(vk::ImageSubresourceRange{
            vk::ImageAspectFlagBits::eColor, 
            0, 1, 0, 1
        })
    ;
    this->imageViews.resize(this->images.size());
    int i = 0;
    for (auto image : this->images) {
        vci.setImage(image);
        this->imageViews[i] = (vkc.device.createImageView(vci));
        i++;
    }

    vk::ImageCreateInfo dici{{}, 
        vk::ImageType::e2D, 
        vk::Format::eD32Sfloat, 
        vk::Extent3D(this->extent, 1), 
        1, 1,
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eDepthStencilAttachment,
    };
    VmaAllocationCreateInfo aci{{}, VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO };
    vmaCreateImage(vkc.allocator, (VkImageCreateInfo*)&dici, &aci, (VkImage*)&this->depthBufferImage, &this->depthBufferAlloc, NULL);

    vk::ImageViewCreateInfo divci{{}, 
        this->depthBufferImage, 
        vk::ImageViewType::e2D, 
        vk::Format::eD32Sfloat, 
        {}, 
        vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}
    };
    this->depthBufferImageView = vkc.device.createImageView(divci);
}
void Swapchain::deinit(VulkanContext &vkc) {
    vmaDestroyImage(vkc.allocator, depthBufferImage, depthBufferAlloc);
    for (auto imageView : imageViews) {
        vkc.device.destroyImageView(imageView);
    }
    vkc.device.destroyImageView(depthBufferImageView);
}