#pragma once

#include <GLFW/glfw3.h>

#include "vulkan/vulkan.hpp"

#include "Context.h"

struct Swapchain : vk::SwapchainKHR {
    std::vector<vk::Image> images;
	std::vector<vk::ImageView> imageViews;
    vk::Format format, depthBufferFormat;
    vk::Extent2D extent;
    vk::Image depthBufferImage;
    VmaAllocation depthBufferAlloc;
    vk::ImageView depthBufferImageView;

    void init(VulkanContext &vkc, vk::Extent2D extent, vk::SwapchainKHR oldSwapchain = nullptr);
    void deinit(VulkanContext &vkc);
};