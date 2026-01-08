#pragma once

#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>

#include "Camera.hpp"
#include "Buffer.h"
#include "Terrain.h"
#include "Swapchain.h"
#include "Context.h"

#include <vector>
#include <memory>

struct Renderer {
	Renderer(GLFWwindow *);
	~Renderer();

	VulkanContext vkc;
	
	std::unique_ptr<Camera> camera;
	Terrain terrain;
	
	#ifdef IMGUI_ENABLE
	void initImGui();
	vk::DescriptorPool imguiPool;
	#endif

	void drawFrame();
	void resize();
	void waitIdle() { vkc.device.waitIdle(); }

	GLFWwindow *window;

	vk::SurfaceKHR surface;
	Swapchain swapchain;
	vk::Pipeline graphicsPipeline;
	vk::DescriptorPool descriptorPool;
	std::vector<vk::DescriptorSet> descriptorSets;
	vk::DescriptorSetLayout descriptorSetLayout;
	vk::PipelineLayout pipelineLayout;
	
	std::vector<vk::Semaphore> renderFinishedSemaphores;
	
	void initGraphicsPipeline();
	vk::ShaderEXT createShader(const std::vector<uint32_t>&, const char *, vk::ShaderStageFlagBits, vk::ShaderStageFlagBits, vk::PushConstantRange);
	void initCommandBuffers();
	void recordCommandBuffer(vk::CommandBuffer, uint32_t imageIndex, uint32_t currentFrame);
	void initSyncObjects();
	void initDescriptorLayout();
	void initDescriptorSets();
	void initUniformBuffers();
	void updateUniformBuffers(AllocBuffer);
};