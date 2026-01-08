// #ifdef __APPLE__
// 	#define VK_ENABLE_BETA_EXTENSIONS
// #endif

#include "vulkan/vulkan.hpp"
#include <GLFW/glfw3.h>
#include <glm/ext/matrix_clip_space.hpp>
#ifdef IMGUI_ENABLE
#include "backends/imgui_impl_vulkan.h"
#endif

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtx/io.hpp>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include <iostream>
#include <fstream>
#include <ranges> // IWYU pragma: keep
#include <print>
#include <stdexcept>
#include <cstdint>

#include "Renderer.h"
#include "Vertex.h"
#include "Debug.h"
extern DebugNameState g_DebugNameState;

#include <Util.hpp>
#include <Terrain.h>

#ifdef IMGUI_ENABLE
static void check_vk_result(VkResult err)
{
    if (err == 0)
        return;
    fprintf(stderr, "[vulkan] Error: VkResult = %d\n", err);
    if (err < 0)
        abort();
}

void Renderer::initImGui() {
	ImGui_ImplVulkan_InitInfo info{};

	std::vector<vk::DescriptorPoolSize> poolSizes = { 
		{ vk::DescriptorType::eSampler, 1000 },
		{ vk::DescriptorType::eCombinedImageSampler, 1000 },
		{ vk::DescriptorType::eSampledImage, 1000 },
		{ vk::DescriptorType::eStorageImage, 1000 },
		{ vk::DescriptorType::eUniformBuffer, 1000 },
		{ vk::DescriptorType::eStorageTexelBuffer, 1000 },
		{ vk::DescriptorType::eUniformBuffer, 1000 },
		{ vk::DescriptorType::eStorageBuffer, 1000 },
		{ vk::DescriptorType::eUniformBufferDynamic, 1000 },
		{ vk::DescriptorType::eStorageBufferDynamic, 1000 },
		{ vk::DescriptorType::eInputAttachment, 1000 } 
	};
	vk::DescriptorPoolCreateInfo poolInfo{
		vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		1000,
		poolSizes
	};
	this->imguiPool = vkc.device.createDescriptorPool(poolInfo);
	info.ApiVersion = vk::ApiVersion14;
	info.Instance = vkc.instance;
	info.PhysicalDevice = vkc.physicalDevice;
	info.Device = vkc.device;
	info.Queue = vkc.graphicsQueue;
	info.DescriptorPool = imguiPool;
	info.MinImageCount = vkc.FRAME_COUNT;
	info.ImageCount = vkc.FRAME_COUNT;
	info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
	info.UseDynamicRendering = true;
	info.CheckVkResultFn = check_vk_result;
	info.Allocator = vkc.allocator->GetAllocationCallbacks();
	vk::PipelineRenderingCreateInfoKHR pipelineInfo{{}, swapchain.format, swapchain.depthBufferFormat};
	info.PipelineRenderingCreateInfo = pipelineInfo;
	info.PipelineCache = nullptr;
	info.QueueFamily = vkc.graphicsQueue.index;
	ImGui_ImplVulkan_Init(&info);
}
#endif

Renderer::Renderer(GLFWwindow *window) {
	this->window = window;
	
	std::println("Initting!");
	vkc.init(window);
	g_DebugNameState.vkc = &vkc;
	terrain.init(vkc);
	
	int w, h;
	glfwGetFramebufferSize(window, &w, &h);
	swapchain.init(vkc, vk::Extent2D{static_cast<uint32_t>(w), static_cast<uint32_t>(h)});
	initCommandBuffers();
	initSyncObjects();
	// initVertexIndexBuffers();
	initUniformBuffers();
	initDescriptorLayout();
	initDescriptorSets();
	initGraphicsPipeline();
}

#define LOG(X) std::println(X)

Renderer::~Renderer() {
	vkc.device.waitIdle();

	#ifdef IMGUI_ENABLE
	ImGui_ImplVulkan_Shutdown();
	vkc.device.destroyDescriptorPool(imguiPool);
	#endif

	const auto dev = vkc.device;

	swapchain.deinit(vkc);
	dev.destroySwapchainKHR(swapchain);
	dev.destroyDescriptorSetLayout(descriptorSetLayout);
	dev.destroyDescriptorPool(descriptorPool);
	dev.destroyPipelineLayout(pipelineLayout);
	for (auto &frame : vkc.frames) {
		dev.destroySemaphore(frame.imageAvailableSemaphore);
		dev.destroyFence(frame.inFlightFence);
		frame.uniformBuffer.deinit(vkc.allocator);
	}
	for (auto &semaphore : renderFinishedSemaphores) {
		dev.destroySemaphore(semaphore);
	}
	dev.destroyPipeline(graphicsPipeline);
	terrain.deinit(vkc);
	vkc.deinit();
}

void Renderer::resize() {
	vkc.device.waitIdle();
	auto oldSwapchain = swapchain;
	int w, h;
	glfwGetFramebufferSize(window, &w, &h);
	swapchain.init(vkc, vk::Extent2D{static_cast<uint32_t>(w), static_cast<uint32_t>(h)}, oldSwapchain);
	oldSwapchain.deinit(vkc);
}

struct UniformBufferObject {
	float time;
	glm::mat4 mvp;
	glm::mat3 normalMatrix;
	glm::vec2 point;
} ubo;

void Renderer::initUniformBuffers() {
	for (auto &frame : vkc.frames) {
		frame.uniformBuffer.init(vkc, 
			sizeof(ubo),
			vk::BufferUsageFlagBits::eUniformBuffer,
			VMA_MEMORY_USAGE_AUTO,
			VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT
		);
	}
}

inline glm::vec4 wdiv(glm::vec4 v) {
	return v / v.w;
}
void Renderer::updateUniformBuffers(AllocBuffer uniformBuffer) {
	static bool once = false;
	using namespace std::chrono;
	static auto start = steady_clock::now();

	auto now = steady_clock::now();
	ubo.time = duration<float, seconds::period>(now - start).count();
	glm::mat4 perspective = glm::infinitePerspective(
		glm::radians(camera->fov), // perspective 
		(float)swapchain.extent.width / (float)swapchain.extent.height, // aspect
		camera->near // near
	);
	perspective[1][1] *= -1;
	
	glm::mat4 view = glm::lookAt(camera->position, camera->position + camera->front, camera->up);
	
	glm::mat4 model = glm::rotate(glm::identity<glm::mat4>(), ubo.time, glm::vec3(0, 1, 0));
	model = glm::rotate(model, ubo.time, glm::vec3(1, 0, 0));
	model = glm::rotate(model, ubo.time, glm::vec3(0, 0, 1));
	// ubo.normalMatrix = glm::mat3(model);
	// ubo.mvp = perspective * view * model;
	ubo.normalMatrix = glm::identity<glm::mat3>();
	ubo.mvp = perspective * view;

	// ubo.point = wdiv(ubo.mvp * glm::vec4(A, A, A, 1.0f));
	int w, h;
	glfwGetFramebufferSize(window, &w, &h);
	auto half_dims = glm::vec2(w,h)/2.f;
	ubo.point *= half_dims;
	ubo.point += half_dims;

	memcpy(uniformBuffer.info.pMappedData, &ubo, sizeof(ubo));
}

template<typename T>
std::vector<T> readFile(const std::string &fileName) {
	std::ifstream fin(fileName, std::ios_base::ate | std::ios_base::binary);

	if (!fin.is_open()) {
		throw std::runtime_error(std::format("File couldn't be opened: {}", fileName));
	}

	size_t size = fin.tellg();
	std::vector<T> buffer(size / sizeof(T));

	fin.seekg(0);
	fin.read((char*)buffer.data(), size);
	fin.close();

	return buffer;
}

vk::ShaderEXT Renderer::createShader(
	const std::vector<uint32_t>& spv, 
	const char *main, 
	vk::ShaderStageFlagBits stage, 
	vk::ShaderStageFlagBits nextStage = {}, 
	vk::PushConstantRange pcRange = {}
) {
	auto ci = vk::ShaderCreateInfoEXT{}
		.setCodeType(vk::ShaderCodeTypeEXT::eSpirv)
		.setCode<uint32_t>(spv)
		.setStage(stage)
		.setNextStage(nextStage)
		.setPName(main)
		.setPushConstantRanges(pcRange);
	auto [result, shader] = vkc.device.createShaderEXT(ci, nullptr, vkc.dldid);
	if (result != vk::Result::eSuccess) {
		throw std::runtime_error("Failed to create shader");
	}
	return shader;
}

void Renderer::initDescriptorLayout() {
	vk::DescriptorSetLayoutBinding layoutBinding{
		0, 
		vk::DescriptorType::eUniformBuffer, 
		1, 
		vk::ShaderStageFlagBits::eVertex
	};
	vk::DescriptorSetLayoutCreateInfo layoutInfo{{}, 
		layoutBinding
	};
	descriptorSetLayout = vkc.device.createDescriptorSetLayout(layoutInfo);
}

void Renderer::initDescriptorSets() {
	vk::DescriptorPoolSize poolSize{vk::DescriptorType::eUniformBuffer, VulkanContext::FRAME_COUNT};
	vk::DescriptorPoolCreateInfo poolInfo{{}, VulkanContext::FRAME_COUNT, poolSize};
	descriptorPool = vkc.device.createDescriptorPool(poolInfo);
	std::vector<vk::DescriptorSetLayout> layouts(vkc.FRAME_COUNT, descriptorSetLayout);
	vk::DescriptorSetAllocateInfo allocInfo{descriptorPool, layouts};
	descriptorSets = vkc.device.allocateDescriptorSets(allocInfo);
	
	for (int i = 0; i < vkc.FRAME_COUNT; i += 1) {
		vk::DescriptorBufferInfo dbi{vkc.frames[i].uniformBuffer.buffer, 0, sizeof(UniformBufferObject)};
		vk::WriteDescriptorSet writeSet{descriptorSets[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer, {}, &dbi};
		vkc.device.updateDescriptorSets(writeSet, {});
	}
}

// vk::ShaderEXT vertexShader, fragmentShader;
// class Shader {
// 	std::vector<std::tuple<vk::ShaderEXT, vk::ShaderStageFlagBits>> objects;
// };

void Renderer::initGraphicsPipeline() {
	auto spv = readFile<uint32_t>("shaders/triangle.slang.spv");
	// vertexShader = createShader(spv, "vertexMain", vk::ShaderStageFlagBits::eVertex, vk::ShaderStageFlagBits::eFragment);
	// fragmentShader = createShader(spv, "fragmentMain", vk::ShaderStageFlagBits::eFragment);

	vk::ShaderModuleCreateInfo moduleInfo{
		{}, spv
	};
	auto module = vkc.device.createShaderModule(moduleInfo);

	std::array shaderStages = {
		vk::PipelineShaderStageCreateInfo{{}, vk::ShaderStageFlagBits::eVertex, module, "vertexMain"},
		vk::PipelineShaderStageCreateInfo{{}, vk::ShaderStageFlagBits::eFragment, module, "fragmentMain"},
	};

	auto vertexBinding = Vertex::getBindingDescription();
	auto vertexAttributes = Vertex::getAttributeDescriptions();
	auto vertexInputInfo = vk::PipelineVertexInputStateCreateInfo{}
		.setVertexAttributeDescriptions(vertexAttributes)
		.setVertexBindingDescriptions(vertexBinding)
	;	
	auto inputAssemblyInfo = vk::PipelineInputAssemblyStateCreateInfo{}
		.setTopology(vk::PrimitiveTopology::eTriangleList)
	;

	auto viewport = vk::Viewport{}
		.setWidth((float) swapchain.extent.width)
		.setHeight((float) swapchain.extent.height)
		.setMaxDepth(1.0f);
	vk::Rect2D scissor{{0, 0}, swapchain.extent};
	vk::PipelineViewportStateCreateInfo viewportInfo{{}, viewport, scissor};

	std::array dynamicStates = {
		vk::DynamicState::eScissor,
		vk::DynamicState::eViewport,
	};
	vk::PipelineDynamicStateCreateInfo dynamicsInfo({}, dynamicStates);

	vk::PipelineRasterizationStateCreateInfo rasterizationInfo{{},
		false,
		false,
		vk::PolygonMode::eFill,
		vk::CullModeFlagBits::eBack,
		vk::FrontFace::eClockwise,
	};
	rasterizationInfo.setLineWidth(1.0f);

	vk::PipelineTessellationStateCreateInfo tesselationInfo{};
	vk::PipelineMultisampleStateCreateInfo multisampleInfo{};
	vk::PipelineDepthStencilStateCreateInfo depthStencilInfo{{}, 
		true, 
		true, 
		vk::CompareOp::eLess
	};

	vk::PipelineColorBlendAttachmentState attachment{};
	attachment.setColorWriteMask((vk::ColorComponentFlagBits) 0b1111);
	vk::PipelineColorBlendStateCreateInfo colorBlendInfo{};
	colorBlendInfo.setAttachments(attachment);

	vk::PipelineLayoutCreateInfo layoutInfo{{}, descriptorSetLayout};
	this->pipelineLayout = vkc.device.createPipelineLayout(layoutInfo);

	vk::PipelineRenderingCreateInfo renderPipelineInfo{};
	renderPipelineInfo.setColorAttachmentFormats(swapchain.format);
	renderPipelineInfo.setDepthAttachmentFormat(swapchain.depthBufferFormat);

	vk::GraphicsPipelineCreateInfo graphicsPipelineInfo{{},
		shaderStages, // shaderStages,
		&vertexInputInfo,
		&inputAssemblyInfo,
		&tesselationInfo,
		&viewportInfo,
		&rasterizationInfo,
		&multisampleInfo,
		&depthStencilInfo, // depthStencil
		&colorBlendInfo,
		&dynamicsInfo,
		this->pipelineLayout,
		nullptr, // renderPass (null because dynamic rendering)
		0, // subpass
	};
	graphicsPipelineInfo.setPNext(&renderPipelineInfo);

	// vk::PipelineCache pipelineCache = device.createPipelineCache({});

	this->graphicsPipeline = Unwrap(vkc.device.createGraphicsPipeline(nullptr, graphicsPipelineInfo), "Could not create graphics pipeline!");

	vkc.device.destroyShaderModule(module);
}

void Renderer::initCommandBuffers() {

	vk::CommandBufferAllocateInfo allocInfo{
		vkc.commandPool,
		vk::CommandBufferLevel::ePrimary,
		VulkanContext::FRAME_COUNT,
	};

	auto buffers = vkc.device.allocateCommandBuffers(allocInfo);
	auto debugName = std::string("Frame Command Buffer 0");
	for (int i = 0; i < vkc.frames.size(); i += 1) {
		vkc.frames[i].commandBuffer = buffers[i];
		vk::DebugUtilsObjectNameInfoEXT nameInfo{
			vk::ObjectType::eCommandBuffer,
			(uint64_t)(VkCommandBuffer)vkc.frames[i].commandBuffer,
			debugName.c_str()
		};
		Unwrap(vkc.device.setDebugUtilsObjectNameEXT(&nameInfo, vkc.dldid), "Command Buffer debug naming failed");
		debugName[21] += 1;
	}
	
}

enum class Timeline {

};

void Renderer::recordCommandBuffer(vk::CommandBuffer commandBuffer, uint32_t imageIndex, uint32_t currentFrame) {
	vk::CommandBufferBeginInfo beginInfo{};
	commandBuffer.begin(beginInfo);

	auto colorBarrier = vk::ImageMemoryBarrier2{}
		.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 })
		.setImage(swapchain.images[imageIndex]);

	auto depthBarrier = vk::ImageMemoryBarrier2{}
		.setSubresourceRange({ vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1 })
		.setImage(swapchain.depthBufferImage);

	colorBarrier
		.setSrcStageMask(vk::PipelineStageFlagBits2::eTopOfPipe).setDstStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput)
		.setSrcAccessMask(vk::AccessFlagBits2::eNone).setDstAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite)
		.setOldLayout(vk::ImageLayout::eUndefined).setNewLayout(vk::ImageLayout::eColorAttachmentOptimal);

	depthBarrier
		.setSrcStageMask(vk::PipelineStageFlagBits2::eTopOfPipe).setDstStageMask(vk::PipelineStageFlagBits2::eEarlyFragmentTests)
		.setSrcAccessMask(vk::AccessFlagBits2::eNone).setDstAccessMask(vk::AccessFlagBits2::eDepthStencilAttachmentWrite)
		.setOldLayout(vk::ImageLayout::eUndefined).setNewLayout(vk::ImageLayout::eDepthAttachmentOptimal);

	vk::DependencyInfo dep1{};
	auto imageBarriers = std::vector<vk::ImageMemoryBarrier2>{colorBarrier, depthBarrier};
	dep1.setImageMemoryBarriers(imageBarriers);
	commandBuffer.pipelineBarrier2(dep1);

	vk::ClearValue clearValue{vk::ClearColorValue(0, 0, 0, 1)};
	vk::RenderingAttachmentInfo colorAttachInfo{
		swapchain.imageViews[imageIndex], // imageView
		vk::ImageLayout::eColorAttachmentOptimal // imageLayout
	};
	colorAttachInfo.setLoadOp(vk::AttachmentLoadOp::eClear);
	colorAttachInfo.setClearValue(clearValue);

	vk::RenderingAttachmentInfo depthAttachInfo{
		swapchain.depthBufferImageView,
		vk::ImageLayout::eDepthAttachmentOptimal,
	};
	depthAttachInfo.setLoadOp(vk::AttachmentLoadOp::eClear);
	depthAttachInfo.setClearValue(vk::ClearValue{vk::ClearDepthStencilValue{1.0f, {}}});

	vk::RenderingInfo renderingInfo{{},
		vk::Rect2D{{0, 0}, swapchain.extent}, // renderArea
		1, // layerCount
		0, // viewMask
		colorAttachInfo,
		&depthAttachInfo
	};

	commandBuffer.beginRendering(&renderingInfo);

	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, this->graphicsPipeline);

	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets[currentFrame], nullptr);

	commandBuffer.setViewport(0,
		vk::Viewport{
			0.0f, 0.0f,
			static_cast<float>(swapchain.extent.width),
			static_cast<float>(swapchain.extent.height),
			0.0f, 1.0f,
		}
	);
	commandBuffer.setScissor(0,
		vk::Rect2D{{0, 0}, swapchain.extent}
	);
	
	terrain.draw(commandBuffer);

	#ifdef IMGUI_ENABLE
	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
	#endif

	commandBuffer.endRendering();

	colorBarrier
		.setSrcStageMask(vk::PipelineStageFlagBits2::eColorAttachmentOutput).setDstStageMask(vk::PipelineStageFlagBits2::eBottomOfPipe)
		.setSrcAccessMask(vk::AccessFlagBits2::eColorAttachmentWrite).setDstAccessMask(vk::AccessFlagBits2::eNone)
		.setOldLayout(vk::ImageLayout::eColorAttachmentOptimal).setNewLayout(vk::ImageLayout::ePresentSrcKHR);

	depthBarrier
		.setSrcStageMask(vk::PipelineStageFlagBits2::eEarlyFragmentTests).setDstStageMask(vk::PipelineStageFlagBits2::eBottomOfPipe)
		.setSrcAccessMask(vk::AccessFlagBits2::eDepthStencilAttachmentWrite).setDstAccessMask(vk::AccessFlagBits2::eNone)
		.setOldLayout(vk::ImageLayout::eDepthAttachmentOptimal).setNewLayout(vk::ImageLayout::eDepthAttachmentOptimal);

	vk::DependencyInfo dep2{};
	auto imageBarriers2 = std::vector<vk::ImageMemoryBarrier2>{colorBarrier, depthBarrier};
	dep2.setImageMemoryBarriers(imageBarriers2);
	commandBuffer.pipelineBarrier2(dep2);

	commandBuffer.end();
}

void Renderer::drawFrame() {
	// static int frameCount = 0;
	// if (frameCount == 20) {
	// 	throw std::runtime_error("Stopping!");
	// }
	// std::cout << "Drawing Frame " << frameCount << "!\n";
	// frameCount += 1;
	auto &frame = vkc.frames[vkc.currentFrame];
	vkc.pruneDestructionQueue(frame.destruction_queue);
    
    std::ignore = vkc.device.waitForFences(frame.inFlightFence, true, UINT64_MAX);
    vkc.device.resetFences(frame.inFlightFence);

    auto [result, imageIndex] = vkc.device.acquireNextImageKHR(swapchain, UINT64_MAX, frame.imageAvailableSemaphore, nullptr);
    if (result != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to acquire next image");
    }

    frame.commandBuffer.reset();

	updateUniformBuffers(frame.uniformBuffer);
    recordCommandBuffer(frame.commandBuffer, imageIndex, vkc.currentFrame);

	std::vector<vk::SemaphoreSubmitInfo> waitInfo {{
		frame.imageAvailableSemaphore,
		{}, 
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
	}};

	vk::CommandBufferSubmitInfo commandBufferInfo{frame.commandBuffer};
	vk::SemaphoreSubmitInfo signalInfo{
		renderFinishedSemaphores[imageIndex],
		{},
		vk::PipelineStageFlagBits2::eAllCommands,
	};

    vk::SubmitInfo2 submitInfo{{},
        waitInfo,
        commandBufferInfo,
        signalInfo,
    };

    {
        std::lock_guard lock(vkc.submitMtx);
        vkc.graphicsQueue.submit2(submitInfo, frame.inFlightFence);

        vk::PresentInfoKHR presentInfo{
            renderFinishedSemaphores[imageIndex],
            this->swapchain,
            imageIndex
        };

        result = vkc.presentQueue.presentKHR(presentInfo);
    }
    if (result != vk::Result::eSuccess) {
        std::cerr << "SUBOPTIMAL_KHR PRESENTATION" << std::endl;
    }

    vkc.currentFrame = (vkc.currentFrame + 1) % vkc.FRAME_COUNT;
}

void Renderer::initSyncObjects() {
	renderFinishedSemaphores.resize(swapchain.images.size());

	vk::SemaphoreCreateInfo semaphoreInfo{};
	vk::FenceCreateInfo fenceInfo{vk::FenceCreateFlagBits::eSignaled};
	
	for (auto &frame : vkc.frames) {
		frame.imageAvailableSemaphore = vkc.device.createSemaphore(semaphoreInfo);
		frame.inFlightFence = vkc.device.createFence(fenceInfo);
	}
	for (auto &semaphore : renderFinishedSemaphores) {
		semaphore = vkc.device.createSemaphore(semaphoreInfo);
	}
}
