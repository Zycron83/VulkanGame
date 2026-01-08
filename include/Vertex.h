#pragma once

#include <vulkan/vulkan.hpp>
#include <glm/vec3.hpp>

struct Index {
	uint32_t val;

	inline operator uint32_t() {
		return val;
	};
	static constexpr vk::IndexType enumType = vk::IndexType::eUint32;
};

struct Vertex {
	glm::vec3 position;
	glm::vec3 color;
    glm::vec3 normal;

	static vk::VertexInputBindingDescription getBindingDescription() {
        vk::VertexInputBindingDescription bindingDescription{
			0,
			sizeof(Vertex),
			vk::VertexInputRate::eVertex
		};

        return bindingDescription;
    }

	static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions() {
		std::array attributeDescriptions{
			vk::VertexInputAttributeDescription{
				0, // loc
				0, // binding
				vk::Format::eR32G32B32Sfloat,
				offsetof(Vertex, position)
			},
			vk::VertexInputAttributeDescription{
				1,
				0,
				vk::Format::eR32G32B32Sfloat,
				offsetof(Vertex, color)
			},
            vk::VertexInputAttributeDescription{
				2,
				0,
				vk::Format::eR32G32B32Sfloat,
				offsetof(Vertex, normal)
			}
		};

		return attributeDescriptions;
	}
};