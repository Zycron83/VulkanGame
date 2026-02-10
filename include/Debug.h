#pragma once

#include <atomic>
#include <vulkan/vulkan.hpp>
#include <vector>

struct Settings {
    struct NoiseSettings {
        int octaves = 4.f;
        float frequency = 1.0f;
        float amplitude = 1.0f;
        float lacunarity = 2.0f;
        float persistence = 0.5f;
        float scale = 1.0f;
        // NoiseSettings() : octaves(4.f), frequency(1.0f), amplitude(1.0f), lacunarity(2.0f), persistence(0.5f) {}
        bool operator==(const NoiseSettings& other) const {
            return octaves == other.octaves &&
                   frequency == other.frequency &&
                   amplitude == other.amplitude &&
                   lacunarity == other.lacunarity &&
                   persistence == other.persistence &&
                   scale == other.scale;
        }
    } Noise;
    std::atomic<size_t> chunkBytes = 0;
    std::atomic<int> x;
    std::atomic<int> y;
    std::atomic<int> z;
};

#define NAME_OBJ_FUNC(obj) \
    void Name##obj(const VulkanContext *vkc, Vk##obj object) { \
        if (this->name == nullptr) return; \
        std::string debugName; \
        for (const auto &n : nameStack) { \
            debugName += n; \
            debugName += "::"; \
        } \
        debugName += this->name; \
        vk::DebugUtilsObjectNameInfoEXT nameInfo{ \
            vk::ObjectType::e##obj, \
            (uint64_t)object, \
            debugName.c_str() \
        }; \
        SetDebugName(vkc, nameInfo); \
        this->name = nullptr; \
    }

struct VulkanContext;

struct DebugNameState {
    std::vector<const char *> nameStack;
    char * name;

    void Push(const char *name) {
        nameStack.push_back(name);
    }
    void Pop() {
        if (!nameStack.empty()) {
            nameStack.pop_back();
        }
    }
    void Name(const char *name) {
        this->name = const_cast<char *>(name);
    }

    NAME_OBJ_FUNC(Buffer)
    NAME_OBJ_FUNC(CommandBuffer)
    NAME_OBJ_FUNC(Queue)
    NAME_OBJ_FUNC(CommandPool)

    void SetDebugName(const VulkanContext *vkc, vk::DebugUtilsObjectNameInfoEXT &) const;
};

struct DebugFrameStats {
    std::atomic<size_t> index_count;
};