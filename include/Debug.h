#pragma once

#include <atomic>
#include <vulkan/vulkan.hpp>

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

struct DebugFrameStats {
    std::atomic<size_t> index_count;
};