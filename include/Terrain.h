#pragma once

#include <array>
#include <condition_variable>
#include <memory>
#include <unordered_map>
#include <queue>

#include <glm/vec3.hpp>
#include <glm/gtx/hash.hpp>

#include "Buffer.h"
#include "Context.h"

constexpr int CHUNK_WIDTH = 32;
constexpr int CHUNK_HEIGHT = 32;
static_assert(CHUNK_HEIGHT <= 64, "Chunk height exceeds 64 (uint32_t)");

enum class Block : uint8_t {
    Invalid = 0, Air = 0, Grass, Dirt, Stone
};
static bool transparent(Block b) {
    return b <= Block::Air;
}

struct Chunk {
    std::array<Block, CHUNK_WIDTH * CHUNK_WIDTH * CHUNK_HEIGHT> data;
    std::array<std::array<uint32_t, CHUNK_WIDTH>, CHUNK_WIDTH> opaquenessMask_Y; // X, Z
    std::array<std::array<uint32_t, CHUNK_HEIGHT>, CHUNK_WIDTH> opaquenessMask_Z; // X, Y
    std::array<std::array<uint32_t, CHUNK_HEIGHT>, CHUNK_WIDTH> opaquenessMask_X; // Z, Y
    AllocBuffer vertexBuffer;
    AllocBuffer indexBuffer;
    size_t vertexCount = 0, indexCount = 0;
    std::atomic<bool> inQueue = false;
    bool filled = false;
    
    glm::ivec3 chunkCoord;

    void init(const VulkanContext &);
    void deinit(VulkanContext &);

    inline Block getBlock(int, int, int) const;
    inline Block getBlock(glm::ivec3) const;

    inline void setBlock(Block, int, int, int);
    inline void setBlock(Block, glm::ivec3);

    inline static glm::ivec3 coordFromIndex(int);

    void fillChunk();
    void writeMesh(VulkanContext &);

    constexpr size_t size();
};

struct Terrain {
    std::unordered_map<glm::ivec3, std::unique_ptr<Chunk>> chunks;
    std::queue<Chunk *> chunkMeshQueue;
    std::mutex queueMtx;
    std::condition_variable queueCond;
    std::thread chunkQueueProcessor;
    std::atomic<bool> stopThread = false;

    void dirtyChunk(Chunk *);
    static void processChunks(Terrain *, VulkanContext *); // chunk processing thread

    void init(const VulkanContext &);

    void loadChunksAround(const VulkanContext &, glm::vec3);
    void loadChunk(const VulkanContext &, glm::ivec3);

    void tickFrame(const VulkanContext &);
    void draw(vk::CommandBuffer);

    void unloadChunk(VulkanContext &, Chunk &);

    void deinit(VulkanContext &);

};

