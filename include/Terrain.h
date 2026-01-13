#pragma once

#include <array>
#include <condition_variable>
#include <memory>
#include <unordered_map>
#include <deque>

#include <glm/vec3.hpp>
#include <glm/gtx/hash.hpp>

#include "Buffer.h"
#include "Camera.hpp"
#include "Context.h"

constexpr int CHUNK_LENGTH = 32;

enum class Block : uint8_t {
    Invalid = 0, Air = 0, Grass, Dirt, Stone
};
static bool transparent(Block b) {
    return b <= Block::Air;
}
enum Dir {
    X_POS, X_NEG,
    Y_POS, Y_NEG,
    Z_POS, Z_NEG,
};

enum QueueOp : uint8_t {
    NONE,
    FILL,
    UPDATE_MESH,
};
namespace std {
    template<typename T, int N, int M = N>
    using array2 = array<array<T, M>, N>;
}

struct Chunk {
    std::unique_ptr<std::array<Block, CHUNK_LENGTH * CHUNK_LENGTH * CHUNK_LENGTH>> data; // doubles as validity check
    std::array2<uint32_t, CHUNK_LENGTH> opaquenessMask_Y; // X, Z
    std::array2<uint32_t, CHUNK_LENGTH> opaquenessMask_Z; // Y, X
    std::array2<uint32_t, CHUNK_LENGTH> opaquenessMask_X; // Z, Y
    // std::array2<Block, CHUNK_LENGTH> border_X_POS; // Z, Y
    // std::array2<Block, CHUNK_LENGTH> border_X_NEG; // Z, Y
    // std::array2<Block, CHUNK_LENGTH> border_Y_POS; // X, Z
    // std::array2<Block, CHUNK_LENGTH> border_Y_NEG; // X, Z
    // std::array2<Block, CHUNK_LENGTH> border_Z_POS; // X, Y
    // std::array2<Block, CHUNK_LENGTH> border_Z_BEG; // X, Y
    std::array<Chunk *, 6> neighbourChunks;
    AllocBuffer vertexBuffer;
    AllocBuffer indexBuffer;
    size_t vertexCount = 0, indexCount = 0;
    std::atomic<bool> inQueue = false;
    std::mutex chunkMtx;
    QueueOp queueOp = QueueOp::NONE;
    
    glm::ivec3 chunkCoord;

    void init(const VulkanContext &);
    void deinit(VulkanContext &);

    inline Block getBlock(int, int, int) const;
    inline Block getBlock(glm::ivec3) const;

    inline void setBlock(Block, int, int, int);
    inline void setBlock(Block, glm::ivec3);

    inline static glm::ivec3 coordFromIndex(int);

    void fillChunk();

    constexpr size_t size();
};

struct Terrain {
    std::unordered_map<glm::ivec3, std::shared_ptr<Chunk>> chunks;
    std::mutex chunkMtx;
    std::deque<std::shared_ptr<Chunk>> chunkMeshQueue;
    std::mutex queueMtx;
    std::condition_variable queueCond;
    std::thread chunkQueueProcessor;
    std::atomic<bool> stopThread = false;

    void dirtyChunk(std::shared_ptr<Chunk>, QueueOp);
    static void processChunks(Terrain *, VulkanContext *); // chunk processing thread

    void init(VulkanContext &);

    void loadChunksAround(VulkanContext &, const glm::ivec3);
    void loadChunk(const VulkanContext &, const glm::ivec3);
    void unloadChunk(VulkanContext &, const glm::ivec3);
    // void writeMesh(VulkanContext &, std::shared_ptr<Chunk>);
    void writeMesh(VulkanContext &, glm::ivec3);

    void tickFrame(VulkanContext &, const Camera &camera);
    void draw(vk::CommandBuffer);


    void deinit(VulkanContext &);

};

