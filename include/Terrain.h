#pragma once

#include <array>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <unordered_map>
// ... Switch to deque eventually?

#include <glm/vec3.hpp>
#include <glm/gtx/hash.hpp>

#include "Buffer.h"
#include "Camera.hpp"
#include "Context.h"

constexpr int CHUNK_LENGTH = 32;
constexpr int CHUNK_BOTTOM = 0;
constexpr int CHUNK_TOP = 2;

enum class Block : uint8_t {
    Invalid = 0, Air = 0, Grass, Dirt, Stone
};
static bool transparent(Block b) {
    return b <= Block::Air;
}
enum Dir {
    X_POS, // Z, Y
    X_NEG, // Z, Y
    Y_POS, // X, Z
    Y_NEG, // X, Z
    Z_POS, // Y, X
    Z_NEG, // Y, X
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
    std::array2<uint32_t, 6, CHUNK_LENGTH> borderMasks;
    // std::optional<std::array<uint32_t, CHUNK_LENGTH>> borderMask_X_POS; 
    // std::optional<std::array<uint32_t, CHUNK_LENGTH>> borderMask_X_NEG; 
    // std::optional<std::array<uint32_t, CHUNK_LENGTH>> borderMask_Y_POS; 
    // std::optional<std::array<uint32_t, CHUNK_LENGTH>> borderMask_Y_NEG; 
    // std::optional<std::array<uint32_t, CHUNK_LENGTH>> borderMask_Z_POS; 
    // std::optional<std::array<uint32_t, CHUNK_LENGTH>> borderMask_Z_BEG; 
    std::array<Chunk *, 6> neighbourChunks;
    AllocBuffer vertexBuffer;
    AllocBuffer indexBuffer;
    size_t vertexCount = 0, indexCount = 0;
    
    glm::ivec3 chunkCoord;

    Chunk();
    void deinit(VulkanContext &);
    ~Chunk();

    inline Block getBlock(int, int, int) const;
    inline Block getBlock(glm::ivec3) const;

    inline void setBlock(Block, int, int, int);
    inline void setBlock(Block, glm::ivec3);

    inline static glm::ivec3 coordFromIndex(int);

    void fillChunk();

    constexpr size_t size();
};

/*
    UNINIT, data == nullopt
    EMPTY, data.value == null
    FILLED, data != null
    MESHED, vertexCount >= 0
*/

struct ThreadPool {
    using QueueItem = std::unique_ptr<Chunk>;
    using ResultItem = QueueItem;

    std::mutex inputMtx, outputMtx;
    std::queue<QueueItem> input, output;
    std::condition_variable queueCond;
    
    std::atomic<bool> stopThreadPool = false;
    std::vector<std::thread> threads;

    void submit(QueueItem item);
    std::vector<ResultItem> &getResults();
};

struct Terrain {
    std::unordered_map<glm::ivec3, std::optional<std::unique_ptr<Chunk>>> chunks; // nullopt when being processed by other threads
    
    ThreadPool threadPool;
    
    VulkanContext *vkc;

    void queueChunk(std::unique_ptr<Chunk>);

    void init(VulkanContext *);

    void loadChunksAround(const glm::ivec3);
    void loadChunk(const glm::ivec3);
    void unloadChunk(std::unique_ptr<Chunk>);
    // void writeMesh(VulkanContext &, std::shared_ptr<Chunk>);
    void writeMesh(Chunk *chunk);

    void tickFrame(const Camera &camera);
    void draw(vk::CommandBuffer);


    void deinit();

};

