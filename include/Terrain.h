#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <deque>
#include <optional>
#include <unordered_map>
#include <format>

#include <glm/vec3.hpp>
#include <glm/gtx/hash.hpp>

#include "Concurrent.hpp"
#include "Buffer.h"
#include "Camera.hpp"
#include "Context.h"

constexpr int CHUNK_LENGTH = 32;
constexpr int CHUNK_BOTTOM = 0;
constexpr int CHUNK_TOP = 2;
using MaskLine = uint32_t;
template<>
struct std::formatter<glm::ivec3> {
    constexpr auto parse(std::format_parse_context &ctx) {
		return ctx.begin();
	}
    auto format(const glm::ivec3 &coord, std::format_context& ctx) const {
        return std::format_to(std::move(ctx.out()), "[{}, {}, {}]", coord.x, coord.y, coord.z);
    }
};
using ChunkCoord = glm::ivec3;
using InnerCoord = glm::ivec3;
struct GlobalCoord {
    inline operator glm::ivec3() {
        return this->chunk * CHUNK_LENGTH + this->inner;
    }
    GlobalCoord() {}
    GlobalCoord(glm::ivec3 v) : chunk(v / CHUNK_LENGTH), inner(v % CHUNK_LENGTH) {
        if (inner.x < 0) inner.x += CHUNK_LENGTH;
        if (inner.y < 0) inner.y += CHUNK_LENGTH;
        if (inner.z < 0) inner.z += CHUNK_LENGTH;
    }

    void operator+=(const glm::ivec3 &rhs) {
        inner += rhs;
        if (inner.x >= CHUNK_LENGTH) {
            inner.x -= CHUNK_LENGTH;
            chunk.x += 1;
        } else if (inner.x < 0) {
            inner.x += CHUNK_LENGTH;
            chunk.x -= 1;
        }
        if (inner.y >= CHUNK_LENGTH) {
            inner.y -= CHUNK_LENGTH;
            chunk.y += 1;
        } else if (inner.y < 0) {
            inner.y += CHUNK_LENGTH;
            chunk.y -= 1;
        }
        if (inner.z >= CHUNK_LENGTH) {
            inner.z -= CHUNK_LENGTH;
            chunk.z += 1;
        } else if (inner.z < 0) {
            inner.z += CHUNK_LENGTH;
            chunk.z -= 1;
        }
    }

    ChunkCoord chunk;
    InnerCoord inner;
};

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
    std::array2<MaskLine, CHUNK_LENGTH> opaquenessMask_Y{0}; // X, Z
    std::array2<MaskLine, CHUNK_LENGTH> opaquenessMask_Z{0}; // Y, X
    std::array2<MaskLine, CHUNK_LENGTH> opaquenessMask_X{0}; // Z, Y
    // std::array2<MaskLine, 6, CHUNK_LENGTH> borderMasks{0};
    // std::optional<std::array<uint32_t, CHUNK_LENGTH>> borderMask_X_POS; 
    // std::optional<std::array<uint32_t, CHUNK_LENGTH>> borderMask_X_NEG; 
    // std::optional<std::array<uint32_t, CHUNK_LENGTH>> borderMask_Y_POS; 
    // std::optional<std::array<uint32_t, CHUNK_LENGTH>> borderMask_Y_NEG; 
    // std::optional<std::array<uint32_t, CHUNK_LENGTH>> borderMask_Z_POS; 
    // std::optional<std::array<uint32_t, CHUNK_LENGTH>> borderMask_Z_NEG; 
    std::array<Chunk *, 6> neighbourChunks{0};
    AllocBuffer vertexBuffer;
    AllocBuffer indexBuffer;
    size_t vertexCount = 0, indexCount = 0;
    std::atomic<bool> filled = false;
    std::atomic<bool> meshed = false;
    
    ChunkCoord chunkCoord;

    Chunk(ChunkCoord);
    void deinit(VulkanContext &);
    ~Chunk();

    inline Block getBlock(int, int, int) const;
    inline Block getBlock(InnerCoord) const;

    inline void setBlock(Block, int, int, int);
    inline void setBlock(Block, InnerCoord);

    inline static InnerCoord coordFromIndex(int);

    void fillChunk();
    void writeMesh(VulkanContext &vkc);

};
// template <glm::length_t L, typename T, glm::qualifier Q> 
// struct std::formatter<glm::vec<L, T, Q>> : std::range_formatter<T> {
// 	constexpr auto parse(auto &ctx) {
// 		return ctx.begin();
// 	}

// 	auto format(const glm::vec<L, T, Q> &vec, auto &ctx) {
// 		return std::range_formatter<T>(std::span(&vec, L), ctx);
// 	}
// };
template<>
struct std::formatter<Chunk> {
    constexpr auto parse(std::format_parse_context &ctx) {
		return ctx.begin();
	}
    auto format(const Chunk &chunk, std::format_context& ctx) const {
        return std::format_to(std::move(ctx.out()), "[{}, {}, {}]", chunk.chunkCoord.x, chunk.chunkCoord.y, chunk.chunkCoord.z);
    }
};

struct ChunkThread {
    enum class QueueAction : bool { FILL, MESH };
    using QueueItem = std::pair<QueueAction, ChunkCoord>;
    using ResultItem = QueueItem;

    std::mutex threadMtx;
    std::condition_variable threadCond;
    std::deque<QueueItem> queue;

    void mesh(ChunkCoord coord) {
        // std::println("push MESH {}", coord);
        queue.push_back(std::make_pair(QueueAction::MESH, coord));
    };

    std::mutex chunksMtx;
    std::unordered_map<ChunkCoord, std::unique_ptr<Chunk>> chunks;

    ValueChannel<ChunkCoord> centerChunkCoord;

    using Edit = std::pair<GlobalCoord, Block>;
    QueueChannel<Edit> editsPending;
    
    std::atomic<bool> stopThread = false;
    std::thread thread;
    
    void loadAround(VulkanContext &, const ChunkCoord);
};

struct Terrain {
    VulkanContext &vkc;
    
    ChunkThread ct;

    Terrain(VulkanContext &);
    ~Terrain();

    void tickFrame(const Camera &camera) noexcept;
    std::optional<Block> getBlock(GlobalCoord) const noexcept;
    void setBlock(GlobalCoord, Block) noexcept;
    void draw(vk::CommandBuffer);

};

