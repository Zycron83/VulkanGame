#include <print>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>
#include <glm/gtx/io.hpp>

#include "SimplexNoise.h"

#include "Vertex.h"
#include "Terrain.h"
#include "Debug.h"
#include "Renderer.h"

extern DebugNameState g_DebugNameState;
extern DebugFrameStats g_DebugFrameStats;
extern Settings g_Settings;
extern Renderer *g_Renderer;

constexpr float A = .5f;
constexpr glm::vec3 WHITE = {1.f, 1.f, 1.f};
constexpr std::array<Vertex, 24> vertices = {
	// +X face
	Vertex{{1.f, 1.f, 1.f}, WHITE, {1.f, 0.f, 0.f}},
	Vertex{{1.f, 1.f, 0.f}, WHITE, {1.f, 0.f, 0.f}},
	Vertex{{1.f, 0.f, 1.f}, WHITE, {1.f, 0.f, 0.f}},
	Vertex{{1.f, 0.f, 0.f}, WHITE, {1.f, 0.f, 0.f}},
	// -X face
	Vertex{{0.f, 1.f, 0.f}, WHITE, {-1.f, 0.f, 0.f}},
	Vertex{{0.f, 1.f, 1.f}, WHITE, {-1.f, 0.f, 0.f}},
	Vertex{{0.f, 0.f, 0.f}, WHITE, {-1.f, 0.f, 0.f}},
	Vertex{{0.f, 0.f, 1.f}, WHITE, {-1.f, 0.f, 0.f}},
	// +Y face
	Vertex{{0.f, 1.f, 0.f}, WHITE, {0.f, 1.f, 0.f}},
	Vertex{{1.f, 1.f, 0.f}, WHITE, {0.f, 1.f, 0.f}},
	Vertex{{0.f, 1.f, 1.f}, WHITE, {0.f, 1.f, 0.f}},
	Vertex{{1.f, 1.f, 1.f}, WHITE, {0.f, 1.f, 0.f}},
	// -Y face
	Vertex{{0.f, 0.f, 1.f}, WHITE, {0.f, -1.f, 0.f}},
	Vertex{{1.f, 0.f, 1.f}, WHITE, {0.f, -1.f, 0.f}},
	Vertex{{0.f, 0.f, 0.f}, WHITE, {0.f, -1.f, 0.f}},
	Vertex{{1.f, 0.f, 0.f}, WHITE, {0.f, -1.f, 0.f}},
	// +Z face
	Vertex{{0.f, 1.f, 1.f}, WHITE, {0.f, 0.f, 1.f}},
	Vertex{{1.f, 1.f, 1.f}, WHITE, {0.f, 0.f, 1.f}},
	Vertex{{0.f, 0.f, 1.f}, WHITE, {0.f, 0.f, 1.f}},
	Vertex{{1.f, 0.f, 1.f}, WHITE, {0.f, 0.f, 1.f}},
	// -Z face
	Vertex{{1.f, 1.f, 0.f}, WHITE, {0.f, 0.f, -1.f}},
	Vertex{{0.f, 1.f, 0.f}, WHITE, {0.f, 0.f, -1.f}},
	Vertex{{1.f, 0.f, 0.f}, WHITE, {0.f, 0.f, -1.f}},
	Vertex{{0.f, 0.f, 0.f}, WHITE, {0.f, 0.f, -1.f}},
};
constexpr std::array<Index, 36> indices = {
	0, 1, 2, 2, 1, 3,
	4, 5, 6, 6, 5, 7,
	8, 9, 10, 10, 9, 11,
	12, 13, 14, 14, 13, 15,
	16, 17, 18, 18, 17, 19,
	20, 21, 22, 22, 21, 23,
};

void Chunk::init(const VulkanContext &vkc) {
    // std::string chunk_name = std::format("Chunk {}, {}", chunkCoord.x, chunkCoord.z);
    // g_DebugNameState.Push(chunk_name.c_str());
    // g_DebugNameState.Name("Vertex");
    // vertexBuffer.initDevice(vkc, 
    //     sizeof(vertices) * CHUNK_LENGTH * CHUNK_LENGTH * CHUNK_LENGTH / 4, 
    //     vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst
    // );
    // g_DebugNameState.Name("Index");
    // indexBuffer.initDevice(vkc, 
    //     sizeof(indices) * CHUNK_LENGTH * CHUNK_LENGTH * CHUNK_LENGTH / 4, 
    //     vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst
    // );
    // g_DebugNameState.Pop();
}
void Chunk::deinit(VulkanContext &vkc) {
    auto &queue = vkc.frames[vkc.currentFrame].destruction_queue;
    if (vertexCount > 0 || indexCount > 0) {
        queue.push_back(vertexBuffer);
        queue.push_back(indexBuffer);
        g_Settings.chunkBytes -= this->vertexBuffer.size + this->indexBuffer.size;
        this->vertexCount = 0;
        this->indexCount = 0;
    }
}

Block Chunk::getBlock(int x, int y, int z) const {
    return data[x * CHUNK_LENGTH * CHUNK_LENGTH + y * CHUNK_LENGTH + z];
}
Block Chunk::getBlock(glm::ivec3 coord) const {
    if (glm::any(glm::greaterThanEqual(coord, {CHUNK_LENGTH, CHUNK_LENGTH, CHUNK_LENGTH}))
     || glm::any(glm::lessThan(coord, {0,0,0}))
    ) {
        return Block::Invalid;
    }
    return data[coord.x * CHUNK_LENGTH * CHUNK_LENGTH + coord.y * CHUNK_LENGTH + coord.z];
}

void Chunk::setBlock(Block b, int x, int y, int z) {
    assert(x < CHUNK_LENGTH && y < CHUNK_LENGTH && z < CHUNK_LENGTH);
    assert(x >= 0 && y >= 0 && z >= 0);
    data[x * CHUNK_LENGTH * CHUNK_LENGTH + y * CHUNK_LENGTH + z] = b;
    if (transparent(b)) { // set 0
        opaquenessMask_Y[x][z] &= ~(1 << y);
        opaquenessMask_Z[x][y] &= ~(1 << z);
        opaquenessMask_X[z][y] &= ~(1 << x);
    } else { // set 1
        opaquenessMask_Y[x][z] |= (1 << y);
        opaquenessMask_Z[x][y] |= (1 << z);
        opaquenessMask_X[z][y] |= (1 << x);
    }
    // queueOp = QueueOp::UPDATE_MESH;
    // Chunk *neighbour = nullptr;
    // if      (x ==               0) neighbour = neighbourChunks[X_NEG];
    // else if (y ==               0) neighbour = neighbourChunks[Y_NEG];
    // else if (z ==               0) neighbour = neighbourChunks[Z_NEG];
    // else if (x == CHUNK_LENGTH - 1) neighbour = neighbourChunks[X_POS]; 
    // else if (y == CHUNK_LENGTH - 1) neighbour = neighbourChunks[Y_POS]; 
    // else if (z == CHUNK_LENGTH - 1) neighbour = neighbourChunks[Z_POS]; 
    // if (neighbour) {
    //     neighbour->queueOp = QueueOp::UPDATE_MESH;
    // }
}
void Chunk::setBlock(Block b, glm::ivec3 coord) {
    data[coord.x * CHUNK_LENGTH * CHUNK_LENGTH + coord.y * CHUNK_LENGTH + coord.z] = b;
}

glm::ivec3 Chunk::coordFromIndex(int i) {
    // invert the mapping from index -> x,y,z used by get/set:
    // i = x * (CHUNK_LENGTH * CHUNK_LENGTH) + y * CHUNK_LENGTH + z
    glm::ivec3 ret; 
    ret.z = i % CHUNK_LENGTH;
    int tmp = i / CHUNK_LENGTH;
    ret.y = tmp % CHUNK_LENGTH;
    ret.x = tmp / CHUNK_LENGTH;
    return ret;
}

void Chunk::fillChunk() {
    const auto simpleSetBlock = [this](Block b, int x, int y, int z) {
        assert(x < CHUNK_LENGTH && y < CHUNK_LENGTH && z < CHUNK_LENGTH);
        assert(x >= 0 && y >= 0 && z >= 0);
        data[x * CHUNK_LENGTH * CHUNK_LENGTH + y * CHUNK_LENGTH + z] = b;
        if (transparent(b)) { // set 0
            opaquenessMask_Y[x][z] &= ~(1 << y);
            opaquenessMask_Z[x][y] &= ~(1 << z);
            opaquenessMask_X[z][y] &= ~(1 << x);
        } else { // set 1
            opaquenessMask_Y[x][z] |= (1 << y);
            opaquenessMask_Z[x][y] |= (1 << z);
            opaquenessMask_X[z][y] |= (1 << x);
        }
    };

    constexpr float scale = 0.005f;
    for (int i = 0; i < CHUNK_LENGTH; i += 1)
    for (int j = 0; j < CHUNK_LENGTH; j += 1) {
        // const float height = simplex.fractal(g_Settings.Noise.octaves, (i + chunkCoord.x * CHUNK_LENGTH) * scale, (j + chunkCoord.z * CHUNK_LENGTH) * scale) * CHUNK_LENGTH;
        const float height = (SimplexNoise::noise((i + chunkCoord.x * CHUNK_LENGTH) * scale, (j + chunkCoord.z * CHUNK_LENGTH) * scale) * CHUNK_LENGTH + CHUNK_LENGTH) / 2.0;
        // const float height = 10;
        for (int k = 0; k < CHUNK_LENGTH; k += 1) {
            if (k < height) {
                simpleSetBlock(Block::Stone, i, k, j);
            } else {
                simpleSetBlock(Block::Air, i, k, j);
            }
        }
    };
}

// constexpr glm::ivec3 unitDir[6] = {
//     {1,0,0}, {-1,0,0},
//     {0,1,0}, {0,-1,0},
//     {0,0,1}, {0,0,-1},
// };

void Chunk::writeMesh(VulkanContext &vkc) {
    constexpr int voxelCount = CHUNK_LENGTH * CHUNK_LENGTH * CHUNK_LENGTH;
    // std::vector<Vertex> vertexData;
    // vertexData.resize(voxelCount * vertices.size());
    // std::vector<Index> indexData;
    // indexData.resize(voxelCount * indices.size());
    static Vertex vertexData[voxelCount * vertices.size()];
    static Index indexData[voxelCount * indices.size()];

    // track where we are writing into the mapped buffers
    size_t vertexOffset = 0; // in vertices
    size_t indexOffset = 0;  // in indices

    const auto writeFace = [&, this](Dir dir, glm::ivec3 coord) {
        for (int j = 0; j < 4; j += 1) {
            auto v = vertices[int{dir} * 4 + j];
            v.position += coord + this->chunkCoord * CHUNK_LENGTH;
            vertexData[vertexOffset + j] = v;
        }
        for (int j = 0; j < 6; j += 1) {
            indexData[indexOffset + j] = Index(indices[j].val + vertexOffset);
        }
        vertexOffset += 4;
        indexOffset += 6;
    };

    // auto start_time = std::chrono::steady_clock::now();
    for (int x = 0; x < CHUNK_LENGTH; x += 1) {
        for (int z = 0; z < CHUNK_LENGTH; z += 1) {
            uint64_t bits = opaquenessMask_Y[x][z];
            bool curr, prev = bits & 1;
            // bits >>= 1;
            int y = 0;
            while (y < CHUNK_LENGTH) {
                curr = bits & 1;
                switch (curr - prev) {
                    case -1: writeFace(Y_POS, {x, y-1, z}); break;
                    case 1: writeFace(Y_NEG, {x, y, z}); break;
                    case 0: break;
                    default: std::unreachable();
                }
                prev = curr;
                uint8_t skip = curr ? std::countr_one(bits) : std::countr_zero(bits);
                y += skip;
                bits >>= skip;
            }
        }
    // }

    // for (int x = 0; x < CHUNK_LENGTH; x += 1) {
        for (int y = 0; y < CHUNK_LENGTH; y += 1) {
            uint64_t bits = opaquenessMask_Z[x][y];
            bool curr, prev = bits & 1;
            // bits >>= 1;
            int z = 0;
            while (z < CHUNK_LENGTH) {
                curr = bits & 1;
                switch (curr - prev) {
                    case -1: writeFace(Z_POS, {x, y, z-1}); break;
                    case 1: writeFace(Z_NEG, {x, y, z}); break;
                    case 0: break;
                    default: std::unreachable();
                }
                prev = curr;
                uint8_t skip = curr ? std::countr_one(bits) : std::countr_zero(bits);
                z += skip;
                bits >>= skip;
            }
        }
    }

    for (int z = 0; z < CHUNK_LENGTH; z += 1) {
        for (int y = 0; y < CHUNK_LENGTH; y += 1) {
            uint64_t bits = opaquenessMask_X[z][y];
            bool curr, prev = bits & 1;
            // bits >>= 1;
            int x = 0;
            while (x < CHUNK_LENGTH) {
                curr = bits & 1;
                switch (curr - prev) {
                    case -1: writeFace(X_POS, {x-1, y, z}); break;
                    case 1: writeFace(X_NEG, {x, y, z}); break;
                    case 0: break;
                    default: std::unreachable();
                }
                prev = curr;
                uint8_t skip = curr ? std::countr_one(bits) : std::countr_zero(bits);
                x += skip;
                bits >>= skip;
            }
        }
    }

    // auto end_time = std::chrono::steady_clock::now();
    // std::println("Passes took {} us", std::chrono::duration<float, std::micro>(end_time - start_time).count());

    if (vertexOffset == 0 || indexOffset == 0) {
        // empty chunk
        this->vertexCount = 0;
        this->indexCount = 0;
        return;
    }

    std::string chunk_name = std::format("Chunk {}, {}", chunkCoord.x, chunkCoord.z);
    g_DebugNameState.Push(chunk_name.c_str());
    g_DebugNameState.Name("Vertex");
    vertexBuffer.initDevice(vkc,
        sizeof(Vertex) * vertexOffset,
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst
    );
    g_DebugNameState.Name("Index");
    indexBuffer.initDevice(vkc,
        sizeof(Index) * indexOffset,
        vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst
    );
    g_DebugNameState.Pop();

    vkc.uploadBuffers({
        { this->indexBuffer, indexData },
        { this->vertexBuffer, vertexData }
    });
    g_Settings.chunkBytes += this->vertexBuffer.size + this->indexBuffer.size;
    this->vertexCount = vertexOffset;
    this->indexCount = indexOffset;
}

constexpr size_t Chunk::size() {
    return sizeof(data);
}

void Terrain::init(const VulkanContext &vkc) {
    using namespace std::chrono;
    auto start = steady_clock::now();
    constexpr int N = 20;
    for (int y = 0; y < 1; y++)
    for (int x = 0; x < N; x++)
    for (int z = 0; z < N; z++) {
        // std::println("Loading: {}", std::chrono::steady_clock::now().time_since_epoch());
        loadChunk(vkc, {x, y, z});
    };
    auto end = steady_clock::now();
    this->stopThread = false;
    this->chunkQueueProcessor = std::thread(processChunks, this, const_cast<VulkanContext*>(&vkc));
    std::println("Initted Terrain in {} milliseconds", duration<float, std::milli>(end - start).count());
}

void Terrain::dirtyChunk(Chunk *chunk, QueueOp queueOp) {
    {
        std::scoped_lock lock(queueMtx);
        if (not chunk->inQueue) {
            chunkMeshQueue.push_back(chunk);
            chunk->inQueue = true;
            chunk->queueOp = queueOp;
        }
    }
    queueCond.notify_one();
}

// Chunk processing thread
void Terrain::processChunks(Terrain *terrain, VulkanContext *vkc) {
    while (not terrain->stopThread) {
        std::unique_lock lock(terrain->queueMtx);
        // std::println("Queue Count: {}", terrain->chunkMeshQueue.size());
        terrain->queueCond.wait(lock, [terrain]() {
            return not terrain->chunkMeshQueue.empty() || terrain->stopThread;
        });
        if (terrain->chunkMeshQueue.empty()) {
            break;
        }
        Chunk *chunk = terrain->chunkMeshQueue.front();
        terrain->chunkMeshQueue.pop_front();
        lock.unlock();

        switch (chunk->queueOp) {
            case FILL: chunk->fillChunk(); 
            case UPDATE_MESH: chunk->writeMesh(*vkc);
            case NONE: 
        }
        // std::println("X,0,0: {:b}, 0,Y,0: {:b}, 0,0,Z: {:b}", chunk->opaquenessMask_X[1][0], chunk->opaquenessMask_Y[1][0], chunk->opaquenessMask_Z[1][0]);
        chunk->inQueue = false;
        chunk->queueOp = QueueOp::NONE;
    }
}

void Terrain::loadChunksAround(VulkanContext &vkc, const glm::ivec3 centerChunkCoord) {
    static glm::ivec3 lastCenter = {0,0,0};
    if (centerChunkCoord == lastCenter) return;
    else lastCenter = centerChunkCoord;
    std::cout << "Loading chunks around " << centerChunkCoord << std::endl;

    constexpr int RENDER_DIST = 10;
    constexpr int UNRENDER_DIST = RENDER_DIST + 2;
    for (int x = -UNRENDER_DIST; x <= UNRENDER_DIST; x++) {
        for (int z = -UNRENDER_DIST; z <= UNRENDER_DIST; z++) {
            glm::ivec3 chunkCoord = centerChunkCoord + glm::ivec3(x, 0, z);
            chunkCoord.y = 0;
            if (x <= RENDER_DIST && x >= -RENDER_DIST && z <= RENDER_DIST && z >= -RENDER_DIST) {
                // load
                if (!chunks.contains(chunkCoord)) {
                    loadChunk(vkc, chunkCoord);
                }
            } else {
                // unload
                if (chunks.contains(chunkCoord)) {
                    unloadChunk(vkc, chunkCoord);
                }
            }
        }
    }
}

void Terrain::loadChunk(const VulkanContext &vkc, const glm::ivec3 chunkCoord) {
    if (chunks.contains(chunkCoord)) return;

    chunks[chunkCoord] = std::make_unique<Chunk>();
    auto chunk = chunks[chunkCoord].get();
    chunk->chunkCoord = chunkCoord;
    chunk->init(vkc);

    dirtyChunk(chunk, QueueOp::FILL);
}

void Terrain::tickFrame(VulkanContext &vkc, const Camera &camera) {
    loadChunksAround(vkc, camera.position / static_cast<float>(CHUNK_LENGTH));
}

void Terrain::draw(vk::CommandBuffer commandBuffer) {
    for (auto& [_, chunk] : chunks) {
        if (chunk->vertexCount == 0 && chunk->indexCount == 0 || chunk->inQueue) continue; // ... bool member `filled` removed
        assert(chunk->vertexBuffer.size > 0 && chunk->indexBuffer.size > 0);
        commandBuffer.bindVertexBuffers(0, chunk->vertexBuffer.buffer, static_cast<vk::DeviceSize>(0));
        commandBuffer.bindIndexBuffer(chunk->indexBuffer.buffer, 0, Index::enumType);

        commandBuffer.drawIndexed(chunk->indexCount, 1, 0, 0, 0);
        g_DebugFrameStats.index_count += chunk->indexCount;
    }
}

void Terrain::unloadChunk(VulkanContext &vkc, const glm::ivec3 chunkCoord) {
    assert(chunks.contains(chunkCoord));
    auto chunk = chunks.at(chunkCoord).get();
    if (chunk->inQueue) {
        auto n = std::ranges::find(chunkMeshQueue, chunk);
        chunkMeshQueue.erase(n);
    }
    chunk->deinit(vkc);
    this->chunks.erase(chunkCoord);
}

void Terrain::deinit(VulkanContext &vkc) {
    auto &queue = vkc.frames[vkc.currentFrame].destruction_queue;
    for (auto& [_, chunk] : chunks) {
        chunk->deinit(vkc);
    }
    this->chunks.clear();
    while (not chunkMeshQueue.empty()) {
        chunkMeshQueue.pop_front();
    }
    this->stopThread = true;
    this->queueCond.notify_one();
    this->chunkQueueProcessor.join();
}

