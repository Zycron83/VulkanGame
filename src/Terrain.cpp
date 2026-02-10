#include <optional>
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

extern DebugFrameStats g_DebugFrameStats;
extern Settings g_Settings;
extern Renderer *g_Renderer;

constexpr int RENDER_DIST = 5;

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

Chunk::Chunk() : data(std::make_unique<typeof(*data)>()) {}
Chunk::~Chunk() {
    assert(data == nullptr && vertexCount == 0);
}

void Chunk::deinit(VulkanContext &vkc) {
    // std::println("Deinitting Chunk [{}, {}, {}]", chunkCoord.x, chunkCoord.y, chunkCoord.z);
    if (vertexCount > 0 || indexCount > 0) {
        vkc.queueDestroy(vertexBuffer);
        vkc.queueDestroy(indexBuffer);
        g_Settings.chunkBytes -= this->vertexBuffer.size + this->indexBuffer.size;
        this->vertexCount = 0;
        this->indexCount = 0;
    }
    this->data.reset();
}

Block Chunk::getBlock(int x, int y, int z) const {
    return (*data)[x * CHUNK_LENGTH * CHUNK_LENGTH + y * CHUNK_LENGTH + z];
}
Block Chunk::getBlock(glm::ivec3 coord) const {
    if (glm::any(glm::greaterThanEqual(coord, {CHUNK_LENGTH, CHUNK_LENGTH, CHUNK_LENGTH}))
     || glm::any(glm::lessThan(coord, {0,0,0}))
    ) {
        return Block::Invalid;
    }
    return (*data)[coord.x * CHUNK_LENGTH * CHUNK_LENGTH + coord.y * CHUNK_LENGTH + coord.z];
}

void Chunk::setBlock(Block b, int x, int y, int z) {
    assert(x < CHUNK_LENGTH && y < CHUNK_LENGTH && z < CHUNK_LENGTH);
    assert(x >= 0 && y >= 0 && z >= 0);
    (*data)[x * CHUNK_LENGTH * CHUNK_LENGTH + y * CHUNK_LENGTH + z] = b;
    if (transparent(b)) { // set 0
        opaquenessMask_Y[x][z] &= ~(1 << y);
        opaquenessMask_Z[y][x] &= ~(1 << z);
        opaquenessMask_X[z][y] &= ~(1 << x);
    } else { // set 1
        opaquenessMask_Y[x][z] |= (1 << y);
        opaquenessMask_Z[y][x] |= (1 << z);
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
    (*data)[coord.x * CHUNK_LENGTH * CHUNK_LENGTH + coord.y * CHUNK_LENGTH + coord.z] = b;
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
    // std::scoped_lock lock{chunkMtx};
    const auto simpleSetBlock = [this](Block b, int x, int y, int z) {
        assert(x < CHUNK_LENGTH && y < CHUNK_LENGTH && z < CHUNK_LENGTH);
        assert(x >= 0 && y >= 0 && z >= 0);
        (*data)[x * CHUNK_LENGTH * CHUNK_LENGTH + y * CHUNK_LENGTH + z] = b;
        if (transparent(b)) { // set 0
            opaquenessMask_Y[x][z] &= ~(1 << y);
            opaquenessMask_Z[y][x] &= ~(1 << z);
            opaquenessMask_X[z][y] &= ~(1 << x);
        } else { // set 1
            opaquenessMask_Y[x][z] |= (1 << y);
            opaquenessMask_Z[y][x] |= (1 << z);
            opaquenessMask_X[z][y] |= (1 << x);
        }
    };

    constexpr float scale = 0.005f;
    for (int i = 0; i < CHUNK_LENGTH; i += 1)
    for (int j = 0; j < CHUNK_LENGTH; j += 1) {
        // const float height = simplex.fractal(g_Settings.Noise.octaves, (i + chunkCoord.x * CHUNK_LENGTH) * scale, (j + chunkCoord.z * CHUNK_LENGTH) * scale) * CHUNK_LENGTH;
        const float height = (SimplexNoise::noise((i + chunkCoord.x * CHUNK_LENGTH) * scale, (j + chunkCoord.z * CHUNK_LENGTH) * scale) * CHUNK_LENGTH + CHUNK_LENGTH);
        // const float height = 10;
        for (int k = 0; k < CHUNK_LENGTH; k += 1) {
            if (k + chunkCoord.y * CHUNK_LENGTH < height) {
                simpleSetBlock(Block::Stone, i, k, j);
            } else {
                simpleSetBlock(Block::Air, i, k, j);
            }
        }
    };
}

constexpr glm::ivec3 unitDir[6] = {
    {1,0,0}, {-1,0,0},
    {0,1,0}, {0,-1,0},
    {0,0,1}, {0,0,-1},
};

thread_local Vertex *vertexData;
thread_local Index *indexData;

// Must own.
void Terrain::writeMesh(Chunk *chunk) {

    size_t vertexOffset = 0; // in vertices
    size_t indexOffset = 0;  // in indices

    const auto writeFace = [&, this](Dir dir, glm::ivec3 coord) {
        for (int j = 0; j < 4; j += 1) {
            auto v = vertices[int{dir} * 4 + j];
            v.position += coord + chunk->chunkCoord * CHUNK_LENGTH;
            vertexData[vertexOffset + j] = v;
        }
        for (int j = 0; j < 6; j += 1) {
            indexData[indexOffset + j] = Index(indices[j].val + vertexOffset);
        }
        vertexOffset += 4;
        indexOffset += 6;
    };

    auto start_time = std::chrono::steady_clock::now();

    for (int x = 0; x < CHUNK_LENGTH; x += 1) {
        auto upMask = chunk->borderMasks[Y_POS][x];
        auto downMask = chunk->borderMasks[Y_NEG][x];
        for (int z = 0; z < CHUNK_LENGTH; z += 1) {
            // up <<|>> down
            uint64_t bits = chunk->opaquenessMask_Y[x][z];
            if (bits == 0) continue;
            bool curr = false;
            bool prev = downMask & 1;
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
            curr = upMask & 1;
            if (curr - prev == -1) writeFace(Y_POS, {x, CHUNK_LENGTH-1, z});
            downMask >>= 1;
            upMask >>= 1;
        }
    }

    for (int x = 0; x < CHUNK_LENGTH; x += 1) {
        for (int y = 0; y < CHUNK_LENGTH; y += 1) {
            uint64_t bits = chunk->opaquenessMask_Z[y][x];
            if (bits == 0) continue;
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
            uint64_t bits = chunk->opaquenessMask_X[z][y];
            if (bits == 0) continue;
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

    auto end_time = std::chrono::steady_clock::now();
    // std::println("Meshing [{}, {}, {}] took {} us", chunkCoord.x, chunkCoord.y, chunkCoord.z, std::chrono::duration<float, std::micro>(end_time - start_time).count());

    if (vertexOffset == 0) {
        assert(indexOffset == 0);
        // empty chunk
        chunk->vertexCount = 0;
        chunk->indexCount = 0;
        return;
    }

    std::string chunk_name = std::format("Chunk {}, {}", chunk->chunkCoord.x, chunk->chunkCoord.z);
    chunk->vertexBuffer.initDevice(std::format("Chunk [{}, {}, {}] Vertex Buffer", chunk->chunkCoord.x, chunk->chunkCoord.y, chunk->chunkCoord.z), *vkc,
        sizeof(Vertex) * vertexOffset,
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst
    );
    
    chunk->indexBuffer.initDevice(std::format("Chunk [{}, {}, {}] Index Buffer", chunk->chunkCoord.x, chunk->chunkCoord.y, chunk->chunkCoord.z), *vkc,
        sizeof(Index) * indexOffset,
        vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst
    );
    
    vkc->uploadBuffers({
        { chunk->indexBuffer, indexData },
        { chunk->vertexBuffer, vertexData }
    });
    g_Settings.chunkBytes += chunk->vertexBuffer.size + chunk->indexBuffer.size;
    chunk->vertexCount = vertexOffset;
    chunk->indexCount = indexOffset;
}

constexpr size_t Chunk::size() {
    return sizeof(data);
}

static void processChunks(Terrain *terrain, VulkanContext *vkc);

void Terrain::init(VulkanContext *vkc) {
    using namespace std::chrono;
    auto start = steady_clock::now();
    auto end = steady_clock::now();
    auto thread_count = 2; //std::thread::hardware_concurrency() - 1;
    this->threadPool.threads.reserve(thread_count);
    for (int i = 0; i < thread_count; i++) {
        this->threadPool.threads.emplace_back(
            processChunks, this, vkc
        );
    }
    this->vkc = vkc;
    std::println("Initted Terrain in {} milliseconds", duration<float, std::milli>(end - start).count());
}

void ThreadPool::submit(std::unique_ptr<Chunk> chunk) {
    {
        std::scoped_lock lock{inputMtx};
        input.push(std::move(chunk));
    }
    queueCond.notify_one();
}

// Chunk processing thread
void processChunks(Terrain *terrain, VulkanContext *vkc) {
    std::println("Reporting from thread {}! o7", std::this_thread::get_id());
    constexpr int voxelCount = CHUNK_LENGTH * CHUNK_LENGTH * CHUNK_LENGTH;
    vertexData = new Vertex[voxelCount * vertices.size()];
    indexData = new Index[voxelCount * indices.size()];
    auto &threadPool = terrain->threadPool;
    while (true) {
        std::unique_lock inputLock{threadPool.inputMtx};
        // std::println("Queue Count: {}", terrain->chunkMeshQueue.size());
        threadPool.queueCond.wait(inputLock, [&threadPool]() {
            return not threadPool.input.empty() || threadPool.stopThreadPool;
        });
        if (threadPool.stopThreadPool) break;

        auto chunk = std::move(threadPool.input.front());
        threadPool.input.pop();
        inputLock.unlock();

        chunk->fillChunk();
        terrain->writeMesh(chunk.get());
        
        // std::array<Chunk *, 6> neighbours{nullptr};
        // {
        //     std::scoped_lock chunksLock{terrain->chunksMtx};
        //     for (Dir dir : {Y_POS, Y_NEG}) {
        //         auto neighbourCoord = chunk->chunkCoord + unitDir[dir];
        //         if (not terrain->chunks.contains(neighbourCoord)) continue;
        //         if (auto &chunk = terrain->chunks.at(chunk->chunkCoord + unitDir[dir]); chunk) {
        //             neighbours[dir] = chunk.get();
        //         }
        //     }
        // }
        // for (Dir dir : {Y_POS, Y_NEG}) {
        //     if (auto neighbour = neighbours[dir]; neighbour) {
        //         std::scoped_lock lock{neighbour->chunkMtx};
        //         chunk->neighbourChunks[dir] = neighbour;
        //         neighbour->neighbourChunks[dir] = chunk.get();
        //     }
        // }
        
        std::scoped_lock outputLock{threadPool.outputMtx};
        threadPool.output.push(std::move(chunk));
        
        // std::println("X,0,0: {:b}, 0,Y,0: {:b}, 0,0,Z: {:b}", chunk->opaquenessMask_X[1][0], chunk->opaquenessMask_Y[1][0], chunk->opaquenessMask_Z[1][0]);
    }
    delete[] vertexData;
    delete[] indexData;
}

void Terrain::loadChunksAround(const glm::ivec3 centerChunkCoord) {
    static glm::ivec3 lastCenter = glm::ivec3{INT_MAX};
    if (centerChunkCoord == lastCenter) return;
    else lastCenter = centerChunkCoord;
    std::println("Loading chunks around [{}, {}, {}]", centerChunkCoord.x, centerChunkCoord.y, centerChunkCoord.z);

    const auto L = [](int x, int z) {
        // return abs(x) + abs(z);
        return sqrt(x*x + z*z);
    };

    for (int x = -RENDER_DIST; x <= RENDER_DIST; x++) {
    for (int y = CHUNK_BOTTOM; y <= CHUNK_TOP  ; y++) {
    for (int z = -RENDER_DIST; z <= RENDER_DIST; z++) {
        if (L(x, z) <= RENDER_DIST) loadChunk(centerChunkCoord + glm::ivec3(x, y - centerChunkCoord.y, z));
    }}};
    
    auto it = chunks.begin();
    while (it != chunks.end()) {
        const glm::ivec3 offset = it->first - centerChunkCoord;
        if (it->second && (L(offset.x, offset.z)) > RENDER_DIST + 2) {
            // unloadChunk(vkc, it->first);
            if (it->second) {
                // std::println("Unloading [{}, {}, {}]", it->first.x, it->first.y, it->first.z);
                it->second.value()->deinit(*vkc);
            }
            it = chunks.erase(it);
        } else {
            it++;
        }
    }
}

// Must aquire chunks
void Terrain::loadChunk(const glm::ivec3 chunkCoord) {
    // std::println("Loading [{}, {}, {}]", chunkCoord.x, chunkCoord.y, chunkCoord.z);
    if (chunkCoord.y < CHUNK_BOTTOM || chunkCoord.y > CHUNK_TOP || chunks.contains(chunkCoord)) return;

    this->chunks.insert(std::pair{chunkCoord, std::nullopt});
    auto chunk = std::make_unique<Chunk>();
    chunk->chunkCoord = chunkCoord;

    threadPool.submit(std::move(chunk));
}

void Terrain::tickFrame(const Camera &camera) {
    loadChunksAround(camera.position / static_cast<float>(CHUNK_LENGTH));
    std::scoped_lock outputLock{threadPool.outputMtx};
    while (not threadPool.output.empty()) {
        auto &chunk = threadPool.output.front();
        // std::println("Moving [{}, {}, {}]", chunk->chunkCoord.x, chunk->chunkCoord.y, chunk->chunkCoord.z);
        chunks.at(chunk->chunkCoord) = std::move(chunk);
        threadPool.output.pop();
    }
}

void Terrain::draw(vk::CommandBuffer commandBuffer) {
    for (auto& [_, chunkOpt] : chunks) {
        if (not chunkOpt) continue;
        auto &chunk = *chunkOpt.value();
        if (chunk.vertexCount == 0) {
            assert(chunk.indexCount == 0);
            // std::println("vertex buffer size is 0 when drawing Chunk [{}, {}, {}]", chunk.chunkCoord.x, chunk.chunkCoord.y, chunk.chunkCoord.z);
            continue;
        }
        commandBuffer.bindVertexBuffers(0, chunk.vertexBuffer.buffer, static_cast<vk::DeviceSize>(0));
        commandBuffer.bindIndexBuffer(chunk.indexBuffer.buffer, 0, Index::enumType);

        commandBuffer.drawIndexed(chunk.indexCount, 1, 0, 0, 0);
        g_DebugFrameStats.index_count += chunk.indexCount;
    }
}

void Terrain::unloadChunk(std::unique_ptr<Chunk> chunk) {
    // std::println("Unloading [{}, {}, {}]", chunk->chunkCoord.x, chunk->chunkCoord.y, chunk->chunkCoord.z);
    chunk->deinit(*vkc);
}

void Terrain::deinit() {
    threadPool.stopThreadPool = true;
    threadPool.queueCond.notify_all();
    for (auto &thread : threadPool.threads) {
        thread.join();
        std::println("Thread {} joining back! o7", std::this_thread::get_id());
    }
    while (not threadPool.input.empty()) {
        threadPool.input.front()->deinit(*vkc);
        threadPool.input.pop();
    }
    while (not threadPool.output.empty()) {
        threadPool.output.front()->deinit(*vkc);
        threadPool.output.pop();
    }
    for (auto& [_, chunkOpt] : chunks) {
        if (chunkOpt.has_value()) chunkOpt.value()->deinit(*vkc);
    }
    this->chunks.clear();
    // threadPool.input.clear();
}

