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
    this->data = std::make_unique<typeof(*data)>();
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
    std::lock_guard lock{chunkMtx};
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
    std::lock_guard lock{chunkMtx};
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

std::vector<float> times;

void Terrain::writeMesh(VulkanContext &vkc, glm::ivec3 chunkCoord) {
    std::lock_guard lock{chunkMtx};
    constexpr int voxelCount = CHUNK_LENGTH * CHUNK_LENGTH * CHUNK_LENGTH;
    static Vertex vertexData[voxelCount * vertices.size()];
    static Index indexData[voxelCount * indices.size()];

    auto chunk = chunks.at(chunkCoord);

    // track where we are writing into the mapped buffers
    size_t vertexOffset = 0; // in vertices
    size_t indexOffset = 0;  // in indices

    const auto writeFace = [&, this](Dir dir, glm::ivec3 coord) {
        for (int j = 0; j < 4; j += 1) {
            auto v = vertices[int{dir} * 4 + j];
            v.position += coord + chunkCoord * CHUNK_LENGTH;
            vertexData[vertexOffset + j] = v;
        }
        for (int j = 0; j < 6; j += 1) {
            indexData[indexOffset + j] = Index(indices[j].val + vertexOffset);
        }
        vertexOffset += 4;
        indexOffset += 6;
    };

    std::array<uint32_t, CHUNK_LENGTH> emptyMask;
    // memset(emptyMask, 0, CHUNK_LENGTH);

    auto start_time = std::chrono::steady_clock::now();
    auto upChunk = chunkCoord + unitDir[Y_POS];
    auto downChunk = chunkCoord + unitDir[Y_NEG];
    auto upZMask = chunks.contains(upChunk) ? chunks.at(upChunk)->opaquenessMask_Z[0] : emptyMask;
    auto downZMask = chunks.contains(downChunk) ? chunks.at(downChunk)->opaquenessMask_Z[0] : emptyMask;
    for (int x = 0; x < CHUNK_LENGTH; x += 1) {
        auto upMask = upZMask[x];
        auto downMask = downZMask[x];
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
    times.push_back(std::chrono::duration<float, std::micro>(end_time - start_time).count());
    // std::println("Meshing [{}, {}, {}] took {} us", chunkCoord.x, chunkCoord.y, chunkCoord.z, std::ranges::fold_left(times, 0, std::plus<float>()) / times.size());

    if (vertexOffset == 0 || indexOffset == 0) {
        // empty chunk
        chunk->vertexCount = 0;
        chunk->indexCount = 0;
        return;
    }

    std::string chunk_name = std::format("Chunk {}, {}", chunkCoord.x, chunkCoord.z);
    g_DebugNameState.Push(chunk_name.c_str());
    g_DebugNameState.Name("Vertex");
    chunk->vertexBuffer.initDevice(vkc,
        sizeof(Vertex) * vertexOffset,
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst
    );
    g_DebugNameState.Name("Index");
    chunk->indexBuffer.initDevice(vkc,
        sizeof(Index) * indexOffset,
        vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst
    );
    g_DebugNameState.Pop();

    vkc.uploadBuffers({
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

void Terrain::init(VulkanContext &vkc) {
    using namespace std::chrono;
    auto start = steady_clock::now();
    auto end = steady_clock::now();
    this->stopThread = false;
    this->chunkQueueProcessor = std::thread(processChunks, this, &vkc);
    std::println("Initted Terrain in {} milliseconds", duration<float, std::milli>(end - start).count());
}

void Terrain::dirtyChunk(std::shared_ptr<Chunk> chunk, QueueOp queueOp) {
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
    while (true) {
        std::unique_lock lock(terrain->queueMtx);
        // std::println("Queue Count: {}", terrain->chunkMeshQueue.size());
        terrain->queueCond.wait(lock, [terrain]() {
            return not terrain->chunkMeshQueue.empty() || terrain->stopThread;
        });
        if (terrain->stopThread) {
            break;
        }
        std::shared_ptr<Chunk> chunk = terrain->chunkMeshQueue.front();
        terrain->chunkMeshQueue.pop_front();
        lock.unlock();
        if (chunk->data == nullptr) {
            assert(chunk.use_count() == 1);
            continue;
        }

        static size_t noop_count = 0;
        switch (chunk->queueOp) {
            case NONE: std::println(stderr, "Queue processing NONE op");
            case FILL: chunk->fillChunk();
            case UPDATE_MESH: terrain->writeMesh(*vkc, chunk->chunkCoord);
        }
        // std::println("X,0,0: {:b}, 0,Y,0: {:b}, 0,0,Z: {:b}", chunk->opaquenessMask_X[1][0], chunk->opaquenessMask_Y[1][0], chunk->opaquenessMask_Z[1][0]);
        chunk->queueOp = QueueOp::NONE;
        chunk->inQueue = false;
    }
}

void Terrain::loadChunksAround(VulkanContext &vkc, const glm::ivec3 centerChunkCoord) {
    static glm::ivec3 lastCenter = glm::ivec3{INT_MAX};
    if (centerChunkCoord == lastCenter) return;
    else lastCenter = centerChunkCoord;
    // std::cout << "Loading chunks around " << centerChunkCoord << std::endl;

    constexpr int RENDER_DIST = 10;
    for (int y = 0; y <= RENDER_DIST; y++)
    for (int z = 0; z <= RENDER_DIST - y; z++)
    for (int x = 0; x <= RENDER_DIST - y - z; x++) {
        loadChunk(vkc, centerChunkCoord + glm::ivec3(+x, +y, +z));
        loadChunk(vkc, centerChunkCoord + glm::ivec3(-x, +y, +z));
        loadChunk(vkc, centerChunkCoord + glm::ivec3(+x, +y, -z));
        loadChunk(vkc, centerChunkCoord + glm::ivec3(-x, +y, -z));
        loadChunk(vkc, centerChunkCoord + glm::ivec3(+x, -y, +z));
        loadChunk(vkc, centerChunkCoord + glm::ivec3(-x, -y, +z));
        loadChunk(vkc, centerChunkCoord + glm::ivec3(+x, -y, -z));
        loadChunk(vkc, centerChunkCoord + glm::ivec3(-x, -y, -z));
    };
    auto it = chunks.begin(); 
    while (it != chunks.end()) {
        if (glm::compAdd(glm::abs(it->first - centerChunkCoord)) > RENDER_DIST + 2) {
            // unloadChunk(vkc, it->first);
            it->second->deinit(vkc);
            it = chunks.erase(it);
        } else {
            it++;
        }
    }
}

void Terrain::loadChunk(const VulkanContext &vkc, const glm::ivec3 chunkCoord) {
    if (chunkCoord.y < 0 || chunkCoord.y > 1 || chunks.contains(chunkCoord)) return;

    chunks[chunkCoord] = std::make_shared<Chunk>();
    auto chunk = chunks[chunkCoord];
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
    std::println("Unloading [{}, {}, {}]", chunkCoord.x, chunkCoord.y, chunkCoord.z);
    assert(chunks.contains(chunkCoord));
    auto chunk = chunks.at(chunkCoord);
    chunk->deinit(vkc);
    this->chunks.erase(chunkCoord);
}

void Terrain::deinit(VulkanContext &vkc) {
    this->stopThread = true;
    this->queueCond.notify_one();
    this->chunkQueueProcessor.join();
    for (auto& [_, chunk] : chunks) {
        chunk->deinit(vkc);
    }
    this->chunks.clear();
    while (not chunkMeshQueue.empty()) {
        chunkMeshQueue.pop_front();
    }
}

