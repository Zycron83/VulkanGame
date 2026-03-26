#include <algorithm>
#include <format>
#include <memory>
#include <optional>
#include <print>
#include <mutex>
#include <thread>
#include <version>
#include <array>

#include "Context.h"
#include "SimplexNoise.h"

#include "Vertex.h"
#include "Terrain.h"
#include "Debug.h"
#include "Renderer.h"
#include "Util.hpp"

#define LOG if (0)

extern DebugFrameStats g_DebugFrameStats;
extern Settings g_Settings;
extern Renderer *g_Renderer;

constexpr int RENDER_DIST = 10;

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

Chunk::Chunk(ChunkCoord coord) : chunkCoord(std::move(coord)), data(std::make_unique<typeof(*data)>()) {
    data->fill(Block::Air);
}
Chunk::~Chunk() {
    if (!(data == nullptr && vertexCount == 0)) {
        std::println("{} was not deinitted before destruction", *this);
    };
}

void Chunk::deinit(VulkanContext &vkc) {
    // std::println("Deinitting Chunk {}", *this);
    if (vertexCount > 0 || indexCount > 0) {
        vkc.queueDestroy(vertexBuffer);
        vkc.queueDestroy(indexBuffer);
        g_Settings.chunkBytes -= this->vertexBuffer.size + this->indexBuffer.size;
        this->vertexCount = 0;
        this->indexCount = 0;
        this->filled = this->meshed = false;
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
void Chunk::setBlock(Block b, InnerCoord coord) {
    (*data)[coord.x * CHUNK_LENGTH * CHUNK_LENGTH + coord.y * CHUNK_LENGTH + coord.z] = b;
}

InnerCoord Chunk::coordFromIndex(int i) {
    // invert the mapping from index -> x,y,z used by get/set:
    // i = x * (CHUNK_LENGTH * CHUNK_LENGTH) + y * CHUNK_LENGTH + z
    InnerCoord ret; 
    ret.z = i % CHUNK_LENGTH;
    int tmp = i / CHUNK_LENGTH;
    ret.y = tmp % CHUNK_LENGTH;
    ret.x = tmp / CHUNK_LENGTH;
    return ret;
}

void Chunk::fillChunk() {
    // Timer timer;
    // timer.start();
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

    // LOG std::println("Filling {}", *this);

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
                // simpleSetBlock(Block::Air, i, k, j);
            }
        }
    };
    // auto time = timer.stop<microseconds>();
    // std::println("Filling took {} us", time);
    filled = true;
}

constexpr glm::ivec3 unitDir[6] = {
    {1,0,0}, {-1,0,0},
    {0,1,0}, {0,-1,0},
    {0,0,1}, {0,0,-1},
};
const Dir opposite[6] = { X_NEG, X_POS, Y_NEG, Y_POS, Z_NEG, Z_POS };
const Dir dirs[6] = { X_POS, X_NEG, Y_POS, Y_NEG, Z_POS, Z_NEG };
const char *dirName[6] = { "X_POS", "X_NEG", "Y_POS", "Y_NEG", "Z_POS", "Z_NEG" };

thread_local Vertex *vertexData;
thread_local Index *indexData;

// Must own.
void Chunk::writeMesh(VulkanContext &vkc) {

    // meshed = false;
    LOG std::println("Meshing {}", *this);

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

    Timer timer;
    timer.start();
    // std::println("Checking {} neighbour of [{}, {}, {}]. Is {}", dirName[Y_POS], chunkCoord.x, chunkCoord.y, chunkCoord.z, (void*)neighbourChunks[Y_POS]);

    uint32_t posMask, negMask;

    for (int x = 0; x < CHUNK_LENGTH; x += 1) {
        // auto upMask = chunk->borderMasks[Y_POS][x];
        // auto downMask = chunk->borderMasks[Y_NEG][x];
        
        posMask = neighbourChunks[Y_POS] ? neighbourChunks[Y_POS]->opaquenessMask_Z[               0][x] : 0;
        negMask = neighbourChunks[Y_NEG] ? neighbourChunks[Y_NEG]->opaquenessMask_Z[CHUNK_LENGTH - 1][x] : 0;
        for (int z = 0; z < CHUNK_LENGTH; z += 1) {
            // up <<|>> down
            uint64_t bits = opaquenessMask_Y[x][z];
            if (bits == 0) {
                negMask >>= 1;
                posMask >>= 1;
                continue;
            }
            bool curr = false;
            bool prev = negMask & 1;
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
            curr = posMask & 1;
            if (curr - prev == -1) writeFace(Y_POS, {x, CHUNK_LENGTH-1, z});
            negMask >>= 1;
            posMask >>= 1;
        }
    }

    for (int x = 0; x < CHUNK_LENGTH; x += 1) {

        posMask = neighbourChunks[Z_POS] ? neighbourChunks[Z_POS]->opaquenessMask_Y[x][               0] : 0;
        negMask = neighbourChunks[Z_NEG] ? neighbourChunks[Z_NEG]->opaquenessMask_Y[x][CHUNK_LENGTH - 1] : 0;
        for (int y = 0; y < CHUNK_LENGTH; y += 1) {
            uint64_t bits = opaquenessMask_Z[y][x];
            if (bits == 0) {
                negMask >>= 1;
                posMask >>= 1;
                continue;
            }
            bool curr = false;
            bool prev = negMask & 1;
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
            curr = posMask & 1;
            if (curr - prev == -1) writeFace(Z_POS, {x, y, CHUNK_LENGTH-1});
            negMask >>= 1;
            posMask >>= 1;
        }
    }

    for (int z = 0; z < CHUNK_LENGTH; z += 1) {

        posMask = neighbourChunks[X_POS] ? neighbourChunks[X_POS]->opaquenessMask_Y[               0][z] : 0;
        negMask = neighbourChunks[X_NEG] ? neighbourChunks[X_NEG]->opaquenessMask_Y[CHUNK_LENGTH - 1][z] : 0;
        for (int y = 0; y < CHUNK_LENGTH; y += 1) {
            uint64_t bits = opaquenessMask_X[z][y];
            if (bits == 0) {
                negMask >>= 1;
                posMask >>= 1;
                continue;
            }
            bool curr = false;
            bool prev = negMask & 1;
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
            curr = posMask & 1;
            if (curr - prev == -1) writeFace(X_POS, {CHUNK_LENGTH-1, y, z});
            negMask >>= 1;
            posMask >>= 1;
        }
    }

    // auto time = timer.stop<microseconds>();
    // std::println("Meshing [{}, {}, {}] took {} us", chunkCoord.x, chunkCoord.y, chunkCoord.z, time);

    if (vertexOffset == 0) {
        assert(indexOffset == 0);
        // empty chunk
        vertexCount = 0;
        indexCount = 0;
        return;
    }

    if (vertexBuffer.buffer == nullptr || sizeof(Vertex) * vertexOffset > vertexBuffer.size) {
        vertexBuffer.initDevice(std::format("Chunk {} Vertex Buffer", *this), vkc,
            sizeof(Vertex) * vertexOffset,
            vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst
        );
        
        indexBuffer.initDevice(std::format("Chunk {} Index Buffer", *this), vkc,
            sizeof(Index) * indexOffset,
            vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst
        );
        g_Settings.chunkBytes += vertexBuffer.size + indexBuffer.size;
    }
    
    vkc.uploadBuffers({
        { indexBuffer, indexData },
        { vertexBuffer, vertexData }
    });
    vertexCount = vertexOffset;
    indexCount = indexOffset;

    meshed = true;
}

// Chunk processing thread
void chunkThreadFunction(VulkanContext &vkc, ChunkThread &ct) {
    LOG std::println("Chunk Mesher reporting from thread {}! o7", std::this_thread::get_id());
    constexpr int voxelCount = CHUNK_LENGTH * CHUNK_LENGTH * CHUNK_LENGTH;
    vertexData = new Vertex[voxelCount * vertices.size()];
    indexData = new Index[voxelCount * indices.size()];
    std::unique_lock queueLock{ct.queueMtx};
    while (true) {
        // std::println("Queue Count: {}", terrain->chunkMeshQueue.size());
        ct.queueCond.wait(queueLock, [&ct]() {
            return !ct.queue.empty() || ct.stopThread || ct.updateCenter;
        });
        if (ct.stopThread) break;

        if (ct.updateCenter.exchange(false)) {
            ct.loadAround(vkc, ct.centerChunkCoord);
        }

        if (ct.queue.empty()) {
            LOG std::println("Chunk queue is somehow empty");
            continue;
        }
        auto [action, chunkCoord] = ct.queue.front();
        ct.queue.pop_front();
        {
            // std::unique_lock chunksLock{ct.chunksMtx};
            if (!ct.chunks.contains(chunkCoord)) {
                LOG std::println("Skipping deleted chunk");
                continue;
            }
            
            auto &chunk = ct.chunks.at(chunkCoord);
            using Action = ChunkThread::Action;
            switch (action) {
                case Action::FILL:
                    chunk->fillChunk();

                    for (Dir dir : dirs) {
                        ChunkCoord neighbourCoord = chunkCoord + unitDir[dir];
                        if (neighbourCoord.y > CHUNK_TOP || neighbourCoord.y < CHUNK_BOTTOM) continue;
                        if (auto it = ct.chunks.find(neighbourCoord); it != ct.chunks.end()) {
                            auto &neighbour = it->second;
                            chunk->neighbourChunks[dir] = neighbour.get();
                            neighbour->neighbourChunks[opposite[dir]] = chunk.get();
                            if (std::ranges::find_if(ct.queue, [&neighbourCoord](auto &pair) { return neighbourCoord == pair.second; }) == ct.queue.end()) {
                                ct.mesh(neighbourCoord);
                            }
                        };
                    }
                    ct.mesh(chunkCoord);
                    // std::println("FILL");
                    // ct.spread(chunkCoord);
                    break;
                case Action::MESH:
                    chunk->writeMesh(vkc);
                    // std::println("MESH");
                    break; 
            }
        }
        
        // std::println("X,0,0: {:b}, 0,Y,0: {:b}, 0,0,Z: {:b}", chunk->opaquenessMask_X[1][0], chunk->opaquenessMask_Y[1][0], chunk->opaquenessMask_Z[1][0]);
    }
    delete[] vertexData;
    delete[] indexData;
}

Terrain::Terrain(VulkanContext &vkc) : vkc(vkc) {
    this->chunkThread.thread = std::thread{chunkThreadFunction, std::ref(vkc), std::ref(chunkThread)};
}

// MUST LOCK queue
void ChunkThread::loadAround(VulkanContext &vkc, const ChunkCoord center) {
    LOG std::println("Loading chunks around [{}, {}, {}]", centerChunkCoord.x, centerChunkCoord.y, centerChunkCoord.z);

    const auto L = [](int x, int z) {
        // return abs(x) + abs(z); // diamond
        // return sqrt(x*x + z*z); // circle
        return std::max(std::abs(x), std::abs(z)); // square
    };

    std::vector<Chunk *> deleted;
    { // Create chunks
        std::scoped_lock chunksLock{chunksMtx};
        for (int x = -RENDER_DIST; x <= RENDER_DIST; x++) {
        for (int y = CHUNK_BOTTOM; y <= CHUNK_TOP  ; y++) {
        for (int z = -RENDER_DIST; z <= RENDER_DIST; z++) {
            if (L(x, z) <= RENDER_DIST) {
                ChunkCoord coord = center + glm::ivec3(x, y - center.y, z);
                if (!chunks.contains(coord)) {
                    LOG std::println("Creating [{}, {}, {}]", coord.x, coord.y, coord.z);
                    {
                        
                        chunks.insert(std::make_pair(coord, std::make_unique<Chunk>(coord)));
                    }
                    fill(coord);
                }
            }
        }}};

     // Delete chunks
        auto it = chunks.begin();
        while (it != chunks.end()) {
            
            const ChunkCoord offset = it->first - centerChunkCoord;
            // LOG std::println("Checking offset {}, from chunk {} at center {}", offset, it->first, centerChunkCoord);
            if (L(offset.x, offset.z) > RENDER_DIST + 2) {
                LOG std::println("Removing {}", it->first);
                auto &chunk = it->second;
                
                for (int d = 0; d < 6; d++) {
                    auto neighbourChunk = chunk->neighbourChunks[d];
                    if (neighbourChunk && !std::ranges::contains(deleted, neighbourChunk)) {
                        neighbourChunk->neighbourChunks[opposite[d]] = nullptr;
                    };
                }
                
                chunk->deinit(vkc);
                deleted.push_back(chunk.get());
                it = chunks.erase(it);
            } else {
                it++;
            }
        }
    }
    deleted.clear();
}

// MUST LOCK chunks
std::optional<Block> Terrain::getBlock(GlobalCoord gCoord) const noexcept {
    if (!chunkThread.chunks.contains(gCoord.chunk)) return std::nullopt;
    return chunkThread.chunks.at(gCoord.chunk)->getBlock(gCoord.inner);
}

bool Terrain::placeBlock(const Camera &camera) noexcept {
    return false;
}
bool Terrain::breakBlock(const Camera &camera) noexcept {
    const auto ray = [&camera](float t) {
        return camera.position + t * camera.front;
    };

    glm::ivec3 step = glm::sign(camera.front);
    

    float t;

    return false;
}

void Terrain::tickFrame(const Camera &camera) noexcept {
    const ChunkCoord currentCenterChunkCoord = camera.position / static_cast<float>(CHUNK_LENGTH);
    if (currentCenterChunkCoord != chunkThread.centerChunkCoord) {
        chunkThread.centerChunkCoord = currentCenterChunkCoord;
        chunkThread.updateCenter = true;
        chunkThread.queueCond.notify_one();
    }

    // {
    //     std::scoped_lock outputLock{threadPool.fillOutputMtx};
    //     auto &queue = threadPool.fillOutput;
    //     for (auto it = queue.begin(); it != queue.end(); it++) {
    //         auto &chunk = *it;
    //         // std::println("Moving [{}, {}, {}]", chunk->chunkCoord.x, chunk->chunkCoord.y, chunk->chunkCoord.z);
    //         for (Dir dir : {Y_POS, Y_NEG}) {
    //             // TODO... skip top and bottom chunk calcs
    //             glm::ivec3 neighbourCoord = chunk->chunkCoord + unitDir[dir];
    //             Chunk *neighbour;
    //             if (chunks.contains(neighbourCoord) && chunks.at(neighbourCoord).has_value()) {
    //                 neighbour = chunks.at(neighbourCoord)->get();
    //                 std::println("[chunks] Setting neighbour for {} and {}", *chunk, *neighbour);
    //                 chunk->neighbourChunks[dir] = neighbour;
    //                 neighbour->neighbourChunks[opposite[dir]] = chunk.get();
    //             }
    //             auto neighbourIt = std::find_if(it, queue.end(), [neighbourCoord](auto &chunk) { 
    //                 return chunk->chunkCoord == neighbourCoord; 
    //             });
    //             if (neighbourIt != queue.end()) {
    //                 neighbour = neighbourIt->get();
    //                 std::println("[fillOutput] Setting neighbour for {} and {}", *chunk, *neighbour);
    //                 chunk->neighbourChunks[dir] = neighbour;
    //                 neighbour->neighbourChunks[opposite[dir]] = chunk.get();
    //             }
    //         }
    //         threadPool.submit(std::move(chunk), ThreadPool::MESH);
    //     }
    //     threadPool.fillOutput.clear();
    // }
}

void Terrain::draw(vk::CommandBuffer commandBuffer) {
    std::scoped_lock chunksLock{chunkThread.chunksMtx};
    for (auto& [_, chunk_ptr] : chunkThread.chunks) {
        auto &chunk = *chunk_ptr;
        if (!chunk.meshed) continue;
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

Terrain::~Terrain() {
    auto &ct = chunkThread;
    ct.stopThread = true;
    ct.queueCond.notify_one();
    auto id = ct.thread.get_id();
    ct.thread.join();
    std::println("Chunk thread {} joined back! o7", id);
    ct.queue.clear();
    for (auto& [_, chunk] : ct.chunks) {
        chunk->deinit(vkc);
    }
    ct.chunks.clear();
}

