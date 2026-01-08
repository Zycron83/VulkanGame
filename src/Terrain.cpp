#include <print>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

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
    //     sizeof(vertices) * CHUNK_WIDTH * CHUNK_WIDTH * CHUNK_HEIGHT / 4, 
    //     vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst
    // );
    // g_DebugNameState.Name("Index");
    // indexBuffer.initDevice(vkc, 
    //     sizeof(indices) * CHUNK_WIDTH * CHUNK_WIDTH * CHUNK_HEIGHT / 4, 
    //     vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst
    // );
    // g_DebugNameState.Pop();
}
void Chunk::deinit(VulkanContext &vkc) {
    auto &queue = vkc.frames[vkc.currentFrame].destruction_queue;
    queue.push_back(vertexBuffer);
    queue.push_back(indexBuffer);
    g_Settings.chunkBytes -= this->vertexBuffer.size + this->indexBuffer.size;
    this->vertexCount = 0;
    this->indexCount = 0;
}

Block Chunk::getBlock(int x, int y, int z) const {
    return data[x * CHUNK_HEIGHT * CHUNK_WIDTH + y * CHUNK_WIDTH + z];
}
Block Chunk::getBlock(glm::ivec3 coord) const {
    if (glm::any(glm::greaterThanEqual(coord, {CHUNK_WIDTH, CHUNK_HEIGHT, CHUNK_WIDTH}))
     || glm::any(glm::lessThan(coord, {0,0,0}))
    ) {
        return Block::Invalid;
    }
    return data[coord.x * CHUNK_HEIGHT * CHUNK_WIDTH + coord.y * CHUNK_WIDTH + coord.z];
}

void Chunk::setBlock(Block b, int x, int y, int z) {
    assert(x < CHUNK_WIDTH && y < CHUNK_HEIGHT && z < CHUNK_WIDTH);
    assert(x >= 0 && y >= 0 && z >= 0);
    data[x * CHUNK_HEIGHT * CHUNK_WIDTH + y * CHUNK_WIDTH + z] = b;
    if (transparent(b)) { // set 0
        opaquenessMask_Y[x][z] &= ~(1 << y);
        opaquenessMask_Z[x][y] &= ~(1 << z);
        opaquenessMask_X[z][y] &= ~(1 << x);
    } else { // set 1
        opaquenessMask_Y[x][z] |= (1 << y);
        opaquenessMask_Z[x][y] |= (1 << z);
        opaquenessMask_X[z][y] |= (1 << x);
    }
}
void Chunk::setBlock(Block b, glm::ivec3 coord) {
    data[coord.x * CHUNK_HEIGHT * CHUNK_WIDTH + coord.y * CHUNK_WIDTH + coord.z] = b;
}

glm::ivec3 Chunk::coordFromIndex(int i) {
    // invert the mapping from index -> x,y,z used by get/set:
    // i = x * (CHUNK_HEIGHT * CHUNK_WIDTH) + y * CHUNK_WIDTH + z
    glm::ivec3 ret; 
    ret.z = i % CHUNK_WIDTH;
    int tmp = i / CHUNK_WIDTH;
    ret.y = tmp % CHUNK_HEIGHT;
    ret.x = tmp / CHUNK_HEIGHT;
    return ret;
}

void Chunk::fillChunk() {
    // std::println("Filling: {}", std::chrono::steady_clock::now().time_since_epoch());
    // auto simplex = SimplexNoise(g_Settings.Noise.frequency, g_Settings.Noise.amplitude, g_Settings.Noise.lacunarity, g_Settings.Noise.persistence);
    constexpr float scale = 0.001f;
    for (int i = 0; i < CHUNK_WIDTH; i += 1)
    for (int j = 0; j < CHUNK_WIDTH; j += 1) {
        // const float height = simplex.fractal(g_Settings.Noise.octaves, (i + chunkCoord.x * CHUNK_WIDTH) * scale, (j + chunkCoord.z * CHUNK_WIDTH) * scale) * CHUNK_HEIGHT;
        const float height = SimplexNoise::noise((i + chunkCoord.x * CHUNK_WIDTH) * scale, (j + chunkCoord.z * CHUNK_WIDTH) * scale) * CHUNK_HEIGHT + 15;
        // const float height = 10;
        for (int k = 0; k < CHUNK_HEIGHT; k += 1) {
            if (k < height) {
                setBlock(Block::Stone, i, k, j);
            } else {
                setBlock(Block::Air, i, k, j);
            }
        }
    }
}

enum Dir {
    X_POS, X_NEG,
    Y_POS, Y_NEG,
    Z_POS, Z_NEG
};
glm::ivec3 unitDir[6] = {
    {1,0,0}, {-1,0,0},
    {0,1,0}, {0,-1,0},
    {0,0,1}, {0,0,-1},
};

void Chunk::writeMesh(VulkanContext &vkc) {
    constexpr int voxelCount = CHUNK_WIDTH * CHUNK_WIDTH * CHUNK_HEIGHT;
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
            v.position += coord + this->chunkCoord * glm::ivec3(CHUNK_WIDTH, 0, CHUNK_WIDTH);
            vertexData[vertexOffset + j] = v;
        }
        for (int j = 0; j < 6; j += 1) {
            indexData[indexOffset + j] = Index(indices[j].val + vertexOffset);
        }
        vertexOffset += 4;
        indexOffset += 6;
    };

    if (this->chunkCoord.x % 2 == 0) {
        for (int x = 0; x < CHUNK_WIDTH ; x += 1) {
            for (int z = 0; z < CHUNK_WIDTH ; z += 1) {
                uint64_t bits = opaquenessMask_Y[x][z];
                int y = 0;
                bool curr;
                while (y < CHUNK_HEIGHT) {
                    curr = bits & 1;
                    if (curr) {
                        for (int d = 0; d < 6; d += 1) {
                            writeFace(Dir(d), {x, y, z});
                        }
                    }
                    bits >>= 1;
                    y += 1;
                }
            }
        }
    } else {

    // auto start_time = std::chrono::steady_clock::now();
    for (int x = 0; x < CHUNK_WIDTH; x += 1) {
        for (int z = 0; z < CHUNK_WIDTH; z += 1) {
            uint64_t bits = opaquenessMask_Y[x][z];
            bool curr, prev = bits & 1;
            // bits >>= 1;
            int y = 0;
            while (y < CHUNK_HEIGHT) {
                curr = bits & 1;
                switch (curr - prev) {
                    case -1: writeFace(Y_POS, {x, y, z}); break;
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

    // for (int x = 0; x < CHUNK_WIDTH; x += 1) {
        for (int y = 0; y < CHUNK_HEIGHT; y += 1) {
            uint64_t bits = opaquenessMask_Z[x][y];
            bool curr, prev = bits & 1;
            // bits >>= 1;
            int z = 0;
            while (z < CHUNK_WIDTH) {
                curr = bits & 1;
                switch (curr - prev) {
                    case -1: writeFace(Z_POS, {x, y, z}); break;
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

    for (int z = 0; z < CHUNK_WIDTH; z += 1) {
        for (int y = 0; y < CHUNK_HEIGHT; y += 1) {
            uint64_t bits = opaquenessMask_X[z][y];
            bool curr, prev = bits & 1;
            // bits >>= 1;
            int x = 0;
            while (x < CHUNK_WIDTH) {
                curr = bits & 1;
                switch (curr - prev) {
                    case -1: writeFace(X_POS, {x, y, z}); break;
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

    }

    // auto end_time = std::chrono::steady_clock::now();
    // std::println("Passes took {} us", std::chrono::duration<float, std::micro>(end_time - start_time).count());

    // writeFace(Y_POS, {0, 0, 0});
    // writeFace(X_NEG, {0, 0, 0});
    // writeFace(Z_NEG, {0, 0, 0});
    // writeFace(Y_POS, {g_Settings.x.load(), g_Settings.y.load(), g_Settings.z.load()});
    // writeFace(X_NEG, {g_Settings.x.load(), g_Settings.y.load(), g_Settings.z.load()});
    // writeFace(Z_NEG, {g_Settings.x.load(), g_Settings.y.load(), g_Settings.z.load()});

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
    constexpr int N = 5;
    for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
        // std::println("Loading: {}", std::chrono::steady_clock::now().time_since_epoch());
        loadChunk(vkc, {i, 0, j});
    };
    auto end = steady_clock::now();
    this->stopThread = false;
    this->chunkQueueProcessor = std::thread(processChunks, this, const_cast<VulkanContext*>(&vkc));
    std::println("Initted Terrain in {} milliseconds", duration<float, std::milli>(end - start).count());
}

void Terrain::dirtyChunk(Chunk *chunk) {
    {
        std::scoped_lock lock(queueMtx);
        if (not chunk->inQueue) {
            chunkMeshQueue.push(chunk);
            chunk->inQueue = true;
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
        terrain->chunkMeshQueue.pop();
        lock.unlock();

        chunk->fillChunk();
        chunk->filled = true;
        chunk->writeMesh(*vkc);
        // std::println("X,0,0: {:b}, 0,Y,0: {:b}, 0,0,Z: {:b}", chunk->opaquenessMask_X[1][0], chunk->opaquenessMask_Y[1][0], chunk->opaquenessMask_Z[1][0]);
        chunk->inQueue = false;
    }
}

void Terrain::loadChunksAround(const VulkanContext &vkc, glm::vec3 center) {}

void Terrain::loadChunk(const VulkanContext &vkc, glm::ivec3 chunkCoord) {
    if (chunks.contains(chunkCoord)) return;
    

    chunks[chunkCoord] = std::make_unique<Chunk>();
    auto chunk = chunks[chunkCoord].get();
    chunk->chunkCoord = chunkCoord;
    chunk->init(vkc);

    dirtyChunk(chunk);
}

void Terrain::tickFrame(const VulkanContext &vkc) {

}

void Terrain::draw(vk::CommandBuffer commandBuffer) {
    for (auto& [_, chunk] : chunks) {
        if (not chunk->filled || chunk->vertexCount == 0 && chunk->indexCount == 0 || chunk->inQueue) continue;
        assert(chunk->vertexBuffer.size > 0 && chunk->indexBuffer.size > 0);
        commandBuffer.bindVertexBuffers(0, chunk->vertexBuffer.buffer, static_cast<vk::DeviceSize>(0));
        commandBuffer.bindIndexBuffer(chunk->indexBuffer.buffer, 0, Index::enumType);

        commandBuffer.drawIndexed(chunk->indexCount, 1, 0, 0, 0);
        g_DebugFrameStats.index_count += chunk->indexCount;
    }
}

void Terrain::unloadChunk(VulkanContext &vkc, Chunk &chunk) {
    chunk.deinit(vkc);
    this->chunks.erase(chunk.chunkCoord);
}

void Terrain::deinit(VulkanContext &vkc) {
    auto &queue = vkc.frames[vkc.currentFrame].destruction_queue;
    for (auto& [_, chunk] : chunks) {
        chunk->deinit(vkc);
    }
    this->chunks.clear();
    while (not chunkMeshQueue.empty()) {
        chunkMeshQueue.pop();
    }
    this->stopThread = true;
    this->queueCond.notify_one();
    this->chunkQueueProcessor.join();
}

