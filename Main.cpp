#include "vulkan/vulkan.hpp"
#include <vulkan/vulkan_to_string.hpp>
#include <GLFW/glfw3.h>

#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/io.hpp>
#include <glm/trigonometric.hpp>

#define IM_VEC2_CLASS_EXTRA \
    ImVec2(const glm::vec2& v) : x(v.x), y(v.y) {}
#ifdef IMGUI_ENABLE
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"
#endif

#include <iostream>
#include <print>
#include <chrono>
#include <thread>

// #include <sanitizer/lsan_interface.h>

#include "Renderer.h"
#include "Debug.h"
#include "Util.hpp"
#include "Terrain.h"

extern Settings g_Settings;
Settings::NoiseSettings prev_NoiseSettings = g_Settings.Noise;
extern DebugFrameStats g_DebugFrameStats;

constexpr int FPS_LIMIT = 165; // frame/s
constexpr steady_clock::rep FRAMETIME_LIMIT = (steady_clock::rep)1000 / FPS_LIMIT; // ms/frame

struct {
    int width;
    int height;
} windowExtent = {800 * 2, 600 * 2};

static GLFWwindow* window = NULL;
Renderer* g_Renderer = NULL;

static bool first_mouse_move = true;
static float speed = 1.f; // 1.0 block/s * [s/frame]= block/frame
static double sensitivity = .1f;

inline static bool pressed(int key) {
    return glfwGetKey(window, key) == GLFW_PRESS;
}
void processInput() {
    auto c = g_Renderer->camera.get();
    const float mult = speed / FPS_LIMIT;
    if (pressed(GLFW_KEY_W))
        c->position += c->front * mult;
    if (pressed(GLFW_KEY_S))
        c->position -= c->front * mult;
    if (pressed(GLFW_KEY_D))
        c->position += glm::cross(c->front, c->up) * mult;
    if (pressed(GLFW_KEY_A))
        c->position -= glm::cross(c->front, c->up) * mult;
    // if (pressed(GLFW_KEY_J)) c->position.x -= speed;
    // if (pressed(GLFW_KEY_I)) c->position.x += speed;
    // if (pressed(GLFW_KEY_K)) c->position.y -= speed;
    // if (pressed(GLFW_KEY_O)) c->position.y += speed;
    // if (pressed(GLFW_KEY_L)) c->position.z -= speed;
    // if (pressed(GLFW_KEY_P)) c->position.z += speed;
}

void MouseScrollEvent(GLFWwindow* window, double xoffset, double yoffset) {
    // Speed adjustment
    speed += yoffset * (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ? 10.f : 1.f);
    if (speed < 1.f) speed = 1.f;
    
}

void KeyEvent(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        switch (key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;
            case GLFW_KEY_Q:
                if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED) {
                    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                } else {
                    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                    first_mouse_move = true;
                }
                break;
            case GLFW_KEY_SEMICOLON:
                if (g_Renderer) g_Renderer->terrain.setBlock(GlobalCoord(g_Renderer->camera->position), Block::Stone);
            case GLFW_KEY_APOSTROPHE:
        }
    }
}
void MousePosEvent(GLFWwindow* window, double xpos, double ypos) {
    if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_NORMAL) return;
    auto c = g_Renderer->camera.get();

    static double last_x = 0, last_y = 0;

    double offset_x = xpos - last_x;
    double offset_y = last_y - ypos;
    last_x = xpos;
    last_y = ypos;
    if (first_mouse_move) {
        [[unlikely]]
        first_mouse_move = false;
        return;
    }
    c->rotXY(offset_x * sensitivity, offset_y * sensitivity);
    
}

void MouseButtonEvent(GLFWwindow* window, int button, int action, int mods) {
    if (action == GLFW_PRESS) {
        auto gcOpt = g_Renderer->terrain.voxelRaycast(*g_Renderer->camera);
        if (!gcOpt) return;
        auto &[gc, optFace] = gcOpt.value();
        if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            if (!optFace) return; 
            gc += *optFace;
            g_Renderer->terrain.setBlock(gc, Block::Stone);
        }
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            g_Renderer->terrain.setBlock(gc, Block::Air);
        }
    }
}
void ErrorEvent(int code, const char* what)
{
    std::cerr << "Couldn't create window: " << code << "::" << what << std::endl;
}

void cleanup() {
    delete g_Renderer;
    g_Renderer = NULL;
    #ifdef IMGUI_ENABLE
    ImGui_ImplGlfw_Shutdown();
    #endif
}

int main(int argc, char* argv[])
{
    glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_WAYLAND);
    glfwInitHint(GLFW_WAYLAND_LIBDECOR, GLFW_FALSE);
    glfwInit();

    glfwSetErrorCallback(ErrorEvent);

    #ifdef IMGUI_ENABLE
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    #endif

    /* Create the window */
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_DECORATED, GLFW_TRUE);
    window = glfwCreateWindow(windowExtent.width, windowExtent.height, "Hello World", nullptr, nullptr);
    if (!window) {
        const char * what;
        int code = glfwGetError(&what);
        std::cerr << "Couldn't create window: " << code << "::" << what << std::endl;
        return 1;
    }

    glfwSetKeyCallback(window, KeyEvent);
    glfwSetCursorPosCallback(window, MousePosEvent);
    glfwSetMouseButtonCallback(window, MouseButtonEvent);
    glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    glfwSetScrollCallback(window, MouseScrollEvent);
    const char *description;
    glfwGetError(&description);
    if (description)std::println("GLFW Error: {}", description);
    #ifdef IMGUI_ENABLE
    ImGui_ImplGlfw_InitForVulkan(window, true); 
    #endif

    // auto start = std::chrono::steady_clock::now();
    // std::array<double, 20> frametimes = {0};
    // int frame_i = 0;
    AverageTimer<20, microseconds> timer;
    timer.start();
    
    try {
        g_Renderer = new Renderer(window);
        g_Renderer->camera = std::make_unique<Camera>(Camera{
            glm::vec3(0,0,0),
            45.0f,
            0.0f,
        });

        #ifdef IMGUI_ENABLE
        g_Renderer->initImGui();
        #endif

        size_t frameCount{0};

        while (!glfwWindowShouldClose(window)) {
            frameCount += 1;
            // Cap framerate
            // auto now = std::chrono::steady_clock::now();
            // auto frame_time = now - start;
            // frametimes[frame_i] = frame_time.count() / 1e6;
            // frame_i = (frame_i + 1) % frametimes.size();
            // if (frame_time < std::chrono::milliseconds((int)FRAMETIME_LIMIT)) {
            //     std::this_thread::sleep_for(std::chrono::milliseconds((int)FRAMETIME_LIMIT) - frame_time);
            // }
            // start = std::chrono::steady_clock::now();
            auto time = timer.stop();
            constexpr steady_clock::rep FRAMETIME_LIMIT = (steady_clock::rep)1000 / FPS_LIMIT;
            if (time < FRAMETIME_LIMIT) {
                std::this_thread::sleep_for(milliseconds(FRAMETIME_LIMIT - time));
            }
            timer.start();

            int w, h;
            glfwGetFramebufferSize(window, &w, &h);
            
            #ifdef IMGUI_ENABLE
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            
            {
                using namespace ImGui;
                Begin("Camera");
                auto c = g_Renderer->camera.get();
                Camera cursorCam{-c->front, c->front};
                const glm::vec3 s_pos = cursorCam.screenPos(w, h, glm::vec3(0));
                SliderFloat("FOV", &c->fov, 0, 360);
                InputFloat2("xy Angles", &c->x_angle);
                InputFloat3("Position", (float*)&c->position);
                InputFloat3("Front", (float*)&c->front);
                InputFloat3("S_POS", (float*)&s_pos);
                End();

                Begin("Info");
                Text("%zu", frameCount);
                Text("Frame Time: %.2f ms", timer.get() / 1000.0);
                Text("Triangles: %zu", g_DebugFrameStats.index_count / 3);
                Text("Chunk Memory: %.2lu MB", g_Settings.chunkBytes.load() / 1024 / 1024);
                Text("Chunks in Fill Queue: %zu", g_Renderer->terrain.ct.queue.size());
                End();

                // Begin("Noise Settings");
                // InputInt("octaves", &g_Settings.Noise.octaves);
                // InputFloat("frequency", &g_Settings.Noise.frequency);
                // InputFloat("amplitude", &g_Settings.Noise.amplitude);
                // InputFloat("lacunarity", &g_Settings.Noise.lacunarity);
                // InputFloat("persistence", &g_Settings.Noise.persistence);
                // InputFloat("scale", &g_Settings.Noise.scale);
                // End();
                // auto perspective = glm::ortho(-2.0f, 2.0f, -2.0f, 2.0f);

                // perspective[1][1] *= -1;
                // auto view = glm::lookAt(-c->front, glm::vec3{0}, glm::vec3(0, 1, 0));
                
                const float aspect = (float)h / w;
                constexpr float A = 30.f;
                const auto ortho = glm::ortho<float>(-A, A, aspect * -A, aspect * A);
                const auto view = glm::lookAt(-c->front * 100.0f, glm::vec3{0}, glm::vec3{0,1,0});
                auto screenPos = [&, h, w](glm::vec3 coord) {
                    auto clipSpace = ortho * view * glm::vec4{coord, 1.0f};
                    if (clipSpace.w != 0) clipSpace /= clipSpace.w;
                    auto screenSpace = glm::vec3{(clipSpace.x + 1.0f) * 0.5f * w, (1.0f - clipSpace.y) * 0.5f * h, clipSpace.z};
                    return screenSpace;
                };
                const auto mid = glm::vec2{screenPos(glm::vec3{0})};
                auto draw = ImGui::GetForegroundDrawList();
                draw->AddLine(
                    mid,
                    glm::vec2{screenPos(glm::vec3{1,0,0})}, 
                    IM_COL32(255,0,0,255), 
                    2.0f
                );
                draw->AddLine(
                    mid,
                    glm::vec2{screenPos(glm::vec3{0,1,0})}, 
                    IM_COL32(0,255,0,255), 
                    2.0f
                );
                draw->AddLine(
                    mid,
                    glm::vec2{screenPos(glm::vec3{0,0,1})}, 
                    IM_COL32(0,0,255,255), 
                    2.0f
                );
            }
            
            
            ImGui::Render();
            #endif
            if (g_Settings.Noise != prev_NoiseSettings) {
                prev_NoiseSettings = g_Settings.Noise;
                g_Renderer->terrain.~Terrain();
                new(&g_Renderer->terrain) Terrain(g_Renderer->vkc);
            }
            g_DebugFrameStats.index_count = 0;
            g_Renderer->terrain.tickFrame(*g_Renderer->camera.get());
            g_Renderer->vkc.waitForTransfers();
            g_Renderer->drawFrame();

            
            glfwPollEvents();
            glfwGetFramebufferSize(window, &w, &h);
            if (w != windowExtent.width || h != windowExtent.height) {
                windowExtent = {w, h};
                g_Renderer->resize();
            }
            processInput();
        }
    }
    catch (vk::SystemError &expt) {
        std::cerr << "[Vulkan System Error!] :: " << expt.what() << " | " << vk::to_string(vk::Result(expt.code().value())) << std::endl;
        cleanup();
        return 1;
    }
    catch (std::runtime_error &expt) {
        std::cerr << "[Runtime Error!] :: " << expt.what() << std::endl;
        cleanup();
        return 1;
    }
    catch (std::logic_error &expt) {
        std::cerr << "[Logic Error!] :: " << expt.what() << std::endl;
        cleanup();
        return 1;
    }
    catch (...) {
        std::cerr << "[THROW]" << std::endl;
        cleanup();
        return 1;
    }

    cleanup();
    return 0;
}
