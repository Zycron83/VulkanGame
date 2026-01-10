#include "vulkan/vulkan.hpp"
#include <GLFW/glfw3.h>

#include <glm/gtx/io.hpp>
#include <glm/trigonometric.hpp>

#ifdef IMGUI_ENABLE
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"
#endif

#include <iostream>
#include <print>
#include <chrono>
#include <thread>
#include <numeric>

// #include <sanitizer/lsan_interface.h>

#include "Renderer.h"
#include "Debug.h"
#include "Util.hpp"

extern Settings g_Settings;
Settings::NoiseSettings prev_NoiseSettings = g_Settings.Noise;
extern DebugFrameStats g_DebugFrameStats;

#define FPS_LIMIT 60
#define FRAMETIME_LIMIT (1000 / FPS_LIMIT)

struct {
    int width;
    int height;
} windowExtent = {800 * 2, 600 * 2};

static GLFWwindow* window = NULL;
Renderer* g_Renderer = NULL;

static bool first_mouse_move = true;
static float speed = 1.f;
static double sensitivity = .1f;

inline static bool pressed(int key) {
    return glfwGetKey(window, key) == GLFW_PRESS;
}
void processInput() {
    auto c = g_Renderer->camera.get();
    if (pressed(GLFW_KEY_W))
        c->position += c->front * speed;
    if (pressed(GLFW_KEY_S))
        c->position -= c->front * speed;
    if (pressed(GLFW_KEY_D))
        c->position += glm::cross(c->front, c->up) * speed;
    if (pressed(GLFW_KEY_A))
        c->position -= glm::cross(c->front, c->up) * speed;
    // if (pressed(GLFW_KEY_J)) c->position.x -= speed;
    // if (pressed(GLFW_KEY_I)) c->position.x += speed;
    // if (pressed(GLFW_KEY_K)) c->position.y -= speed;
    // if (pressed(GLFW_KEY_O)) c->position.y += speed;
    // if (pressed(GLFW_KEY_L)) c->position.z -= speed;
    // if (pressed(GLFW_KEY_P)) c->position.z += speed;
}

void MouseScrollEvent(GLFWwindow* window, double xoffset, double yoffset) {
    // Speed adjustment
    speed += yoffset * 0.1f;
    if (speed < 0.1f) speed = 0.1f;
    
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
            case GLFW_KEY_J: g_Settings.x -= 1; break;
            case GLFW_KEY_I: g_Settings.x += 1; break;
            case GLFW_KEY_K: g_Settings.y -= 1; break;
            case GLFW_KEY_O: g_Settings.y += 1; break;
            case GLFW_KEY_L: g_Settings.z -= 1; break;
            case GLFW_KEY_P: g_Settings.z += 1; break;
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
    return; // TODO: Remove
    if (action == GLFW_PRESS) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
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
    glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
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
    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
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

    auto start = std::chrono::steady_clock::now();
    std::array<double, 20> frametimes = {0};
    int frame_i = 0;
    
    try {
        g_Renderer = new Renderer(window);
        g_Renderer->camera = std::make_unique<Camera>(Camera{
            glm::vec3(-5, 5, -5),
            45.0f,
            -30.0f,
        });

        #ifdef IMGUI_ENABLE
        g_Renderer->initImGui();
        #endif

        size_t frameCount{0};

        while (!glfwWindowShouldClose(window)) {
            frameCount += 1;
            // Cap framerate
            auto now = std::chrono::steady_clock::now();
            auto frame_time = now - start;
            frametimes[frame_i] = frame_time.count() / 1e6;
            frame_i = (frame_i + 1) % frametimes.size();
            constexpr auto target_frame_time = std::chrono::milliseconds(FRAMETIME_LIMIT);
            if (frame_time < target_frame_time) {
                std::this_thread::sleep_for(target_frame_time - frame_time);
            }
            start = std::chrono::steady_clock::now();

            int w, h;
            // glfwGetFramebufferSize(window, &w, &h);
            // const glm::vec3 s_pos = g_Renderer->camera->screenPos(w, h, glm::vec3(.5, .5, .5)) * glm::vec3(w, h, 1);
            
            #ifdef IMGUI_ENABLE
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            
            {
                using namespace ImGui;
                Begin("Camera");
                auto c = g_Renderer->camera.get();
                SliderFloat("FOV", &c->fov, 0, 360);
                InputFloat2("xy Angles", &c->x_angle);
                InputFloat3("Position", (float*)&c->position);
                // InputFloat3("S_POS", (float*)&s_pos);
                End();

                Begin("Info");
                Text("%zu", frameCount);
                Text("Frame Time: %.2f ms", std::accumulate(frametimes.begin(), frametimes.end(), 0.0) / frametimes.size());
                Text("Triangles: %zu", g_DebugFrameStats.index_count / 3);
                Text("Chunk Memory: %.2lu MB", g_Settings.chunkBytes.load() / 1024 / 1024);
                Text("Chunks in Queue: %zu", g_Renderer->terrain.chunkMeshQueue.size());
                End();

                // Begin("Noise Settings");
                // InputInt("octaves", &g_Settings.Noise.octaves);
                // InputFloat("frequency", &g_Settings.Noise.frequency);
                // InputFloat("amplitude", &g_Settings.Noise.amplitude);
                // InputFloat("lacunarity", &g_Settings.Noise.lacunarity);
                // InputFloat("persistence", &g_Settings.Noise.persistence);
                // InputFloat("scale", &g_Settings.Noise.scale);
                // End();   
            }
            
            
            ImGui::Render();
            #endif
            if (g_Settings.Noise != prev_NoiseSettings) {
                prev_NoiseSettings = g_Settings.Noise;
                g_Renderer->terrain.deinit(g_Renderer->vkc);
                g_Renderer->terrain.init(g_Renderer->vkc);
            }
            g_DebugFrameStats.index_count = 0;
            g_Renderer->terrain.tickFrame(g_Renderer->vkc, *g_Renderer->camera.get());
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
        std::cerr << "[Vulkan System Error!] :: " << expt.what() << " | " << vkResultString(vk::Result(expt.code().value())) << std::endl;
        cleanup();
        return 1;
    }
    catch (std::runtime_error &expt) {
        std::cerr << "[Runtime Error!] :: " << expt.what() << std::endl;
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
