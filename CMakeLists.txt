cmake_minimum_required(VERSION 3.28)
project(VulkanGame)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(BUILD_SHARED_LIBS ON)
# set(CMAKE_CXX_SCAN_FOR_MODULES ON)
set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_CXX_FLAGS "-fdiagnostics-color=always")
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")

add_executable(VulkanGame)
file(GLOB SRC_FILES "src/*.cpp")
target_sources(VulkanGame PRIVATE Main.cpp ${SRC_FILES})
file(GLOB ImGui_SRC_FILES "imgui/*.cpp")
target_sources(VulkanGame PRIVATE 
    ${ImGui_SRC_FILES}
    imgui/backends/imgui_impl_glfw.cpp imgui/backends/imgui_impl_vulkan.cpp
    simplex-noise/SimplexNoise.cpp
)
# file(GLOB CPPM_FILES "modules/*.cppm")
# target_sources(VulkanGame PRIVATE 
#     FILE_SET modules 
#     TYPE CXX_MODULES
#     FILES ${CPPM_FILES}
# )

# target_compile_options(VulkanGame PRIVATE "-fsanitize=address" "-fno-omit-frame-pointer")
# target_link_options(VulkanGame PRIVATE "-fsanitize=address")
target_precompile_headers(VulkanGame PRIVATE 
    <vulkan/vulkan.hpp>
    <vulkan/vulkan_structs.hpp>
    <vk_mem_alloc.h>
    <iostream>
    <ranges>
    
)
add_compile_definitions(
    VULKAN_HPP_NO_TO_STRING
    GLM_ENABLE_EXPERIMENTAL
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    GLM_FORCE_XYZW_ONLY
    IMGUI_ENABLE
)

#set(VULKAN_SDK "~/VulkanSDK/1.4.304.1/macOS")
include(FindVulkan)

target_include_directories(VulkanGame PRIVATE 
    include
    "${Vulkan_INCLUDE_DIRS}"
    "GLM/include"
    "imgui"
    "simplex-noise/include"
)
if(APPLE)
    target_include_directories(VulkanGame PRIVATE "GLFW_3.4/include")
    find_library(GLFW_LIB glfw3 PATHS "GLFW_3.4/lib-arm64")
else()
    target_link_libraries(VulkanGame PRIVATE glfw)
endif()
# add_subdirectory("VMA_3.3")
set(VulkanMemoryAllocator_DIR VMA_3.3/build/install/share/cmake/VulkanMemoryAllocator)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
target_link_libraries(VulkanGame PRIVATE Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator)

# include(FetchContent)
# FetchContent_Declare(
#   cpptrace
#   GIT_REPOSITORY https://github.com/jeremy-rifkin/cpptrace.git
#   GIT_TAG        v1.0.4 # <HASH or TAG>
# )
# FetchContent_MakeAvailable(cpptrace)
# target_link_libraries(VulkanGame PRIVATE cpptrace::cpptrace)

if(APPLE)
    find_library(COCOA_LIB Cocoa)
    find_library(IOKIT_LIB IOKit)
    target_link_libraries(VulkanGame PRIVATE ${COCOA_LIB} ${IOKIT_LIB})
    target_compile_definitions(VulkanGame PRIVATE VK_ENABLE_BETA_EXTENSIONS)
endif()
    
    
    # file(COPY "GLFW_3.4/lib-arm64/libglfw.3.dylib" DESTINATION "${CMAKE_BINARY_DIR}")