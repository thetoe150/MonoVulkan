#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define VMA_IMPLEMENTATION
#include "vma/vk_mem_alloc.h"
#include "vulkan/vulkan.hpp"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_vulkan.h"

#include "Tracy.hpp"
#include "TracyVulkan.hpp"

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <array>
#include <optional>
#include <set>
#include <unordered_map>

constexpr uint32_t WIDTH = 1500;
constexpr uint32_t HEIGHT = 1000;
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

static float s_scale[3] = {1.f, 1.f, 1.f};
static float s_rotate[3] = {0.f, 0.f, 0.f};
static float s_translate[3] = {0.f, 0.f, 1.f};
static float s_viewPos[3] = {10.f, 0.f, 0.f};
static float s_nearPlane = 0.1f;
static float s_farPlane = 100.f;
