#include "glm/trigonometric.hpp"
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
// already included in tiny_obj_loader.h
// #include "stb_image.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include <tiny_gltf.h>

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
#include <ctime>
#include <cstdint>
#include <limits>
#include <array>
#include <math.h>
#include <optional>
#include <set>
#include <unordered_map>

#define CHECK_VK_RESULT(f, msg)																	\
{																								\
	if(VkResult res = f){																		\
		throw std::runtime_error(msg + vk::to_string((vk::Result)res));							\
	}																							\
}																								\

constexpr uint32_t WIDTH = 1600;
constexpr uint32_t HEIGHT = 900;
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

constexpr float c_towerScale[3] = {0.7f, 0.7f, 0.7f};
constexpr float c_towerRotate[3] = {0.f, 0.f, 0.f};
constexpr float c_towerTranslate[3] = {0.f, -5.f, 0.f};

static float s_snowScale[3] = {0.001f, 0.001f, 0.001f};
static float s_snowRotate[3] = {0.f, 0.f, 0.f};
static float s_snowTranslate[3] = {0.f, 5.f, 0.f};

static float s_viewPos[3] = {20.f, 15.f, 0.f};
static float s_nearPlane = 0.1f;
static float s_farPlane = 100.f;

const std::string TOWER_MODEL_PATH = "res/models/wooden_watch_tower2.obj";
const std::string SNOWFLAKE_MODEL_PATH = "res/models/Snowflake.obj";
const std::string TOWER_TEXTURE_PATH = "res/textures/Wood_Tower_Col.jpg";
// const std::string SNOWFLAKE_TEXTURE_PATH = "res/textures/Wood_Tower_Col.jpg";

constexpr int SNOWFLAKE_COUNT = 2048;
constexpr int MAX_VORTEX_COUNT = 10;

static auto startTime = std::chrono::high_resolution_clock::now();
static std::array<float, MAX_VORTEX_COUNT> s_baseRadius;
static std::array<float, MAX_VORTEX_COUNT> s_basePhase;
static std::array<float, MAX_VORTEX_COUNT> s_baseForce;

enum class ObjectType{
	TOWER,
	SNOWFLAKE
};


struct Vortex {
	alignas(16) glm::vec3 pos;
	alignas(4) float force;
	alignas(4) float radius;
	alignas(4) float height;
};

inline auto getVortexRadius = [](float currentValue, float delta) -> float {
	return 2 + 3 * std::sin(delta);
};

inline auto getVortexVelocity = [](float currentValue, float delta) -> float {
	return 2 + 3 * std::sin(delta);
};

struct Snowflake {
	glm::vec3 position;
	float weight;
};

struct ComputePushConstant{
	int snowflakeCount = SNOWFLAKE_COUNT;
	float deltaTime;
};

struct GraphicPushConstant{
	alignas(4) bool useTexture;
};

struct SpecializationConstant{
	alignas(4) bool useTexture{true};
}s_specConstant;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
	VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

template <std::size_t Last = 0, typename TF, typename TArray, typename... TRest>
constexpr auto with_acc_sizes(TF&& f, const TArray& array, const TRest&... rest)
{
    f(array, std::integral_constant<std::size_t, Last>{});

    if constexpr(sizeof...(TRest) != 0)
    {
        with_acc_sizes<Last + std::tuple_size_v<TArray>>(f, rest...); 
    }
}

template<typename T, std::size_t... Sizes>
constexpr auto concat(const std::array<T, Sizes>&... arrays)
{
    std::array<T, (Sizes + ...)> result{};

    with_acc_sizes([&](const auto& arr, auto offset)
    {
        std::copy(arr.begin(), arr.end(), result.begin() + offset);
    }, arrays...);

    return result;
}

float generateRandomFloat(float low, float high){
	return low + (static_cast<float>(rand()) / (RAND_MAX / (high - low)));
}
 
void testAlignment()
{
	struct Struct1{
		glm::vec3 pos;
	};

	struct Struct2{
		alignas(16) glm::vec3 pos;
	};

	struct Struct3{
		alignas(16) glm::vec3 pos;
		float weight;
	};

	struct Struct4{
		glm::vec3 pos;
		float weight;
	};

	struct Struct5{
		float weight;
		alignas(16) glm::vec3 pos;
	};

	struct Struct6{
		alignas(16) glm::vec3 pos;
		alignas(16) glm::vec3 weight;
	};

		std::cout << "\nsize of Struct1 is: " << sizeof(Struct1)
				<< "\nsize of Struct2 is:" << sizeof(Struct2)
				<< "\nsize of Struct3 is:" << sizeof(Struct3)
				<< "\noffset of weight is:" << offsetof(Struct3, weight)
				<< "\nsize of Struct4 is:" << sizeof(Struct4)
				<< "\noffset of weight is:" << offsetof(Struct4, weight)
				<< "\nnsize of Struct5 is:" << sizeof(Struct5)
				<< "\noffset of weight is:" << offsetof(Struct5, weight)
				<< "\nsize of Struct6 is:" << sizeof(Struct6)
				<< "\noffset of weight is:" << offsetof(Struct6, weight) << std::endl;

}
