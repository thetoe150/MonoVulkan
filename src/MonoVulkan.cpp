#include "MonoVulkan.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "vulkan/vulkan_core.h"

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

void CheckImGuiResult(VkResult res){
	if(res == VK_SUCCESS){
		std::cout << "ImGui: success.\n";
	}
	else{
		std::cout << "ImGui: fail.\n";
	}
}

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicFamily;
    std::optional<uint32_t> computeFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicFamily.has_value() && computeFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct Vertex {
	alignas(16) glm::vec3 pos;
    alignas(16) glm::vec3 color;
    alignas(8)  glm::vec2 texCoord;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

        return attributeDescriptions;
    }

    bool operator==(const Vertex& other) const {
        return pos == other.pos && color == other.color && texCoord == other.texCoord;
    }
};


struct VertexInstance {
	alignas(16) glm::vec3 pos;

	static VkVertexInputBindingDescription getBindingDescription(){
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 4;
		bindingDescription.stride = sizeof(VertexInstance);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 1> getAttributeDescriptions(){
		std::array<VkVertexInputAttributeDescription, 1> attributeDescriptions{};
		attributeDescriptions[0].binding = 4;
		attributeDescriptions[0].location = 4;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		// attributeDescriptions[0].offset = offsetof(VertexInstance, pos);
		attributeDescriptions[0].offset = 0;

		return attributeDescriptions;
	}

	bool operator==(const VertexInstance& other) const{
		return pos == other.pos;
	}
};

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
        }
    };
}

struct TransformUniform {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::mat4 dummy;
};

struct LightingUniform {
    alignas(16) glm::vec3 lightPos;
    alignas(16) glm::vec3 camPos;
    alignas(16) glm::mat3 dummy1;
    alignas(16) glm::mat3 dummy2;
};

class MonoVulkan {
public:
	void init(){
		initContext();
        initGLFW();
        initVulkan();
		initTracy();
		initImGui();
	}

	void clean(){
		cleanUpImGui();
		cleanUpTracy();
        cleanUpVulkan();
		cleanUpGLFW();
	}

    void run() {
		init();
		updateContext();
        mainLoop();
		clean();
    }

private:
    GLFWwindow* window;
	tracy::VkCtx* tracyContext;

    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
    VkDevice device;
	VmaAllocator m_allocator;

	VkPhysicalDeviceProperties m_physicalDeviceProperties;

    VkQueue m_graphicQueue;
    VkQueue m_computeQueue;
    VkQueue m_presentQueue;

    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
	struct {
		std::vector<VkFramebuffer> base;
		struct {
			std::vector<VkFramebuffer> horizontal;
			std::vector<VkFramebuffer> vertical;
		} bloom;
		std::vector<VkFramebuffer> combine;
	} m_frameBuffers;

	struct {
		VkRenderPass base;
		VkRenderPass bloom;
		VkRenderPass combine;
	} m_renderPasses;

	struct {
		VkDescriptorSetLayout snowflake;
		struct {
			VkDescriptorSetLayout tranformUniform;
			VkDescriptorSetLayout meshMaterial;
		} candles;

		VkDescriptorSetLayout bloom;
		VkDescriptorSetLayout combine;
	} m_graphicDescriptorSetLayouts;

	struct {
		VkPipelineLayout snowflake;
		VkPipelineLayout candles;
		VkPipelineLayout bloom;
		VkPipelineLayout combine;
	}
    m_graphicPipelineLayouts;

	VkDescriptorSetLayout m_computeDescriptorSetLayout;
    VkPipelineLayout m_computePipelineLayout;

	VkPipelineCache m_pipelineCache;
	std::vector<char> pipelineCacheBlob;

    VkPipeline m_computePipeline;
	struct {
		VkPipeline snowflake;
		VkPipeline candles;
		struct {
			VkPipeline vertical;
			VkPipeline horizontal;
		} bloom;
		VkPipeline combine;
	}m_graphicPipelines;

    VkCommandPool m_graphicCommandPool;
    VkCommandPool m_computeCommandPool;
	VkQueryPool timestampPool;

	std::map<Object, tinygltf::Model> m_model;
	std::map<Object, std::vector< glm::mat4>> m_modelMeshTransforms;

	std::map<Object, std::vector<std::vector< glm::vec3>>> m_modelMeshAnimPositions;
	std::map<Object, std::vector<std::vector< float>>> m_modelMeshFrameWeights;

	typedef struct {
		VkImage image;
		VmaAllocation allocation;
		VkImageView view;
	} Image;

	typedef struct {
		Image baseImage;
		Image normalImage;
		Image emissiveImage;
	} MeshImages;

	std::map<Object, std::vector<MeshImages>> m_modelImages;

	struct {
		struct {
			Image colorRT;
			Image colorResRT;
			Image bloomThresholdRT;
			Image bloomThresholdResRT;
			Image depthRT;
		} base;
		Image bloom1;
		Image bloom2;
	} m_renderTargets;

    uint32_t mipLevels;

	struct {
		VkSampler snowflake;
		VkSampler candles;
		VkSampler quad;
	}
	m_samplers;

	typedef struct {
		void* raw;
		VkBuffer buffer;
		VmaAllocation allocation;
	} Buffer;

	struct {
		std::vector<Buffer> snowflake;
		std::vector<Buffer> candles;
		std::vector<Buffer> quad;
	} m_vertexBuffers;

	struct {
		std::vector<Buffer> snowflake;
		std::vector<Buffer> candles;
		std::vector<Buffer> quad;
	} m_indexBuffers;

	std::map<Object, std::map<int, VkBuffer>> m_animBuffer;
	std::map<Object, std::map<int, VmaAllocation>> m_animBufferAlloc;

	struct {
		std::map<Object, std::vector<void*>> transform;
		std::map<Object, std::vector<void*>> lighting;
	} m_graphicUniformBuffersMapped;

	struct {
		std::map<Object, std::vector<VkBuffer>> transform;
		std::map<Object, std::vector<VkBuffer>> lighting;
	} m_graphicUniformBuffers;

	struct {
		std::map<Object, std::vector<VmaAllocation>> transform;
		std::map<Object, std::vector<VmaAllocation>> lighting;
	} m_uniformBuffersAlloc;

    std::vector<VertexInstance> m_towerInstanceRaw;
	VkBuffer m_towerInstanceBuffer;
	VmaAllocation instanceBufferAlloc;


	// Compute buffers
	VkBuffer m_storageBuffer;
	VmaAllocation m_storageBufferAlloc;

	SpecializationConstant m_graphicSpecConstant;
	GraphicPushConstant m_graphicPushConstant;

	ComputePushConstant m_computePushConstant;

	VkBuffer m_vortexUniformBuffer;
	VmaAllocation m_vortexUniformBufferAlloc;
	void* m_vortexUniformBufferMapped;

    VkDescriptorPool m_descriptorPool;

	struct {
		std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> snowflake;
		struct {
			std::vector<std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT>> meshMaterial; // per mesh of candles model
			std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> tranformUniform; // 1 for candles model, update every frame
		} candles;
		std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> bloom1;
		std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> bloom2;
		std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> combine;
	} m_graphicDescriptorSets;

    VkDescriptorSet m_computeDescriptorSets;

    std::vector<VkCommandBuffer> m_graphicCommandBuffers;
    VkCommandBuffer m_computeCommandBuffer;
    VkCommandBuffer tracyCommandBuffer;

    std::vector<VkSemaphore> m_imageAvailableSemaphores;
    std::vector<VkSemaphore> m_renderFinishedSemaphores;
    std::vector<VkSemaphore> m_computeStartingSemaphores;

    std::vector<VkFence> m_inFlightGraphicFences;
    VkFence m_inFlightComputeFences;
	VkSemaphore m_computeFinishedSemaphore;
    uint32_t m_currentFrame = 0;

	float m_lastTime;
	float m_currentDeltaTime = 0;
	float m_currentAnimTime = 0;

	VkDescriptorPool imguiDescriptorPool;

    bool framebufferResized = false;

	void initContext() {
        auto now = std::chrono::high_resolution_clock::now();
        float currentTime = std::chrono::duration<float, std::chrono::seconds::period>(now - startTime).count();

        m_lastTime = currentTime;

        loadModels();
		computeAnimation(Object::CANDLE);
	}

    void initGLFW() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
		glfwSetKeyCallback(window, keyCallback);
		glfwSetCursorPosCallback(window, mouseCallback);
		glfwSetScrollCallback(window, scrollCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<MonoVulkan*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

	static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods){
		if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
			glfwSetWindowShouldClose(window, true);
		if(key == GLFW_KEY_Z && action == GLFW_PRESS)
			std::cout << "Callback key Z is pressed.\n";
	}

	static void mouseCallback(GLFWwindow* window, double xpos, double ypos){
		if (firstMouse) {
			lastX = xpos;
			lastY = ypos;
			firstMouse = false;
		}
		
		float xoffset = xpos - lastX;
		float yoffset = lastY - ypos;
		lastX = xpos;
		lastY = ypos;

		if (s_moveCam)
			g_camera.processMouseMovement(xoffset, yoffset);
	}

	static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset){
		g_camera.processMouseScroll(static_cast<float>(yoffset));
	}

	void initTracy(){
		tracyContext = TracyVkContextCalibrated(instance, physicalDevice, device, m_graphicQueue, tracyCommandBuffer, vkGetInstanceProcAddr, vkGetDeviceProcAddr);
		
		// VkQueryPoolCreateInfo poolInfo;
		// poolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO; 
		// poolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
		// poolInfo.queryCount = 1024 * 8;
		// poolInfo.pNext = nullptr;
		// if(vkCreateQueryPool(device, &poolInfo, nullptr, &timestampPool) != VK_SUCCESS)
		// {
        //     throw std::runtime_error("failed to create query pool!");
		// }
	}

	void initImGui(){
		// Setup Dear ImGui context
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO();
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
		
		ImGui_ImplGlfw_InitForVulkan(window, true);

		std::array<VkDescriptorPoolSize, 1> poolSize;
		poolSize[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSize[0].descriptorCount = 10;

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
		poolInfo.maxSets = 10;
		poolInfo.pPoolSizes = poolSize.data();
		poolInfo.poolSizeCount = poolSize.size();

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &imguiDescriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }

		ImGui_ImplVulkan_InitInfo info{};
		info.Instance = instance;
		info.PhysicalDevice = physicalDevice;
		info.Device = device;
		info.QueueFamily = findQueueFamilies(physicalDevice).graphicFamily.value();
		info.Queue = m_graphicQueue;
		info.DescriptorPool = imguiDescriptorPool;
		info.MinImageCount = swapChainImages.size();
		info.ImageCount = swapChainImages.size();
		// info.RenderPass = m_renderPasses.base;
		// info.MSAASamples = getMaxUsableSampleCount();
		info.RenderPass = m_renderPasses.combine;
		info.MSAASamples = VK_SAMPLE_COUNT_1_BIT; 
		
		// info.CheckVkResultFn = CheckImGuiResult;
		ImGui_ImplVulkan_Init(&info);
	}

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
		createAllocator();
        createSwapChain();
        createSwapchainImageViews();
        createRenderPasses();
        createDescriptorSetLayouts();
		createPipelineCache();
		createPipelineLayouts();
		createPipelines();
        createCommandPools();
        createModelImages();
        createRenderTargets();
        createFramebuffers();
        createSamplers();
		loadInstanceData();
        createVertexBuffers();
        createIndexBuffers();
		createInstanceBuffer();
		createUniformBuffers();
		createAnimationBuffers();
		createStorageBuffer();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();

		// printPhysicalDeviceProperties();
		// printMemoryStatistics();
		// printMemoryBudget();
		// printQueueFamilyProperties();
    }

	void processInput(){
		glfwPollEvents();

		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
			glfwSetWindowShouldClose(window, true);

		//// key for camera
		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
			g_camera.processKeyboard(MovementDirection::FORWARD, m_currentDeltaTime);
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
			g_camera.processKeyboard(MovementDirection::BACKWARD, m_currentDeltaTime);
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
			g_camera.processKeyboard(MovementDirection::LEFT, m_currentDeltaTime);
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
			g_camera.processKeyboard(MovementDirection::RIGHT, m_currentDeltaTime);
		if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
			g_camera.processKeyboard(MovementDirection::UP, m_currentDeltaTime);
		if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
			g_camera.processKeyboard(MovementDirection::DOWN, m_currentDeltaTime);

		if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS) {
			s_moveCam = true;
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		}
		if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
			s_moveCam = false;
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}

		if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
			recreatePipelines();
		}

		//// key for rotate object
		//float rotateVelocity = deltaTime * 40.0f;
		//if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS)
		//    yAngle -= rotateVelocity;
		//if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS)
		//    yAngle += rotateVelocity;
		//if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS)
		//    xAngle += rotateVelocity;
		//if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS)
		//    xAngle -= rotateVelocity;
	}

    void updateGraphicUniformBuffer() {
		ZoneScopedN("Update Graphic Transform Uniform Buffer");
		for (unsigned int i = 0; i < Object::COUNT; i++){
			Object objIdx = static_cast<Object>(i);
			tinygltf::Model& model = m_model[objIdx];

			// transform uniform
			{
				unsigned int meshCount = model.meshes.size();
				TransformUniform ubo{};

				if (objIdx == Object::CANDLE) {
					ubo.model = glm::mat4(1.0f);
					ubo.model = glm::translate(ubo.model, glm::vec3(c_towerTranslate[0], c_towerTranslate[1], c_towerTranslate[2]));
					ubo.model = glm::scale(ubo.model, glm::vec3(c_towerScale[0], c_towerScale[1], c_towerScale[2]));
				}
				else if (objIdx == Object::SNOWFLAKE) {
					ubo.model = glm::mat4(1.0f);
					ubo.model = glm::translate(ubo.model, glm::vec3(s_snowTranslate[0], s_snowTranslate[1], s_snowTranslate[2]));
					if(s_snowRotate[0] != 0.f || s_snowRotate[1] != 0.f || s_snowRotate[2] != 0.f)
						ubo.model = glm::rotate(ubo.model, m_lastTime * glm::radians(90.0f), glm::vec3(s_snowRotate[0], s_snowRotate[1], s_snowRotate[2]));
					ubo.model = glm::scale(ubo.model, glm::vec3(s_snowScale[0], s_snowScale[1], s_snowScale[2]));
				}
				
				// glm::mat4 view = glm::lookAt(glm::vec3(s_viewPos[0], s_viewPos[1], s_viewPos[2]), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
				// glm::mat4 proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float) swapChainExtent.height, s_nearPlane, s_farPlane);
				glm::mat4 view = g_camera.getViewMatrix();
				glm::mat4 proj = glm::perspective(g_camera.getZoom(), swapChainExtent.width / (float) swapChainExtent.height, s_nearPlane, s_farPlane);
				proj[1][1] *= -1;

				ubo.view = view;
				ubo.proj = proj;

				TransformUniform* tranformUBO = (TransformUniform*)m_graphicUniformBuffersMapped.transform[objIdx][m_currentFrame];
				for (unsigned int i = 0; i < meshCount; i++){
					tranformUBO[i] = ubo;
				}
			}

			// lighting uniform
			{
				LightingUniform* lightingUBO = (LightingUniform*)m_graphicUniformBuffersMapped.lighting[objIdx][m_currentFrame];
				lightingUBO->lightPos = glm::vec3(s_lightPos[0], s_lightPos[1], s_lightPos[2]);
				lightingUBO->camPos = g_camera.getPostion();
			}
		}
    }
	
	void updateComputeUniformBuffer() {
		ZoneScopedN("Update Compute Vortex Uniform Buffer");
		for(unsigned int i = 0; i < MAX_VORTEX_COUNT; i++){
			Vortex& vortex = ((Vortex*)m_vortexUniformBufferMapped)[i];

			vortex.radius = s_baseRadius[i] * std::abs(std::sin(m_lastTime * 0.1f + s_basePhase[i]));
			vortex.force = s_baseForce[i] * std::sin(m_lastTime * 0.2f);
		}
	}

	void updateComputePushConstant() {
		m_computePushConstant.snowflakeCount = SNOWFLAKE_COUNT;
		m_computePushConstant.deltaTime = m_currentDeltaTime;
	}

	void updateContext() {
        auto now = std::chrono::high_resolution_clock::now();
        float currentTime = std::chrono::duration<float, std::chrono::seconds::period>(now - startTime).count();

		m_currentDeltaTime = currentTime - m_lastTime;
        m_lastTime = currentTime;

		// updateGraphicUniformBuffer();
		// updateComputeUniformBuffer();
		// updateComputePushConstant();

		computeAnimation(Object::CANDLE);
	}

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
			std::cout << std::endl << ">>>>>>> New Frame Start <<<<<<<<" << std::endl;
			processInput();
			updateContext();
            drawFrame();
			FrameMark;
        }

        vkDeviceWaitIdle(device);
    }

	void savePipelineCache() {
		size_t dataSize{};
		vkGetPipelineCacheData(device, m_pipelineCache, &dataSize, nullptr);
		if(dataSize){
			char* data = new char[dataSize]();
			vkGetPipelineCacheData(device, m_pipelineCache, &dataSize, data);
			writeFile("../../res/cache/pipeline_cache.blob", data, dataSize);
		}
	}

	void cleanUpImGui(){
		ImGui_ImplVulkan_DestroyFontsTexture();
		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}

	void cleanUpGLFW(){
        glfwDestroyWindow(window);
        glfwTerminate();
	}
	
	void cleanUpTracy(){
		TracyVkDestroy(tracyContext);
	}

    void cleanUpVulkan() {
        cleanupSwapChain();

		savePipelineCache();
        // vkDestroyPipeline(device, m_graphicPipelines.snowflake, nullptr);
        vkDestroyPipeline(device, m_graphicPipelines.candles, nullptr);
        vkDestroyPipeline(device, m_graphicPipelines.bloom.vertical, nullptr);
        vkDestroyPipeline(device, m_graphicPipelines.bloom.horizontal, nullptr);
        vkDestroyPipeline(device, m_graphicPipelines.combine, nullptr);
        vkDestroyPipeline(device, m_computePipeline, nullptr);
        vkDestroyPipelineCache(device, m_pipelineCache, nullptr);
        // vkDestroyPipelineLayout(device, m_graphicPipelineLayouts.snowflake, nullptr);
        vkDestroyPipelineLayout(device, m_graphicPipelineLayouts.candles, nullptr);
        vkDestroyPipelineLayout(device, m_graphicPipelineLayouts.bloom, nullptr);
        vkDestroyPipelineLayout(device, m_graphicPipelineLayouts.combine, nullptr);
        vkDestroyPipelineLayout(device, m_computePipelineLayout, nullptr);
        vkDestroyRenderPass(device, m_renderPasses.base, nullptr);
        vkDestroyRenderPass(device, m_renderPasses.bloom, nullptr);
        vkDestroyRenderPass(device, m_renderPasses.combine, nullptr);

		vkDestroyBuffer(device, m_vortexUniformBuffer, nullptr);
		vmaUnmapMemory(m_allocator, m_vortexUniformBufferAlloc);
		vmaFreeMemory(m_allocator, m_vortexUniformBufferAlloc);

		vkDestroyBuffer(device, m_storageBuffer, nullptr);
		vmaFreeMemory(m_allocator, m_storageBufferAlloc);

        vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
        vkDestroyDescriptorPool(device, imguiDescriptorPool, nullptr);

        // vkDestroyDescriptorSetLayout(device, m_graphicDescriptorSetLayouts.snowflake, nullptr);
        vkDestroyDescriptorSetLayout(device, m_graphicDescriptorSetLayouts.candles.tranformUniform, nullptr);
        vkDestroyDescriptorSetLayout(device, m_graphicDescriptorSetLayouts.candles.meshMaterial, nullptr);
        vkDestroyDescriptorSetLayout(device, m_graphicDescriptorSetLayouts.bloom, nullptr);
        vkDestroyDescriptorSetLayout(device, m_graphicDescriptorSetLayouts.combine, nullptr);
        vkDestroyDescriptorSetLayout(device, m_computeDescriptorSetLayout, nullptr);

		for(auto& buffer : m_vertexBuffers.snowflake) {
			vkDestroyBuffer(device, buffer.buffer, nullptr);
			vmaFreeMemory(m_allocator, buffer.allocation);
		}
		for(auto& buffer : m_vertexBuffers.candles) {
			vkDestroyBuffer(device, buffer.buffer, nullptr);
			vmaFreeMemory(m_allocator, buffer.allocation);
		}
		for(auto& buffer : m_vertexBuffers.quad) {
			vkDestroyBuffer(device, buffer.buffer, nullptr);
			vmaFreeMemory(m_allocator, buffer.allocation);
		}

		for(auto& buffer : m_indexBuffers.snowflake) {
			vkDestroyBuffer(device, buffer.buffer, nullptr);
			vmaFreeMemory(m_allocator, buffer.allocation);
		}
		for(auto& buffer : m_indexBuffers.candles) {
			vkDestroyBuffer(device, buffer.buffer, nullptr);
			vmaFreeMemory(m_allocator, buffer.allocation);
		}
		for(auto& buffer : m_indexBuffers.quad) {
			vkDestroyBuffer(device, buffer.buffer, nullptr);
			vmaFreeMemory(m_allocator, buffer.allocation);
		}

		for (unsigned int i = 0; i < Object::COUNT; i++){
			Object objIdx = static_cast<Object>(i);
			for (auto& bufferIdx : m_animBuffer[objIdx]) {
				vkDestroyBuffer(device, bufferIdx.second, nullptr);
			}
			for (auto& bufferAllocIdx : m_animBufferAlloc[objIdx]) {
				vmaFreeMemory(m_allocator, bufferAllocIdx.second);
			}

			for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
				vkDestroyBuffer(device, m_graphicUniformBuffers.transform[objIdx][i], nullptr);
				vmaUnmapMemory(m_allocator, m_uniformBuffersAlloc.transform[objIdx][i]);
				vmaFreeMemory(m_allocator, m_uniformBuffersAlloc.transform[objIdx][i]);

				vkDestroyBuffer(device, m_graphicUniformBuffers.lighting[objIdx][i], nullptr);
				vmaUnmapMemory(m_allocator, m_uniformBuffersAlloc.lighting[objIdx][i]);
				vmaFreeMemory(m_allocator, m_uniformBuffersAlloc.lighting[objIdx][i]);
			}

			for (auto& meshImage : m_modelImages[objIdx]) {
				vkDestroyImage(device, meshImage.baseImage.image, nullptr);
				vkDestroyImage(device, meshImage.normalImage.image, nullptr);
				vkDestroyImage(device, meshImage.emissiveImage.image, nullptr);

				vmaFreeMemory(m_allocator, meshImage.baseImage.allocation);
				vmaFreeMemory(m_allocator, meshImage.normalImage.allocation);
				vmaFreeMemory(m_allocator, meshImage.emissiveImage.allocation);

				vkDestroyImageView(device, meshImage.baseImage.view, nullptr);
				vkDestroyImageView(device, meshImage.normalImage.view, nullptr);
				vkDestroyImageView(device, meshImage.emissiveImage.view, nullptr);
			}
		}

		// vkDestroySampler(device, m_samplers.snowflake, nullptr);
		vkDestroySampler(device, m_samplers.candles, nullptr);
		vkDestroySampler(device, m_samplers.quad, nullptr);

        vkDestroyBuffer(device, m_towerInstanceBuffer, nullptr);
        vmaFreeMemory(m_allocator, instanceBufferAlloc);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, m_renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, m_imageAvailableSemaphores[i], nullptr);
            vkDestroySemaphore(device, m_computeStartingSemaphores[i], nullptr);
            vkDestroyFence(device, m_inFlightGraphicFences[i], nullptr);
        }
		vkDestroyFence(device, m_inFlightComputeFences, nullptr);
		vkDestroySemaphore(device, m_computeFinishedSemaphore, nullptr);

        vkDestroyCommandPool(device, m_graphicCommandPool, nullptr);
        vkDestroyCommandPool(device, m_computeCommandPool, nullptr);
        // vkDestroyCommandPool(device, timestampPool, nullptr);
		vmaDestroyAllocator(m_allocator);

        vkDestroyDevice(device, nullptr);

        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
    }

    void cleanupSwapChain() {
        vkDestroyImageView(device, m_renderTargets.base.colorRT.view, nullptr);
        vkDestroyImage(device, m_renderTargets.base.colorRT.image, nullptr);
        vmaFreeMemory(m_allocator, m_renderTargets.base.colorRT.allocation);

        vkDestroyImageView(device, m_renderTargets.base.colorResRT.view, nullptr);
        vkDestroyImage(device, m_renderTargets.base.colorResRT.image, nullptr);
        vmaFreeMemory(m_allocator, m_renderTargets.base.colorResRT.allocation);

        vkDestroyImageView(device, m_renderTargets.base.depthRT.view, nullptr);
        vkDestroyImage(device, m_renderTargets.base.depthRT.image, nullptr);
        vmaFreeMemory(m_allocator, m_renderTargets.base.depthRT.allocation);

        vkDestroyImageView(device, m_renderTargets.base.bloomThresholdRT.view, nullptr);
        vkDestroyImage(device, m_renderTargets.base.bloomThresholdRT.image, nullptr);
        vmaFreeMemory(m_allocator, m_renderTargets.base.bloomThresholdRT.allocation);

        vkDestroyImageView(device, m_renderTargets.base.bloomThresholdResRT.view, nullptr);
        vkDestroyImage(device, m_renderTargets.base.bloomThresholdResRT.image, nullptr);
        vmaFreeMemory(m_allocator, m_renderTargets.base.bloomThresholdResRT.allocation);

        vkDestroyImageView(device, m_renderTargets.bloom1.view, nullptr);
        vkDestroyImage(device, m_renderTargets.bloom1.image, nullptr);
        vmaFreeMemory(m_allocator, m_renderTargets.bloom1.allocation);

        vkDestroyImageView(device, m_renderTargets.bloom2.view, nullptr);
        vkDestroyImage(device, m_renderTargets.bloom2.image, nullptr);
        vmaFreeMemory(m_allocator, m_renderTargets.bloom2.allocation);

        for (auto framebuffer : m_frameBuffers.base) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        for (auto framebuffer : m_frameBuffers.bloom.horizontal) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        for (auto framebuffer : m_frameBuffers.bloom.vertical) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        for (auto framebuffer : m_frameBuffers.combine) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }

    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device);

        cleanupSwapChain();

        createSwapChain();
        createSwapchainImageViews();
        createRenderTargets();
        createFramebuffers();
    }

    void createInstance() {
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "MonoVulkan";
        appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 1, 3, 0);
        appInfo.pEngineName = "MonoVulkan";
        appInfo.engineVersion = VK_MAKE_API_VERSION(0, 1, 3, 0);
        appInfo.apiVersion = VK_API_VERSION_1_3;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
        } else {
            createInfo.enabledLayerCount = 0;

            createInfo.pNext = nullptr;
        }

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }
    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }

    void setupDebugMessenger() {
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    void createSurface() {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                msaaSamples = getMaxUsableSampleCount();
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }

        vkGetPhysicalDeviceProperties(physicalDevice, &m_physicalDeviceProperties);
    }

    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {indices.graphicFamily.value(), indices.computeFamily.value(), indices.presentFamily.value()};

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.samplerAnisotropy = VK_TRUE;

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        createInfo.pEnabledFeatures = &deviceFeatures;

        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }

		VkPhysicalDeviceVertexAttributeDivisorFeaturesEXT divisorFeature{};
		divisorFeature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT;
		divisorFeature.vertexAttributeInstanceRateDivisor = true;
		divisorFeature.vertexAttributeInstanceRateZeroDivisor = true;

		createInfo.pNext = &divisorFeature;

		VkPhysicalDeviceRobustness2FeaturesEXT robustFeature{};
		robustFeature.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT;
		robustFeature.nullDescriptor = true;
		robustFeature.robustBufferAccess2 = false;
		robustFeature.robustImageAccess2 = false;

		divisorFeature.pNext = &robustFeature;
		robustFeature.pNext = nullptr;

		// VkPhysicalDeviceRobustness2PropertiesEXT robustProperties{};
		// robustProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_PROPERTIES_EXT;
		// robustProperties.robustUniformBufferAccessSizeAlignment = 256;

		// robustFeature.pNext = &robustProperties;

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(device, indices.graphicFamily.value(), 0, &m_graphicQueue);
        vkGetDeviceQueue(device, indices.computeFamily.value(), 0, &m_computeQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &m_presentQueue);

		std::cout << "\nQueue graphic family Index: " << indices.graphicFamily.value()
				<< "\nQueue compute family Index: " << indices.computeFamily.value()
				<< "\nQueue present family Index: " << indices.presentFamily.value() << std::endl;
    }

	void createAllocator(){
		VmaAllocatorCreateInfo allocatorInfo{};
		allocatorInfo.flags = VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT | VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
		allocatorInfo.physicalDevice = physicalDevice;
		allocatorInfo.device = device;
		allocatorInfo.preferredLargeHeapBlockSize = 0;
		allocatorInfo.pAllocationCallbacks = nullptr;
		allocatorInfo.pDeviceMemoryCallbacks = nullptr;
		allocatorInfo.pHeapSizeLimit = nullptr;
		allocatorInfo.pVulkanFunctions = nullptr;
		allocatorInfo.instance = instance;
		allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;

        if (vmaCreateAllocator(&allocatorInfo, &m_allocator) != VK_SUCCESS) {
            throw std::runtime_error("fail to create memory allocator");
        }
	}

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;

        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {indices.graphicFamily.value(), indices.presentFamily.value()};

        if (indices.graphicFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());
		std::cout << "swapchain extent width: " << extent.width << "\n";
		std::cout << "swapchain extent height: " << extent.height << "\n";
		std::cout << "frames in flight count:" << MAX_FRAMES_IN_FLIGHT << "\n";
		std::cout << "swapchain images count:" << imageCount << "\n";

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    void createSwapchainImageViews() {
        swapChainImageViews.resize(swapChainImages.size());

        for (uint32_t i = 0; i < swapChainImages.size(); i++) {
            swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
        }
    }

    void createRenderPasses() {
		// base render pass
		{
			VkAttachmentDescription colorAttachment{};
			colorAttachment.format = swapChainImageFormat;
			colorAttachment.samples = msaaSamples;
			colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

			VkAttachmentDescription bloomThresholdAttachment{};
			bloomThresholdAttachment.format = swapChainImageFormat;
			bloomThresholdAttachment.samples = msaaSamples;
			bloomThresholdAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			bloomThresholdAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			bloomThresholdAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			bloomThresholdAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			bloomThresholdAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			bloomThresholdAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

			VkAttachmentDescription depthAttachment{};
			depthAttachment.format = findDepthFormat();
			depthAttachment.samples = msaaSamples;
			depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

			VkAttachmentDescription colorAttachmentResolve{};
			colorAttachmentResolve.format = swapChainImageFormat;
			colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
			colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

			VkAttachmentDescription bloomThresholdAttachmentResolve{};
			bloomThresholdAttachmentResolve.format = swapChainImageFormat;
			bloomThresholdAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
			bloomThresholdAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			bloomThresholdAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			bloomThresholdAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			bloomThresholdAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			bloomThresholdAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			bloomThresholdAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

			VkAttachmentReference colorAttachmentRef{};
			colorAttachmentRef.attachment = 0;
			colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

			VkAttachmentReference bloomThresholdAttachmentRef{};
			bloomThresholdAttachmentRef.attachment = 1;
			bloomThresholdAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

			std::array<VkAttachmentReference, 2> colorRefs = {colorAttachmentRef, bloomThresholdAttachmentRef};

			VkAttachmentReference depthAttachmentRef{};
			depthAttachmentRef.attachment = 2;
			depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

			VkAttachmentReference colorAttachmentResolveRef{};
			colorAttachmentResolveRef.attachment = 3;
			colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

			VkAttachmentReference bloomThresholdAttachmentResolveRef{};
			bloomThresholdAttachmentResolveRef.attachment = 4;
			bloomThresholdAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

			std::array<VkAttachmentReference, 2> colorResolveRefs = {colorAttachmentResolveRef, bloomThresholdAttachmentResolveRef};

			VkSubpassDescription subpass{};
			subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
			subpass.colorAttachmentCount = colorRefs.size();
			subpass.pColorAttachments = colorRefs.data();
			subpass.pDepthStencilAttachment = &depthAttachmentRef;
			subpass.pResolveAttachments = colorResolveRefs.data();

			std::array<VkSubpassDependency, 2> dependencies;

			// original
			dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
			dependencies[0].dstSubpass = 0;
			dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
			dependencies[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
			dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

			dependencies[1].srcSubpass = 0;
			dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
			dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
			dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

			std::array<VkAttachmentDescription, 5> attachments = {colorAttachment, bloomThresholdAttachment, 
				depthAttachment, colorAttachmentResolve, bloomThresholdAttachmentResolve};
			VkRenderPassCreateInfo renderPassInfo{};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
			renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
			renderPassInfo.pAttachments = attachments.data();
			renderPassInfo.subpassCount = 1;
			renderPassInfo.pSubpasses = &subpass;
			renderPassInfo.dependencyCount = dependencies.size();
			renderPassInfo.pDependencies = dependencies.data();

			if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &m_renderPasses.base) != VK_SUCCESS) {
				throw std::runtime_error("failed to create render pass!");
			}
		}

		// bloom pass
		{
			VkAttachmentDescription blurAttachment{};
			blurAttachment.format = swapChainImageFormat;
			blurAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
			blurAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			blurAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			blurAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			blurAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			blurAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			blurAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

			VkAttachmentReference blurRef{};
			blurRef.attachment = 0;
			blurRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

			VkSubpassDescription blurSubpass{};
			blurSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
			blurSubpass.colorAttachmentCount = 1;
			blurSubpass.pColorAttachments = &blurRef;

			// WARNING: seem wrong
			std::array<VkSubpassDependency, 2> blurDeps{};
			blurDeps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
			blurDeps[0].dstSubpass = 0;
			blurDeps[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			blurDeps[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			blurDeps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			blurDeps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

			blurDeps[1].srcSubpass = 0;
			blurDeps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
			blurDeps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			blurDeps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			blurDeps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
			blurDeps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			VkRenderPassCreateInfo bloomPassInfo{};
			bloomPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO; 
			bloomPassInfo.subpassCount = 1;
			bloomPassInfo.pSubpasses = &blurSubpass;
			bloomPassInfo.attachmentCount = 1;
			bloomPassInfo.pAttachments = &blurAttachment;
			bloomPassInfo.dependencyCount = blurDeps.size();
			bloomPassInfo.pDependencies = blurDeps.data();

			if (vkCreateRenderPass(device, &bloomPassInfo, nullptr, &m_renderPasses.bloom) != VK_SUCCESS) {
				throw std::runtime_error("failed to create render pass!");
			}
		}

		// combine pass
		{
			VkAttachmentDescription combineAttachment{};
			combineAttachment.format = swapChainImageFormat;
			combineAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
			combineAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			combineAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			combineAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			combineAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			combineAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			combineAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

			VkAttachmentReference combineRef{};
			combineRef.attachment = 0;
			combineRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

			VkSubpassDescription combineSubpass{};
			combineSubpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
			combineSubpass.colorAttachmentCount = 1;
			combineSubpass.pColorAttachments = &combineRef;

			// WARNING: seem wrong
			std::array<VkSubpassDependency, 2> combineDeps{};
			combineDeps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
			combineDeps[0].dstSubpass = 0;
			combineDeps[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			combineDeps[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			combineDeps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			combineDeps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

			combineDeps[1].srcSubpass = 0;
			combineDeps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
			combineDeps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			combineDeps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			combineDeps[1].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			combineDeps[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

			VkRenderPassCreateInfo combinePassInfo{};
			combinePassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO; 
			combinePassInfo.subpassCount = 1;
			combinePassInfo.pSubpasses = &combineSubpass;
			combinePassInfo.attachmentCount = 1;
			combinePassInfo.pAttachments = &combineAttachment;
			combinePassInfo.dependencyCount = combineDeps.size();
			combinePassInfo.pDependencies = combineDeps.data();
			if (vkCreateRenderPass(device, &combinePassInfo, nullptr, &m_renderPasses.combine) != VK_SUCCESS) {
				throw std::runtime_error("failed to create render pass!");
			}
		}
    }

    void createDescriptorSetLayouts() {
		createGraphicDescriptorSetLayouts();	
		createComputeDescriptorSetLayouts();
    }

	void createGraphicDescriptorSetLayouts() {
		{
			// for candles: 2 descriptor set layouts, 1 for texture+sampler(change for each mesh), 1 for uniform buffer (change each frame)
			{
				VkDescriptorSetLayoutBinding uboLayoutBinding{};
				uboLayoutBinding.binding = 0;
				uboLayoutBinding.descriptorCount = 1;
				uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
				uboLayoutBinding.pImmutableSamplers = nullptr;
				uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

				VkDescriptorSetLayoutBinding lightBinding{};
				lightBinding.binding = 1;
				lightBinding.descriptorCount = 1;
				lightBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				lightBinding.pImmutableSamplers = nullptr;
				lightBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

				std::array<VkDescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, lightBinding};
				VkDescriptorSetLayoutCreateInfo layoutInfo{};
				layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
				layoutInfo.pBindings = bindings.data();

				if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_graphicDescriptorSetLayouts.candles.tranformUniform) != VK_SUCCESS) {
					throw std::runtime_error("failed to create descriptor set layout!");
				}
			}

			{
				VkDescriptorSetLayoutBinding samplerLayoutBinding{};
				samplerLayoutBinding.binding = 2;
				samplerLayoutBinding.descriptorCount = 1;
				samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				samplerLayoutBinding.pImmutableSamplers = nullptr;
				samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

				VkDescriptorSetLayoutBinding normalBinding{};
				normalBinding.binding = 3;
				normalBinding.descriptorCount = 1;
				normalBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				normalBinding.pImmutableSamplers = nullptr;
				normalBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

				VkDescriptorSetLayoutBinding emissiveBinding{};
				emissiveBinding.binding = 4;
				emissiveBinding.descriptorCount = 1;
				emissiveBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				emissiveBinding.pImmutableSamplers = nullptr;
				emissiveBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

				std::array<VkDescriptorSetLayoutBinding, 3> bindings = {samplerLayoutBinding, normalBinding, emissiveBinding};
				VkDescriptorSetLayoutCreateInfo layoutInfo{};
				layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
				layoutInfo.pBindings = bindings.data();

				if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_graphicDescriptorSetLayouts.candles.meshMaterial) != VK_SUCCESS) {
					throw std::runtime_error("failed to create descriptor set layout!");
				}
			}
		}

		// for bloom
		{
			VkDescriptorSetLayoutBinding baseBinding{};
			baseBinding.binding = 0;
			baseBinding.descriptorCount = 1;
			baseBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			baseBinding.pImmutableSamplers = nullptr;
			baseBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

			VkDescriptorSetLayoutCreateInfo layoutInfo{};
			layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
			layoutInfo.bindingCount = 1;
			layoutInfo.pBindings = &baseBinding;

			CHECK_VK_RESULT(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_graphicDescriptorSetLayouts.bloom)
				, "fail to create bloom descriptor set layout");
		}

		// for combine
		{
			VkDescriptorSetLayoutBinding baseBinding{};
			baseBinding.binding = 0;
			baseBinding.descriptorCount = 1;
			baseBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			baseBinding.pImmutableSamplers = nullptr;
			baseBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

			VkDescriptorSetLayoutBinding bloomBinding{};
			bloomBinding.binding = 1;
			bloomBinding.descriptorCount = 1;
			bloomBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			bloomBinding.pImmutableSamplers = nullptr;
			bloomBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

			std::array<VkDescriptorSetLayoutBinding, 2> bindings{baseBinding, bloomBinding};

			VkDescriptorSetLayoutCreateInfo layoutInfo{};
			layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
			layoutInfo.bindingCount = bindings.size();
			layoutInfo.pBindings = bindings.data();

			CHECK_VK_RESULT(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_graphicDescriptorSetLayouts.combine)
				, "fail to create bloom descriptor set layout");
		}
	}

	void createComputeDescriptorSetLayouts() {
		VkDescriptorSetLayoutBinding storageBinding{};
		storageBinding.binding = 0;
		storageBinding.descriptorCount = 1;
		storageBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		storageBinding.pImmutableSamplers = nullptr;
		storageBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding uboBinding{};
		uboBinding.binding = 1;
		uboBinding.descriptorCount = 1;
		uboBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboBinding.pImmutableSamplers = nullptr;
		uboBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		std::array<VkDescriptorSetLayoutBinding, 2> bindings = {storageBinding, uboBinding};
		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
		layoutInfo.pBindings = bindings.data();

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_computeDescriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor set layout!");
		}
	}

    void createPipelineCache() {
		if(!isFileExist("../../res/cache/pipeline_cache.blob"))
			makeFile("../../res/cache/pipeline_cache.blob");

		pipelineCacheBlob = readFile("../../res/cache/pipeline_cache.blob");

		VkPipelineCacheCreateInfo pipelineCacheInfo{};
		pipelineCacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
		pipelineCacheInfo.pInitialData = static_cast<void*>(pipelineCacheBlob.data());
		pipelineCacheInfo.initialDataSize = pipelineCacheBlob.size() * sizeof(char);


		vkCreatePipelineCache(device, &pipelineCacheInfo, nullptr, &m_pipelineCache);
	}

	void createPipelineLayouts() {
		createGraphicPipelineLayouts();
		createComputePipelineLayouts();
	}

	void recreatePipelines() {
        vkDeviceWaitIdle(device);

        // vkDestroyPipeline(device, m_graphicPipelines.snowflake, nullptr);
        vkDestroyPipeline(device, m_graphicPipelines.candles, nullptr);
        vkDestroyPipeline(device, m_graphicPipelines.bloom.vertical, nullptr);
        vkDestroyPipeline(device, m_graphicPipelines.bloom.horizontal, nullptr);
        vkDestroyPipeline(device, m_graphicPipelines.combine, nullptr);
        vkDestroyPipeline(device, m_computePipeline, nullptr);

        createGraphicPipelines();
		createComputePipelines();
	}

	void createPipelines() {
        createGraphicPipelines();
		createComputePipelines();
	}

	void createGraphicPipelineLayouts() {
		{
			// candles
			VkPushConstantRange pushConstant{};
			pushConstant.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
			pushConstant.size = sizeof(GraphicPushConstant);
			pushConstant.offset = 0;

			VkDescriptorSetLayout layouts[2] = {m_graphicDescriptorSetLayouts.candles.tranformUniform, m_graphicDescriptorSetLayouts.candles.meshMaterial};

			VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
			pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			pipelineLayoutInfo.setLayoutCount = 2;
			pipelineLayoutInfo.pSetLayouts = layouts;
			pipelineLayoutInfo.pushConstantRangeCount = 1;
			pipelineLayoutInfo.pPushConstantRanges = &pushConstant;

			if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &m_graphicPipelineLayouts.candles) != VK_SUCCESS) {
				throw std::runtime_error("failed to create graphic pipeline layout!");
			}
		}

		// bloom
		{
			VkDescriptorSetLayout layouts[] = {m_graphicDescriptorSetLayouts.bloom};

			VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
			pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			pipelineLayoutInfo.setLayoutCount = 1;
			pipelineLayoutInfo.pSetLayouts = layouts;
			pipelineLayoutInfo.pushConstantRangeCount = 0;
			pipelineLayoutInfo.pPushConstantRanges = nullptr;

			CHECK_VK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &m_graphicPipelineLayouts.bloom)
				, "fail to create bloom pipeline layout");
		}

		// combine
		{
			VkPushConstantRange pushConstant{};
			pushConstant.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
			pushConstant.size = sizeof(GraphicPushConstant);
			pushConstant.offset = 0;

			VkDescriptorSetLayout layouts[] = {m_graphicDescriptorSetLayouts.combine};

			VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
			pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			pipelineLayoutInfo.setLayoutCount = 1;
			pipelineLayoutInfo.pSetLayouts = layouts;
			pipelineLayoutInfo.pushConstantRangeCount = 1;
			pipelineLayoutInfo.pPushConstantRanges = &pushConstant;

			CHECK_VK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &m_graphicPipelineLayouts.combine)
				, "fail to create bloom pipeline layout");
		}
	}

    void createGraphicPipelines() {
		// snowflake pipeline
		// {
		// 	VkPipelineVertexInputDivisorStateCreateInfoEXT divisor{};
		// 	divisor.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT;

		// 	VkVertexInputBindingDivisorDescriptionEXT divisorDescription{};
		// 	divisorDescription.binding = 4;
		// 	divisorDescription.divisor = 1;

		// 	divisor.vertexBindingDivisorCount = 1;
		// 	divisor.pVertexBindingDivisors = &divisorDescription;

		// 	vertexInputInfo.pNext = &divisor;

		// 	VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		// 	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		// 	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		// 	inputAssembly.primitiveRestartEnable = VK_FALSE;

		// 	VkPipelineViewportStateCreateInfo viewportState{};
		// 	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		// 	viewportState.viewportCount = 1;
		// 	viewportState.scissorCount = 1;

		// 	VkPipelineRasterizationStateCreateInfo rasterizer{};
		// 	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		// 	rasterizer.depthClampEnable = VK_FALSE;
		// 	rasterizer.rasterizerDiscardEnable = VK_FALSE;
		// 	rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		// 	rasterizer.lineWidth = 1.0f;
		// 	rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		// 	rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		// 	rasterizer.depthBiasEnable = VK_FALSE;

		// 	VkPipelineMultisampleStateCreateInfo multisampling{};
		// 	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		// 	multisampling.sampleShadingEnable = VK_FALSE;
		// 	multisampling.rasterizationSamples = msaaSamples;

		// 	VkPipelineDepthStencilStateCreateInfo depthStencil{};
		// 	depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		// 	depthStencil.depthTestEnable = VK_TRUE;
		// 	depthStencil.depthWriteEnable = VK_TRUE;
		// 	depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
		// 	depthStencil.depthBoundsTestEnable = VK_FALSE;
		// 	depthStencil.stencilTestEnable = VK_FALSE;

		// 	VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		// 	colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		// 	colorBlendAttachment.blendEnable = VK_TRUE;
		// 	colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
		// 	colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		// 	colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		// 	colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_MIN;
		// 	colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		// 	colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;

		// 	// 2 attachments for 2 framebuffer attachments
		// 	std::array<VkPipelineColorBlendAttachmentState, 2> blendAttachments{colorBlendAttachment, colorBlendAttachment};

		// 	VkPipelineColorBlendStateCreateInfo colorBlending{};
		// 	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		// 	colorBlending.logicOpEnable = VK_FALSE;
		// 	colorBlending.logicOp = VK_LOGIC_OP_COPY;
		// 	colorBlending.attachmentCount = static_cast<uint32_t>(blendAttachments.size());
		// 	colorBlending.pAttachments = blendAttachments.data();
		// 	// colorBlending.blendConstants[0] = 0.0f;
		// 	// colorBlending.blendConstants[1] = 0.0f;
		// 	// colorBlending.blendConstants[2] = 0.0f;
		// 	// colorBlending.blendConstants[3] = 0.0f;

		// 	std::vector<VkDynamicState> dynamicStates = {
		// 		VK_DYNAMIC_STATE_VIEWPORT,
		// 		VK_DYNAMIC_STATE_SCISSOR
		// 	};
		// 	VkPipelineDynamicStateCreateInfo dynamicState{};
		// 	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		// 	dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		// 	dynamicState.pDynamicStates = dynamicStates.data();

		// 	auto vertShaderCode = readFile("../../src/shaders/candles.vert.spv");
		// 	auto fragShaderCode = readFile("../../src/shaders/candles.frag.spv");

		// 	VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		// 	VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		// 	struct SpecializationConstant{
		// 		alignas(4) bool useTexture{true};
		// 	}specConstant;

		// 	std::array<VkSpecializationMapEntry, 1> specEntries;
		// 	specEntries[0].constantID = 0;
		// 	specEntries[0].offset = 0;
		// 	specEntries[0].size = sizeof(SpecializationConstant);

		// 	VkSpecializationInfo specInfo{};
		// 	specInfo.dataSize = sizeof(SpecializationConstant);
		// 	specInfo.mapEntryCount = static_cast<uint32_t>(specEntries.size());
		// 	specInfo.pMapEntries = specEntries.data();
		// 	specInfo.pData = &specConstant;

		// 	VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		// 	vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		// 	vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		// 	vertShaderStageInfo.module = vertShaderModule;
		// 	vertShaderStageInfo.pName = "main";

		// 	VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		// 	fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		// 	fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		// 	fragShaderStageInfo.module = fragShaderModule;
		// 	fragShaderStageInfo.pName = "main";
		// 	fragShaderStageInfo.pSpecializationInfo = &specInfo;

		// 	VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

		// 	VkGraphicsPipelineCreateInfo pipelineInfo{};
		// 	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		// 	pipelineInfo.stageCount = 2;
		// 	pipelineInfo.pStages = shaderStages;
		// 	pipelineInfo.pVertexInputState = &vertexInputInfo;
		// 	pipelineInfo.pInputAssemblyState = &inputAssembly;
		// 	pipelineInfo.pViewportState = &viewportState;
		// 	pipelineInfo.pRasterizationState = &rasterizer;
		// 	pipelineInfo.pMultisampleState = &multisampling;
		// 	pipelineInfo.pDepthStencilState = &depthStencil;
		// 	pipelineInfo.pColorBlendState = &colorBlending;
		// 	pipelineInfo.pDynamicState = &dynamicState;
		// 	pipelineInfo.layout = m_graphicPipelineLayout;
		// 	pipelineInfo.renderPass = m_renderPasses.base;
		// 	pipelineInfo.subpass = 0;
		// 	pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

		// 	if(Object::CANDLE == Object::SNOWFLAKE)
		// 		specConstant.useTexture = false;

		// 	if (vkCreateGraphicsPipelines(device, m_pipelineCache, 1, &pipelineInfo, nullptr, &m_graphicPipelines.candles) != VK_SUCCESS) {
		// 		throw std::runtime_error("failed to create graphics pipeline!");
		// 	}

		// 	vkDestroyShaderModule(device, fragShaderModule, nullptr);
		// 	vkDestroyShaderModule(device, vertShaderModule, nullptr);
		// }
			// candles pipeline
		{
			tinygltf::Model& model = m_model[Object::CANDLE];
			VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
			vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

			auto vertexDef = getModelVertexDescriptions(Object::CANDLE);

			// instance attribute is the same for snowflake and candle
			auto instanceBindingDescription = VertexInstance::getBindingDescription();
			auto instanceAttributeDescription = VertexInstance::getAttributeDescriptions();

			std::array<VkVertexInputBindingDescription, 5> totalBindingDescriptions = 
				{vertexDef["POSITION"].first, vertexDef["NORMAL"].first, vertexDef["TANGENT"].first, vertexDef["TEXCOORD_0"].first, instanceBindingDescription};
			std::array<VkVertexInputAttributeDescription, 5> totalAttributeDescriptions = 
				{vertexDef["POSITION"].second, vertexDef["NORMAL"].second, vertexDef["TANGENT"].second, vertexDef["TEXCOORD_0"].second, instanceAttributeDescription[0]};

			vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(totalBindingDescriptions.size());
			vertexInputInfo.pVertexBindingDescriptions = totalBindingDescriptions.data();
			vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(totalAttributeDescriptions.size());
			vertexInputInfo.pVertexAttributeDescriptions = totalAttributeDescriptions.data();

			VkPipelineVertexInputDivisorStateCreateInfoEXT divisor{};
			divisor.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT;

			VkVertexInputBindingDivisorDescriptionEXT divisorDescription{};
			divisorDescription.binding = 4;
			divisorDescription.divisor = 1;

			divisor.vertexBindingDivisorCount = 1;
			divisor.pVertexBindingDivisors = &divisorDescription;

			vertexInputInfo.pNext = &divisor;

			VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
			inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
			inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
			inputAssembly.primitiveRestartEnable = VK_FALSE;

			VkPipelineViewportStateCreateInfo viewportState{};
			viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
			viewportState.viewportCount = 1;
			viewportState.scissorCount = 1;

			VkPipelineRasterizationStateCreateInfo rasterizer{};
			rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
			rasterizer.depthClampEnable = VK_FALSE;
			rasterizer.rasterizerDiscardEnable = VK_FALSE;
			rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
			rasterizer.lineWidth = 1.0f;
			rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
			rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
			rasterizer.depthBiasEnable = VK_FALSE;

			VkPipelineMultisampleStateCreateInfo multisampling{};
			multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
			multisampling.sampleShadingEnable = VK_FALSE;
			multisampling.rasterizationSamples = msaaSamples;

			VkPipelineDepthStencilStateCreateInfo depthStencil{};
			depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
			depthStencil.depthTestEnable = VK_TRUE;
			depthStencil.depthWriteEnable = VK_TRUE;
			depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
			depthStencil.depthBoundsTestEnable = VK_FALSE;
			depthStencil.stencilTestEnable = VK_FALSE;

			VkPipelineColorBlendAttachmentState colorBlendAttachment{};
			colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
			colorBlendAttachment.blendEnable = VK_TRUE;
			colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
			colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
			colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
			colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_MIN;
			colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
			colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;

			// 2 attachments for 2 framebuffer attachments
			std::array<VkPipelineColorBlendAttachmentState, 2> blendAttachments{colorBlendAttachment, colorBlendAttachment};

			VkPipelineColorBlendStateCreateInfo colorBlending{};
			colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
			colorBlending.logicOpEnable = VK_FALSE;
			colorBlending.logicOp = VK_LOGIC_OP_COPY;
			colorBlending.attachmentCount = static_cast<uint32_t>(blendAttachments.size());
			colorBlending.pAttachments = blendAttachments.data();
			// colorBlending.blendConstants[0] = 0.0f;
			// colorBlending.blendConstants[1] = 0.0f;
			// colorBlending.blendConstants[2] = 0.0f;
			// colorBlending.blendConstants[3] = 0.0f;

			std::vector<VkDynamicState> dynamicStates = {
				VK_DYNAMIC_STATE_VIEWPORT,
				VK_DYNAMIC_STATE_SCISSOR
			};
			VkPipelineDynamicStateCreateInfo dynamicState{};
			dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
			dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
			dynamicState.pDynamicStates = dynamicStates.data();

			auto vertShaderCode = readFile("../../src/shaders/candles.vert.spv");
			auto fragShaderCode = readFile("../../src/shaders/candles.frag.spv");

			VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
			VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

			struct SpecializationConstant{
				alignas(4) bool useTexture{true};
			}specConstant;

			std::array<VkSpecializationMapEntry, 1> specEntries;
			specEntries[0].constantID = 0;
			specEntries[0].offset = 0;
			specEntries[0].size = sizeof(SpecializationConstant);

			VkSpecializationInfo specInfo{};
			specInfo.mapEntryCount = static_cast<uint32_t>(specEntries.size());
			specInfo.pMapEntries = specEntries.data();
			specInfo.dataSize = sizeof(SpecializationConstant);
			specInfo.pData = &specConstant;

			VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
			vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
			vertShaderStageInfo.module = vertShaderModule;
			vertShaderStageInfo.pName = "main";

			VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
			fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
			fragShaderStageInfo.module = fragShaderModule;
			fragShaderStageInfo.pName = "main";
			fragShaderStageInfo.pSpecializationInfo = &specInfo;

			VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

			VkGraphicsPipelineCreateInfo pipelineInfo{};
			pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
			pipelineInfo.stageCount = 2;
			pipelineInfo.pStages = shaderStages;
			pipelineInfo.pVertexInputState = &vertexInputInfo;
			pipelineInfo.pInputAssemblyState = &inputAssembly;
			pipelineInfo.pViewportState = &viewportState;
			pipelineInfo.pRasterizationState = &rasterizer;
			pipelineInfo.pMultisampleState = &multisampling;
			pipelineInfo.pDepthStencilState = &depthStencil;
			pipelineInfo.pColorBlendState = &colorBlending;
			pipelineInfo.pDynamicState = &dynamicState;
			pipelineInfo.layout = m_graphicPipelineLayouts.candles;
			pipelineInfo.renderPass = m_renderPasses.base;
			pipelineInfo.subpass = 0;
			pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

			if(Object::CANDLE == Object::SNOWFLAKE)
				specConstant.useTexture = false;

			if (vkCreateGraphicsPipelines(device, m_pipelineCache, 1, &pipelineInfo, nullptr, &m_graphicPipelines.candles) != VK_SUCCESS) {
				throw std::runtime_error("failed to create graphics pipeline!");
			}

			vkDestroyShaderModule(device, fragShaderModule, nullptr);
			vkDestroyShaderModule(device, vertShaderModule, nullptr);
		}

		// bloom & combine pipeline
		{ 
			// shared states
			VkVertexInputBindingDescription vertexBindings{};
			vertexBindings.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
			vertexBindings.binding = 0;
			vertexBindings.stride = 5 * sizeof(float);

			VkVertexInputAttributeDescription vertexAttributePos{};
			vertexAttributePos.binding = 0;
			vertexAttributePos.location = 0;
			vertexAttributePos.offset = 0;
			vertexAttributePos.format = VK_FORMAT_R32G32B32_SFLOAT;

			VkVertexInputAttributeDescription vertexAttributeTex{};
			vertexAttributeTex.binding = 0;
			vertexAttributeTex.location = 1;
			vertexAttributeTex.offset = 12;
			vertexAttributeTex.format = VK_FORMAT_R32G32_SFLOAT;

			std::array<VkVertexInputAttributeDescription, 2> vertexAttribute{vertexAttributePos, vertexAttributeTex};

			VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
			vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
			vertexInputInfo.vertexBindingDescriptionCount = 1;
			vertexInputInfo.pVertexBindingDescriptions = &vertexBindings;
			vertexInputInfo.vertexAttributeDescriptionCount = vertexAttribute.size();
			vertexInputInfo.pVertexAttributeDescriptions = vertexAttribute.data();

			VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo{};
			inputAssemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
			// inputAssemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
			inputAssemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
			inputAssemblyInfo.primitiveRestartEnable = VK_FALSE;

			VkPipelineViewportStateCreateInfo viewportInfo{};
			viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
			viewportInfo.scissorCount = 1;
			viewportInfo.viewportCount = 1;
			// ignore this since viewport state is dynamic
			// viewportInfo.pScissors
			// viewportInfo.pViewports

			VkPipelineRasterizationStateCreateInfo rasterizationInfo{};
			rasterizationInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
			rasterizationInfo.rasterizerDiscardEnable = VK_FALSE;
			rasterizationInfo.lineWidth = 1.0f;
			rasterizationInfo.cullMode = VK_CULL_MODE_NONE;
			rasterizationInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
			rasterizationInfo.polygonMode = VK_POLYGON_MODE_FILL;
			rasterizationInfo.depthBiasEnable = VK_FALSE;
			rasterizationInfo.depthClampEnable = VK_FALSE;

			VkPipelineMultisampleStateCreateInfo multisampleInfo{};
			multisampleInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
			multisampleInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
			multisampleInfo.sampleShadingEnable = VK_FALSE;

			VkPipelineDepthStencilStateCreateInfo depthStencilInfo{};
			depthStencilInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
			depthStencilInfo.depthTestEnable = VK_FALSE;
			depthStencilInfo.depthWriteEnable = VK_FALSE;
			depthStencilInfo.stencilTestEnable = VK_FALSE;
			depthStencilInfo.depthBoundsTestEnable = VK_FALSE;

			VkPipelineColorBlendAttachmentState blendAttachmentInfo{};
			// WTF: we have to set value for this `colorWriteMask` flag for some reason???
			blendAttachmentInfo.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT;
			blendAttachmentInfo.blendEnable = VK_FALSE;

			VkPipelineColorBlendStateCreateInfo blendInfo{};
			blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
			blendInfo.attachmentCount = 1;
			blendInfo.pAttachments = &blendAttachmentInfo;

			std::vector<VkDynamicState> dynamicStates{
				VK_DYNAMIC_STATE_VIEWPORT,
				VK_DYNAMIC_STATE_SCISSOR
			};

			VkPipelineDynamicStateCreateInfo dynamicInfo{};
			dynamicInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
			dynamicInfo.dynamicStateCount = dynamicStates.size();
			dynamicInfo.pDynamicStates = dynamicStates.data();

			auto vertShaderCode = readFile("../../src/shaders/bloom.vert.spv");
			auto fragShaderCode = readFile("../../src/shaders/bloom.frag.spv");

			VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
			VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

			VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
			vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
			vertShaderStageInfo.module = vertShaderModule;
			vertShaderStageInfo.pName = "main";

			struct SpecializationConstant{
				alignas(4) int isHorizontal{0};
			} specConstant;

			std::array<VkSpecializationMapEntry, 1> specEntries;
			specEntries[0].constantID = 0;
			specEntries[0].offset = 0;
			specEntries[0].size = sizeof(SpecializationConstant);

			VkSpecializationInfo specInfo{};
			specInfo.mapEntryCount = specEntries.size();
			specInfo.pMapEntries = specEntries.data();
			specInfo.dataSize = sizeof(SpecializationConstant);
			specInfo.pData = &specConstant;

			VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
			fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
			fragShaderStageInfo.module = fragShaderModule;
			fragShaderStageInfo.pName = "main";
			fragShaderStageInfo.pSpecializationInfo = &specInfo;

			VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

			VkGraphicsPipelineCreateInfo pipelineInfo{};
			pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
			pipelineInfo.pVertexInputState = &vertexInputInfo;
			pipelineInfo.pInputAssemblyState = &inputAssemblyInfo;
			pipelineInfo.pViewportState = &viewportInfo;
			pipelineInfo.pColorBlendState = &blendInfo;
			pipelineInfo.pMultisampleState = &multisampleInfo;
			pipelineInfo.pDepthStencilState = &depthStencilInfo;
			pipelineInfo.pRasterizationState = &rasterizationInfo;
			pipelineInfo.stageCount = 2;
			pipelineInfo.pStages = shaderStages;
			pipelineInfo.pDynamicState = &dynamicInfo;
			pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
			pipelineInfo.renderPass = m_renderPasses.bloom;
			pipelineInfo.subpass = 0;
			pipelineInfo.layout = m_graphicPipelineLayouts.bloom;

			// vertical bloom pass
			{
				specConstant.isHorizontal = 0;
				CHECK_VK_RESULT(vkCreateGraphicsPipelines(device, m_pipelineCache, 1, &pipelineInfo, nullptr, &m_graphicPipelines.bloom.vertical)
					   , "fail to create bloom pipeline");
			}

			// horizontal bloom pass
			{
				specConstant.isHorizontal = 1;
				CHECK_VK_RESULT(vkCreateGraphicsPipelines(device, m_pipelineCache, 1, &pipelineInfo, nullptr, &m_graphicPipelines.bloom.horizontal)
					   , "fail to create bloom pipeline");
			}

			// combine pass
			vkDestroyShaderModule(device, vertShaderModule, nullptr);
			vkDestroyShaderModule(device, fragShaderModule, nullptr);

			auto combineVertShaderCode = readFile("../../src/shaders/combine.vert.spv");
			auto combineFragShaderCode = readFile("../../src/shaders/combine.frag.spv");

			VkShaderModule combineVertShaderModule = createShaderModule(combineVertShaderCode);
			VkShaderModule combineFragShaderModule = createShaderModule(combineFragShaderCode);

			VkPipelineShaderStageCreateInfo combineVertShaderStageInfo{};
			combineVertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			combineVertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
			combineVertShaderStageInfo.module = combineVertShaderModule;
			combineVertShaderStageInfo.pName = "main";

			VkPipelineShaderStageCreateInfo combineFragShaderStageInfo{};
			combineFragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			combineFragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
			combineFragShaderStageInfo.module = combineFragShaderModule;
			combineFragShaderStageInfo.pName = "main";

			VkPipelineShaderStageCreateInfo combineShaderStages[] = {combineVertShaderStageInfo, combineFragShaderStageInfo};

			pipelineInfo.stageCount = 2;
			pipelineInfo.pStages = combineShaderStages;
			pipelineInfo.renderPass = m_renderPasses.combine;
			pipelineInfo.subpass = 0;
			pipelineInfo.layout = m_graphicPipelineLayouts.combine;

			CHECK_VK_RESULT(vkCreateGraphicsPipelines(device, m_pipelineCache, 1, &pipelineInfo, nullptr, &m_graphicPipelines.combine)
				   , "fail to create combine pipeline");

			vkDestroyShaderModule(device, combineVertShaderModule, nullptr);
			vkDestroyShaderModule(device, combineFragShaderModule, nullptr);
		}
    }

	void createComputePipelineLayouts() {
		VkPushConstantRange pushConstant{};
		pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		pushConstant.size = sizeof(ComputePushConstant);
		pushConstant.offset = 0;

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &m_computeDescriptorSetLayout;
		pipelineLayoutInfo.pushConstantRangeCount = 1;
		pipelineLayoutInfo.pPushConstantRanges = &pushConstant;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &m_computePipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("failed to create compute pipeline layout!");
	}

	void createComputePipelines() {
        auto snowflakeCompShaderCode = readFile("../../src/shaders/snowflake.comp.spv");
		VkShaderModule computeShaderModule = createShaderModule(snowflakeCompShaderCode);

		VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
        computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        computeShaderStageInfo.module = computeShaderModule;
        computeShaderStageInfo.pName = "main";

		VkComputePipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		pipelineInfo.flags = 0;
		pipelineInfo.stage = computeShaderStageInfo;
		pipelineInfo.layout = m_computePipelineLayout;

		if (vkCreateComputePipelines(device, m_pipelineCache, 1, &pipelineInfo, nullptr, &m_computePipeline) != VK_SUCCESS)
            throw std::runtime_error("failed to create compute pipeline!");

        vkDestroyShaderModule(device, computeShaderModule, nullptr);
	}

    void createFramebuffers() {
		// base
		{
			m_frameBuffers.base.resize(swapChainImageViews.size());

			for (size_t i = 0; i < swapChainImageViews.size(); i++) {
				std::array<VkImageView, 5> attachments = {
					m_renderTargets.base.colorRT.view,
					m_renderTargets.base.bloomThresholdRT.view,
					m_renderTargets.base.depthRT.view,
					// swapChainImageViews[i],
					m_renderTargets.base.colorResRT.view,
					m_renderTargets.base.bloomThresholdResRT.view
				};

				VkFramebufferCreateInfo framebufferInfo{};
				framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
				framebufferInfo.renderPass = m_renderPasses.base;
				framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
				framebufferInfo.pAttachments = attachments.data();
				framebufferInfo.width = swapChainExtent.width;
				framebufferInfo.height = swapChainExtent.height;
				framebufferInfo.layers = 1;

				if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &m_frameBuffers.base[i]) != VK_SUCCESS) {
					throw std::runtime_error("failed to create framebuffer!");
				}
			}
		}

		// bloom
		{
			m_frameBuffers.bloom.horizontal.resize(swapChainImageViews.size());
			m_frameBuffers.bloom.vertical.resize(swapChainImageViews.size());

			for (size_t i = 0; i < swapChainImageViews.size(); i++) {
				std::array<VkImageView, 1> bloom1Attachments = {
					m_renderTargets.bloom1.view,
				};

				VkFramebufferCreateInfo framebufferInfo{};
				framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
				framebufferInfo.renderPass = m_renderPasses.bloom;
				framebufferInfo.attachmentCount = static_cast<uint32_t>(bloom1Attachments.size());
				framebufferInfo.pAttachments = bloom1Attachments.data();
				framebufferInfo.width = swapChainExtent.width;
				framebufferInfo.height = swapChainExtent.height;
				framebufferInfo.layers = 1;

				if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &m_frameBuffers.bloom.horizontal[i]) != VK_SUCCESS) {
					throw std::runtime_error("failed to create framebuffer!");
				}

				std::array<VkImageView, 1> bloom2Attachments = {
					m_renderTargets.bloom2.view,
				};
				framebufferInfo.pAttachments = bloom2Attachments.data();

				if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &m_frameBuffers.bloom.vertical[i]) != VK_SUCCESS) {
					throw std::runtime_error("failed to create framebuffer!");
				}
			}
		}

		{
			m_frameBuffers.combine.resize(swapChainImageViews.size());

			for (size_t i = 0; i < swapChainImageViews.size(); i++) {
				std::array<VkImageView, 1> attachments = {
					swapChainImageViews[i]
				};

				VkFramebufferCreateInfo framebufferInfo{};
				framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
				framebufferInfo.renderPass = m_renderPasses.combine;
				framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
				framebufferInfo.pAttachments = attachments.data();
				framebufferInfo.width = swapChainExtent.width;
				framebufferInfo.height = swapChainExtent.height;
				framebufferInfo.layers = 1;

				if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &m_frameBuffers.combine[i]) != VK_SUCCESS) {
					throw std::runtime_error("failed to create framebuffer!");
				}
			}
		}
    }

    void createCommandPools() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo graphicPoolInfo{};
        graphicPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        graphicPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        graphicPoolInfo.queueFamilyIndex = queueFamilyIndices.graphicFamily.value();

        if (vkCreateCommandPool(device, &graphicPoolInfo, nullptr, &m_graphicCommandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics command pool!");
        }

		VkCommandPoolCreateInfo computePoolInfo{};
		computePoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		computePoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		computePoolInfo.queueFamilyIndex = queueFamilyIndices.computeFamily.value();

		if (vkCreateCommandPool(device, &computePoolInfo, nullptr, &m_computeCommandPool) != VK_SUCCESS){
			throw std::runtime_error("failed to create compute command pool!");
		}
    }

    void createRenderTargets() {
        VkFormat colorFormat = swapChainImageFormat;

		// for base framebuffer
        createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, colorFormat, 
					VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, 
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_renderTargets.base.colorRT.image, m_renderTargets.base.colorRT.allocation);
        m_renderTargets.base.colorRT.view = createImageView(m_renderTargets.base.colorRT.image, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);

        createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, colorFormat, 
					VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, 
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_renderTargets.base.bloomThresholdRT.image, m_renderTargets.base.bloomThresholdRT.allocation);
        m_renderTargets.base.bloomThresholdRT.view = createImageView(m_renderTargets.base.bloomThresholdRT.image, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);

        VkFormat depthFormat = findDepthFormat();

        createImage(swapChainExtent.width, swapChainExtent.height, 1, msaaSamples, depthFormat,
					VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, 
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_renderTargets.base.depthRT.image, m_renderTargets.base.depthRT.allocation);
        m_renderTargets.base.depthRT.view = createImageView(m_renderTargets.base.depthRT.image, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);

        createImage(swapChainExtent.width, swapChainExtent.height, 1, VK_SAMPLE_COUNT_1_BIT, colorFormat, 
					VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_renderTargets.base.colorResRT.image, m_renderTargets.base.colorResRT.allocation);
        m_renderTargets.base.colorResRT.view = createImageView(m_renderTargets.base.colorResRT.image, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);

        createImage(swapChainExtent.width, swapChainExtent.height, 1, VK_SAMPLE_COUNT_1_BIT, colorFormat, 
					VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_renderTargets.base.bloomThresholdResRT.image, m_renderTargets.base.bloomThresholdResRT.allocation);
        m_renderTargets.base.bloomThresholdResRT.view = createImageView(m_renderTargets.base.bloomThresholdResRT.image, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);

		// for bloom framebuffer
        createImage(swapChainExtent.width, swapChainExtent.height, 1, VK_SAMPLE_COUNT_1_BIT, colorFormat, 
					VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_renderTargets.bloom1.image, m_renderTargets.bloom1.allocation);
        m_renderTargets.bloom1.view = createImageView(m_renderTargets.bloom1.image, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);

        createImage(swapChainExtent.width, swapChainExtent.height, 1, VK_SAMPLE_COUNT_1_BIT, colorFormat, 
					VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_renderTargets.bloom2.image, m_renderTargets.bloom2.allocation);
        m_renderTargets.bloom2.view = createImageView(m_renderTargets.bloom2.image, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
    }

    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
        for (VkFormat format : candidates) {
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

            if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
                return format;
            } else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }

        throw std::runtime_error("failed to find supported format!");
    }

    VkFormat findDepthFormat() {
        return findSupportedFormat(
            {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
            VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
        );
    }

    bool hasStencilComponent(VkFormat format) {
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
    }

    void createModelImages() {
		for (unsigned int i = 0; i < Object::COUNT; i++){
			Object objIdx = static_cast<Object>(i);
			tinygltf::Model& model = m_model[objIdx];

			MeshImages meshImages{};
			if (model.images.empty()) {
				continue;
			}

			// WARNING: recreate vk handles even if resource is the same.
			int meshIdx = 0;
			for (auto& mesh : model.meshes) {
				const tinygltf::Material& material = model.materials[mesh.primitives[0].material];

				const tinygltf::Texture& baseTexture = model.textures[material.pbrMetallicRoughness.baseColorTexture.index];
				 meshImages.baseImage = createModelImageFromGltf(objIdx, baseTexture, true);

				if (material.normalTexture.index == -1) {
					// add a dummy image
					meshImages.normalImage = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
					std::cout << "No normal mapping image for this mesh of model" << std::endl;
				}
				else {
					const tinygltf::Texture& normalTexture = model.textures[material.normalTexture.index];
					meshImages.normalImage = createModelImageFromGltf(objIdx, normalTexture, true);
				}

				if (material.emissiveTexture.index == -1) {
					// add a dummy image
					meshImages.emissiveImage = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
					std::cout << "No emissive mapping image for this mesh of model" << std::endl;
				}
				else {
					const tinygltf::Texture& emissiveTexture = model.textures[material.emissiveTexture.index];
					meshImages.emissiveImage = createModelImageFromGltf(objIdx, emissiveTexture, true);
				}

				m_modelImages[objIdx].push_back(meshImages);
			}
		}
    }

	Image createModelImageFromGltf(Object objIdx, const tinygltf::Texture& tex, bool isMipmap) {
		tinygltf::Image image = m_model[objIdx].images[tex.source];

		int texWidth = image.width;
		int texHeight = image.height;
		int texChannels = image.component;
		
		VkDeviceSize imageSize = texWidth * texHeight * 4;
		mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

		unsigned char* pixels = image.image.data();
		if (!pixels) {
			throw std::runtime_error("failed to load texture image!");
		}

		VkBuffer stagingBuffer{};
		VmaAllocation stagingBufferAlloc{};
		VkImage textureImage{};
		VmaAllocation textureImageAlloc{};
		createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferAlloc);

		void* data;
		vmaMapMemory(m_allocator, stagingBufferAlloc, &data);
			memcpy(data, pixels, static_cast<size_t>(imageSize));
		vmaUnmapMemory(m_allocator, stagingBufferAlloc);

		// stbi_image_free(pixels);

		createImage(texWidth, texHeight, mipLevels, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, 
			VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
			textureImage, textureImageAlloc);

		transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
		copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));

		//transitioned to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL while generating mipmaps
		if (isMipmap)
			generateMipmaps(textureImage, VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight, mipLevels);
		else
			transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, mipLevels);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		//vkFreeMemory(device, stagingBufferMemory, nullptr);
		vmaFreeMemory(m_allocator, stagingBufferAlloc);

		VkImageView textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
		return Image{textureImage, textureImageAlloc, textureImageView};
	}

    void generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels) {
        // Check if image format supports linear blitting
        VkFormatProperties formatProperties;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, imageFormat, &formatProperties);

        if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
            throw std::runtime_error("texture image format does not support linear blitting!");
        }

        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.image = image;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.subresourceRange.levelCount = 1;

        int32_t mipWidth = texWidth;
        int32_t mipHeight = texHeight;

        for (uint32_t i = 1; i < mipLevels; i++) {
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

            vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                0, nullptr,
                0, nullptr,
                1, &barrier);

            VkImageBlit blit{};
            blit.srcOffsets[0] = {0, 0, 0};
            blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
            blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.srcSubresource.mipLevel = i - 1;
            blit.srcSubresource.baseArrayLayer = 0;
            blit.srcSubresource.layerCount = 1;
            blit.dstOffsets[0] = {0, 0, 0};
            blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
            blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.dstSubresource.mipLevel = i;
            blit.dstSubresource.baseArrayLayer = 0;
            blit.dstSubresource.layerCount = 1;

            vkCmdBlitImage(commandBuffer,
                image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &blit,
                VK_FILTER_LINEAR);

            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                0, nullptr,
                0, nullptr,
                1, &barrier);

            if (mipWidth > 1) mipWidth /= 2;
            if (mipHeight > 1) mipHeight /= 2;
        }

        barrier.subresourceRange.baseMipLevel = mipLevels - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
            0, nullptr,
            0, nullptr,
            1, &barrier);

        endSingleTimeCommands(commandBuffer);
    }

    VkSampleCountFlagBits getMaxUsableSampleCount() {
        VkPhysicalDeviceProperties physicalDeviceProperties;
        vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

        VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
        if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
        if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
        if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
        if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
        if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
        if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

        return VK_SAMPLE_COUNT_1_BIT;
    }

    void createSamplers() {
		// TODO: these are a fake ass sampler bro
		// if(m_model[objIdx].samplers.empty()){
		// 	m_samplers[objIdx] = VK_NULL_HANDLE;
		// 	continue;
		// }
		
		// assume there is 1 texture sampler per model
		// TODO: set sampler according to gltf model
		// tinygltf::Sampler& modelSampler = m_model[objIdx].samplers[0];

		// candles sampler
		{
			VkSamplerCreateInfo samplerInfo{};
			samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
			samplerInfo.magFilter = VK_FILTER_LINEAR;
			samplerInfo.minFilter = VK_FILTER_LINEAR;
			samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
			samplerInfo.anisotropyEnable = VK_TRUE;
			samplerInfo.maxAnisotropy = m_physicalDeviceProperties.limits.maxSamplerAnisotropy;
			samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
			samplerInfo.unnormalizedCoordinates = VK_FALSE;
			samplerInfo.compareEnable = VK_FALSE;
			samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
			samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
			samplerInfo.minLod = 0.0f;
			samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
			samplerInfo.mipLodBias = 0.0f;

			if (vkCreateSampler(device, &samplerInfo, nullptr, &m_samplers.candles) != VK_SUCCESS) {
				throw std::runtime_error("failed to create texture sampler!");
			}
		}

		// quad sampler
		{
			VkSamplerCreateInfo samplerInfo{};
			samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
			samplerInfo.magFilter = VK_FILTER_LINEAR;
			samplerInfo.minFilter = VK_FILTER_LINEAR;
			// for bloom
			samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
			samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
			samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
			samplerInfo.anisotropyEnable = VK_TRUE;
			samplerInfo.maxAnisotropy = m_physicalDeviceProperties.limits.maxSamplerAnisotropy;
			samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
			samplerInfo.unnormalizedCoordinates = VK_FALSE;
			samplerInfo.compareEnable = VK_FALSE;
			samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
			samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

			if (vkCreateSampler(device, &samplerInfo, nullptr, &m_samplers.quad) != VK_SUCCESS) {
				throw std::runtime_error("failed to create texture sampler!");
			}
		}
    }

    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspectFlags;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = mipLevels;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        VkImageView imageView;
        if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image view!");
        }

        return imageView;
    }

    void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VmaAllocation& imageAlloc) {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = mipLevels;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.samples = numSamples;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VmaAllocationCreateInfo	imageAllocInfo{};
		imageAllocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		// imageAllocInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
		imageAllocInfo.priority = 1.0f;

		vmaCreateImage(m_allocator, &imageInfo, &imageAllocInfo, &image, &imageAlloc, nullptr);

        // if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        //     throw std::runtime_error("failed to create image!");
        // }

        // VkMemoryRequirements memRequirements;
        // vkGetImageMemoryRequirements(device, image, &memRequirements);

        // VkMemoryAllocateInfo allocInfo{};
        // allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        // allocInfo.allocationSize = memRequirements.size;
        // allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        // if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        //     throw std::runtime_error("failed to allocate image memory!");
        // }

        // vkBindImageMemory(device, image, imageMemory, 0);
    }

    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = mipLevels;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }

        vkCmdPipelineBarrier(
            commandBuffer,
            sourceStage, destinationStage,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );

        endSingleTimeCommands(commandBuffer);
    }

    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		{
			// TracyVkZone(tracyContext, commandBuffer, "transferBufferToImage");

			VkBufferImageCopy region{};
			region.bufferOffset = 0;
			region.bufferRowLength = 0;
			region.bufferImageHeight = 0;
			region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			region.imageSubresource.mipLevel = 0;
			region.imageSubresource.baseArrayLayer = 0;
			region.imageSubresource.layerCount = 1;
			region.imageOffset = {0, 0, 0};
			region.imageExtent = {
				width,
				height,
				1
			};

			vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
		}

        endSingleTimeCommands(commandBuffer);
    }

	std::map<std::string, std::pair<VkVertexInputBindingDescription, VkVertexInputAttributeDescription>> 
		getModelVertexDescriptions(Object obj) {
		tinygltf::Model& model = m_model[obj];
		std::map<std::string, std::pair<VkVertexInputBindingDescription, VkVertexInputAttributeDescription>> vertexDescription;
		unsigned int idx = 0;
		for (auto& attribute : model.meshes[0].primitives[0].attributes) {
			// WANRING: hardcode each buffer binding for each attribute
			VkVertexInputBindingDescription bindingDescription{};
			bindingDescription.binding = idx;
			bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

			VkVertexInputAttributeDescription attributeDescription{};

			attributeDescription.binding = idx;
			attributeDescription.location = idx;

			if (model.accessors[attribute.second].type == TINYGLTF_TYPE_VEC2) {
				attributeDescription.format = VK_FORMAT_R32G32_SFLOAT;
				bindingDescription.stride = 8;
			}
			else if (model.accessors[attribute.second].type == TINYGLTF_TYPE_VEC3) {
				attributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
				bindingDescription.stride = 12;
			}
			else if (model.accessors[attribute.second].type == TINYGLTF_TYPE_VEC4) {
				attributeDescription.format = VK_FORMAT_R32G32B32A32_SFLOAT;
				bindingDescription.stride = 16;
			}

			attributeDescription.offset = 0;

			vertexDescription[attribute.first] = {bindingDescription, attributeDescription};
			++idx;
		}
		return vertexDescription;
	}

	std::vector<int> findModelVertexBufferView(Object obj) {
		tinygltf::Model& model = m_model[obj];
		std::vector<int> vertexViewIdx;
		for (auto& mesh : model.meshes) {
			for (auto& primitive : mesh.primitives) {
				for (auto& attribute : primitive.attributes) {
					int bufferViewIdx = model.accessors[attribute.second].bufferView;
					if (std::find(vertexViewIdx.begin(), vertexViewIdx.end(), bufferViewIdx) == vertexViewIdx.end()){
						vertexViewIdx.push_back(bufferViewIdx);
					}
				}
			}
		}
		return vertexViewIdx;
	}

	std::vector<int> findModelIndexBufferView(Object obj) {
		tinygltf::Model& model = m_model[obj];
		std::vector<int> indexViewIdx;
		for (auto& mesh : model.meshes) {
			for (auto& primitive : mesh.primitives) {
				int bufferViewIdx = model.accessors[primitive.indices].bufferView;
				if(std::find(indexViewIdx.begin(), indexViewIdx.end(), bufferViewIdx) == indexViewIdx.end()) {
					indexViewIdx.push_back(bufferViewIdx);
				}
			}
		}
		return indexViewIdx;
	}

#if 1
	void computeAnimation(Object obj) {
		tinygltf::Model& model = m_model[obj];
		m_modelMeshAnimPositions[obj].resize(model.meshes.size());
		
		for (unsigned int meshIdx = 0; meshIdx < model.meshes.size(); meshIdx++) {
			auto weights = computeWeights(obj, meshIdx);
			if (!weights.empty()) {
				computeMorphTargets(obj, meshIdx, weights);
			}
			else {
				m_modelMeshAnimPositions[obj][meshIdx] = {};
			}
		}
	}

	std::vector<float> computeWeights(Object obj, unsigned int meshIdx) {
		// sample animation
		tinygltf::Model& model = m_model[obj];
		// WANRING: assume only 1 animation per object
		tinygltf::Animation& anims = model.animations[0];
		std::vector<tinygltf::AnimationChannel>& channels = anims.channels;
		std::vector<tinygltf::AnimationChannel>::iterator channel = std::find_if(channels.begin(), channels.end(), [&model, meshIdx](tinygltf::AnimationChannel& i_channel){
			auto& node = model.nodes[i_channel.target_node];
			if(node.mesh == meshIdx)
				return true;
			return false;
		});

		if (channel == channels.end())
			return {};

		auto& sampler = anims.samplers[channel->sampler];
		const tinygltf::Accessor& inputAcc = model.accessors[sampler.input];
		const tinygltf::BufferView& inputView = model.bufferViews[inputAcc.bufferView];
		const tinygltf::Buffer& inputBuffer = model.buffers[inputView.buffer];
		const unsigned char* pInData = inputBuffer.data.data() + inputView.byteOffset + inputAcc.byteOffset;	

		m_currentAnimTime += m_currentDeltaTime * CANDLE_ANIMATION_SPEED;
		if (m_currentAnimTime > inputAcc.maxValues[0])
			m_currentAnimTime -= inputAcc.maxValues[0];

		const float* inputWeights = reinterpret_cast<const float*>(pInData);
		unsigned int hi = 1;
		for (; hi < inputAcc.count; hi++) {
			if(inputWeights[hi] > m_currentAnimTime)
				break;
		}

		float ratio = (m_currentAnimTime - inputWeights[hi-1]) / (inputWeights[hi] - inputWeights[hi-1]);

		const tinygltf::Accessor& outputAcc = model.accessors[sampler.output];
		const tinygltf::BufferView& outputView = model.bufferViews[outputAcc.bufferView];
		const tinygltf::Buffer& outputBuffer = model.buffers[outputView.buffer];
		const unsigned char* pOutData = outputBuffer.data.data() + outputView.byteOffset + outputAcc.byteOffset;	

		const float* outputWeights = reinterpret_cast<const float*>(pOutData);
		const float* liWeights = outputWeights + (hi - 1) * inputAcc.count;
		const float* hiWeights = outputWeights + hi * inputAcc.count;
		
		std::vector<float> res{};
		res.resize(inputAcc.count);
		for (unsigned int i = 0; i < res.size(); i++) {
			res[i] = hiWeights[i] * ratio + liWeights[i] * (1 - ratio);
			// std::cout << "hiWeights[" << i << "]" << " = " << hiWeights[i] << "\n";
			// std::cout << "liWeights[" << i << "]" << " = " << liWeights[i] << "\n";
			// std::cout << "res[" << i << "]" << " = " << res[i] << "\n";
		}

		return res;
	}

	void computeMorphTargets(Object obj, unsigned int meshIdx, std::vector<float> weights) {
		tinygltf::Model& model = m_model[obj];
		auto& mesh = model.meshes[meshIdx];

		// set original position
		const tinygltf::Accessor& posAccessor = model.accessors[mesh.primitives[0].attributes["POSITION"]];
		const tinygltf::BufferView& posView = model.bufferViews[posAccessor.bufferView];
		const tinygltf::Buffer& posBuffer = model.buffers[posView.buffer];
		
		const unsigned char* pData = posBuffer.data.data() + posView.byteOffset + posAccessor.byteOffset;
		const glm::vec3* pPos = reinterpret_cast<const glm::vec3*>(pData);
		m_modelMeshAnimPositions[obj][meshIdx].resize(posAccessor.count);

		memcpy(m_modelMeshAnimPositions[obj][meshIdx].data(), pPos, posAccessor.count * sizeof(glm::vec3));

		// compute morph target
		auto& morphTargets = mesh.primitives[0].targets;
		for (unsigned int morphIdx = 0; morphIdx < morphTargets.size(); morphIdx++) {
			unsigned int morphAccessorIdx = morphTargets[morphIdx]["POSITION"];
			const tinygltf::Accessor& morphAccessor = model.accessors[morphAccessorIdx];
			const tinygltf::BufferView& bufferView = model.bufferViews[morphAccessor.bufferView];
			const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
			const unsigned char* pData = buffer.data.data() + bufferView.byteOffset + morphAccessor.byteOffset;
			const glm::vec3* pPos = reinterpret_cast<const glm::vec3*>(pData);

			for (unsigned int vertexIdx = 0; vertexIdx < morphAccessor.count; vertexIdx++){
				m_modelMeshAnimPositions[obj][meshIdx][vertexIdx] += pPos[vertexIdx] * weights[morphIdx];
			}
		}
	}
#endif

	void traverseModelNodesForTransform(Object obj, tinygltf::Node node, glm::mat4 mat) {
		tinygltf::Model& model = m_model[obj];
		if (node.children.empty()) {
			if (node.mesh != -1) {
				m_modelMeshTransforms[obj][node.mesh] = mat;
				// std::cout << "m_modelMeshTransforms at mesh " << node.mesh << " is:" << glm::to_string(mat) << "\n";
				return;
			}
		}
		
		if(!node.matrix.empty()) {
			glm::mat4 nodeMat = glm::make_mat4(node.matrix.data());
			// nodeMat = glm::transpose(nodeMat);
			mat = nodeMat * mat;
		} else if(!node.scale.empty() || !node.rotation.empty() || !node.translation.empty()) {
			if (!node.translation.empty()) {
				glm::vec3 translateVec = glm::make_vec3(node.translation.data());
				mat = glm::translate(mat, translateVec);
			}
			if (!node.scale.empty()) {
				glm::vec3 scaleVec = glm::make_vec3(node.scale.data());
				mat = glm::scale(mat, scaleVec);
			}
		}

		for (auto& childIdx : node.children) {
			tinygltf::Node& child = model.nodes[childIdx];
			traverseModelNodesForTransform(obj, child, mat);
		}
	}

    void loadModels() {
		// trace();
		std::cout << "start loading models \n";
		// loadObjectModel(Object::TOWER);
		// loadObjectModel(Object::SNOWFLAKE);

		for (unsigned int i = 0; i < Object::COUNT; i++){
			Object objIdx = static_cast<Object>(i);
			tinygltf::Model& model = m_model[objIdx];

			std::string path{};
			if (objIdx == Object::CANDLE)
				path = CANDLE_MODEL_PATH;
			else if (objIdx == Object::SNOWFLAKE)
				path = SNOWFLAKE_MODEL_PATH;

			loadGltfModel(model, path.c_str());

			m_modelMeshTransforms[objIdx].resize(model.meshes.size());
			traverseModelNodesForTransform(objIdx, model.nodes[0], glm::mat4(1.0f));
		}

		std::cout << "finish loading models \n";
	}

	void loadGltfModel(tinygltf::Model &model, const char *filename) {
		tinygltf::TinyGLTF loader;
		std::string err;
		std::string warn;

		bool res = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
		if (!warn.empty()) {
		std::cout << "WARN: " << warn << std::endl;
		}
		if (!err.empty()) {
		std::cout << "ERR: " << err << std::endl;
		}
		if (!res)
		std::cout << "Failed to load glTF: " << filename << std::endl;
		else
		std::cout << "Loaded glTF: " << filename << std::endl;
	}

    // void loadObjectModel(Object type) {
	// 	std::string modelPath;
	// 	if (type == Object::TOWER){
	// 		modelPath = TOWER_MODEL_PATH;
	// 	}
	// 	else {
	// 		modelPath = SNOWFLAKE_MODEL_PATH;
	// 	}

    //     tinyobj::attrib_t attrib;
    //     std::vector<tinyobj::shape_t> shapes;
    //     std::vector<tinyobj::material_t> materials;
    //     std::string warn, err;

    //     if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, modelPath.c_str())) {
    //         throw std::runtime_error(warn + err);
    //     }

    //     std::unordered_map<Vertex, uint32_t> uniqueVertices{};

    //     for (const auto& shape : shapes) {
    //         for (const auto& index : shape.mesh.indices) {
    //             Vertex vertex{};

    //             vertex.pos = {
    //                 attrib.vertices[3 * index.vertex_index + 0],
    //                 attrib.vertices[3 * index.vertex_index + 1],
    //                 attrib.vertices[3 * index.vertex_index + 2]
    //             };

    //             vertex.texCoord = {
    //                 attrib.texcoords[2 * index.texcoord_index + 0],
    //                 1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
    //             };

    //             vertex.color = {0.5f, 0.5f, 1.0f};

    //             if (uniqueVertices.count(vertex) == 0) {
	// 				if (type == Object::TOWER){
	// 					uniqueVertices[vertex] = static_cast<uint32_t>(m_vertexRaw[Object::TOWER].size());
	// 					m_vertexRaw[Object::TOWER].push_back(vertex);
	// 				}
	// 				else {
	// 					uniqueVertices[vertex] = static_cast<uint32_t>(m_vertexRaw[Object::SNOWFLAKE].size());
	// 					m_vertexRaw[Object::SNOWFLAKE].push_back(vertex);
	// 				}
    //             }

	// 			if (type == Object::TOWER)
	// 				m_indexRaw[Object::TOWER].push_back(uniqueVertices[vertex]);
	// 			else
	// 				m_indexRaw[Object::SNOWFLAKE].push_back(uniqueVertices[vertex]);
    //         }
    //     }

	// 	std::cout << "Size of tower index buffer: " << m_indexRaw[Object::TOWER].size() << std::endl;
	// 	std::cout << "Size of snowflake index buffer: " << m_indexRaw[Object::SNOWFLAKE].size() << std::endl;
	// }

	void loadInstanceData() {
		std::ifstream file("../../res/instance_position.csv");
		if(file.is_open()) {
			std::string line;
			while(std::getline(file, line)){
				VertexInstance vInstance{};
				unsigned int offset = 0;
				unsigned int space = line.find(" ");
				float x = stof(line.substr(offset, space - offset));
				offset = space + 1;
				space = line.find(" ", offset);
				float y = stof(line.substr(offset, space - offset));
				offset = space + 1;
				space = line.find(" ", offset);
				float z = stof(line.substr(offset, space - offset));
				vInstance.pos = {x, y, z};
				m_towerInstanceRaw.push_back(vInstance);
			}
		}
	}

	void createVertexBuffers() {
		{
			// Candles
			Object objIdx = Object::CANDLE;
			tinygltf::Model& model = m_model[objIdx];
			auto bufferViews = findModelVertexBufferView(objIdx);
			// use gltf buffer view index value as m_vertexBuffer index
			int maxVal = *std::max_element(bufferViews.begin(), bufferViews.end());
			m_vertexBuffers.candles.resize(maxVal + 1);

			for (auto viewIdx : bufferViews) {
				tinygltf::BufferView view = model.bufferViews[viewIdx];
				Buffer newBuffer{};
				VkBuffer stagingBuffer;
				VmaAllocation stagingBufferAlloc{};

				VkBuffer vertexBuffer;
				VmaAllocation vertexBufferAlloc{};

				createBuffer(view.byteLength, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferAlloc);
				void* data;
				vmaMapMemory(m_allocator, stagingBufferAlloc, &data);
					memcpy(data, &model.buffers[view.buffer].data.at(0) + view.byteOffset, view.byteLength);
				vmaUnmapMemory(m_allocator, stagingBufferAlloc);
				createBuffer(view.byteLength, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferAlloc);
				copyBuffer(stagingBuffer, vertexBuffer, view.byteLength);

				newBuffer.buffer = vertexBuffer;
				newBuffer.allocation = vertexBufferAlloc;

				vkDestroyBuffer(device, stagingBuffer, nullptr);
				vmaFreeMemory(m_allocator, stagingBufferAlloc);

				m_vertexBuffers.candles[viewIdx] = newBuffer;
			}
		}

		{
			// Quad
			VkBuffer stagingBuffer;
			VmaAllocation stagingBufferAlloc{};

			VkBuffer vertexBuffer;
			VmaAllocation vertexBufferAlloc{};

			m_vertexBuffers.quad.resize(1);

			int size = sizeof(quadListVertices);
			createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferAlloc);
			void* data;
			vmaMapMemory(m_allocator, stagingBufferAlloc, &data);
				memcpy(data, quadListVertices, size);
			vmaUnmapMemory(m_allocator, stagingBufferAlloc);
			createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferAlloc);
			copyBuffer(stagingBuffer, vertexBuffer, size);

			m_vertexBuffers.quad[0].buffer = vertexBuffer;
			m_vertexBuffers.quad[0].allocation = vertexBufferAlloc;

			vkDestroyBuffer(device, stagingBuffer, nullptr);
			vmaFreeMemory(m_allocator, stagingBufferAlloc);
		}
	}

	void createIndexBuffers() {
		Object objIdx = Object::CANDLE;
		tinygltf::Model& model = m_model[objIdx];
		auto bufferViews = findModelIndexBufferView(objIdx);

		// use gltf buffer view index value as m_vertexBuffer index
		int maxVal = *std::max_element(bufferViews.begin(), bufferViews.end());
		m_indexBuffers.candles.resize(maxVal + 1);

		for (auto viewIdx : bufferViews) {
			tinygltf::BufferView view = model.bufferViews[viewIdx];
			Buffer newBuffer{};
			VkBuffer stagingBuffer;
			VmaAllocation stagingBufferAloc{};
			VkBuffer indexBuffer;
			VmaAllocation indexBufferAloc{};

			createBuffer(view.byteLength, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferAloc);
			void* data;
			vmaMapMemory(m_allocator, stagingBufferAloc, &data);
				memcpy(data, &model.buffers[view.buffer].data.at(0) + view.byteOffset, view.byteLength);
			vmaUnmapMemory(m_allocator, stagingBufferAloc);
			createBuffer(view.byteLength, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferAloc);
			copyBuffer(stagingBuffer, indexBuffer, view.byteLength);

			newBuffer.buffer = indexBuffer;
			newBuffer.allocation = indexBufferAloc;

			vkDestroyBuffer(device, stagingBuffer, nullptr);
			vmaFreeMemory(m_allocator, stagingBufferAloc);
			m_indexBuffers.candles[viewIdx] = newBuffer;
		}
	}

	void createUniformBuffers(){
        createGraphicUniformBuffers();
		createComputeUniformBuffers();
	}

    void createGraphicUniformBuffers() {
		for (unsigned int i = 0; i < Object::COUNT; i++){
			Object objIdx = static_cast<Object>(i);
			tinygltf::Model& model = m_model[objIdx];

			// transform uniform
			{
				unsigned int meshCount = model.meshes.size();
				VkDeviceSize bufferSize = sizeof(TransformUniform) * meshCount;

				m_graphicUniformBuffers.transform[objIdx].resize(MAX_FRAMES_IN_FLIGHT);
				m_uniformBuffersAlloc.transform[objIdx].resize(MAX_FRAMES_IN_FLIGHT);
				m_graphicUniformBuffersMapped.transform[objIdx].resize(MAX_FRAMES_IN_FLIGHT);

				for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
					createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
						, m_graphicUniformBuffers.transform[objIdx][i], m_uniformBuffersAlloc.transform[objIdx][i]);

					vmaMapMemory(m_allocator, m_uniformBuffersAlloc.transform[objIdx][i], &m_graphicUniformBuffersMapped.transform[objIdx][i]);
				}
			}

			// lighting uniform
			{
				VkDeviceSize bufferSize = sizeof(LightingUniform);

				m_graphicUniformBuffers.lighting[objIdx].resize(MAX_FRAMES_IN_FLIGHT);
				m_uniformBuffersAlloc.lighting[objIdx].resize(MAX_FRAMES_IN_FLIGHT);
				m_graphicUniformBuffersMapped.lighting[objIdx].resize(MAX_FRAMES_IN_FLIGHT);

				for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
					createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
						, m_graphicUniformBuffers.lighting[objIdx][i], m_uniformBuffersAlloc.lighting[objIdx][i]);

					vmaMapMemory(m_allocator, m_uniformBuffersAlloc.lighting[objIdx][i], &m_graphicUniformBuffersMapped.lighting[objIdx][i]);
				}
			}
		}
    }

	void createAnimationBuffers() {
		for (unsigned int i = 0; i < Object::COUNT; i++){
			Object objIdx = static_cast<Object>(i);
			if (m_modelMeshAnimPositions.find(objIdx) == m_modelMeshAnimPositions.end()) {
				continue;
			}
			for (unsigned int meshIdx = 0; meshIdx < m_model[objIdx].meshes.size(); meshIdx++) {
				if (!m_modelMeshAnimPositions[objIdx][meshIdx].empty()){
					unsigned int size = m_modelMeshAnimPositions[objIdx][meshIdx].size() * sizeof(glm::vec3);
					createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
						  , m_animBuffer[objIdx][meshIdx], m_animBufferAlloc[objIdx][meshIdx]);
				}
			}
		}
	}

    void createComputeUniformBuffers() {
		m_vortexUniformBufferMapped = static_cast<void*>(new Vortex[MAX_VORTEX_COUNT]);

		VkDeviceSize bufferSize = sizeof(Vortex) * MAX_VORTEX_COUNT;
		createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
				, m_vortexUniformBuffer, m_vortexUniformBufferAlloc);
		vmaMapMemory(m_allocator, m_vortexUniformBufferAlloc, &m_vortexUniformBufferMapped);

		for(unsigned int i = 0; i < MAX_VORTEX_COUNT; i++){
			Vortex& vortex = ((Vortex*)m_vortexUniformBufferMapped)[i];
			vortex.pos.x = generateRandomFloat(-5.f, 5.f);
			vortex.pos.y = generateRandomFloat(-5.f, 5.f);
			vortex.pos.z = generateRandomFloat(-5.f, 5.f);
			vortex.height = generateRandomFloat(5.f, 10.f);

			s_basePhase[i] = generateRandomFloat(0.f, 3.14f);
			s_baseForce[i] = generateRandomFloat(2.f, 4.f);
			s_baseRadius[i] = generateRandomFloat(5.f, 15.f);
			vortex.force = s_baseForce[i];
			vortex.radius = s_baseRadius[i];
		}
	}

	void createInstanceBuffer() {
		VkDeviceSize bufferSize = sizeof(m_towerInstanceRaw[0]) * m_towerInstanceRaw.size();
		VkBuffer stagingBuffer{};
		VmaAllocation stagingBufferAlloc{};

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, stagingBuffer, stagingBufferAlloc);

		void* data;
		vmaMapMemory(m_allocator, stagingBufferAlloc, &data);
		memcpy(data, m_towerInstanceRaw.data(), bufferSize);
		vmaUnmapMemory(m_allocator, stagingBufferAlloc);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_towerInstanceBuffer, instanceBufferAlloc);

		copyBuffer(stagingBuffer, m_towerInstanceBuffer, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vmaFreeMemory(m_allocator, stagingBufferAlloc);
	}

	void createStorageBuffer() {
		void* data = static_cast<void*>(new Snowflake[SNOWFLAKE_COUNT]);
		VkDeviceSize bufferSize = sizeof(Snowflake) * SNOWFLAKE_COUNT;
		VkBuffer stagingBuffer{};
		VmaAllocation stagingBufferAlloc{};

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, stagingBuffer, stagingBufferAlloc);

		vmaMapMemory(m_allocator, stagingBufferAlloc, &data);
		// only work when set data when mapped like this
		for(unsigned int i = 0; i < SNOWFLAKE_COUNT; i++) {
			Snowflake& snow = ((Snowflake*)data)[i];
			snow.position.x = generateRandomFloat(-15.f, 15.f);
			snow.position.y = generateRandomFloat(-15.f, 15.f);
			snow.position.z = generateRandomFloat(-15.f, 15.f);
			snow.weight = generateRandomFloat(0.5f, 1.5f);
		}
		vmaUnmapMemory(m_allocator, stagingBufferAlloc);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
			   , m_storageBuffer, m_storageBufferAlloc);

		copyBuffer(stagingBuffer, m_storageBuffer, bufferSize);
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vmaFreeMemory(m_allocator, stagingBufferAlloc);
	}

	// Pool use for both graphic and compute descriptors
    void createDescriptorPool() {
		unsigned int materialCount = 0;
		for (unsigned int i = 0; i < Object::COUNT; i++){
			Object objIdx = static_cast<Object>(i);
			tinygltf::Model& model = m_model[objIdx];

			materialCount += model.meshes.size();
		}

		std::array<VkDescriptorPoolSize, 3> poolSizes{};
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * Object::COUNT) * 2 + 1; // for mesh transform + light uniform +1 for compute uniform
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT * materialCount) * 3 + /*for bloom */(1 + 2) * 2; // for base, normal and emissive texture + bloom texture
		poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSizes[2].descriptorCount = static_cast<uint32_t>(1);

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		// Object::COUNT for the number of uniform buffer
		// materialCount for the number of mesh material
		// 1 for compute descriptor set
		// 1 for imgui descriptor set
		// 4 for other passes with each frame in flight
		poolInfo.maxSets = static_cast<uint32_t>((materialCount + Object::COUNT) * MAX_FRAMES_IN_FLIGHT * 2) + 3 + 4; // for graphics and compute

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor pool!");
		}
    }

    void createDescriptorSets() {
		createGraphicDescriptorSets();
		createComputeDescriptorSets();
	}

	void createGraphicDescriptorSets(){
		{
			Object objIdx = Object::CANDLE;
			if (m_modelImages.find(objIdx) == m_modelImages.end()) {
				return;
			}

			tinygltf::Model& model = m_model[objIdx];
			m_graphicDescriptorSets.candles.meshMaterial.resize(model.meshes.size());
			int meshIdx = 0;

			for (auto& mesh : model.meshes) {
				std::array<VkDescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> 
					layouts = {m_graphicDescriptorSetLayouts.candles.meshMaterial, m_graphicDescriptorSetLayouts.candles.meshMaterial};
				VkDescriptorSetAllocateInfo allocInfo{};
				allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				allocInfo.descriptorPool = m_descriptorPool;
				allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
				allocInfo.pSetLayouts = layouts.data();

				if (vkAllocateDescriptorSets(device, &allocInfo, m_graphicDescriptorSets.candles.meshMaterial[meshIdx].data()) != VK_SUCCESS) {
					throw std::runtime_error("failed to allocate graphic descriptor sets!");
				}

				// TODO: does it need 2 descriptor here since we only read image?
				for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
					std::array<VkWriteDescriptorSet, 3> descriptorWrites{};

					VkDescriptorImageInfo imageInfo{};
					imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
					imageInfo.imageView = m_modelImages[objIdx][meshIdx].baseImage.view;
					// assume 1 sampler per object type
					imageInfo.sampler = m_samplers.candles;

					descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					descriptorWrites[0].dstSet = m_graphicDescriptorSets.candles.meshMaterial[meshIdx][i];
					descriptorWrites[0].dstBinding = 2;
					descriptorWrites[0].dstArrayElement = 0;
					descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
					descriptorWrites[0].descriptorCount = 1;
					descriptorWrites[0].pImageInfo = &imageInfo;
					
					VkDescriptorImageInfo normalImageInfo{};
					normalImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
					normalImageInfo.imageView = m_modelImages[objIdx][meshIdx].normalImage.view;
					// assume 1 sampler per object type
					normalImageInfo.sampler = m_samplers.candles;

					descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					descriptorWrites[1].dstSet = m_graphicDescriptorSets.candles.meshMaterial[meshIdx][i];
					descriptorWrites[1].dstBinding = 3;
					descriptorWrites[1].dstArrayElement = 0;
					descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
					descriptorWrites[1].descriptorCount = 1;
					descriptorWrites[1].pImageInfo = &normalImageInfo;

					VkDescriptorImageInfo emissiveImageInfo{};
					emissiveImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
					emissiveImageInfo.imageView = m_modelImages[objIdx][meshIdx].emissiveImage.view;
					// assume 1 sampler per object type
					emissiveImageInfo.sampler = m_samplers.candles;

					descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					descriptorWrites[2].dstSet = m_graphicDescriptorSets.candles.meshMaterial[meshIdx][i];
					descriptorWrites[2].dstBinding = 4;
					descriptorWrites[2].dstArrayElement = 0;
					descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
					descriptorWrites[2].descriptorCount = 1;
					descriptorWrites[2].pImageInfo = &emissiveImageInfo;

					vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
				}
				meshIdx++;
			}

			// allocate and update data for OBJECT UNIFORM tranform
			std::array<VkDescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> 
				layouts = {m_graphicDescriptorSetLayouts.candles.tranformUniform, m_graphicDescriptorSetLayouts.candles.tranformUniform};
			VkDescriptorSetAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			allocInfo.descriptorPool = m_descriptorPool;
			allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
			allocInfo.pSetLayouts = layouts.data();

			if (vkAllocateDescriptorSets(device, &allocInfo, m_graphicDescriptorSets.candles.tranformUniform.data()) != VK_SUCCESS) {
				throw std::runtime_error("failed to allocate graphic descriptor sets!");
			}

			for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
				std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

				VkDescriptorBufferInfo bufferInfo{};
				bufferInfo.buffer = m_graphicUniformBuffers.transform[objIdx][i];
				bufferInfo.offset = 0;
				bufferInfo.range = sizeof(TransformUniform);

				descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[0].dstSet = m_graphicDescriptorSets.candles.tranformUniform[i];
				descriptorWrites[0].dstBinding = 0;
				descriptorWrites[0].dstArrayElement = 0;
				descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
				descriptorWrites[0].descriptorCount = 1;
				descriptorWrites[0].pBufferInfo = &bufferInfo;

				VkDescriptorBufferInfo lightBufferInfo{};
				lightBufferInfo.buffer = m_graphicUniformBuffers.lighting[objIdx][i];
				lightBufferInfo.offset = 0;
				lightBufferInfo.range = sizeof(LightingUniform);

				descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[1].dstSet = m_graphicDescriptorSets.candles.tranformUniform[i];
				descriptorWrites[1].dstBinding = 1;
				descriptorWrites[1].dstArrayElement = 0;
				descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorWrites[1].descriptorCount = 1;
				descriptorWrites[1].pBufferInfo = &lightBufferInfo;

				vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
			}
		}

		// bloom
		{
			VkDescriptorSetAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			std::array<VkDescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> 
				layouts = {m_graphicDescriptorSetLayouts.bloom, m_graphicDescriptorSetLayouts.bloom};
			allocInfo.pSetLayouts = layouts.data();
			allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
			allocInfo.descriptorPool = m_descriptorPool;

			CHECK_VK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, m_graphicDescriptorSets.bloom1.data())
				, "failed to allocate bloom1 graphic descriptor sets!");

			CHECK_VK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, m_graphicDescriptorSets.bloom2.data())
				, "failed to allocate bloom2 graphic descriptor sets!");

			for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
				std::array<VkWriteDescriptorSet, 1> descriptorWrites{};

				VkDescriptorImageInfo bloom1ImageInfo{};
				bloom1ImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				bloom1ImageInfo.imageView = m_renderTargets.base.bloomThresholdResRT.view;
				bloom1ImageInfo.sampler = m_samplers.quad;

				descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[0].dstSet = m_graphicDescriptorSets.bloom1[i];
				descriptorWrites[0].dstBinding = 0;
				descriptorWrites[0].dstArrayElement = 0;
				descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descriptorWrites[0].descriptorCount = 1;
				descriptorWrites[0].pImageInfo = &bloom1ImageInfo;

				vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);

				// for bloom2
				VkDescriptorImageInfo bloom2ImageInfo{};
				bloom2ImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				bloom2ImageInfo.imageView = m_renderTargets.bloom1.view;
				bloom2ImageInfo.sampler = m_samplers.quad;

				descriptorWrites[0].pImageInfo = &bloom2ImageInfo;
				descriptorWrites[0].dstSet = m_graphicDescriptorSets.bloom2[i];
				vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
			}
		}

		// combine
		{
			VkDescriptorSetAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			std::array<VkDescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> 
				layouts = {m_graphicDescriptorSetLayouts.combine, m_graphicDescriptorSetLayouts.combine};
			allocInfo.pSetLayouts = layouts.data();
			allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
			allocInfo.descriptorPool = m_descriptorPool;

			CHECK_VK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, m_graphicDescriptorSets.combine.data())
				, "failed to allocate graphic descriptor sets!");

			for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
				std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

				VkDescriptorImageInfo baseImageInfo{};
				baseImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				baseImageInfo.imageView = m_renderTargets.base.colorResRT.view;
				baseImageInfo.sampler = m_samplers.quad;

				descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[0].dstSet = m_graphicDescriptorSets.combine[i];
				descriptorWrites[0].dstBinding = 0;
				descriptorWrites[0].dstArrayElement = 0;
				descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descriptorWrites[0].descriptorCount = 1;
				descriptorWrites[0].pImageInfo = &baseImageInfo;
                                 
				VkDescriptorImageInfo bloomImageInfo{};
				bloomImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				bloomImageInfo.imageView = m_renderTargets.bloom2.view;
				bloomImageInfo.sampler = m_samplers.quad;

				descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[1].dstSet = m_graphicDescriptorSets.combine[i]; 
				descriptorWrites[1].dstBinding = 1;
				descriptorWrites[1].dstArrayElement = 0;
				descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descriptorWrites[1].descriptorCount = 1;
				descriptorWrites[1].pImageInfo = &bloomImageInfo;

				vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
			}
		}
	}

	void createComputeDescriptorSets() {
		std::array<VkDescriptorSetLayout, 1> 
			layouts = {m_computeDescriptorSetLayout};

		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorSetCount = layouts.size();
		allocInfo.descriptorPool = m_descriptorPool;
		allocInfo.pSetLayouts = layouts.data();

		if (vkAllocateDescriptorSets(device, &allocInfo, &m_computeDescriptorSets) != VK_SUCCESS)
			throw std::runtime_error("failed to allocate compute descriptor sets!");

		VkDescriptorBufferInfo storageBufferInfo{};
		storageBufferInfo.buffer = m_storageBuffer;
		storageBufferInfo.offset = 0;
		// FIXME: range is sus
		storageBufferInfo.range = VK_WHOLE_SIZE;

		VkDescriptorBufferInfo uboBufferInfo{};
		uboBufferInfo.buffer = m_vortexUniformBuffer;
		uboBufferInfo.offset = 0;
		// FIXME: range is sus
		uboBufferInfo.range = VK_WHOLE_SIZE;

		std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

		descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[0].dstSet = m_computeDescriptorSets;
		descriptorWrites[0].dstBinding = 0;
		descriptorWrites[0].dstArrayElement = 0;
		descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[0].descriptorCount = 1;
		descriptorWrites[0].pBufferInfo = &storageBufferInfo;

		descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[1].dstSet = m_computeDescriptorSets;
		descriptorWrites[1].dstBinding = 1;
		descriptorWrites[1].dstArrayElement = 0;
		descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descriptorWrites[1].descriptorCount = 1;
		descriptorWrites[1].pBufferInfo = &uboBufferInfo;

		vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, 0);
	}

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VmaAllocation& allocation) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VmaAllocationCreateInfo	allocInfo{};
		if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
		{
			allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
			allocInfo.priority = 0.0f;
		}
		else
		{
			// allocInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT | VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
			allocInfo.priority = 1.0f;
		}

		allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
		allocInfo.requiredFlags = 0;
		allocInfo.preferredFlags = properties;
		allocInfo.memoryTypeBits = 0;
		allocInfo.pool = VK_NULL_HANDLE;
		allocInfo.pUserData = nullptr;

		if (vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate buffer memory!");
		}

        // if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        //     throw std::runtime_error("failed to create buffer!");
        // }

        // VkMemoryRequirements memRequirements;
        // vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        // VkMemoryAllocateInfo allocInfo{};
        // allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        // allocInfo.allocationSize = memRequirements.size;
        // allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        // if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        //     throw std::runtime_error("failed to allocate buffer memory!");
        // }

        // vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    VkCommandBuffer beginSingleTimeCommands() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = m_graphicCommandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
		// TracyVkCollect(tracyContext, commandBuffer);
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(m_graphicQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(m_graphicQueue);

        vkFreeCommandBuffers(device, m_graphicCommandPool, 1, &commandBuffer);
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		{
			// TracyVkZone(tracyContext, commandBuffer, "transferBuffer");
			VkBufferCopy copyRegion{};
			copyRegion.size = size;
			vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
		}

        endSingleTimeCommands(commandBuffer);
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createCommandBuffers() {
        m_graphicCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo graphicAllocInfo{};
        graphicAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        graphicAllocInfo.commandPool = m_graphicCommandPool;
        graphicAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        graphicAllocInfo.commandBufferCount = (uint32_t) m_graphicCommandBuffers.size();

        if (vkAllocateCommandBuffers(device, &graphicAllocInfo, m_graphicCommandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }

		VkCommandBufferAllocateInfo	computeAllocInfo{};
        computeAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        computeAllocInfo.commandPool = m_computeCommandPool;
        computeAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        computeAllocInfo.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(device, &computeAllocInfo, &m_computeCommandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }

        VkCommandBufferAllocateInfo tracyAllocInfo{};
        tracyAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        tracyAllocInfo.commandPool = m_graphicCommandPool;
        tracyAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        tracyAllocInfo.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(device, &tracyAllocInfo, &tracyCommandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
        
    }

	void renderModel(VkCommandBuffer commandBuffer, Object object) {

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicPipelines.candles);

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float) swapChainExtent.width;
		viewport.height = (float) swapChainExtent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

		VkRect2D scissor{};
		scissor.offset = {0, 0};
		scissor.extent = swapChainExtent;
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

		tinygltf::Model& model = m_model[object];

		int meshIdx = 0;
		// factor out tangent
		auto& attribute = model.meshes[0].primitives[0].attributes;
		VkBuffer tangentBuffer = m_vertexBuffers.candles[model.accessors[attribute["TANGENT"]].bufferView].buffer; 
		
		for (auto& mesh : model.meshes) {
			// assume there is 1 primitive per mesh
			auto& attributes = mesh.primitives[0].attributes;

			VkBuffer positionBuffer = m_vertexBuffers.candles[model.accessors[attributes["POSITION"]].bufferView].buffer;
			size_t positionBufferOffset = model.accessors[attributes["POSITION"]].byteOffset;
			bool hasAnimation = m_modelMeshAnimPositions.find(object) !=  m_modelMeshAnimPositions.end();
			if (hasAnimation && !m_modelMeshAnimPositions[object][meshIdx].empty()) {
				positionBuffer = m_animBuffer[object][meshIdx];
				positionBufferOffset = 0;
			}

			size_t normalBufferOffset = model.accessors[attributes["NORMAL"]].byteOffset;
			VkBuffer normalBuffer = m_vertexBuffers.candles[model.accessors[attributes["NORMAL"]].bufferView].buffer;

			size_t texCordBufferOffset = model.accessors[attributes["TEXCOORD_0"]].byteOffset;
			VkBuffer texCordBuffer = m_vertexBuffers.candles[model.accessors[attributes["TEXCOORD_0"]].bufferView].buffer;

			size_t tangentBufferOffset = model.accessors[attributes["TANGENT"]].byteOffset; 
			// some mesh of the model don't have tangent attribute
			// if(attribute.find("TANGENT") != attribute.end()) {
			// 	tangentBuffer = m_vertexBuffer[object][model.accessors[attribute["TANGENT"]].bufferView];
			// 	tangentBufferOffset = model.accessors[attribute["TANGENT"]].byteOffset;
			// }

			VkBuffer instanceBuffer;
			uint32_t instanceCount{};
			if (object == Object::SNOWFLAKE) {
				instanceBuffer = m_storageBuffer;
				instanceCount = SNOWFLAKE_COUNT;
			}
			else if (object == Object::CANDLE) {
				instanceBuffer = m_towerInstanceBuffer;
				instanceCount = m_towerInstanceRaw.size();
			}

			VkBuffer vertexBuffers[5] = {positionBuffer, normalBuffer, tangentBuffer, texCordBuffer, instanceBuffer};
			VkDeviceSize vertexBufferOffsets[5] = {positionBufferOffset, normalBufferOffset, tangentBufferOffset, texCordBufferOffset, 0};

			vkCmdBindVertexBuffers(commandBuffer, 0, sizeof(vertexBuffers) / sizeof(VkBuffer), vertexBuffers, vertexBufferOffsets);

			auto& indexAccessoridx = mesh.primitives[0].indices;
			VkBuffer indexBuffer = m_indexBuffers.candles[model.accessors[indexAccessoridx].bufferView].buffer;
			uint64_t indexBufferOffsets = model.accessors[indexAccessoridx].byteOffset;
			vkCmdBindIndexBuffer(commandBuffer, indexBuffer, indexBufferOffsets, VK_INDEX_TYPE_UINT32);

			// mesh local transform
			TransformUniform* uniformMapped = (TransformUniform*)m_graphicUniformBuffersMapped.transform[object][m_currentFrame];
			uniformMapped[meshIdx].model = uniformMapped[meshIdx].model * m_modelMeshTransforms[object][meshIdx];

			// some mesh in the model don't normal mapp
			if(attributes["TANGENT"] == 0) {
				m_graphicPushConstant.isNormalMapping = 0;
			}
			else {
				m_graphicPushConstant.isNormalMapping = 1;
			}

			vkCmdPushConstants(commandBuffer, m_graphicPipelineLayouts.candles, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(GraphicPushConstant), (void*)&m_graphicPushConstant);

			uint32_t DynamicOffset{};
			// this dynamic offset have to be 256 byte aligned
			DynamicOffset = sizeof(TransformUniform) * meshIdx;
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicPipelineLayouts.candles, 
						   0, 1, &m_graphicDescriptorSets.candles.tranformUniform[m_currentFrame], 1, &DynamicOffset);

			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicPipelineLayouts.candles, 
						   1, 1, &m_graphicDescriptorSets.candles.meshMaterial[meshIdx][m_currentFrame], 0, 0);

			// is this count right?
			vkCmdDrawIndexed(commandBuffer, model.accessors[indexAccessoridx].count, instanceCount, 0, 0, 0);
			meshIdx++;
		}
	}

	void renderBloomHorizontal(VkCommandBuffer commandBuffer) {
		VkDeviceSize offsets{0};

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicPipelines.bloom.horizontal);
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, &m_vertexBuffers.quad[0].buffer, &offsets);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicPipelineLayouts.bloom,
						0, 1, &m_graphicDescriptorSets.bloom1[m_currentFrame], 0, 0);
		vkCmdDraw(commandBuffer, 6, 1, 0, 0);
	}

	void renderBloomVertical(VkCommandBuffer commandBuffer) {
		VkDeviceSize offsets{0};

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicPipelines.bloom.vertical);
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, &m_vertexBuffers.quad[0].buffer, &offsets);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicPipelineLayouts.bloom,
						0, 1, &m_graphicDescriptorSets.bloom2[m_currentFrame], 0, 0);
		vkCmdDraw(commandBuffer, 6, 1, 0, 0);
	}

	void renderCombine(VkCommandBuffer commandBuffer) {
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicPipelines.combine);

		VkDeviceSize offsets{0};
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, &m_vertexBuffers.quad[0].buffer, &offsets);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicPipelineLayouts.combine,
						0, 1, &m_graphicDescriptorSets.combine[m_currentFrame], 0, 0);
		vkCmdDraw(commandBuffer, 6, 1, 0, 0);
	}

	void transferBuffers(VkCommandBuffer commandBuffer) {
		for (unsigned int i = 0; i < Object::COUNT; i++){
			Object objIdx = static_cast<Object>(i);
			if (m_modelMeshAnimPositions.find(objIdx) == m_modelMeshAnimPositions.end()) {
				continue;
			}
			std::vector<VkBufferMemoryBarrier> animBarriers{};
			std::vector<VkBuffer> stagingBuffers{};
			std::vector<VmaAllocation> stagingAllocs{};
			for (unsigned int meshIdx = 0; meshIdx < m_model[objIdx].meshes.size(); meshIdx++) {
				if (!m_modelMeshAnimPositions[objIdx][meshIdx].empty()){
					// Transfer vertex position animation data
					VkBuffer stagingBuffer;
					VmaAllocation stagingAlloc;
					unsigned int size = m_modelMeshAnimPositions[objIdx][meshIdx].size() * sizeof(glm::vec3);

					createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingAlloc);

					void* data;
					vmaMapMemory(m_allocator, stagingAlloc, &data);
						memcpy(data, m_modelMeshAnimPositions[objIdx][meshIdx].data(), static_cast<size_t>(size));
					vmaUnmapMemory(m_allocator, stagingAlloc);

					VkBufferCopy copyRegion{};
					copyRegion.size = size;

					vkCmdCopyBuffer(commandBuffer, stagingBuffer, m_animBuffer[objIdx][meshIdx], 1, &copyRegion);

					stagingBuffers.push_back(stagingBuffer);
					stagingAllocs.push_back(stagingAlloc);

					VkBufferMemoryBarrier animBarrier{};
					animBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
					animBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; 
					animBarrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
					animBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
					animBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
					animBarrier.buffer = m_animBuffer[objIdx][meshIdx];
					animBarrier.size = size;
					animBarrier.offset = 0;
				 
					animBarriers.push_back(animBarrier);
				}
			}

			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0,
				0, nullptr,
				animBarriers.size(), animBarriers.data(),
				0, nullptr);

			for(auto& buffer : stagingBuffers) {
					vkDestroyBuffer(device, buffer, nullptr);
			}
			for(auto& alloc : stagingAllocs) {
					vmaFreeMemory(m_allocator, alloc);
			}
		}
	}

    void recordGraphicCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

		{
			TracyVkZone(tracyContext, commandBuffer, "Transfer animation buffers");
			transferBuffers(commandBuffer);
		}

		{
			VkRenderPassBeginInfo basePassInfo{};
			basePassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			basePassInfo.renderPass = m_renderPasses.base;
			basePassInfo.framebuffer = m_frameBuffers.base[imageIndex];
			basePassInfo.renderArea.offset = {0, 0};
			basePassInfo.renderArea.extent = swapChainExtent;

			std::array<VkClearValue, 5> clearValues{};
			clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
			clearValues[1].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
			clearValues[2].depthStencil = {1.0f, 0};
			clearValues[3].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
			clearValues[4].color = {{0.0f, 0.0f, 0.0f, 1.0f}};

			basePassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
			basePassInfo.pClearValues = clearValues.data();

			vkCmdBeginRenderPass(commandBuffer, &basePassInfo, VK_SUBPASS_CONTENTS_INLINE);

				renderModel(commandBuffer, Object::CANDLE);

			vkCmdEndRenderPass(commandBuffer);

////////////////////////////////////////////////////////////////
			VkRenderPassBeginInfo bloomPassInfo{};
			bloomPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			bloomPassInfo.renderPass = m_renderPasses.bloom;
			bloomPassInfo.framebuffer = m_frameBuffers.bloom.horizontal[imageIndex];
			bloomPassInfo.renderArea.offset = {0, 0};
			bloomPassInfo.renderArea.extent = swapChainExtent;

			std::array<VkClearValue, 1> bloomClearValues{};
			bloomClearValues[0].color = {0.0f, 0.0f, 0.0f, 1.0f};

			bloomPassInfo.clearValueCount = static_cast<uint32_t>(bloomClearValues.size());
			bloomPassInfo.pClearValues = bloomClearValues.data();

			vkCmdBeginRenderPass(commandBuffer, &bloomPassInfo, VK_SUBPASS_CONTENTS_INLINE);

				renderBloomHorizontal(commandBuffer);

			vkCmdEndRenderPass(commandBuffer);

			bloomPassInfo.framebuffer = m_frameBuffers.bloom.vertical[imageIndex];
			vkCmdBeginRenderPass(commandBuffer, &bloomPassInfo, VK_SUBPASS_CONTENTS_INLINE);

				renderBloomVertical(commandBuffer);

			vkCmdEndRenderPass(commandBuffer);

			VkRenderPassBeginInfo combinePassInfo{};
			combinePassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			combinePassInfo.renderPass = m_renderPasses.combine;
			combinePassInfo.framebuffer = m_frameBuffers.combine[imageIndex];
			combinePassInfo.renderArea.offset = {0, 0};
			combinePassInfo.renderArea.extent = swapChainExtent;

			std::array<VkClearValue, 1> combineClearValues{};
			combineClearValues[0].color = {0.0f, 0.0f, 0.0f, 1.0f};

			combinePassInfo.clearValueCount = static_cast<uint32_t>(combineClearValues.size());
			combinePassInfo.pClearValues = combineClearValues.data();

			vkCmdBeginRenderPass(commandBuffer, &combinePassInfo, VK_SUBPASS_CONTENTS_INLINE);

				renderCombine(commandBuffer);

				{
					TracyVkZone(tracyContext, commandBuffer, "Draw ImGui");
					ImGui::Render();
					ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer, VK_NULL_HANDLE);
				}


			vkCmdEndRenderPass(commandBuffer);
		
			TracyVkCollect(tracyContext, commandBuffer);
		}

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record graphic command buffer!");
        }
    }

	void recordComputeCommandBuffer(VkCommandBuffer commandBuffer){
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS){
            throw std::runtime_error("failed to begin recording command buffer!");
		}

		{
			TracyVkZone(tracyContext, commandBuffer, "Dispatch Snowflake Compute");
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipeline);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipelineLayout, 0, 1, &m_computeDescriptorSets, 0, nullptr);
			vkCmdPushConstants(commandBuffer, m_computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePushConstant), (void*)&m_computePushConstant);
			// FIXME: choose right number of workgroups
			vkCmdDispatch(commandBuffer, 1024, 1, 1);
			TracyVkCollect(tracyContext, commandBuffer);
		}

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record compute command buffer!");
        }
	}

    void createSyncObjects() {
        m_imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        m_renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		m_computeStartingSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        m_inFlightGraphicFences.resize(MAX_FRAMES_IN_FLIGHT);

		// we can submit an empty command buffer to signal the m_renderFinishedSemaphores but
		// doing this way creating a timeline semaphore is cooler
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		// VkSemaphoreTypeCreateInfo semaphoreTypeInfo{};
		// semaphoreTypeInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
		// semaphoreTypeInfo.initialValue = 1;
		// semaphoreTypeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        // semaphoreInfo.pNext = &semaphoreTypeInfo;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &m_imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &m_renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &m_computeStartingSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &m_inFlightGraphicFences[i]) != VK_SUCCESS ){
                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }
		CHECK_VK_RESULT(vkCreateFence(device, &fenceInfo, nullptr, &m_inFlightComputeFences)
					, "fail to create Compute fence");

		// signal the last index computeStartingSemaphore because if we don't do manually, noone do :(
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &m_computeStartingSemaphores[MAX_FRAMES_IN_FLIGHT - 1];
		CHECK_VK_RESULT(vkQueueSubmit(m_graphicQueue, 1, &submitInfo, VK_NULL_HANDLE)
				  ,"fail to submit semaphore signaling to queue");
		CHECK_VK_RESULT(vkQueueWaitIdle(m_graphicQueue)
				  ,"fail to wait for semaphore signling queuing");

		VkSemaphoreCreateInfo computeSemaphoreInfo{};
        computeSemaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		CHECK_VK_RESULT(vkCreateSemaphore(device, &computeSemaphoreInfo, nullptr, &m_computeFinishedSemaphore)
				  , "failed to create compute synchronization objects for a frame!");
    }


	void processImGui(){
        // ImGui::SeparatorText("Watch Tower Model");
		// ImGui::SliderFloat3("Translate", s_translate, -10.f, 10.f, "%.2f");
		// ImGui::SliderFloat3("Rotate", s_rotate, -10.f, 10.f, "%.2f");
		// ImGui::SliderFloat3("Scale", s_scale, -10.f, 10.f, "%.2f");

        ImGui::SeparatorText("Time");
		ImGui::Text("Current time: (%f)", m_lastTime);
		ImGui::Text("Delta time: (%f)", m_currentDeltaTime);
		ImGui::Text("FPS: (%f)", 1 / m_currentDeltaTime);

        ImGui::SeparatorText("Camera");
		ImGui::Text("Camera front: (%f), (%f), (%f)", g_camera.getFront().x, g_camera.getFront().y, g_camera.getFront().z);
		ImGui::Text("Camera position: (%f), (%f), (%f)", g_camera.getPostion().x, g_camera.getPostion().y, g_camera.getPostion().z);

		ImGui::Spacing();

        ImGui::SeparatorText("Snowflake Model");
		ImGui::SliderFloat3("Translate", s_snowTranslate, -10.f, 10.f, "%.2f");
		ImGui::SliderFloat3("Rotate", s_snowRotate, -10.f, 10.f, "%.2f");
		ImGui::SliderFloat3("Scale", s_snowScale, -10.f, 10.f, "%.2f");

        ImGui::SeparatorText("Light");
		ImGui::SliderFloat3("Light Direction", s_lightPos, -20.f, 20.f, "%.2f");
        ImGui::SeparatorText("Projection");
		ImGui::SliderFloat("Near Plane", &s_nearPlane, -10.f, 10.f, "%.5f");
		ImGui::SliderFloat("Far Plane", &s_farPlane, -10.f, 100.f, "%.5f");
	}

    void drawFrame() {
		ZoneScopedN("Update&Render&Present");

		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		// ImGui::ShowDemoWindow();
		processImGui();

		uint32_t imageIndex;
		VkResult result{};
		{
			ZoneScopedN("Submit Compute Command Buffer");
			{
				ZoneScopedN("Wait for Compute Fence");
				vkWaitForFences(device, 1, &m_inFlightComputeFences, VK_TRUE, UINT64_MAX);
				vkResetFences(device, 1, &m_inFlightComputeFences);
			}
			{
				// NOTE: only update Uniform buffer after the command buffer with the same m_currentFrame (the last 2 frames) have FINISHED.
				// have to update uniform after WaitForFence or else uniform are override within that frame
				updateComputeUniformBuffer();
				updateComputePushConstant();

				ZoneScopedN("Dispatch Compute Command Buffer");
				vkResetCommandBuffer(m_computeCommandBuffer, /*VkCommandBufferResetFlagBits*/ 0);
				recordComputeCommandBuffer(m_computeCommandBuffer);

				VkSubmitInfo computeSubmitInfo{};
				VkSemaphore computeWaitSemaphores[] = {m_computeStartingSemaphores[(m_currentFrame - 1) % 2]};
				VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};
				VkSemaphore computeSignalSemaphores[] = {m_computeFinishedSemaphore};
				computeSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

				computeSubmitInfo.waitSemaphoreCount = sizeof(computeWaitSemaphores) / sizeof(VkSemaphore);
				computeSubmitInfo.pWaitSemaphores = computeWaitSemaphores;
				computeSubmitInfo.pWaitDstStageMask = waitStages;
				computeSubmitInfo.signalSemaphoreCount = sizeof(computeSignalSemaphores) / sizeof(VkSemaphore);
				computeSubmitInfo.pSignalSemaphores = computeSignalSemaphores;
				computeSubmitInfo.commandBufferCount = 1;
				computeSubmitInfo.pCommandBuffers = &m_computeCommandBuffer;

				CHECK_VK_RESULT(vkQueueSubmit(m_computeQueue, 1, &computeSubmitInfo, m_inFlightComputeFences)
					, "fail to submit compute command buffer");
				// vkQueueWaitIdle(m_computeQueue);
			}
		}

		{
			ZoneScopedN("Submit Graphic Command Buffer");
			{
				ZoneScopedN("Accquire Next Image");
				result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, m_imageAvailableSemaphores[m_currentFrame], VK_NULL_HANDLE, &imageIndex);
				if (result == VK_ERROR_OUT_OF_DATE_KHR) {
					recreateSwapChain();
					return;
				} else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
					throw std::runtime_error("failed to acquire swap chain image!");
				}
			}
			{
				ZoneScopedN("Wait for Graphic Fence");
				vkWaitForFences(device, 1, &m_inFlightGraphicFences[m_currentFrame], VK_TRUE, UINT64_MAX);
				vkResetFences(device, 1, &m_inFlightGraphicFences[m_currentFrame]);
			}

			// NOTE: only update Uniform buffer after the command buffer with the same m_currentFrame (the last 2 frames) have FINISHED.
			// have to update uniform after WaitForFence or else uniform are override within that frame
			updateGraphicUniformBuffer();

			vkResetCommandBuffer(m_graphicCommandBuffers[m_currentFrame], /*VkCommandBufferResetFlagBits*/ 0);
			recordGraphicCommandBuffer(m_graphicCommandBuffers[m_currentFrame], imageIndex);

			VkSubmitInfo submitInfo{};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

			VkSemaphore waitSemaphores[] = {m_imageAvailableSemaphores[m_currentFrame], m_computeFinishedSemaphore};
			VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT};
			submitInfo.waitSemaphoreCount = sizeof(waitSemaphores) / sizeof(VkSemaphore);
			submitInfo.pWaitSemaphores = waitSemaphores;
			submitInfo.pWaitDstStageMask = waitStages;

			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &m_graphicCommandBuffers[m_currentFrame];

			VkSemaphore signalSemaphores[] = {m_renderFinishedSemaphores[m_currentFrame], m_computeStartingSemaphores[m_currentFrame]};
			submitInfo.signalSemaphoreCount = sizeof(signalSemaphores) / sizeof(VkSemaphore);
			submitInfo.pSignalSemaphores = signalSemaphores;

			if (VkResult res = vkQueueSubmit(m_graphicQueue, 1, &submitInfo, m_inFlightGraphicFences[m_currentFrame])) {
				std::string msg = "failed to submit graphic command buffer with CODE: " + vk::to_string((vk::Result)res);
				throw std::runtime_error(msg);
			}
		}	

		{
			ZoneScopedN("Submit present image");
			VkPresentInfoKHR presentInfo{};
			presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

			presentInfo.waitSemaphoreCount = 1;
			presentInfo.pWaitSemaphores = &m_renderFinishedSemaphores[m_currentFrame];

			VkSwapchainKHR swapChains[] = {swapChain};
			presentInfo.swapchainCount = 1;
			presentInfo.pSwapchains = swapChains;

			presentInfo.pImageIndices = &imageIndex;

			result = vkQueuePresentKHR(m_presentQueue, &presentInfo);

			if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
				framebufferResized = false;
				recreateSwapChain();
			} else if (result != VK_SUCCESS) {
				throw std::runtime_error("failed to present swap chain image!");
			}
		}

        m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        } else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    bool isDeviceSuitable(VkPhysicalDevice device) {
        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        VkPhysicalDeviceFeatures supportedFeatures;
        vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

        return indices.isComplete() && extensionsSupported && swapChainAdequate  && supportedFeatures.samplerAnisotropy;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
			// std::cout << extension.extensionName << "\n";
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicFamily = i;
            }

			if(queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)
                indices.computeFamily = i;

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

            if (presentSupport) {
                indices.presentFamily = i;
            }

            if (indices.isComplete()) {
                break;
            }

            i++;
        }

        return indices;
    }

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    bool checkValidationLayerSupport() {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers) {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }

        return true;
    }

	void printPhysicalDeviceProperties(){
		std::cout << "Physical device INFO: " <<
		"\n apiVersion: \n" << m_physicalDeviceProperties.apiVersion <<
		"\n driverVersion: \n" << m_physicalDeviceProperties.driverVersion <<
		"\n vendorID: \n" << m_physicalDeviceProperties.vendorID <<
		"\n deviceID: \n" << m_physicalDeviceProperties.deviceID <<
		"\n deviceType: \n" << m_physicalDeviceProperties.deviceType <<
		// "\n deviceName: \n" << m_physicalDeviceProperties.deviceID <<
		"\n limit properties: \n" <<
		"\n maxImageDimension1D: " << m_physicalDeviceProperties.limits.maxImageDimension1D <<
		"\n maxImageDimension2D: "     << m_physicalDeviceProperties.limits.maxImageDimension2D <<
		"\n maxImageDimension3D: "     << m_physicalDeviceProperties.limits.maxImageDimension3D <<
		"\n maxImageDimensionCube: "   << m_physicalDeviceProperties.limits.maxImageDimensionCube <<
		"\n maxImageArrayLayers: " << m_physicalDeviceProperties.limits.maxImageArrayLayers <<
		"\n maxTexelBufferElements: "  << m_physicalDeviceProperties.limits.maxTexelBufferElements <<
		"\n maxUniformBufferRange: "   << m_physicalDeviceProperties.limits.maxUniformBufferRange <<
		"\n maxStorageBufferRange: "   << m_physicalDeviceProperties.limits.maxStorageBufferRange <<
		"\n maxPushConstantsSize: "    << m_physicalDeviceProperties.limits.maxPushConstantsSize <<
		"\n maxMemoryAllocationCount: "    << m_physicalDeviceProperties.limits.maxMemoryAllocationCount <<
		"\n maxSamplerAllocationCount: "   << m_physicalDeviceProperties.limits.maxSamplerAllocationCount <<
		"\n bufferImageGranularity: "<< m_physicalDeviceProperties.limits.bufferImageGranularity <<
		"\n sparseAddressSpaceSize: "<< m_physicalDeviceProperties.limits.sparseAddressSpaceSize <<
		"\n maxBoundDescriptorSets: "  << m_physicalDeviceProperties.limits.maxBoundDescriptorSets <<
		"\n maxPerStageDescriptorSamplers: "   << m_physicalDeviceProperties.limits.maxPerStageDescriptorSamplers <<
		"\n maxPerStageDescriptorUniformBuffers: "     << m_physicalDeviceProperties.limits.maxPerStageDescriptorUniformBuffers <<
		"\n maxPerStageDescriptorStorageBuffers: "     << m_physicalDeviceProperties.limits.maxPerStageDescriptorStorageBuffers <<
		"\n maxPerStageDescriptorSampledImages: "  << m_physicalDeviceProperties.limits.maxPerStageDescriptorSampledImages <<
		"\n maxPerStageDescriptorStorageImages: "  << m_physicalDeviceProperties.limits.maxPerStageDescriptorStorageImages <<
		"\n maxPerStageDescriptorInputAttachments: "   << m_physicalDeviceProperties.limits.maxPerStageDescriptorInputAttachments <<
		"\n maxPerStageResources: "    << m_physicalDeviceProperties.limits.maxPerStageResources <<
		"\n maxDescriptorSetSamplers: "    << m_physicalDeviceProperties.limits.maxDescriptorSetSamplers <<
		"\n maxDescriptorSetUniformBuffers: "  << m_physicalDeviceProperties.limits.maxDescriptorSetUniformBuffers <<
		"\n maxDescriptorSetUniformBuffersDynamic: "   << m_physicalDeviceProperties.limits.maxDescriptorSetUniformBuffersDynamic <<
		"\n maxDescriptorSetStorageBuffers: "  << m_physicalDeviceProperties.limits.maxDescriptorSetStorageBuffers <<
		"\n maxDescriptorSetStorageBuffersDynamic: "   << m_physicalDeviceProperties.limits.maxDescriptorSetStorageBuffersDynamic <<
		"\n maxDescriptorSetSampledImages: "   << m_physicalDeviceProperties.limits.maxDescriptorSetSampledImages <<
		"\n maxDescriptorSetStorageImages: "   << m_physicalDeviceProperties.limits.maxDescriptorSetStorageImages <<
		"\n maxDescriptorSetInputAttachments: "    << m_physicalDeviceProperties.limits.maxDescriptorSetInputAttachments <<
		"\n maxVertexInputAttributes: "    << m_physicalDeviceProperties.limits.maxVertexInputAttributes <<
		"\n maxVertexInputBindings: "  << m_physicalDeviceProperties.limits.maxVertexInputBindings <<
		"\n maxVertexInputAttributeOffset: "   << m_physicalDeviceProperties.limits.maxVertexInputAttributeOffset <<
		"\n maxVertexInputBindingStride: "     << m_physicalDeviceProperties.limits.maxVertexInputBindingStride <<
		"\n maxVertexOutputComponents: "   << m_physicalDeviceProperties.limits.maxVertexOutputComponents <<
		"\n maxTessellationGenerationLevel: "  << m_physicalDeviceProperties.limits.maxTessellationGenerationLevel <<
		"\n maxTessellationPatchSize: "    << m_physicalDeviceProperties.limits.maxTessellationPatchSize <<
		"\n maxTessellationControlPerVertexInputComponents: "  << m_physicalDeviceProperties.limits.maxTessellationControlPerVertexInputComponents <<
		"\n maxTessellationControlPerVertexOutputComponents: "     << m_physicalDeviceProperties.limits.maxTessellationControlPerVertexOutputComponents <<
		"\n maxTessellationControlPerPatchOutputComponents: "  << m_physicalDeviceProperties.limits.maxTessellationControlPerPatchOutputComponents <<
		"\n maxTessellationControlTotalOutputComponents: "     << m_physicalDeviceProperties.limits.maxTessellationControlTotalOutputComponents <<
		"\n maxTessellationEvaluationInputComponents: "    << m_physicalDeviceProperties.limits.maxTessellationEvaluationInputComponents <<
		"\n maxTessellationEvaluationOutputComponents: "   << m_physicalDeviceProperties.limits.maxTessellationEvaluationOutputComponents <<
		"\n maxGeometryShaderInvocations: "    << m_physicalDeviceProperties.limits.maxGeometryShaderInvocations <<
		"\n maxGeometryInputComponents: "  << m_physicalDeviceProperties.limits.maxGeometryInputComponents <<
		"\n maxGeometryOutputComponents: "     << m_physicalDeviceProperties.limits.maxGeometryOutputComponents <<
		"\n maxGeometryOutputVertices: "   << m_physicalDeviceProperties.limits.maxGeometryOutputVertices <<
		"\n maxGeometryTotalOutputComponents: "    << m_physicalDeviceProperties.limits.maxGeometryTotalOutputComponents <<
		"\n maxFragmentInputComponents: "  << m_physicalDeviceProperties.limits.maxFragmentInputComponents <<
		"\n maxFragmentOutputAttachments: "    << m_physicalDeviceProperties.limits.maxFragmentOutputAttachments <<
		"\n maxFragmentDualSrcAttachments: "   << m_physicalDeviceProperties.limits.maxFragmentDualSrcAttachments <<
		"\n maxFragmentCombinedOutputResources: "  << m_physicalDeviceProperties.limits.maxFragmentCombinedOutputResources <<
		"\n maxComputeSharedMemorySize: "  << m_physicalDeviceProperties.limits.maxComputeSharedMemorySize <<
		"\n maxComputeWorkGroupCount[0]: "    << m_physicalDeviceProperties.limits.maxComputeWorkGroupCount[0] <<
		"\n maxComputeWorkGroupInvocations: "  << m_physicalDeviceProperties.limits.maxComputeWorkGroupInvocations <<
		"\n maxComputeWorkGroupSize[0]: "     << m_physicalDeviceProperties.limits.maxComputeWorkGroupSize[0] <<
		"\n subPixelPrecisionBits: "   << m_physicalDeviceProperties.limits.subPixelPrecisionBits <<
		"\n subTexelPrecisionBits: "   << m_physicalDeviceProperties.limits.subTexelPrecisionBits <<
		"\n mipmapPrecisionBits: "     << m_physicalDeviceProperties.limits.mipmapPrecisionBits <<
		"\n maxDrawIndexedIndexValue: "    << m_physicalDeviceProperties.limits.maxDrawIndexedIndexValue <<
		"\n maxDrawIndirectCount: "    << m_physicalDeviceProperties.limits.maxDrawIndirectCount <<
		"\n maxSamplerLodBias: "       << m_physicalDeviceProperties.limits.maxSamplerLodBias <<
		"\n maxSamplerAnisotropy: "        << m_physicalDeviceProperties.limits.maxSamplerAnisotropy <<
		"\n maxViewports: "    << m_physicalDeviceProperties.limits.maxViewports <<
		"\n maxViewportDimensions[0]: "   << m_physicalDeviceProperties.limits.maxViewportDimensions[0] <<
		"\n viewportBoundsRange[0]: "     << m_physicalDeviceProperties.limits.viewportBoundsRange[0] <<
		"\n viewportSubPixelBits: "    << m_physicalDeviceProperties.limits.viewportSubPixelBits <<
		"\n minMemoryMapAlignment: "       << m_physicalDeviceProperties.limits.minMemoryMapAlignment <<
		"\n minTexelBufferOffsetAlignment: "<< m_physicalDeviceProperties.limits.minTexelBufferOffsetAlignment <<
		"\n minUniformBufferOffsetAlignment: " << m_physicalDeviceProperties.limits.minUniformBufferOffsetAlignment <<
		"\n minStorageBufferOffsetAlignment: " << m_physicalDeviceProperties.limits.minStorageBufferOffsetAlignment <<
		"\n minTexelOffset: "      << m_physicalDeviceProperties.limits.minTexelOffset <<
		"\n maxTexelOffset: "  << m_physicalDeviceProperties.limits.maxTexelOffset <<
		"\n minTexelGatherOffset: "    << m_physicalDeviceProperties.limits.minTexelGatherOffset <<
		"\n maxTexelGatherOffset: "    << m_physicalDeviceProperties.limits.maxTexelGatherOffset <<
		"\n minInterpolationOffset: "      << m_physicalDeviceProperties.limits.minInterpolationOffset <<
		"\n maxInterpolationOffset: "      << m_physicalDeviceProperties.limits.maxInterpolationOffset <<
		"\n subPixelInterpolationOffsetBits: "     << m_physicalDeviceProperties.limits.subPixelInterpolationOffsetBits <<
		"\n maxFramebufferWidth: "     << m_physicalDeviceProperties.limits.maxFramebufferWidth <<
		"\n maxFramebufferHeight: "    << m_physicalDeviceProperties.limits.maxFramebufferHeight <<
		"\n maxFramebufferLayers: "    << m_physicalDeviceProperties.limits.maxFramebufferLayers <<
		"\n framebufferColorSampleCounts: " << m_physicalDeviceProperties.limits.framebufferColorSampleCounts <<
		"\n framebufferDepthSampleCounts: "<< m_physicalDeviceProperties.limits.framebufferDepthSampleCounts <<
		"\n framebufferStencilSampleCounts: "<< m_physicalDeviceProperties.limits.framebufferStencilSampleCounts <<
		"\n framebufferNoAttachmentsSampleCounts:  "<< m_physicalDeviceProperties.limits.framebufferNoAttachmentsSampleCounts <<
		"\n maxColorAttachments: "     << m_physicalDeviceProperties.limits.maxColorAttachments <<
		"\n sampledImageColorSampleCounts: "<< m_physicalDeviceProperties.limits.sampledImageColorSampleCounts <<
		"\n sampledImageIntegerSampleCounts: " << m_physicalDeviceProperties.limits.sampledImageIntegerSampleCounts <<
		"\n sampledImageDepthSampleCounts: "<< m_physicalDeviceProperties.limits.sampledImageDepthSampleCounts <<
		"\n sampledImageStencilSampleCounts: "<< m_physicalDeviceProperties.limits.sampledImageStencilSampleCounts <<
		"\n storageImageSampleCounts: " << m_physicalDeviceProperties.limits.storageImageSampleCounts <<
		"\n maxSampleMaskWords: "  << m_physicalDeviceProperties.limits.maxSampleMaskWords <<
		"\n timestampComputeAndGraphics: "     << m_physicalDeviceProperties.limits.timestampComputeAndGraphics <<
		"\n timestampPeriod: "     << m_physicalDeviceProperties.limits.timestampPeriod <<
		"\n maxClipDistances: "    << m_physicalDeviceProperties.limits.maxClipDistances <<
		"\n maxCullDistances: "    << m_physicalDeviceProperties.limits.maxCullDistances <<
		"\n maxCombinedClipAndCullDistances: "     << m_physicalDeviceProperties.limits.maxCombinedClipAndCullDistances <<
		"\n discreteQueuePriorities: "     << m_physicalDeviceProperties.limits.discreteQueuePriorities <<
		"\n pointSizeGranularity: "        << m_physicalDeviceProperties.limits.pointSizeGranularity <<
		"\n lineWidthGranularity: "        << m_physicalDeviceProperties.limits.lineWidthGranularity <<
		"\n strictLines: "     << m_physicalDeviceProperties.limits.strictLines <<
		"\n standardSampleLocations: "     << m_physicalDeviceProperties.limits.standardSampleLocations <<
		"\n optimalBufferCopyOffsetAlignment: "<< m_physicalDeviceProperties.limits.optimalBufferCopyOffsetAlignment <<
		"\n optimalBufferCopyRowPitchAlignment: "<< m_physicalDeviceProperties.limits.optimalBufferCopyRowPitchAlignment <<
		"\n nonCoherentAtomSize: " << m_physicalDeviceProperties.limits.nonCoherentAtomSize;
	}

	void printQueueFamilyProperties(){
		uint32_t count{};
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, nullptr);
		std::vector<VkQueueFamilyProperties> queueProperties(count);
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, queueProperties.data());

		for(unsigned int i = 0; i < count; i++){
			std::cout << "Queue Family index " << i << ": " << vk::to_string((vk::QueueFlags)queueProperties[i].queueFlags) << "\n";
		}

	}

	void printMemoryBudget(){
		VmaBudget* budgets = new VmaBudget[m_allocator->GetMemoryHeapCount()];
		vmaGetHeapBudgets(m_allocator, budgets);
		
		std::cout << "Number of Heaps is: " << m_allocator->GetMemoryHeapCount() << "\n" <<
					"Number of Types is: " <<m_allocator->GetMemoryTypeCount() << "\n";

		for(unsigned int i = 0; i < m_allocator->GetMemoryHeapCount(); i++)
		{
			std::cout <<
			"Heap index: " << i << "\n" <<
			"Number of `VkDeviceMemory` objects						: " << budgets[i].statistics.blockCount << "\n" <<
			"Number of #VmaAllocation objects allocated				: " << budgets[i].statistics.allocationCount << "\n" <<
			"Number of bytes allocated in `VkDeviceMemory` blocks	: " << budgets[i].statistics.blockBytes << "\n" <<
			"Number of bytes occupied by all #VmaAllocation objects	: " << budgets[i].statistics.allocationBytes << "\n" <<
			"Estimated current memory usage of the program, in bytes: " << budgets[i].usage << "\n" <<
			"Estimated amount of memory available to the program, in bytes: " << budgets[i].budget << "\n"
			"\n";
		}

		delete[] budgets;
	}

	void printMemoryStatistics(){
		enum class StatType{
			TYPE,
			HEAP,
			TOTAL,
		};

		VkPhysicalDeviceMemoryProperties memProperties{};
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

		auto l_printStat = [&memProperties](VmaDetailedStatistics* stats, unsigned int size, StatType type) -> void{
			for(unsigned int i = 0; i < size; i++)
			{
				if(type == StatType::TOTAL){
				}
				else if(type == StatType::HEAP){
					std::cout << "Heap index "<< i << ": " << vk::to_string((vk::MemoryHeapFlags)memProperties.memoryHeaps[i].flags) 
					<< ", with size: " << memProperties.memoryHeaps[i].size << "\n";
				}
				else if (type == StatType::TYPE){
					std::cout << "Type index "<< i << ": " << vk::to_string((vk::MemoryPropertyFlags)memProperties.memoryTypes[i].propertyFlags)
					<< ", belong to heap index: " << memProperties.memoryTypes[i].heapIndex  << "\n";
				}

				VmaStatistics basicStat = stats[i].statistics;
				std::cout << "Current usage: \n" <<
				"Number of `VkDeviceMemory` objects - Vulkan memory blocks allocated: " << basicStat.blockCount << "\n" <<
				"Number of #VmaAllocation objects allocated: " << basicStat.allocationCount << "\n" <<
				"Number of bytes allocated in `VkDeviceMemory` blocks: " << basicStat.blockBytes << "\n" <<
				"Total number of bytes occupied by all #VmaAllocation objects: " << basicStat.allocationBytes << "\n" <<
				"Number of free ranges of memory between allocations: " << stats[i].unusedRangeCount << "\n" <<
				"Smallest allocation size: " << stats[i].allocationSizeMin << "\n" <<
				"Largest allocation size: " << stats[i].allocationSizeMax << "\n" <<
				"Smallest empty range size: " << stats[i].unusedRangeSizeMin << "\n" <<
				"Largest empty range size: " << stats[i].unusedRangeSizeMax << "\n" <<
				"\n";
			}
		};

		VmaTotalStatistics stats{};
		vmaCalculateStatistics(m_allocator, &stats);

		std::cout << "\n ######## Total statistics: ########\n";
		l_printStat(&stats.total, 1, StatType::TOTAL);

		std::cout << "\n ######## Heap statistics: ########\n";
		unsigned int heapCount = m_allocator->GetMemoryHeapCount();
		l_printStat(stats.memoryHeap, heapCount, StatType::HEAP);

		std::cout << "\n ######## Type statistics: ########\n";
		unsigned int typeCount = m_allocator->GetMemoryTypeCount();
		l_printStat(stats.memoryType, typeCount, StatType::TYPE);

		// l_printStat(stats.total, typeCount);
	}

	static bool isFileExist(const std::string& filename){
		std::ifstream file(filename);

		if(file.good()){
			file.close();
			return true;
		}
		file.close();
		return false;
	}

	static bool makeFile(const std::string& filename) {
		std::ofstream file(filename, std::ios::ate | std::ios::binary);

		if(file.good()){
			file.flush();
			file.close();
			std::cout << "@@@@@ create file at path: " << filename << "\n";
			return true;
		}
		file.close();
		std::cout << "@@@@@ FAIL to create file at path: " << filename << "\n";
		return false;
	}

	static void writeFile(const std::string& filename, char* data, size_t size){
		std::ofstream file(filename, std::ios::binary | std::ofstream::trunc);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

		std::cout << "@@@@@ write file at path: " << filename << ", with size: " << size << "\n";

		file.write(data, size);
		file.close();
	}

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file - " + filename);
        }

        size_t fileSize = (size_t) file.tellg();
        std::vector<char> buffer(fileSize);

		std::cout << "@@@@@ read file at path: " << filename << ", with size: " << fileSize << "\n";
        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
		if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT || messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
			std::cerr << "\n##### " << pCallbackData->pMessage << std::endl;
		}

        return VK_FALSE;
    }
};

int main() {
	srand(static_cast<unsigned>(time(0)));
	// testAlignment();
    MonoVulkan app;

    try {
		std::cout << "Start Rendering" << std::endl;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
