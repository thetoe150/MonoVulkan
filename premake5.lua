workspace "MonoVulkan"
	configurations {"Debug", "Release"}
	location "build/dummy"

project "MonoVulkan"
	kind "ConsoleApp"
	language "C++"
	cppdialect "C++17"
	toolset "msc"
	location "build/dummy"
	targetname "MONO"
	architecture "x86_64"

	includedirs {"inc", "tracy/public/tracy", "inc/vma", "inc/imgui"}
	files {"src/**.cpp", "**.hpp", "tracy/public/TracyClient.cpp"}
	removefiles {"src/cpptrace/**", "src/shaders"}

	libdirs {"lib", "C:/VulkanSDK/1.3.268.0/Lib"}
	links {"glfw3dll", "vulkan-1"}

	buildoptions {"-std=c++17"}
	linkoptions {"-std=c++17"}
	
	filter "configurations:Debug"
		defines {"DEBUG", "TRACY_ENABLE", "_WIN32_WINNT=0x0602", "WINVER=0x0602", "TRACY_VK_USE_SYMBOL_TABLE"}
		symbols "On"
		targetdir "bin/debug"
		-- for tracy
		links {"ws2_32", "imagehlp"}

	filter "configurations:Release"
		defines {"NDEBUG"}
		optimize "On"
		targetdir "bin/release"
