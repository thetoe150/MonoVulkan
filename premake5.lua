workspace "MonoVulkan"
	configurations {"Debug", "Release"}
	-- platforms {"Windows", "Unix"}
	location "build"

project "MonoVulkan"
	kind "ConsoleApp"
	language "C++"
	cppdialect "C++17"
	toolset "msc"
	targetname "MONO"
	architecture "x86_64"

	-- gcc makefile have cwd at binary, msvc have cwd at project for some reason
	-- this is for loading resource at the right path
	location "build/MonoVulkan"
	-- filter "options:cc=msc"
	-- 	location "build/VisualStudio"

	includedirs {"inc/", "tracy/public/tracy", "inc/vma", "inc/imgui", "src/meshoptimizer"}
	files {"src/**.cpp", "**.hpp", "tracy/public/TracyClient.cpp", "src/spirv_reflect.c", "src/spirv_reflect_output.cpp"}
	removefiles {"src/cpptrace/**", "src/shaders/**", "src/meshoptimizer/**"}

	defines {"TRACY_ENABLE", "TRACY_VK_USE_SYMBOL_TABLE", "ENABLE_OPTIMIZE_MESH"}
	libdirs {"lib", "build/meshoptimizer/bin"}
	links {"meshoptimizer"}

	filter "system:windows"
		libdirs {"C:/VulkanSDK/1.3.268.0/Lib"}
		links {"vulkan-1"}
		-- for tracy
		defines {"_WIN32_WINNT=0x0602", "WINVER=0x0602"}
		links {"ws2_32", "imagehlp"}
		links {"glfw3dll"}
	filter {}

	filter "system:linux"
		-- local vulanLib = os.findlib("vulkan")
		libdirs {"/usr/local/bin/1.3.296.0/x86_64/lib"}
		links {"vulkan"}
	filter {}

	buildoptions {"-std=c++17"}
	linkoptions {"-std=c++17"}

	filter "configurations:Release"
		defines {"NDEBUG"}
		optimize "On"
		targetdir "bin/release"
	filter {}
	
	filter "configurations:Debug"
		defines {"DEBUG"}
		symbols "On"
		targetdir "bin/debug"
	filter {}

-----------------------------------------------------------------------------------------------

project "MeshOptimizer"
	kind "StaticLib"
	language "C++"
	cppdialect "C++17"
	targetname "meshoptimizer"
	toolset "msc"
	architecture "x86_64"

	location "build/meshoptimizer"
	includedirs {"src/meshoptimizer"}
	files {"src/meshoptimizer/**.cpp"}
	removefiles {"src/meshoptimizer/nanite.cpp", "src/meshoptimizer/tests.cpp",  "src/meshoptimizer/main.cpp",  "src/meshoptimizer/ansi.c"}

	-- defines {}
	optimize "On"
	targetdir "build/meshoptimizer/bin"

-----------------------------------------------------------------------------------------------
project "GLFW"
	kind "StaticLib"
	language "C++"
	targetname "GLFW"
	architecture "x86_64"

	location "build/GLFW"
	includedirs {"inc/GLFW", "src/GLFW"}
	files {"src/GLFW/init.c", "src/GLFW/context.c", "src/GLFW/input.c", "src/GLFW/vulkan.c", "src/GLFW/window.c", "src/GLFW/platform.c", "src/GLFW/monitor.c"}

	filter "system:Linux"
		files {"src/GLFW/x11/*.c"}
		includedirs {"src/GLFW/x11"}
		-- files {"posix_module.c", "posix_thread.c", "posix_time.c", "posix_poll.c", "linux_joystick.c"}
		-- files {"x11_init.c", "x11_window.c", "x11_monitor.c", "x11_platform.c", "xkb_unicode.c", "glx_context.c"}
		defines {"_GLFW_X11"}
	filter {}

	filter "system:Windows"
	filter {}

	optimize "On"
	targetdir "build/GLFW/bin"

newaction {
	trigger = "clean",
	description = "clean object files",
	execute = function ()
		os.execute("./clean.ps1")
	end
}

newoption {
	trigger = "cc",
	value = "compiler",
	description = "Choose compiler to compile code",
	allowed = {
		{"gcc", "GCC"},
		{"clang", "CLANG"},
		{"msc", "MSVC"}
	},
	default = "gcc"
}
