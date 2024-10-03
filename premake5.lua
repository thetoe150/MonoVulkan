workspace "MonoVulkan"
	configurations {"Debug", "Release"}
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
	location "build"
	filter "options:cc=msc"
		location "build/VisualStudio"

	includedirs {"inc/", "tracy/public/tracy", "inc/vma", "inc/imgui"}
	files {"src/**.cpp", "**.hpp", "tracy/public/TracyClient.cpp", "src/spirv_reflect.c", "src/spirv_reflect_output.cpp"}
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
