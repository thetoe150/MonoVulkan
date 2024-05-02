CC=g++
 
CFLAGS=-Wall -std=c++17 -Iinc -Iinc/imgui -Iinc/vma -Itracy/public/tracy
LFLAGS=-Wall -std=c++17 -Llib -lglfw3dll -LC:/VulkanSDK/1.3.268.0/Lib -lvulkan-1

SRC_FILES=$(wildcard src/*.cpp)
OBJ_FILES=$(patsubst src/%, obj/%, $(patsubst %.cpp, %.o, $(SRC_FILES)))
IMGUI_SRC_FILES=$(wildcard src/imgui/*.cpp)
IMGUI_OBJ_FILES=$(patsubst src/imgui/%, obj/imgui/%, $(patsubst %.cpp, %.o, $(IMGUI_SRC_FILES)))
TRACY_OBJ=obj/tracy/TracyClient.o
TRACY_SRC=tracy/public/TracyClient.cpp
SHADER_SRC=$(wildcard src/shaders/*.glsl)
SHADER_BIN=$(patsubst %.glsl, %.spv, $(SHADER_SRC))
HEADER_ONLY_FILES=inc/vma/vk_mem_alloc.h
#
EXE=bin/main

debug: CFLAGS += -DDEBUG -DTRACY_ENABLE -O0 -ggdb -D_WIN32_WINNT=0x0602 -DWINVER=0x0602 -DTRACY_VK_USE_SYMBOL_TABLE
debug: LFLAGS += -DDEBUG -DTRACY_ENABLE -O0 -ggdb -lws2_32 -limagehlp
debug: OBJ_FILES += $(IMGUI_OBJ_FILES) 
debug: $(EXE) $(SHADER_BIN)
	./$(EXE)

release: $(EXE)
	./$(EXE)

$(EXE): $(OBJ_FILES) $(IMGUI_OBJ_FILES) $(TRACY_OBJ)
	$(CC) $^ -o $@ $(LFLAGS)

obj/%.o: src/%.cpp inc/%.hpp
	$(CC) -c $< -o $@ $(CFLAGS)

obj/imgui/%.o: src/imgui/%.cpp
	$(CC) -c $< -o $@ $(CFLAGS)

$(TRACY_OBJ): $(TRACY_SRC) tracy/public/tracy/Tracy.hpp
	$(CC) -c $< -o $@ $(CFLAGS)

src/shaders/final_vs.spv: src/shaders/final_vs.glsl
	glslc -fshader-stage=vertex ./src/shaders/final_vs.glsl -o ./src/shaders/final_vs.spv

src/shaders/final_fs.spv: src/shaders/final_fs.glsl
	glslc -fshader-stage=fragment ./src/shaders/final_fs.glsl -o ./src/shaders/final_fs.spv
