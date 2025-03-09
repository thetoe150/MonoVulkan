#!/bin/bash

echo "Script's PID: $$"
echo "Number of arguments passed to script: $#"
echo "All arguments passed to script: $@"
echo "Script's arguments separated into different variables: $1 $2 $3"

function compile_spirv()
{
	# if [[ $1 == "all_layers" ]]; then
	# 	export VK_LOADER_LAYERS_ENABLE=*api_dump,*synchronization2,renderdoc_capture
	# else
	# 	export VK_LOADER_LAYERS_ENABLE=
	# fi

	glslc -fshader-stage=vertex src/shaders/snowflake.vert -o src/shaders/snowflake.vert.spv
	glslc -fshader-stage=vertex src/shaders/candles.vert -o src/shaders/candles.vert.spv
	glslc -fshader-stage=vertex src/shaders/shadow_batch.vert -o src/shaders/shadow_batch.vert.spv
	glslc -fshader-stage=vertex src/shaders/quad.vert -o src/shaders/quad.vert.spv
	glslc -fshader-stage=vertex src/shaders/floor.vert -o src/shaders/floor.vert.spv
	glslc -fshader-stage=vertex src/shaders/skybox.vert -o src/shaders/skybox.vert.spv
	glslc -fshader-stage=fragment src/shaders/snowflake.frag -o src/shaders/snowflake.frag.spv
	glslc -fshader-stage=fragment src/shaders/candles.frag -o src/shaders/candles.frag.spv
	glslc -fshader-stage=fragment src/shaders/shadow_viewport.frag -o src/shaders/shadow_viewport.frag.spv
	glslc -fshader-stage=fragment src/shaders/bloom.frag -o src/shaders/bloom.frag.spv
	glslc -fshader-stage=fragment src/shaders/combine.frag -o src/shaders/combine.frag.spv
	glslc -fshader-stage=fragment src/shaders/floor.frag -o src/shaders/floor.frag.spv
	glslc -fshader-stage=fragment src/shaders/skybox.frag -o src/shaders/skybox.frag.spv
	glslc -fshader-stage=compute src/shaders/snowflake.comp -o src/shaders/snowflake.comp.spv

	spirv-dis src/shaders/snowflake.vert.spv > src/shaders/snowflake.vert.spvasm
	spirv-dis src/shaders/floor.vert.spv > src/shaders/floor.vert.spvasm
	spirv-dis src/shaders/skybox.vert.spv > src/shaders/skybox.vert.spvasm
	spirv-dis src/shaders/candles.vert.spv > src/shaders/candles.vert.spvasm
	spirv-dis src/shaders/quad.vert.spv > src/shaders/quad.vert.spvasm
	spirv-dis src/shaders/candles.frag.spv > src/shaders/candles.frag.spvasm
	spirv-dis src/shaders/snowflake.frag.spv > src/shaders/snowflake.frag.spvasm
	spirv-dis src/shaders/shadow_batch.vert.spv > src/shaders/shadow_batch.vert.spvasm
	spirv-dis src/shaders/shadow_viewport.frag.spv > src/shaders/shadow_viewport.frag.spvasm
	spirv-dis src/shaders/bloom.frag.spv > src/shaders/bloom.frag.spvasm
	spirv-dis src/shaders/floor.frag.spv > src/shaders/floor.frag.spvasm
	spirv-dis src/shaders/skybox.frag.spv > src/shaders/skybox.frag.spvasm
	spirv-dis src/shaders/combine.frag.spv > src/shaders/combine.frag.spvasm
	spirv-dis src/shaders/snowflake.comp.spv > src/shaders/snowflake.comp.spvasm
}

cmd="./MONO"

export VK_LOADER_LAYERS_ENABLE=
for arg in $@ 
do
	case "$arg" in
		"-d") 
			export VK_LOADER_LAYERS_ENABLE=*api_dump,*synchronization2,renderdoc_capture;;
		"-r") 
			export VK_LOADER_LAYERS_ENABLE=renderdoc_capture
			cmd="~/renderdoc_1.36/bin/qrenderdoc";;
		"-c") 
			cd build/MonoVulkan
			make
			cd ../..;;
		"-l")
			cmd="$cmd > output.log";;
		"-s")
			compile_spirv
			return_val=$?;;

	esac
done

cd ./bin/debug
eval $cmd
