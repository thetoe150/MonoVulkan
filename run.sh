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

	glslc -fshader-stage=vertex src/shaders/candles.vert -o src/shaders/candles.vert.spv
	glslc -fshader-stage=fragment src/shaders/candles.frag -o src/shaders/candles.frag.spv
	glslc -fshader-stage=vert src/shaders/snowflake.vert -o src/shaders/snowflake.vert.spv
	glslc -fshader-stage=frag src/shaders/snowflake.frag -o src/shaders/snowflake.frag.spv
	glslc -fshader-stage=comp src/shaders/snowflake.comp -o src/shaders/snowflake.comp.spv
	glslc -fshader-stage=vertex src/shaders/bloom.vert -o src/shaders/bloom.vert.spv
	glslc -fshader-stage=fragment src/shaders/bloom.frag -o src/shaders/bloom.frag.spv
	glslc -fshader-stage=vertex src/shaders/combine.vert -o src/shaders/combine.vert.spv
	glslc -fshader-stage=fragment src/shaders/combine.frag -o src/shaders/combine.frag.spv

	spirv-dis src/shaders/candles.vert.spv > src/shaders/candles.vert.spvasm
	spirv-dis src/shaders/candles.frag.spv > src/shaders/candles.frag.spvasm
	spirv-dis src/shaders/snowflake.vert.spv > src/shaders/snowflake.vert.spvasm
	spirv-dis src/shaders/snowflake.frag.spv > src/shaders/snowflake.frag.spvasm
	spirv-dis src/shaders/snowflake.comp.spv > src/shaders/snowflake.comp.spvasm
	spirv-dis src/shaders/bloom.vert.spv > src/shaders/bloom.vert.spvasm
	spirv-dis src/shaders/bloom.frag.spv > src/shaders/bloom.frag.spvasm
	spirv-dis src/shaders/combine.vert.spv > src/shaders/combine.vert.spvasm
	spirv-dis src/shaders/combine.frag.spv > src/shaders/combine.frag.spvasm
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

