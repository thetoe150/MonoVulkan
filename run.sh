#!/bin/bash

echo "Script's PID: $$"
echo "Number of arguments passed to script: $#"
echo "All arguments passed to script: $@"
echo "Script's arguments separated into different variables: $1 $2 $3"

# if [[ $1 == "all_layers" ]]; then
# 	export VK_LOADER_LAYERS_ENABLE=*api_dump,*synchronization2,renderdoc_capture
# else
# 	export VK_LOADER_LAYERS_ENABLE=
# fi

cmd="./MONO"

export VK_LOADER_LAYERS_ENABLE=
for arg in $@ 
do
	case "$arg" in
		"-l") 
			export VK_LOADER_LAYERS_ENABLE=*api_dump,*synchronization2,renderdoc_capture;;
		"-r") 
			export VK_LOADER_LAYERS_ENABLE=renderdoc_capture;;
		"-c") 
			cd build/MonoVulkan
			make
			cd ../..;;
		"-d")
			cmd="$cmd > output.log";

	esac
done

cd ./bin/debug
eval $cmd
