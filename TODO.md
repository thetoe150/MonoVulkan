### TODO:
- change LOD base on distance
- frustum culling object on CPU
- add Tracy on Debian
- add directional shadow
- add point shadow
- add point light
- add dynamic light/shadow
- add skymap
- add multi-view port
- separate candles base and flame
- remove phong shading with flame
- adjust face culling
- refactor vertex attribute definition
- add barrier when compute and graphic queue is not the same 
- more *physic-based* realistic vortex snowfall
- shader reflection
- print total vertex + fragments calculation a frame
- move animation calculation to compute shader

### BUGS:
- renderDoc always choose on board card
- mouse scroll go crazy go stupid

### Done
- add animation for candle - done
- Push constants - done
- Specialization constants - done
- Instanced rendering - done
- Dynamic uniforms - done
- Pipeline cache - done
- Compute shader
    - Shared memory - done
- this makefile script is banana - no re-compile sometime - done, add premnake
- make a better snowflake model and import with gltf - done
- add light - done
- hot reload - done
- fix that Vulkan instance extensions layers - belong to tracy, don't care
- add LODs - done

### Note
- smaller the data structure used in a loop - the faster it runs for some reason
    + bring the whole map<vector<>> into a loop and iterate is 15ms
    + bring the vector<> into a loop and iterate is 4ms
    + bring the only the only pointer to vector<>.data and access through [] takes 2ms

### Big Problem
- manage and free memory, too many malloc - a allocator?
- data conflict between shader - gltf model - vk object -> reflection + model descriptor after load?
- synchronization ?
