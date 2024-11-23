### TODO:
- change LOD base on distance
- frustum culling object on CPU
- add shadow
- add skymap
- add tesselated terrain
- add multi-view port
- add dof when moving cam?
- add barrier when compute and graphic queue is not the same 
- more *physic-based* realistic vortex snow-fall
- PBR
- shader reflection
- Vulkan features:
    - Separate images and sampler descriptors
    - Multi-threaded command buffer generation
    - Multiple subpasses
    - Compute shaders
        - Asynchronous compute
        - Atomic operations
        - Subgroups
- print total vertex + fragments calculation a frame
- move animation calculation to compute shader

### BUGS:
- candles flame animation crash renderDoc
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
