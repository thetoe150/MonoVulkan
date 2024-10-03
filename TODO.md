### TODO:
- add HDR and bloom
- add shadow
- add skymap
- add tesselated terrain
- add multi-view port
- add dof when moving cam?
- add barrier when compute and graphic queue is not the same 
- more *physic-based* realistic vortex snow-fall
- PBR
- shader reflection
- fix that Vulkan instance extensions layers
- Vulkan features:
    - Separate images and sampler descriptors
    - Multi-threaded command buffer generation
    - Multiple subpasses
    - Compute shaders
        - Asynchronous compute
        - Atomic operations
        - Subgroups

### BUGS:
- candles flame animation crash renderDoc
- candles base look weird

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
