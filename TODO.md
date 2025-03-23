### TODO:
- change LOD base on distance
- frustum culling object on CPU
- add Tracy on Debian
- add point shadow
- add point light
- add dynamic light/shadow
- separate candles base and flame
- remove phong shading with flame
- adjust face culling
- add barrier when compute and graphic queue is not the same 
- more *physic-based* realistic vortex snowfall
- refactor vertex attribute definition
- shader reflection - in progress
- print total vertex + fragments calculation a frame
- move animation calculation to compute shader

### BUGS:
- renderDoc always choose on board card
- renderDoc fail to capture postFx
- mouse scroll go crazy go stupid

### Note
- smaller the data structure used in a loop - the faster it runs for some reason
    + bring the whole map<vector<>> into a loop and iterate is 15ms
    + bring the vector<> into a loop and iterate is 4ms
    + bring the only the only pointer to vector<>.data and access through [] takes 2ms

### Big Problem
- manage and free memory, too many malloc - a allocator?
- data conflict between shader - gltf model - vk object -> reflection + model descriptor after load?
- synchronization ?
