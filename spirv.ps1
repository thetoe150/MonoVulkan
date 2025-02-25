glslc -fshader-stage=vertex src/shaders/snowflake.vert -o src/shaders/snowflake.vert.spv
glslc -fshader-stage=vertex src/shaders/candles.vert -o src/shaders/candles.vert.spv
glslc -fshader-stage=vertex src/shaders/shadow_batch.vert -o src/shaders/shadow_batch.vert.spv
glslc -fshader-stage=vertex src/shaders/quad.vert -o src/shaders/quad.vert.spv
glslc -fshader-stage=fragment src/shaders/snowflake.frag -o src/shaders/snowflake.frag.spv
glslc -fshader-stage=fragment src/shaders/candles.frag -o src/shaders/candles.frag.spv
glslc -fshader-stage=fragment src/shaders/shadow_viewport.frag -o src/shaders/shadow_viewport.frag.spv
glslc -fshader-stage=fragment src/shaders/bloom.frag -o src/shaders/bloom.frag.spv
glslc -fshader-stage=fragment src/shaders/combine.frag -o src/shaders/combine.frag.spv
glslc -fshader-stage=compute src/shaders/snowflake.comp -o src/shaders/snowflake.comp.spv

spirv-dis src/shaders/snowflake.vert.spv > src/shaders/snowflake.vert.spvasm
spirv-dis src/shaders/candles.vert.spv > src/shaders/candles.vert.spvasm
spirv-dis src/shaders/quad.vert.spv > src/shaders/quad.vert.spvasm
spirv-dis src/shaders/candles.frag.spv > src/shaders/candles.frag.spvasm
spirv-dis src/shaders/snowflake.frag.spv > src/shaders/snowflake.frag.spvasm
spirv-dis src/shaders/shadow_batch.vert.spv > src/shaders/shadow_batch.vert.spvasm
spirv-dis src/shaders/shadow_viewport.frag.spv > src/shaders/shadow_viewport.frag.spvasm
spirv-dis src/shaders/bloom.frag.spv > src/shaders/bloom.frag.spvasm
spirv-dis src/shaders/combine.frag.spv > src/shaders/combine.frag.spvasm
spirv-dis src/shaders/snowflake.comp.spv > src/shaders/snowflake.comp.spvasm
