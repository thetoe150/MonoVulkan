Set-PSDebug -Trace 1

glslc -fshader-stage=vertex src/shaders/candles.vert -o src/shaders/candles.vert.spv
glslc -fshader-stage=fragment src/shaders/candles.frag -o src/shaders/candles.frag.spv
glslc -fshader-stage=comp src/shaders/snowflake.comp -o src/shaders/snowflake.comp.spv
glslc -fshader-stage=vertex src/shaders/bloom.vert -o src/shaders/bloom.vert.spv
glslc -fshader-stage=fragment src/shaders/bloom.frag -o src/shaders/bloom.frag.spv
glslc -fshader-stage=vertex src/shaders/combine.vert -o src/shaders/combine.vert.spv
glslc -fshader-stage=fragment src/shaders/combine.frag -o src/shaders/combine.frag.spv

spirv-dis src/shaders/candles.vert.spv > src/shaders/candles.vert.spvasm
spirv-dis src/shaders/candles.frag.spv > src/shaders/candles.frag.spvasm
spirv-dis src/shaders/snowflake.comp.spv > src/shaders/snowflake.comp.spvasm
spirv-dis src/shaders/bloom.vert.spv > src/shaders/bloom.vert.spvasm
spirv-dis src/shaders/bloom.frag.spv > src/shaders/bloom.frag.spvasm
spirv-dis src/shaders/combine.vert.spv > src/shaders/combine.vert.spvasm
spirv-dis src/shaders/combine.frag.spv > src/shaders/combine.frag.spvasm
