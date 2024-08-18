Set-PSDebug -Trace 1

glslc -fshader-stage=vertex src/shaders/candles.vert -o src/shaders/candles.vert.spv
glslc -fshader-stage=fragment src/shaders/candles.frag -o src/shaders/candles.frag.spv
glslc -fshader-stage=comp src/shaders/snowflake.comp -o src/shaders/snowflake.comp.spv

spirv-dis src/shaders/candles.vert.spv > src/shaders/candles.vert.spvasm
spirv-dis src/shaders/candles.frag.spv > src/shaders/candles.frag.spvasm
spirv-dis src/shaders/snowflake.comp.spv > src/shaders/snowflake.comp.spvasm
