Set-PSDebug -Trace 1

glslc -fshader-stage=vertex src/shaders/model.vert -o src/shaders/model.vert.spv
glslc -fshader-stage=fragment src/shaders/model.frag -o src/shaders/model.frag.spv
glslc -fshader-stage=comp src/shaders/snowflake.comp -o src/shaders/snowflake.comp.spv

spirv-dis src/shaders/model.vert.spv > src/shaders/model.vert.spvasm
spirv-dis src/shaders/model.frag.spv > src/shaders/model.frag.spvasm
spirv-dis src/shaders/snowflake.comp.spv > src/shaders/snowflake.comp.spvasm
