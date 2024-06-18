glslc -fshader-stage=vertex src/shaders/model.vert -o src/shaders/model.vert.spv
glslc -fshader-stage=fragment src/shaders/model.frag -o src/shaders/model.frag.spv
glslc -fshader-stage=comp src/shaders/snowflake.comp -o src/shaders/snowflake.comp.spv
