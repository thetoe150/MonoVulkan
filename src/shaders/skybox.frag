#version 450

layout (location = 0) in vec3 v_pos;
layout (location = 0) out vec4 outFragColor;

layout (set = 0, binding = 1) uniform samplerCube cubeMap;

void main() {
	outFragColor = texture(cubeMap, v_pos);
}
