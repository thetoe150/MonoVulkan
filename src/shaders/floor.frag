#version 450

layout(set = 1, binding = 1) uniform sampler2D directionalShadowMap;

layout(location = 0) in vec3 vPos;
layout(location = 0) out vec4 outFragColor;

void main() {
	outFragColor = vec4(1.0);
}
