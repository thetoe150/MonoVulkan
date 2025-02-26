#version 450

layout(set = 0, binding = 1) uniform sampler2D directionalShadowMap;

layout(location = 0) in vec3 vPos;
layout(location = 0) out vec4 outFragColor;
layout(location = 1) out vec4 outBloomThreadhold;

void main() {
	outFragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
