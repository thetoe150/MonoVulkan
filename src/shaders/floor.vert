#version 450

layout(location = 0) in vec3 aPos;
layout(location = 0) out vec3 vPos;
layout(set = 0, binding = 0) uniform Transform{
	mat4 model;
	mat4 viewProj;
} u_transform;

void main() {
	gl_Position = u_transform.viewProj * u_transform.model * vec4(aPos, 1.0);
}
