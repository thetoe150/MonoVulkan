#version 450

layout (location = 0) in vec3 a_pos;
layout (location = 0) out vec3 v_pos;

layout (set = 0, binding = 0) uniform Transform{
	mat4 camViewProj;
} u_transform;

void main() {
	v_pos = a_pos;
	vec4 pos = u_transform.camViewProj * vec4(a_pos, 1.0);
	gl_Position = pos.xyww;
}
