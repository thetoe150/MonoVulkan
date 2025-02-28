#version 450

layout(location = 0) in vec3 a_pos;

layout(location = 0) out vec3 v_worldSpaceFragPos;
layout(location = 1) out vec4 v_lightSpaceFragPos;

layout(set = 0, binding = 0) uniform Transform{
	mat4 model;
	mat4 camViewProj;
	mat4 lightViewProj;
	vec3 camPos;
	vec3 lightPos;
} u_transform;

void main(){
	vec4 worldPos = u_transform.model * vec4(a_pos, 1.0);
	v_lightSpaceFragPos = u_transform.lightViewProj * worldPos;

	gl_Position = u_transform.camViewProj * worldPos;
}
