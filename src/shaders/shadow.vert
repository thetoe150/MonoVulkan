#version 450

layout (constant_id = 0) const uint c_meshCapacity = 10;
layout (constant_id = 1) const uint c_instanceCapacity = 10;

layout (push_constant) uniform Count{
	uint vertex;
	uint instance;
} p_count;

layout (set = 0, binding = 0) uniform ShadowUniform {
    mat4 viewProj;
} u_shadowUniform;

layout (set = 0, binding = 1) uniform PerMeshTransform{
	mat4 value[c_meshCapacity];
} u_perMeshTransform;

layout (set = 0, binding = 2) uniform PerInstanceTransform{
	mat4 value[c_instanceCapacity];
} u_perInstanceTransform;

layout (location = 0) in vec4 a_position;

void main() {
	gl_Position = u_shadowUniform.viewProj * u_perInstanceTransform.value[gl_InstanceIndex] * u_perMeshTransform.value[int(a_position.w)] * vec4(a_position.xyz, 1.0);
	// gl_VertexIndex;
}
