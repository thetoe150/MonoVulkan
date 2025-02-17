#version 450

layout (set = 0, binding = 0) uniform ShadowUniform {
    mat4 viewProj;
} u_shadowUniform;

layout (set = 0, binding = 1) uniform ShadowPerMeshUniform {
    mat4 candlesModel;
} u_perMeshUniform;

layout (location = 0) in vec3 a_position;

void main() {
	gl_Position = u_shadowUniform.viewProj * u_perMeshUniform.candlesModel * vec4(a_position, 1.0);
	// gl_VertexIndex;
}
