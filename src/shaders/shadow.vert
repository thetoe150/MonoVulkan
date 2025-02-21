#version 450

layout (constant_id = 0) const int meshCount = 10;
layout (push_constant) uniform vertexCount{
	int ;
} pc_vertexCount;

layout (set = 0, binding = 0) uniform ShadowUniform {
    mat4 viewProj;
} u_shadowUniform;

layout (set = 0, binding = 1) uniform ShadowPerMeshUniform {
    mat4 candlesPerMeshModel[k_meshCount];
} u_perMeshUniform;

layout (std140, binding = 2) buffer ShadowMeshLookup{
	unsigned char[ ];
};

layout (location = 0) in vec3 a_position;

void main() {
	unsigned char meshTransformIdx = ShadowMeshLookup[gl_VertexID];
	
	gl_Position = u_shadowUniform.viewProj * candlesPerMeshModel[meshTransformIdx] * vec4(a_position, 1.0);
	// gl_VertexIndex;
}
