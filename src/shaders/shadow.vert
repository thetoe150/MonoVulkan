#version 450

layout (set = 0, binding = 0) uniform ShadowUniform {
    mat4 viewProj;
    mat4 lightPos;
    mat4 candlesModel;
} u_shadowUniform;

layout (location = 0) in vec3 a_position;

void main() {
	gl_Position = ushadowUniform.viewProj * candlesModel * vec4(a_position, 1.0);
	// gl_VertexIndex;
}
