#version 450

layout(set = 0, binding = 0) uniform CandlesPerMeshTransform {
    mat4 model;
} u_perMesh;

layout(set = 0, binding = 1) uniform CandlesLightingTransform {
    mat4 viewProj;
    vec3 lightPos;
    vec3 camPos;
} u_lightingTransform;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec4 a_tangent;
layout(location = 3) in vec2 a_texCoord;

layout(location = 4) in vec3 instancePos;

layout(location = 0) out vec2 v_texCoord;
layout(location = 1) out vec3 v_tangentFragPos;
layout(location = 2) out vec3 v_tangentLightPos;
layout(location = 3) out vec3 v_tangentCamPos;
// in case not using normal mapping
layout(location = 4) out vec3 v_fragPosition;
layout(location = 5) out vec3 v_normal;

// just for testing
layout(location = 6) out vec3 v_tangent;

void main() {
	mat4 instanceModel;
	instanceModel[0] = vec4(1.0f, 0.0f, 0.0f, 0.0f);
	instanceModel[1] = vec4(0.0f, 1.0f, 0.0f, 0.0f);
	instanceModel[2] = vec4(0.0f, 0.0f, 1.0f, 0.0f);
	instanceModel[3] = vec4(instancePos.x, instancePos.y, instancePos.z, 1.0f);

	mat4 model = instanceModel * u_perMesh.model;
	vec3 fragPos = vec3(model * vec4(a_position, 1.0));

	mat3 worldModel = transpose(inverse(mat3(model)));
	vec3 N = normalize(worldModel * a_normal);
	vec3 T = normalize(worldModel * vec3(a_tangent));
	// ???
	T = normalize(T - dot(T, N) * N);
	vec3 B = cross(N, T);
	mat3 TBN = transpose(mat3(T, B, N));

	v_tangentFragPos = TBN * fragPos;
	v_tangentLightPos = TBN * u_lightingTransform.lightPos;
	v_tangentCamPos = TBN * u_lightingTransform.camPos;

    v_texCoord = a_texCoord;
    v_fragPosition = fragPos;
	// normalize normal here won't help if use MSAA with 16bit format
	// white artifact happen maybe because MSAA rasterization make normalized normal not normalized anymore
    v_normal = a_normal;

	// just for tessting
	v_tangent = vec3(a_tangent);
    gl_Position = u_lightingTransform.viewProj * model * vec4(a_position, 1.0);
}
