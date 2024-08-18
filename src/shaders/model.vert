#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} u_transform;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec4 a_tangent;
layout(location = 3) in vec2 a_texCoord;

layout(location = 4) in vec3 instancePos;

layout(location = 0) out vec2 fragTexCoord;

void main() {
	mat4 instanceModel;
	instanceModel[0] = vec4(1.0f, 0.0f, 0.0f, 0.0f);
	instanceModel[1] = vec4(0.0f, 1.0f, 0.0f, 0.0f);
	instanceModel[2] = vec4(0.0f, 0.0f, 1.0f, 0.0f);
	instanceModel[3] = vec4(instancePos.x, instancePos.y, instancePos.z, 1.0f);
	// equivalent to this
	// mat4 model = u_transform.model;
	// model[3][0] = model[3][0] + instancePos.x;
	// model[3][1] = model[3][1] + instancePos.y;
	// model[3][2] = model[3][2] + instancePos.z;

	mat4 model = instanceModel * u_transform.model;

    gl_Position = u_transform.proj * u_transform.view * model * vec4(a_position, 1.0);
    fragTexCoord = a_texCoord;
}
