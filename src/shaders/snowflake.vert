#version 450

layout(set = 0, binding = 0) uniform UniformTransform {
	mat4 model;
	mat4 view;
	mat4 proj;
} u_transform;

layout(location = 0) in vec3 a_position;

layout(location = 1) in vec3 a_instancePos;

void main() {
	mat4 instanceModel;
	instanceModel[0] = vec4(1.0f, 0.0f, 0.0f, 0.0f);
	instanceModel[1] = vec4(0.0f, 1.0f, 0.0f, 0.0f);
	instanceModel[2] = vec4(0.0f, 0.0f, 1.0f, 0.0f);
	instanceModel[3] = vec4(a_instancePos.x, a_instancePos.y, a_instancePos.z, 1.0f);

	mat4 model = instanceModel * u_transform.model;
	gl_Position = u_transform.proj * u_transform.view * model * vec4(a_position, 1.0);
}
