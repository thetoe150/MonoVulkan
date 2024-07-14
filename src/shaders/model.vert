#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;

layout(location = 3) in vec3 instancePos;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;

void main() {
	mat4 model = ubo.model;
	model[3][0] = model[3][0] + instancePos.x;
	model[3][1] = model[3][1] + instancePos.y;
	model[3][2] = model[3][2] + instancePos.z;

    gl_Position = ubo.proj * ubo.view * model * vec4(inPosition, 1.0);
    fragColor = inColor;
    fragTexCoord = inTexCoord;
}
