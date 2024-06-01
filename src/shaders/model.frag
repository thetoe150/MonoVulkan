#version 450

layout(binding = 1) uniform sampler2D u_texSampler;

layout(location = 0) in vec3 a_fragColor;
layout(location = 1) in vec2 a_fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(u_texSampler, a_fragTexCoord);
}
