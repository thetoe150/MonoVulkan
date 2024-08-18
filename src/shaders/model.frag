#version 450

layout(set = 0, binding = 1) uniform LightingObject {
    vec3 lightDir;
    vec3 camPos;
} u_lighting;

layout(set = 1, binding = 2) uniform sampler2D u_texSampler;
layout(set = 1, binding = 3) uniform sampler2D u_normalSampler;

layout(location = 0) in vec2 v_fragTexCoord;

layout(location = 0) out vec4 outColor;

layout (constant_id = 0) const bool useTexture = true;

void main() {
	if(useTexture)
		outColor = texture(u_texSampler, v_fragTexCoord);
	else
		outColor = vec4(vec3(0.9f, 0.9f, 1.0f), 0.5f);
}
