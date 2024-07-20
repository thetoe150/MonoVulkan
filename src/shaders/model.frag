#version 450

layout(set = 1, binding = 1) uniform sampler2D u_texSampler;

layout(location = 0) in vec3 v_fragColor;
layout(location = 1) in vec2 v_fragTexCoord;

layout(location = 0) out vec4 outColor;

layout (constant_id = 0) const bool useTexture = true;

void main() {
	if(useTexture)
		outColor = texture(u_texSampler, v_fragTexCoord);
	else
		outColor = vec4(vec3(0.9f, 0.9f, 1.0f), 0.5f);
}
