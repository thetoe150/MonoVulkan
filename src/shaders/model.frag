#version 450

layout(binding = 1) uniform sampler2D u_texSampler;

layout(location = 0) in vec3 v_fragColor;
layout(location = 1) in vec2 v_fragTexCoord;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform GraphicPushConstant{
	bool useTexture;
} p_const;

void main() {
	if(p_const.useTexture)
		outColor = texture(u_texSampler, v_fragTexCoord);
	else
		outColor = vec4(v_fragColor, 0.5f);
}
