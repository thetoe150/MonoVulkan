#version 450

layout(location = 0) in vec2 vTexCoords;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D baseSampler;
layout(set = 0, binding = 1) uniform sampler2D bloomSampler;

layout (push_constant) uniform DataPushConstant{
	int useBloom;
} p_const;

void main() {

	outColor = vec4(1.0, 1.0, 0.0, 1.0);
	return;
	vec3 baseColor = texture(baseSampler, vTexCoords).rgb;
	outColor = vec4(baseColor, 1.0);
	return;

	baseColor += texture(bloomSampler, vTexCoords).rgb;

    // tone mapping
    // vec3 result = vec3(1.0) - exp(-hdrColor * exposure);
    // also gamma correct while we're at it       
    // const float gamma = 2.2;
    // result = pow(result, vec3(1.0 / gamma));
	outColor = vec4(baseColor, 1.0);
}
