#version 450

layout(location = 0) in vec2 vTexCoords;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D baseSampler;
layout(set = 0, binding = 1) uniform sampler2D bloomSampler;

layout (push_constant) uniform DataPushConstant{
	float exposure;
} p_const;

const float gamma = 2.2;

void main() {
	if (true) {
		vec3 baseColor = pow(texture(baseSampler, vTexCoords).rgb, vec3(gamma));
		vec3 bloomColor = pow(texture(bloomSampler, vTexCoords).rgb, vec3(gamma));

		vec3 hdrColor = baseColor + bloomColor;

		// vec3 mapped = hdrColor / (hdrColor + vec3(1.0));
		vec3 mapped = vec3(1.0) - exp(-hdrColor * p_const.exposure);
		vec3 result = pow(mapped, vec3(1.0 / gamma));
		outColor = vec4(result, 1.0);
	}
	else {
		vec3 baseColor = texture(baseSampler, vTexCoords).rgb;
		vec3 bloomColor = texture(bloomSampler, vTexCoords).rgb;

		vec3 hdrColor = baseColor + bloomColor;
		// vec3 mapped = hdrColor / (hdrColor + vec3(1.0));
		vec3 mapped = vec3(1.0) - exp(-hdrColor * p_const.exposure);

		outColor = vec4(mapped, 1.0);
	}
}
