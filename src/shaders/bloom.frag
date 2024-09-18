#version 450

layout(location = 0) in vec2 vTexCoords;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D u_texSampler;
layout (constant_id = 0) const bool isHorizontal = true;

float weight[5] = float[] (0.2270270270, 0.1945945946, 0.1216216216, 0.0540540541, 0.0162162162);

void main() {
	outColor = vec4(texture(u_texSampler, vTexCoords).rgb, 1.0);
	return;

	vec2 texOffset = 1 / textureSize(u_texSampler, 0);
	vec3 res = texture(u_texSampler, vTexCoords).rgb * weight[0];
	if (isHorizontal) {
		for(int i = 1; i < 5; i++) {
			res += texture(u_texSampler, vTexCoords + vec2(texOffset.x * i, 0.0)).rgb * weight[i];
			res += texture(u_texSampler, vTexCoords - vec2(texOffset.x * i, 0.0)).rgb * weight[i];
		}
	}
	else {
		for(int i = 1; i < 5; i++) {
			res += texture(u_texSampler, vTexCoords + vec2(0.0, texOffset.y * i)).rgb * weight[i];
			res += texture(u_texSampler, vTexCoords - vec2(0.0, texOffset.y * i)).rgb * weight[i];
		}
	}

	outColor = vec4(res, 1.0);
}
