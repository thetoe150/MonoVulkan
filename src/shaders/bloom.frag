#version 450

layout(location = 0) in vec2 vTexCoords;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D u_texSampler;
layout (constant_id = 0) const int isHorizontal = 0;

// From the OpenGL Super bible
const float weights[] = float[](0.0896631113333857,
								0.0874493212267511,
								0.0811305381519717,
								0.0715974486241365,
								0.0601029809166942,
								0.0479932050577658,
								0.0364543006660986,
								0.0263392293891488,
								0.0181026699707781,
								0.0118349786570722,
								0.0073599963704157,
								0.0043538453346397,
								0.0024499299678342);

void main() {
	float blurScale = 0.0006;
	const float blurStrength = 1.0;

	float ar = 1.0;
	// Aspect ratio for vertical blur pass
	if (isHorizontal == 0)
	{
		// vec2 ts = textureSize(u_texSampler, 0);
		// ar = ts.y / ts.x;
		blurScale = 0.0012;
	}

	vec2 P = vTexCoords.yx - vec2(0, (weights.length() >> 1) * ar * blurScale);

	// we can't use texUnit for some reason
	vec2 texUnit = 1 / textureSize(u_texSampler, 0);
	vec3 res = texture(u_texSampler, vTexCoords).rgb * weights[0];
	if (isHorizontal == 1) {
		// blurScale = texUnit.x * 3;
		for(int i = 1; i < 13; i++) {
			vec2 du = vec2(i * blurScale, 0.0) * ar;
			res += texture(u_texSampler, vTexCoords + du).rgb * weights[i];
			res += texture(u_texSampler, vTexCoords - du).rgb * weights[i];
		}
	}
	else {
		// blurScale = texUnit.y * 3;
		for(int i = 1; i < 13; i++) {
			vec2 dv = vec2(0.0, i * blurScale) * ar;
			res += texture(u_texSampler, vTexCoords + dv).rgb * weights[i];
			res += texture(u_texSampler, vTexCoords - dv).rgb * weights[i];
		}
	}

	outColor = vec4(res, 1.0);
	// outColor = vec4(1.0, 1.0, 0.0, 1.0);
}

// void main1()
// {
// 	// From the OpenGL Super bible
// 	const float weights[] = float[](0.0024499299678342,
// 									0.0043538453346397,
// 									0.0073599963704157,
// 									0.0118349786570722,
// 									0.0181026699707781,
// 									0.0263392293891488,
// 									0.0364543006660986,
// 									0.0479932050577658,
// 									0.0601029809166942,
// 									0.0715974486241365,
// 									0.0811305381519717,
// 									0.0874493212267511,
// 									0.0896631113333857,
// 									0.0874493212267511,
// 									0.0811305381519717,
// 									0.0715974486241365,
// 									0.0601029809166942,
// 									0.0479932050577658,
// 									0.0364543006660986,
// 									0.0263392293891488,
// 									0.0181026699707781,
// 									0.0118349786570722,
// 									0.0073599963704157,
// 									0.0043538453346397,
// 									0.0024499299678342);
// 
// 
// 	const float blurScale = 0.003;
// 	const float blurStrength = 1.0;
// 
// 	float ar = 1.0;
// 	// Aspect ratio for vertical blur pass
// 	if (!isHorizontal)
// 	{
// 		vec2 ts = textureSize(u_texSampler, 0);
// 		ar = ts.y / ts.x;
// 	}
// 
// 	vec2 P = vTexCoords.yx - vec2(0, (weights.length() >> 1) * ar * blurScale);
// 
// 	vec4 color = vec4(0.0);
// 	for (int i = 0; i < weights.length(); i++)
// 	{
// 		vec2 dv = vec2(0.0, i * blurScale) * ar;
// 		color += texture(u_texSampler, P + dv) * weights[i] * blurStrength;
// 	}
// 
// 	outColor = color;
// }
