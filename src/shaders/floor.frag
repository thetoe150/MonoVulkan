#version 450

layout(set = 0, binding = 0) uniform Transform{
	mat4 model;
	mat4 camViewProj;
	mat4 lightViewProj;
	vec3 camPos;
	vec3 lightPos;
} u_transform;

layout(set = 0, binding = 1) uniform sampler2D directionalShadowMap;

layout(location = 0) in vec3 v_worldSpaceFragPos;
layout(location = 1) in vec4 v_lightSpaceFragPos;

layout(location = 0) out vec4 outFragColor;
layout(location = 1) out vec4 outBloomThreadhold;

float calculateShadow(vec3 projCoord) {
	vec3 lightDir = normalize(u_transform.lightPos - v_worldSpaceFragPos);
	const vec3 normal = vec3(0.0, 1.0, 0.0);
	float bias = max(0.05 * 1.0 - dot(normal, lightDir), 0.005);
	float currentDepth = projCoord.z;
	vec2 texelSize = 1.0 / textureSize(directionalShadowMap, 0);
	float shadow = 0.0;
	for (int x = -1; x < 1; x++) {
		for (int y = -1; y < 1; y++) {
			float pcfDepth = texture(directionalShadowMap, projCoord.xy + texelSize * vec2(x, y)).r;
			shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
		}
	}

	return shadow / 9.0;
}

void main() {
	vec3 projCoord = v_lightSpaceFragPos.xyz / v_lightSpaceFragPos.w;
	projCoord = projCoord * 0.5 + 0.5;

	// without this code, mapped depth return 1 but current z > 0
	// will result in pixel in shadow
	if (projCoord.z > 1.0 || projCoord.z < -1.0) {
		outFragColor = vec4(1.0, 1.0, 1.0, 1.0);
		return;
	}

	if (projCoord.x > 1.0 || projCoord.y > 1.0 || 
		projCoord.x < 0.0 || projCoord.y < 0.0) {

		outFragColor = vec4(1.0, 1.0, 1.0, 1.0);
		return;
	}

	float shadow = 1.0 - calculateShadow(projCoord) - 0.1;
	outFragColor = vec4(shadow, shadow, shadow, 1.0);
}
