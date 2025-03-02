#version 450

layout(set = 0, binding = 1) uniform sampler2D directionalShadowMap;

layout(location = 0) in vec3 v_worldSpaceFragPos;
layout(location = 1) in vec4 v_lightSpaceFragPos;

layout(location = 0) out vec4 outFragColor;
layout(location = 1) out vec4 outBloomThreadhold;

bool calculateDirectionalShadow() {
	vec3 projCoord = v_lightSpaceFragPos.xyz / v_lightSpaceFragPos.w;
	projCoord = projCoord * 0.5 + 0.5;

	float closestDepth = texture(directionalShadowMap, projCoord.xy).r;
	float currentDepth = projCoord.z;
	return currentDepth > closestDepth ? true : false;
}

void main() {
	if (calculateDirectionalShadow()) {
		outFragColor = vec4(0.01, 0.01, 0.01, 1.0);
	}
	else {
		outFragColor = vec4(1.0, 1.0, 1.0, 1.0);
	}
}

// void main() {
// 	vec3 projCoord = v_lightSpaceFragPos.xyz / v_lightSpaceFragPos.w;
// 	projCoord = projCoord * 0.5 + 0.5;
// 
// 	float closestDepth = texture(directionalShadowMap, projCoord.xy).r;
// 	float currentDepth = projCoord.z;
// 	if(currentDepth > closestDepth)
// 		outFragColor = vec4(0.01, 0.01, 0.01, 1.0);
// 	else if(currentDepth < closestDepth)
// 		outFragColor = vec4(1.0, 1.0, 1.0, 1.0);
// 	else
// 		outFragColor = vec4(1.0, 0.0, 0.0, 1.0);
// }

// float ShadowCalculation(vec4 fragPosLightSpace)
// {
//     // perform perspective divide
//     vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
//     // transform to [0,1] range
//     projCoords = projCoords * 0.5 + 0.5;
//     // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
//     float closestDepth = texture(shadowMap, projCoords.xy).r; 
//     // get depth of current fragment from light's perspective
//     float currentDepth = projCoords.z;
//     // calculate bias (based on depth map resolution and slope)
//     vec3 normal = normalize(fs_in.Normal);
//     vec3 lightDir = normalize(lightPos - fs_in.FragPos);
//     float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);
//     // check whether current frag pos is in shadow
//     // float shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;
//     // PCF
//     float shadow = 0.0;
//     vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
//     for(int x = -1; x <= 1; ++x)
//     {
//         for(int y = -1; y <= 1; ++y)
//         {
//             float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r; 
//             shadow += currentDepth - bias > pcfDepth  ? 1.0 : 0.0;        
//         }    
//     }
//     shadow /= 9.0;
//     
//     // keep the shadow at 0.0 when outside the far_plane region of the light's frustum.
//     if(projCoords.z > 1.0)
//         shadow = 0.0;
//         
//     return shadow;
// }
