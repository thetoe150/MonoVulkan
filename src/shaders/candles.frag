#version 450

layout(set = 1, binding = 2) uniform sampler2D u_texSampler;
layout(set = 1, binding = 3) uniform sampler2D u_normalSampler;

layout(location = 0) in vec2 v_fragTexCoord;
layout(location = 1) in vec3 v_tangentFragPos;
layout(location = 2) in vec3 v_tangentLightPos;
layout(location = 3) in vec3 v_tangentCamPos;

layout(location = 0) out vec4 outColor;

layout (constant_id = 0) const bool useTexture = true;

void main() {
	vec3 normal = texture(u_normalSampler, v_fragTexCoord).rgb;
	vec3 n = normalize(normal * 2 - 1.0);
	vec3 color = texture(u_texSampler, v_fragTexCoord).rgb;

	vec3 l = normalize(v_tangentLightPos - v_tangentFragPos);
	vec3 c = normalize(v_tangentCamPos - v_tangentFragPos);
	vec3 r = reflect(-l, n);
	vec3 h = normalize(l + c);

	vec3 ambient = 0.2 * color;

	float diff = max(dot(l, n), 0.0);
	vec3 diffuse = diff * color;

	float spec = pow(max(dot(n, h), 0.0), 32.0);
	vec3 specular = vec3(0.1) * spec;

	outColor = vec4(ambient + diffuse + specular, 1.0);
}