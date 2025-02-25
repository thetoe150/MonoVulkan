#version 450

layout(location = 0) in vec2 vTexCoords;
layout(location = 0) out vec4 outFragColor;

layout(set = 0, binding = 0) uniform sampler2D u_texSampler;

layout (constant_id = 0) const float zFar = 15;

float LinearizeDepth(float depth)
{
  float n = 1;
  float f = zFar;
  float z = depth;
  return (2.0 * n) / (f + n - z * (f - n));	
}

void main() {
	float d = texture(u_texSampler, vTexCoords).r;
	outFragColor = vec4(vec3(1.0 - LinearizeDepth(d)), 1.0);
}
