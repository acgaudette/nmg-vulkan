#version 450

#define MAX_SOFTBODY_VERT 10
#define MAX_INSTANCE_LIGHTS 4
#define AMBIENT 0.1

layout(binding = 1, std140) uniform instance_ubo {
  mat4 model;
  vec4 lights[MAX_INSTANCE_LIGHTS * 2];
} this_data;

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
  outColor = vec4(fragColor, 1);
}
