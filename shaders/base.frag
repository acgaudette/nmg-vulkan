#version 450

#define MAX_SOFTBODY_VERT 10
#define MAX_INSTANCE_LIGHTS 4
#define AMBIENT 0.1

struct Light {
  vec3 vector;
  float radius;
  vec3 color;
  float intensity;
};

layout(binding = 1, std140) uniform instance_ubo {
  mat4 _;
  Light lights[MAX_INSTANCE_LIGHTS];
} instance;

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
  vec3 total_light = vec3(0);

  for (int i = 0; i < MAX_INSTANCE_LIGHTS; ++i) {
    // Ignore lights with no radius
    if (instance.lights[i].radius == 0) continue;

    float light = instance.lights[i].intensity;
    float radius = instance.lights[i].radius;

    if (radius == -1) { // Directional
      light *= max(0, dot(fragNormal, instance.lights[i].vector));
    }

    else { // Point
      vec3 diff = instance.lights[i].vector - fragPosition;
      float dist = length(diff);

      // Compute attenuation
      float atten = max(0, 1 - (dist * dist) / (radius * radius));
      atten *= atten;

      light *= max(0.0, dot(fragNormal, diff / dist))
        * atten;
    }

    total_light += instance.lights[i].color * light;
  }

  total_light = max(vec3(AMBIENT), total_light);
  outColor = vec4(fragColor * total_light, 1);
}
