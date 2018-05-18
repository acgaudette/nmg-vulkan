#version 450

#define MAX_SOFTBODY_VERT 10
#define MAX_INSTANCE_LIGHTS 4
#define AMBIENT 0.1

layout(binding = 1, std140) uniform instance_ubo {
  mat4 model;
  vec4 lights[MAX_INSTANCE_LIGHTS * 2];
} this_data;

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
  vec3 total_light = vec3(0);

  for (int i = 0; i < 2 * MAX_INSTANCE_LIGHTS; i += 2) {
    float radius = this_data.lights[i].w;
    if (radius == 0) continue;

    vec3 light_vector = this_data.lights[i].xyz;
    float light = this_data.lights[i + 1].w; // Intensity

    if (radius == -1) { // Directional
      light *= max(0.0, dot(fragNormal, light_vector));
    }

    else { // Point
      vec3 diff = light_vector - fragPosition;
      float dist = distance(light_vector, fragPosition);

      light *= (1 - dist / radius) * max(0.0, dot(fragNormal, diff / dist));
    }

    total_light += this_data.lights[i + 1].rgb * light;
  }

  total_light = max(vec3(AMBIENT), total_light);
  outColor = vec4(fragColor * total_light, 1);
}
