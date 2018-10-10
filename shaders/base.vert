#version 450

#define MAX_SOFTBODY_VERT 10
#define MAX_INSTANCE_LIGHTS 4

layout(binding = 0) uniform shared_ubo {
  mat4 view;
  mat4 projection;
} shared_data;

layout(binding = 1, std140) uniform instance_ubo {
  mat4 model;
  vec4 lights[MAX_INSTANCE_LIGHTS * 2];
  vec3 position_offsets[MAX_SOFTBODY_VERT];
  vec3 normal_offsets[MAX_SOFTBODY_VERT];
} instance;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec3 fragPosition;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;

out gl_PerVertex {
  vec4 gl_Position;
};

void main() {
  vec4 position = instance.model
    * vec4(inPosition + instance.position_offsets[gl_VertexIndex], 1);

  fragPosition = position.xyz;
  fragColor = inColor;

  fragNormal = (
    instance.model
      * vec4(inNormal + instance.normal_offsets[gl_VertexIndex], 0)
  ).xyz;
  fragNormal = normalize(fragNormal);

  gl_Position = shared_data.projection * shared_data.view * position;
}
