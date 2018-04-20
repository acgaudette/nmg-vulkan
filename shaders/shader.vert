#version 450

#define MAX_SOFTBODY_VERT 12

layout(binding = 0) uniform shared_ubo {
  mat4 view;
  mat4 projection;
} shared_data;

layout(binding = 1, std140) uniform instance_ubo {
  mat4 model;
  vec3 offsets[MAX_SOFTBODY_VERT];
} this_data;

layout(location = 0) in  vec3 inPosition;
layout(location = 1) in  vec3 inNormal;
layout(location = 2) in  vec3 inColor;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec3 fragColor;

out gl_PerVertex {
  vec4 gl_Position;
};

void main() {
  gl_Position = shared_data.projection * shared_data.view
    * this_data.model
    * vec4(inPosition + this_data.offsets[gl_VertexIndex], 1);

  fragColor = inColor;
}
