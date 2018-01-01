#version 450

layout(binding = 0) uniform VP {
  mat4 view;
  mat4 projection;
} shared;

layout(binding = 1) uniform M {
  mat4 model;
} this;

layout(location = 0) in  vec3 inPosition;
layout(location = 1) in  vec3 inColor;
layout(location = 0) out vec3 fragColor;

out gl_PerVertex {
  vec4 gl_Position;
};

void main() {
  gl_Position = shared.projection
    * shared.view
    * this.model
    * vec4(inPosition, 1);

  fragColor = inColor;
}
