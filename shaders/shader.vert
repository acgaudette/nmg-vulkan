#version 450

layout(binding = 0) uniform UBO {
  mat4 model;
  mat4 view;
  mat4 projection;
} ubo;

layout(location = 0) in  vec3 inPosition;
layout(location = 1) in  vec3 inColor;
layout(location = 0) out vec3 fragColor;

out gl_PerVertex {
  vec4 gl_Position;
};

void main() {
  gl_Position = ubo.projection * ubo.view * ubo.model * vec4(inPosition, 1);
  fragColor = inColor;
}
