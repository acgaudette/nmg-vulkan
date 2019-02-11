#version 450 core

layout(binding = 0) uniform font_ubo {
  mat4 proj_model;
} instance;

layout (location = 0) in  vec2 inPos;
layout (location = 1) in  vec2 inUV;
layout (location = 0) out vec2 outUV;

out gl_PerVertex {
  vec4 gl_Position;
};

void main(void)
{
  // Apply 3d transformations in 2d space
  gl_Position = instance.proj_model * vec4(inPos, 0.0, 1.0);

  outUV = inUV;
}
