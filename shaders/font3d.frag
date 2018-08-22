#version 450

layout (binding = 1) uniform sampler2D samplerFont;

layout (location = 0) in vec2 inUV;
layout (location = 0) out vec4 outColor;

void main(void) {
  float color = texture(samplerFont, inUV).r;
  outColor = vec4(color);
}
