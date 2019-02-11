#version 450 core

layout (binding = 1) uniform sampler2D samplerFont;

layout (location = 0) in  vec2 inUV;
layout (location = 0) out vec4 outFragColor;

void main(void)
{
    float color = texture(samplerFont, inUV).r;
    outFragColor = vec4(color);
}
