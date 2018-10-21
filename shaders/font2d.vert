#version 450 core

layout(binding = 0) uniform font_ubo {
  mat4 model;
} instance;

layout (location = 0) in vec2 inPos;
layout (location = 1) in vec2 inUV;
layout (location = 0) out vec2 outUV;

out gl_PerVertex {
	vec4 gl_Position;   
};

const float PERSPECTIVE_TRANSFORM = -1.0;

void main(void)
{
    // Apply 3d transformations in 2d space
    gl_Position = instance.model * vec4(inPos, 0.0, 1.0);
    // Flips UV textures as text will look upside down without transform
    gl_Position.y = gl_Position.y * PERSPECTIVE_TRANSFORM;
    outUV = inUV;
}
