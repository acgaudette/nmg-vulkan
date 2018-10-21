#version 450 core

layout(binding = 0) uniform font_ubo {
  mat4 model;
} model_data;

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec2 inUV;
layout (location = 0) out vec2 outUV;

out gl_PerVertex {
	vec4 gl_Position;   
};

const float PERSPECTIVE_TRANSFORM = -1.0;

void main(void)
{
    gl_Position = model_data.model * vec4(inPos, 1.0);
    gl_Position.y = gl_Position.y * PERSPECTIVE_TRANSFORM;
    gl_Position.z = 0;
    outUV = inUV;
}
