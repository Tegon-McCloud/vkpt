#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include "../shader_include/definitions.glsl"

layout(location = 0) rayPayloadInEXT pathInfo payload;


void main() {
    payload.radiance += vec3(0.286, 0.545, 0.960);
}