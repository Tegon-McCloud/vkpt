#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include "../shader_include/definitions.glsl"

layout(location = 0) rayPayloadInEXT shadowInfo payload;

void main() {
    payload.blocked = false;
}

