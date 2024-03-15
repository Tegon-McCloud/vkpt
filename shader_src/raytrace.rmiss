#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include "../shader_include/definitions.glsl"

layout(set = 0, binding = 0) uniform sampler2D environmentMap;

layout(location = 0) rayPayloadInEXT pathInfo payload;

void main() {

    

    payload.radiance += payload.weight * vec3(0.286, 0.545, 0.960);
    payload.depth = max_depth; // ends the path

}