#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include "../shader_include/definitions.glsl"

layout(location = 0) rayPayloadInEXT pathInfo payload;

void main() {

    vec3 direction = gl_WorldRayDirectionEXT;
    
    vec3 env_radiance = vec3(0.5);

    payload.radiance += payload.weight * env_radiance;
    payload.depth = max_depth; // end the path

}