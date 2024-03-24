#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include "../shader_include/definitions.glsl"

layout(location = 0) rayPayloadInEXT pathInfo payload;

void main() {

    vec3 direction = gl_WorldRayDirectionEXT;

    // vec2 uv = vec2(
    //     atan(direction.x, direction.z) / (2.0 * pi) + 0.5,
    //     acos(direction.y) / pi
    // );
    
    vec3 color = vec3(0.5);

    payload.radiance += payload.weight * color;
    // payload.radiance += payload.weight * vec3(0.286, 0.545, 0.960);
    payload.depth = max_depth; // end the path

}