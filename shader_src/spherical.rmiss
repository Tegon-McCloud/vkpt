#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include "../shader_include/definitions.glsl"

layout(set = 0, binding = 2) uniform sampler2D environment_map;

layout(location = 0) rayPayloadInEXT pathInfo payload;

void main() {

    vec3 direction = gl_WorldRayDirectionEXT;

    vec2 uv = vec2(
        atan(direction.x, direction.z) / (2.0 * pi) + 0.5,
        acos(direction.y) / pi
    );

    vec3 env_color = texture(environment_map, uv).rgb;
    
    payload.radiance += payload.weight * env_color;
    payload.depth = max_depth; // end the path

}