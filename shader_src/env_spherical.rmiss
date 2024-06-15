#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include "../shader_include/definitions.glsl"

layout(set = 0, binding = 2) uniform usampler2D environment_map;

layout(location = 0) rayPayloadInEXT pathInfo payload;

void main() {

    payload.depth = max_depth; // end the path
    
    if (payload.emit == 0) {
        return;
    }

    vec3 direction = gl_WorldRayDirectionEXT;

    vec2 uv = vec2(
        atan(direction.x, direction.z) / (2.0 * pi),
        acos(direction.y) / pi
    );

    uvec4 env_rgbe = texture(environment_map, uv);
    vec3 env_radiance = (1.0 / 256.0) * vec3(env_rgbe.rgb) * pow(2.0, float(env_rgbe.a) - 128.0);

    payload.radiance += payload.weight * env_radiance;
}