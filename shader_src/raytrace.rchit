#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable


layout(location = 0) rayPayloadInEXT vec3 color;
hitAttributeEXT vec2 attribs;

void main() {
    vec3 bc = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    color = bc;
}
