#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable


layout(location = 0) rayPayloadInEXT vec3 color;
hitAttributeEXT vec3 attribs;

void main() {
    color = vec3(1.0, 0.0, 0.0);
}
