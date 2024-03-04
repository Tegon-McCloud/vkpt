#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_GOOGLE_include_directive : enable

#include "../shader_include/definitions.glsl"

layout(buffer_reference, std430, buffer_reference_align = 16) buffer indexBuffer {
    uint indices[];
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer vec4Buffer {
    vec4 arr[];
};

layout(shaderRecordEXT, std430) buffer sbtData {
    uint materialIndex;
    indexBuffer indices;
    vec4Buffer positions;
    vec4Buffer normals;
};

layout(location = 0) rayPayloadInEXT pathInfo payload;

layout(location = 0) callableDataEXT brdfEvaluation evaluation;

hitAttributeEXT vec2 attribs;

uvec3 getFace() {
    uint indexOffset = 3 * gl_PrimitiveID;

    return uvec3(
        indices.indices[indexOffset + 0],
        indices.indices[indexOffset + 1],
        indices.indices[indexOffset + 2]
    );
}



vec3 evaluateBrdf(vec3 wi, vec3 wo) {
    evaluation.wi = wi;
    evaluation.wo = wo;

    executeCallableEXT(materialIndex, 0);

    return evaluation.weight;
}

void main() {
    vec3 bc = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    uvec3 face = getFace();

    vec3 n0 = normals.arr[face.x].xyz;
    vec3 n1 = normals.arr[face.y].xyz;
    vec3 n2 = normals.arr[face.z].xyz;

    vec3 normal = normalize(gl_ObjectToWorldEXT * vec4(bc.x * n0 + bc.y * n1 + bc.z * n2, 0.0));
    
    vec3 wo = gl_WorldRayDirectionEXT;
    
    vec3 light_dir = normalize(vec3(-1.0, -1.0, -1.0));

    vec3 brdf = evaluateBrdf(light_dir, gl_WorldRayDirectionEXT);
    
    payload.weight *= brdf;
    payload.radiance += payload.weight * max(dot(-light_dir, normal), 0.0);
}
