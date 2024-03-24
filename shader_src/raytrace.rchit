#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_GOOGLE_include_directive : enable

#include "../shader_include/definitions.glsl"
#include "../shader_include/random.glsl"
#include "../shader_include/util.glsl"

layout(buffer_reference, std430, buffer_reference_align = 16) buffer indexBuffer {
    uint indices[];
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer vec4Buffer {
    vec4 arr[];
};

layout(shaderRecordEXT, std430) buffer sbtData {
    uint material_index;
    indexBuffer indices;
    vec4Buffer positions;
    vec4Buffer normals;
};

layout(location = 0) rayPayloadInEXT pathInfo payload;

layout(location = 0) callableDataEXT brdfEvaluation evaluation;

hitAttributeEXT vec2 attribs;

uvec3 getFace() {
    uint index_offset = 3 * gl_PrimitiveID;

    return uvec3(
        indices.indices[index_offset + 0],
        indices.indices[index_offset + 1],
        indices.indices[index_offset + 2]
    );
}

vec3 evaluateBsdfCos(vec3 wi, vec3 wo) {
    evaluation.wi = wi;
    evaluation.wo = wo;
    evaluation.rand_state = payload.rand_state;
    
    executeCallableEXT(2 * material_index, 0);

    payload.rand_state = evaluation.rand_state;

    return evaluation.weight;
}

vec3 sampleBsdfCos(out vec3 wi, vec3 wo) {
    evaluation.wo = wo;
    evaluation.rand_state = payload.rand_state;
    
    executeCallableEXT(2 * material_index + 1, 0);

    wi = evaluation.wi;
    payload.rand_state = evaluation.rand_state;

    return evaluation.weight;
}

void main() {
    vec3 bc = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    vec3 hit_pos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;

    uvec3 face = getFace();

    vec3 n0 = normals.arr[face.x].xyz;
    vec3 n1 = normals.arr[face.y].xyz;
    vec3 n2 = normals.arr[face.z].xyz;

    vec3 normal = normalize(gl_ObjectToWorldEXT * vec4(bc.x * n0 + bc.y * n1 + bc.z * n2, 0.0));

    // local coordinate system from normal
    mat3 tangent_to_world = getTbnMatrix(normal);
    mat3 world_to_tangent = transpose(tangent_to_world);
    
    vec3 wo = -world_to_tangent * gl_WorldRayDirectionEXT;

    // vec3 light_dir = world_to_tangent * normalize(vec3(-1.0, -1.0, -1.0));

    // vec3 bsdf = evaluateBsdfCos(-light_dir, wo);
    
    // payload.radiance += payload.weight * bsdf * pi;
    // payload.emit = 0;
    
    vec3 wi;

    payload.weight *= sampleBsdfCos(wi, wo);

    if (wi == vec3(0.0)) {
        payload.depth = max_depth;
        return;
    }
    
    payload.position = hit_pos;
    payload.direction = tangent_to_world * wi;
    
    payload.depth += 1;
}

