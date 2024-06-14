#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require
#extension GL_GOOGLE_include_directive : enable

#include "../shader_include/definitions.glsl"
#include "../shader_include/random.glsl"
#include "../shader_include/util.glsl"


layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

layout(buffer_reference, std430, buffer_reference_align = 16) buffer uintBuffer {
    uint indices[];
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer vec4Buffer {
    vec4 arr[];
};

layout(shaderRecordEXT, std430) buffer sbtData {
    uint material_index;
    uintBuffer indices;
    vec4Buffer positions;
    vec4Buffer normals;
};

layout(location = 0) rayPayloadInEXT pathInfo payload;
layout(location = 1) rayPayloadEXT shadowInfo shadow_info;

layout(location = 0) callableDataEXT brdfEvaluation brdf_evaluation;
layout(location = 1) callableDataEXT lightSample light_sample;

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
    brdf_evaluation.wi = wi;
    brdf_evaluation.wo = wo;
    brdf_evaluation.rand_state = payload.rand_state;
    brdf_evaluation.error_state = payload.error_state;
    
    executeCallableEXT(2 * material_index, 0);

    payload.rand_state = brdf_evaluation.rand_state;
    payload.error_state = brdf_evaluation.error_state;

    return brdf_evaluation.weight;
}

vec3 sampleBsdfCos(out vec3 wi, vec3 wo) {
    brdf_evaluation.wo = wo;
    brdf_evaluation.rand_state = payload.rand_state;
    brdf_evaluation.error_state = payload.error_state;
    
    executeCallableEXT(2 * material_index + 1, 0);

    wi = brdf_evaluation.wi;
    payload.rand_state = brdf_evaluation.rand_state;
    payload.error_state = brdf_evaluation.error_state;

    return brdf_evaluation.weight;
}

vec3 sampleLight(out vec3 wi_world, vec3 position) {

    light_sample.position = position;
    light_sample.rand_state = payload.rand_state;

    executeCallableEXT(14, 1);

    // light_sample.wi_world = normalize(vec3(1.0, 1.0, 1.0));
    // light_sample.value = vec3(1.0);

    payload.rand_state = light_sample.rand_state;
    wi_world = light_sample.wi_world;

    shadow_info.blocked = true;

    const uint ray_flags = gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsTerminateOnFirstHitEXT;
    const uint cull_mask = 0xff;

    traceRayEXT(tlas,
        ray_flags,          // rayFlags
        cull_mask,          // cullMask
        0,                  // sbtRecordOffset
        0,                  // sbtRecordStride
        1,                  // missIndex
        position,           // ray origin
        tmin,               // ray min range
        wi_world,           // ray direction
        tmax,               // ray max range
        1                   // payload (location = 1)
    );

    if(shadow_info.blocked) {
        return vec3(0.0);
    }

    return light_sample.value;
}

void sampleDirect(vec3 position, vec3 wo, mat3 world_to_tangent) {

    vec3 wi_world;
    vec3 light_value = sampleLight(wi_world, position);
    vec3 wi = world_to_tangent * wi_world;    

    vec3 bsdf_cos = evaluateBsdfCos(wi, wo);

    payload.radiance += payload.weight * bsdf_cos * light_value;
}

void sampleIndirect(vec3 position, vec3 wo, mat3 tangent_to_world) {
    vec3 wi;

    payload.weight *= sampleBsdfCos(wi, wo);

    if (wi == vec3(0.0)) {
        payload.depth = max_depth;
        return;
    }

    payload.position = position;
    payload.direction = tangent_to_world * wi;

    payload.depth += 1;
}


void main() {

    vec3 bc = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    vec3 position = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;

    uvec3 face = getFace();

    vec3 n0 = normals.arr[face.x].xyz;
    vec3 n1 = normals.arr[face.y].xyz;
    vec3 n2 = normals.arr[face.z].xyz;

    vec3 normal = normalize(gl_ObjectToWorldEXT * vec4(bc.x * n0 + bc.y * n1 + bc.z * n2, 0.0));
    
    // local coordinate system from normal
    mat3 tangent_to_world = getTbnMatrix(normal);
    mat3 world_to_tangent = transpose(tangent_to_world);
    
    vec3 wo = -(world_to_tangent * gl_WorldRayDirectionEXT);

    // sampleDirect(position, wo, world_to_tangent);
    sampleIndirect(position, wo, tangent_to_world);


    payload.emit = 1;

}

