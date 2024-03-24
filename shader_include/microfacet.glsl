#ifndef MICROFACET_GLSL
#define MICROFACET_GLSL

#include "../shader_include/definitions.glsl"
#include "../shader_include/random.glsl"

float geometryGgx(float cos_theta, float alpha_sq) {
    return 2.0 * cos_theta / (cos_theta + sqrt(alpha_sq + (1.0 - alpha_sq) * cos_theta * cos_theta));
}

float maskingShadowingGgx(float cos_theta_i, float cos_theta_o, float alpha_sq) {
    return geometryGgx(cos_theta_i, alpha_sq) * geometryGgx(cos_theta_o, alpha_sq);
}

float ndfGgx(float cos_theta_m, float alpha_sq) {
    float denom = cos_theta_m * cos_theta_m * (alpha_sq - 1.0) + 1.0;

    return alpha_sq / (pi * denom * denom);
}

vec3 sampleNdfGgx(float alpha_sq, inout uint rand_state) {
    float u0 = rnd(rand_state);
    float u1 = rnd(rand_state);

    float cos_theta_m = sqrt((1.0 - u0) / (u0 * (alpha_sq - 1.0) + 1.0));
    float sin_theta_m = sqrt(max(1.0 - cos_theta_m * cos_theta_m, 0.0));
    float phi = 2.0 * pi * u1;

    return vec3(
        sin_theta_m * cos(phi),
        sin_theta_m * sin(phi),
        cos_theta_m
    );
}

vec3 sampleVndfGgx(vec3 w, float alpha, inout uint rand_state) {
    vec3 wi_std = normalize(vec3(w.xy * alpha, w.z));

    float phi = 2.0 * pi * rnd(rand_state);

    float z = (1.0 - rnd(rand_state)) * (1.0 + w.z) - w.z;
    float sin_theta = sqrt(max(1.0 - z * z, 0.0));
    float x = sin_theta * cos(phi);
    float y = sin_theta * sin(phi);

    vec3 wm_std = w + vec3(x, y, z);

    return normalize(vec3(wm_std.xy * alpha, wm_std.z));
}

// float smithLambdaGgx() {
    
// }

// float sampleHeight(float prevHeight, vec3 w, out leave, inout uint rand_state) {

//     float u = rnd(rand_state);

//     if u <


// }

// vec3 ggxRandomWalk(vec3 wi) {
//     float height = 1e8;
//     float energy = 1.0;

//     vec3 w = -wi;
    
//     while(true) {

//     }



// }


float reflectanceFresnel(float cos_theta_i, float cos_theta_t, float ior_i, float ior_t) {
    float a_ii = ior_i * cos_theta_i;
    float a_tt = ior_t * cos_theta_t;
    float a_it = ior_i * cos_theta_t;
    float a_ti = ior_t * cos_theta_i;

    float r_perp = (a_ii - a_tt) / (a_ii + a_tt);
    float r_para = (a_ti - a_it) / (a_ti + a_it);

    return 0.5 * ((r_perp * r_perp) + (r_para * r_para));
}

#endif