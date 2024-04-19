#ifndef MICROFACET_GLSL
#define MICROFACET_GLSL

#include "../shader_include/definitions.glsl"
#include "../shader_include/random.glsl"

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

    vec3 w_std = normalize(vec3(w.xy * alpha, w.z));

    float phi = 2.0 * pi * rnd(rand_state);

    float z = (1.0 - rnd(rand_state)) * (1.0 + w_std.z) - w_std.z;
    float sin_theta = sqrt(max(1.0 - z * z, 0.0));
    float x = sin_theta * cos(phi);
    float y = sin_theta * sin(phi);

    // unnormalized!
    vec3 wm_std = w_std + vec3(x, y, z);

    return normalize(vec3(wm_std.xy * alpha, wm_std.z));
}

float lambdaGgx(float cos_theta, float alpha_sq) {
    float cos_theta_sq = cos_theta * cos_theta;
    float sin_theta_sq = max(1.0 - cos_theta_sq, 0.0);
    float tan_theta_sq = sin_theta_sq / cos_theta_sq;

    return 0.5 * (sign(cos_theta) * sqrt(1.0 + alpha_sq * tan_theta_sq) - 1.0);
}

float geometryGgx(float cos_theta, float alpha_sq) {
    return 1.0 / (1.0 + lambdaGgx(cos_theta, alpha_sq));
}

float maskingShadowingGgx(float cos_theta_i, float cos_theta_o, float alpha_sq) {
    return 1.0 / (1.0 + lambdaGgx(cos_theta_i, alpha_sq) + lambdaGgx(cos_theta_o, alpha_sq));
}

float heightCdfUniform(float height) {
    return clamp(0.0, 1.0, 0.5 * (height + 1.0));
}

float invHeightCdfUniform(float u) {
    return clamp(2.0 * u - 1.0, -1.0, 1.0);
}

float sampleHeightGgxUniform(
    vec3 w,
    float height,
    out bool escaped,
    float alpha_sq,
    inout uint rand_state
) {

    float u = rnd(rand_state);

    if (w.z > 0.9999) {
        escaped = true;
        return 0.0;
    }
    
    if (w.z < -0.9999) {
        escaped = false;
        return invHeightCdfUniform(u * heightCdfUniform(height));
    }

    if (abs(w.z) < 0.0001) {
        escaped = false;
        return height;
    }

    float lambda = lambdaGgx(w.z, alpha_sq);

    float cdf = heightCdfUniform(height);
    float geometry;

    if (w.z <= 0.0) {
        geometry = 0.0;
    } else {
        geometry = pow(cdf, lambda);
    }

    if(u >= 1.0 - geometry) {
        escaped = true;
        return 0.0;
    }

    float denom = pow(1.0 - u, 1.0 / lambda);
    escaped = false;
    return invHeightCdfUniform(heightCdfUniform(height) / denom);
}

vec3 sampleDielectricPhaseFunction(
    vec3 wo,
    inout float weight,
    out bool transmitted,
    float alpha,
    float ior_rel,
    inout uint rand_state
) {
    vec3 wm = sampleVndfGgx(wo, alpha, rand_state);
    float cos_theta_om = dot(wo, wm);

    float reflectance = 1.0;

    float sin_theta_tm_sq = ior_rel * ior_rel * (1.0 - cos_theta_om * cos_theta_om);
    float cos_theta_tm;

    if (sin_theta_tm_sq < 1.0) {
        cos_theta_tm = -sqrt(1.0 - sin_theta_tm_sq);
        reflectance = reflectanceFresnel(cos_theta_om, -cos_theta_tm, ior_rel, 1.0);
    }

    vec3 wi;
    
    if(rnd(rand_state) < reflectance) {
        wi = 2.0 * cos_theta_om * wm - wo;
        transmitted = false;
    } else {
        wi = (ior_rel * cos_theta_om + cos_theta_tm) * wm - ior_rel * wo;
        transmitted = true;
    }

    return wi;
}


#endif