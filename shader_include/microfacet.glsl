#ifndef MICROFACET_GLSL
#define MICROFACET_GLSL

#include "../shader_include/definitions.glsl"
#include "../shader_include/random.glsl"

float ndfGgx(float cos_theta_m, float alpha_sq) {
    float denom = cos_theta_m * cos_theta_m * (alpha_sq - 1.0) + 1.0;

    return alpha_sq / (pi * denom * denom);
}

float projectedArea(vec3 w, float alpha_sq) {
    if(w.z > 0.999f) {
        return 1.0;
    } 

    if(w.z < -0.999f) {
        return 0.0;
    } 

    float cos_theta_sq = w.z * w.z;
    float sin_theta_sq = 1.0 - cos_theta_sq;

    return 0.5 * (w.z + sqrt(cos_theta_sq + sin_theta_sq * alpha_sq));
}

float vndfGgx(vec3 w, vec3 wm, float alpha_sq) {

    if (wm.z < 0.0) {
        return 0.0;
    }

    float projected_area = projectedArea(w, alpha_sq);

    if (projected_area == 0.0) {
        return 0.0;
    }

    return max(dot(w, wm), 0.0) * ndfGgx(wm.z, alpha_sq) / projected_area;
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

vec3 sampleVndfStdGgx(vec3 wi, inout uint rand_state) {

    float phi = 2.0 * pi * rnd(rand_state);

    float z = (1.0 - rnd(rand_state)) * (1.0 + wi.z) - wi.z;
    float sin_theta = sqrt(max(1.0 - z * z, 0.0));
    float x = sin_theta * cos(phi);
    float y = sin_theta * sin(phi);

    return wi + vec3(x, y, z); // unnormalized!
}

vec3 sampleVndfGgx(vec3 wi, float alpha, inout uint rand_state) {
    vec3 wi_std = normalize(vec3(wi.xy * alpha, wi.z));
    vec3 wm_std = sampleVndfStdGgx(wi_std, rand_state);

    return normalize(vec3(wm_std.xy * alpha, wm_std.z));
}


float lambdaGgx(float cos_theta, float alpha_sq) {
    float cos_theta_sq = cos_theta * cos_theta;
    float sin_theta_sq = max(1.0 - cos_theta_sq, 0.0);
    float tan_theta_sq = sin_theta_sq / cos_theta_sq;

    return 0.5 * (sign(cos_theta) * sqrt(1.0 + alpha_sq * tan_theta_sq) - 1.0);
}

float maskingGgx(float cos_theta, float alpha_sq) {
    return 1.0 / (1.0 + lambdaGgx(cos_theta, alpha_sq));
}

float maskingShadowingGgx(float cos_theta_i, float cos_theta_o, float alpha_sq) {
    return 1.0 / (1.0 + lambdaGgx(cos_theta_i, alpha_sq) + lambdaGgx(cos_theta_o, alpha_sq));
}

float conductorBsdfCosGgx(vec3 wi, vec3 wo, float ior, float alpha_sq) {

    float cos_theta_i = wi.z;
    float cos_theta_o = wo.z;

    vec3 wm = normalize(sign(cos_theta_i)*(wo + wi));

    float cos_theta_m = wm.z;
    if (cos_theta_m < 1e-5) {
        return 0.0;
    }

    float distribution = ndfGgx(cos_theta_m, alpha_sq);
    float geometry = maskingShadowingGgx(abs(cos_theta_i), abs(cos_theta_o), alpha_sq);

    return distribution * geometry / (4.0 * abs(cos_theta_o));
}

float dielectricBsdfCosGgx(vec3 wi, vec3 wo, float ior, float alpha) {
    
    float alpha_sq = alpha * alpha;

    float cos_theta_i = wi.z;
    float cos_theta_o = wo.z;

    bool outside = cos_theta_i > 0.0;

    float ior_rel = outside ? 1.0 / ior : ior;
    
    bool transmit = cos_theta_i * cos_theta_o < 0.0;

    vec3 wm = normalize(transmit ? ior_rel * wi + wo : wi + wo);

    if (wm.z < 0.0) { // pick wm so it is consistent with the macro normal
        wm = -wm;
    }

    float cos_theta_m = wm.z;
    // if (cos_theta_m < 1e-5) {
    //     return 0.0;
    // }

    float cos_theta_im = dot(wm, wi);
    // if (cos_theta_im * cos_theta_i < 0.0) {
    //     return 0.0;
    // }

    float cos_theta_om = dot(wm, wo);
    float cos_ratio_o = cos_theta_om / cos_theta_o;
    // if (cos_ratio_o <= 0.0) {
    //     return 0.0;
    // }

    float fresnel = 1.0;
    if (transmit) {
        fresnel = 1.0 - reflectanceFresnel(abs(cos_theta_im), abs(cos_theta_om), ior_rel, 1.0);
    } else {
        float sin_theta_tm_sq = ior_rel * ior_rel * (1.0 - cos_theta_im * cos_theta_im);

        if (sin_theta_tm_sq < 1.0) {
            float cos_theta_tm = sqrt(1.0 - sin_theta_tm_sq);
            fresnel = reflectanceFresnel(abs(cos_theta_im), cos_theta_tm, ior_rel, 1.0);
        }
    }

    float cos_weight;

    if (transmit) {
        float a = outside ? cos_theta_im + ior * cos_theta_om : ior * cos_theta_im + cos_theta_om;
        cos_weight = cos_ratio_o * abs(cos_theta_im)  / (a * a);
    } else {
        cos_weight = 1.0 / (4.0 * abs(cos_theta_om));
    }

    float distribution = ndfGgx(cos_theta_m, alpha_sq);
    float geometry = maskingShadowingGgx(abs(cos_theta_i), abs(cos_theta_o), alpha_sq);

    return fresnel * distribution * geometry * cos_weight;
}



float heightCdfUniform(float height) {
    return clamp(0.5 * (height + 1.0), 0.0, 1.0);
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
    // lambda function has numerical issues in these cases
    if (abs(w.z) < 0.001) {
        escaped = false;
        return height;
    }

    if (w.z > 0.999) {
        escaped = true;
        return 0.0;
    }

    float u = rnd(rand_state);

    if (w.z < -0.999) {
        escaped = false;
        return invHeightCdfUniform(u * heightCdfUniform(height));
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
    escaped = false;

    float denom = pow(1.0 - u, 1.0 / lambda);
    return invHeightCdfUniform(heightCdfUniform(height) / denom);
}

#endif