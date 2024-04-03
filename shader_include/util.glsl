#ifndef UTIL_GLSL
#define UTIL_GLSL

mat3 getTbnMatrix(vec3 normal) {
    float signbit = sign(normal.z + 1.0e-16f);
    float a = -1.0f/(1.0f + abs(normal.z));
    float b = normal.x*normal.y*a;

    return mat3(
        vec3(1.0f + normal.x*normal.x*a, b, -signbit*normal.x),
        vec3(signbit*b, signbit*(1.0f + normal.y*normal.y*a), -normal.y),
        normal
    );
}

float reflectanceSchlick(float r0, float cos_theta) {
    float a = 1.0 - cos_theta;
    return r0 + (1.0 - r0) * (a * a) * (a * a) * a;
}

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