#ifndef RANDOM_GLSL
#define RANDOM_GLSL

#include "../shader_include/definitions.glsl"

// PRNG xorshift seed generator by NVIDIA
uint tea(uint v0, uint v1)
{
    const uint N = 16u; // User specified number of iterations
    uint s0 = 0u;
    for(uint n = 0u; n < N; n++) {
        s0 += 0x9e3779b9u;
        v0 += ((v1<<4u)+0xa341316cu)^(v1+s0)^((v1>>5u)+0xc8013ea4u);
        v1 += ((v0<<4u)+0xad90777du)^(v0+s0)^((v0>>5u)+0x7e95761eu);
    }

    return v0;
}

// Generate random unsigned int in [0, 2^31-1)
void mcg31(inout uint rand_state) {
    const uint LCG_A = 1977654935u; // Multiplier from Hui-Ching Tang [EJOR 2007]
    rand_state = (LCG_A * rand_state) & 0x7FFFFFFEu;
}

// Generate random float in [0, 1)
float rnd(inout uint rand_state) {
    mcg31(rand_state);
    return float(rand_state) / float(0x80000000u);
}

float sampleExponential(float rate, inout uint rand_state) {
    return -log(1.0 - rnd(rand_state)) / rate;
}

vec3 sampleCosineHemisphere(inout uint rand_state) {
    float cos_theta = sqrt(rnd(rand_state));
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    float phi = 2.0 * pi * rnd(rand_state);

    return vec3(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta
    );
}

vec3 sampleUniformSphere(inout uint rand_state) {

    float phi = 2.0 * pi * rnd(rand_state);

    float cos_theta = 2.0 * rnd(rand_state) - 1.0;
    float sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));

    return vec3(
        sin_theta * cos(phi),
        sin_theta * sin(phi),
        cos_theta
    );
}

#endif