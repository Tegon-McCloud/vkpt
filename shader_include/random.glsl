#ifndef RANDOM_GLSL
#define RANDOM_GLSL

#include "../shader_include/definitions.glsl"

uint rand_state;

// PRNG xorshift seed generator by NVIDIA
void tea(uint v0, uint v1)
{
    const uint N = 16u; // User specified number of iterations
    uint s0 = 0u;
    for(uint n = 0u; n < N; n++) {
        s0 += 0x9e3779b9u;
        v0 += ((v1<<4u)+0xa341316cu)^(v1+s0)^((v1>>5u)+0xc8013ea4u);
        v1 += ((v0<<4u)+0xad90777du)^(v0+s0)^((v0>>5u)+0x7e95761eu);
    }

    rand_state = v0;
}

// Generate random unsigned int in [0, 2^31-1)
uint mcg31() {
    const uint LCG_A = 1977654935u; // Multiplier from Hui-Ching Tang [EJOR 2007]
    rand_state = (LCG_A * rand_state) & 0x7FFFFFFEu;
    return rand_state;
}

// Generate random float in [0, 1)
float rnd() {
    return float(mcg31()) / float(0x80000000u);
}

vec3 sampleCosineHemisphere() {

    float cosTheta = sqrt(rnd());
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    float phi = 2.0 * pi * rnd();

    return vec3(
        sinTheta * cos(phi),
        sinTheta * sin(phi),
        cosTheta
    );
}

#endif