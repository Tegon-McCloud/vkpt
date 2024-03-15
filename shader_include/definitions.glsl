#ifndef DEFINITIONS_GLSL
#define DEFINITIONS_GLSL

const float pi = 3.14159265359;

const float tmin = 1e-3;
const float tmax = 1e8;

const uint max_depth = 32;

struct pathInfo {
    vec3 position;
    uint rand_state;
    vec3 direction;
    uint depth;
    vec3 radiance;
    uint emit;
    vec3 weight;
    uint channel;
};

struct intersection {
    vec3 position;
    vec3 normal;
};

struct brdfEvaluation {
    vec3 wi;
    vec3 wo;
    vec3 weight;
};

struct lightSample {
    vec3 wi;
    vec3 weight;
};

#endif