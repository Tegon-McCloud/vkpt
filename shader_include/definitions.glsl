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
    uint error_state;
};

struct shadowInfo {
    bool blocked;
};

struct brdfEvaluation {
    vec3 wi;
    uint rand_state;
    vec3 wo;
    uint error_state;
    vec3 weight;
};

struct lightSample {
    vec3 wi_world;
    uint rand_state;
    vec3 position;
    float dist;
    vec3 value;
};

#endif