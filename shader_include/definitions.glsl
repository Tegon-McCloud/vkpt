
const float pi = 3.14159265359;

struct pathInfo {
    vec3 radiance;
    vec3 weight;
};

struct brdfEvaluation {
    vec3 wi;
    vec3 wo;
    vec3 weight;
};
