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


#endif