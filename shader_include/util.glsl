#ifndef UTIL_GLSL
#define UTIL_GLSL

// Given a direction vector v sampled around the z-axis of a
// local coordinate system, this function applies the same
// rotation to v as is needed to rotate the z-axis to the
// actual direction n that v should have been sampled around
// [Frisvad, Journal of Graphics Tools 16, 2012;
//  Duff et al., Journal of Computer Graphics Techniques 6, 2017].
vec3 rotateToNormal(vec3 v, vec3 normal) {

    float signbit = sign(normal.z + 1.0e-16f);
    float a = -1.0f/(1.0f + abs(normal.z));
    float b = normal.x*normal.y*a;

    return vec3(1.0f + normal.x*normal.x*a, b, -signbit*normal.x)*v.x
        + vec3(signbit*b, signbit*(1.0f + normal.y*normal.y*a), -normal.y)*v.y
        + normal*v.z;
}

#endif