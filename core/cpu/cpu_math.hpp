// Vector and matrix types with Metal Shading Language semantics, so the CPU backend
// can carry the kernel math over from msplat_metal.metal line by line. Matrices are
// column-major as in MSL: floatNxN(a, b, c, ...) fills column 0 first and M[i] is
// column i.
#ifndef MSPLAT_CPU_MATH_HPP
#define MSPLAT_CPU_MATH_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace msl {

struct float2 {
    float x = 0.f, y = 0.f;
    float2() = default;
    constexpr float2(float a, float b) : x(a), y(b) {}
    explicit constexpr float2(float s) : x(s), y(s) {}
    float &operator[](int i) { return i == 0 ? x : y; }
    float operator[](int i) const { return i == 0 ? x : y; }
};

struct float3 {
    float x = 0.f, y = 0.f, z = 0.f;
    float3() = default;
    constexpr float3(float a, float b, float c) : x(a), y(b), z(c) {}
    explicit constexpr float3(float s) : x(s), y(s), z(s) {}
    float &operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
    float operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
};

struct float4 {
    float x = 0.f, y = 0.f, z = 0.f, w = 0.f;
    float4() = default;
    constexpr float4(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {}
};

#define MSL_VEC_OPS(op)                                                                          \
    inline float2 operator op(float2 a, float2 b) { return {a.x op b.x, a.y op b.y}; }           \
    inline float2 operator op(float2 a, float b) { return {a.x op b, a.y op b}; }                \
    inline float2 operator op(float a, float2 b) { return {a op b.x, a op b.y}; }                \
    inline float3 operator op(float3 a, float3 b) { return {a.x op b.x, a.y op b.y, a.z op b.z}; } \
    inline float3 operator op(float3 a, float b) { return {a.x op b, a.y op b, a.z op b}; }        \
    inline float3 operator op(float a, float3 b) { return {a op b.x, a op b.y, a op b.z}; }
MSL_VEC_OPS(+)
MSL_VEC_OPS(-)
MSL_VEC_OPS(*)
MSL_VEC_OPS(/)
#undef MSL_VEC_OPS

inline float2 operator-(float2 a) { return {-a.x, -a.y}; }
inline float3 operator-(float3 a) { return {-a.x, -a.y, -a.z}; }
inline float2 &operator+=(float2 &a, float2 b) { a = a + b; return a; }
inline float3 &operator+=(float3 &a, float3 b) { a = a + b; return a; }
inline float3 &operator-=(float3 &a, float3 b) { a = a - b; return a; }
inline float3 &operator*=(float3 &a, float b) { a = a * b; return a; }

inline float dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }
inline float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float length(float3 a) { return std::sqrt(dot(a, a)); }
inline float3 normalize(float3 a) { return a * (1.f / std::sqrt(dot(a, a))); }
inline float3 exp(float3 a) { return {std::exp(a.x), std::exp(a.y), std::exp(a.z)}; }
inline float3 max(float3 a, float b) { return {std::max(a.x, b), std::max(a.y, b), std::max(a.z, b)}; }
inline float3 min(float3 a, float b) { return {std::min(a.x, b), std::min(a.y, b), std::min(a.z, b)}; }
inline float saturate(float a) { return std::min(std::max(a, 0.f), 1.f); }
inline float3 saturate(float3 a) { return {saturate(a.x), saturate(a.y), saturate(a.z)}; }
inline float clampf(float v, float lo, float hi) { return std::min(std::max(v, lo), hi); }
inline float fmaf_(float a, float b, float c) { return a * b + c; }
inline float3 fma3(float3 a, float b, float3 c) { return a * b + c; }
inline float3 fma3(float3 a, float3 b, float3 c) { return a * b + c; }
inline float mixf(float a, float b, float t) { return a + (b - a) * t; }
inline float signf(float v) { return v > 0.f ? 1.f : (v < 0.f ? -1.f : 0.f); }
inline float rsqrtf_(float v) { return 1.f / std::sqrt(v); }

struct float2x2 {
    float2 c[2];
    float2x2() = default;
    // Column-major, as in MSL
    float2x2(float a, float b, float d, float e) { c[0] = {a, b}; c[1] = {d, e}; }
    float2 &operator[](int i) { return c[i]; }
    const float2 &operator[](int i) const { return c[i]; }
};

inline float2x2 operator*(const float2x2 &A, const float2x2 &B) {
    float2x2 R;
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++)
            R.c[j][i] = A.c[0][i] * B.c[j][0] + A.c[1][i] * B.c[j][1];
    return R;
}
inline float2x2 operator*(float s, const float2x2 &A) {
    float2x2 R = A;
    for (int j = 0; j < 2; j++) R.c[j] = R.c[j] * s;
    return R;
}

struct float3x3 {
    float3 c[3];
    float3x3() = default;
    explicit float3x3(float diag) { c[0] = {diag, 0.f, 0.f}; c[1] = {0.f, diag, 0.f}; c[2] = {0.f, 0.f, diag}; }
    // Column-major, as in MSL
    float3x3(float a0, float a1, float a2, float b0, float b1, float b2, float d0, float d1, float d2) {
        c[0] = {a0, a1, a2};
        c[1] = {b0, b1, b2};
        c[2] = {d0, d1, d2};
    }
    float3x3(float3 c0, float3 c1, float3 c2) { c[0] = c0; c[1] = c1; c[2] = c2; }
    float3 &operator[](int i) { return c[i]; }
    const float3 &operator[](int i) const { return c[i]; }
};

inline float3x3 operator*(const float3x3 &A, const float3x3 &B) {
    float3x3 R;
    for (int j = 0; j < 3; j++)
        for (int i = 0; i < 3; i++)
            R.c[j][i] = A.c[0][i] * B.c[j][0] + A.c[1][i] * B.c[j][1] + A.c[2][i] * B.c[j][2];
    return R;
}
inline float3 operator*(const float3x3 &A, float3 v) { return A.c[0] * v.x + A.c[1] * v.y + A.c[2] * v.z; }
inline float3x3 operator*(float s, const float3x3 &A) { return {A.c[0] * s, A.c[1] * s, A.c[2] * s}; }
inline float3x3 operator+(const float3x3 &A, const float3x3 &B) {
    return {A.c[0] + B.c[0], A.c[1] + B.c[1], A.c[2] + B.c[2]};
}
inline float3x3 transpose(const float3x3 &A) {
    return float3x3(A.c[0].x, A.c[1].x, A.c[2].x, A.c[0].y, A.c[1].y, A.c[2].y, A.c[0].z, A.c[1].z, A.c[2].z);
}

}  // namespace msl

#endif
