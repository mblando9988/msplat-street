// CPU backend: every entry point of bindings.h without Metal, so msplat builds, trains
// and renders on Linux (and anywhere a C++17 compiler runs). The math is carried over
// from msplat_metal.metal kernel by kernel, quirks included, so a CPU run and a Metal
// run of the same scene train the same way; the matching kernel is named above each
// function here. Work is spread over a thread pool (MSPLAT_THREADS overrides its size).

#include "bindings.h"
#include "cpu_math.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace msl;
using msplat_cpu::parallel_for;

namespace {

constexpr int BLOCK_X = 16, BLOCK_Y = 16;
constexpr int MAX_TILE_ELEMS = 2048;

constexpr float SH_C0 = 0.28209479177387814f;
constexpr float SH_C1 = 0.4886025119029199f;
constexpr float SH_C2[] = {1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
                           -1.0925484305920792f, 0.5462742152960396f};
constexpr float SH_C3[] = {-0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f,
                           0.3731763325901154f, -0.4570457994644658f, 1.445305721320277f,
                           -0.5900435899266435f};
constexpr float SH_C4[] = {2.5033429417967046f, -1.7701307697799304f, 0.9461746957575601f,
                           -0.6690465435572892f, 0.10578554691520431f, -0.6690465435572892f,
                           0.47308734787878004f, -1.7701307697799304f, 0.6258357354491761f};

constexpr int SSIM_WIN = 11, SSIM_HALF_WIN = 5;
constexpr float SSIM_C1 = 0.0001f, SSIM_C2 = 0.0009f;
constexpr float GAUSS_1D[11] = {
    0.0010283801f, 0.0075987581f, 0.0360007721f, 0.1093606895f, 0.2130055377f,
    0.2660117249f,
    0.2130055377f, 0.1093606895f, 0.0360007721f, 0.0075987581f, 0.0010283801f};

inline uint32_t num_sh_bases(uint32_t degree) {
    return degree == 0 ? 1 : degree == 1 ? 4 : degree == 2 ? 9 : degree == 3 ? 16 : 25;
}

// Packed buffer access (the Metal read/write_packed_floatN helpers)
inline float2 rd2(const float *a, int64_t i) { return {a[2 * i], a[2 * i + 1]}; }
inline float3 rd3(const float *a, int64_t i) { return {a[3 * i], a[3 * i + 1], a[3 * i + 2]}; }
inline float4 rd4(const float *a, int64_t i) { return {a[4 * i], a[4 * i + 1], a[4 * i + 2], a[4 * i + 3]}; }
inline void wr2(float *a, int64_t i, float2 v) { a[2 * i] = v.x; a[2 * i + 1] = v.y; }
inline void wr3(float *a, int64_t i, float3 v) { a[3 * i] = v.x; a[3 * i + 1] = v.y; a[3 * i + 2] = v.z; }

template <typename T> T *ptr(MTensor &t) { return t.data<T>(); }
template <typename T> const T *cptr(const MTensor &t) { return t.data<T>(); }

// ── Projection math (msplat_metal.metal helpers) ────────────────────────────

inline float3 transform_4x3(const float *m, float3 p) {
    return {m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3],
            m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7],
            m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]};
}

inline float4 transform_4x4(const float *m, float3 p) {
    return {m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3],
            m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7],
            m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11],
            m[12] * p.x + m[13] * p.y + m[14] * p.z + m[15]};
}

// Quaternion stored (w, x, y, z); returns R (column-major)
inline float3x3 quat_to_rotmat(float4 quat) {
    float s = rsqrtf_(quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z);
    float w = quat.x * s, x = quat.y * s, y = quat.z * s, z = quat.w * s;
    return float3x3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y + w * z), 2.f * (x * z - w * y),
        2.f * (x * y - w * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z + w * x),
        2.f * (x * z + w * y), 2.f * (y * z - w * x), 1.f - 2.f * (x * x + y * y));
}

inline float3x3 scale_to_mat(float3 scale, float glob_scale) {
    float3x3 S(1.f);
    S[0][0] = glob_scale * scale.x;
    S[1][1] = glob_scale * scale.y;
    S[2][2] = glob_scale * scale.z;
    return S;
}

inline void scale_rot_to_cov3d(float3 scale, float glob_scale, float4 quat, float *cov3d) {
    float3x3 R = quat_to_rotmat(quat);
    float3x3 M = R * scale_to_mat(scale, glob_scale);
    float3x3 tmp = M * transpose(M);
    cov3d[0] = tmp[0][0]; cov3d[1] = tmp[0][1]; cov3d[2] = tmp[0][2];
    cov3d[3] = tmp[1][1]; cov3d[4] = tmp[1][2]; cov3d[5] = tmp[2][2];
}

// EWA projection of the 3D covariance (with the 0.3 px low-pass)
inline float3 project_cov3d_ewa(const float *cov3d, const float *viewmat, float fx, float fy,
                                float tan_fovx, float tan_fovy, float3 p_view) {
    float lim_x = 1.3f * tan_fovx, lim_y = 1.3f * tan_fovy;
    p_view.x = p_view.z * std::min(lim_x, std::max(-lim_x, p_view.x / p_view.z));
    p_view.y = p_view.z * std::min(lim_y, std::max(-lim_y, p_view.y / p_view.z));
    float rz = 1.f / p_view.z, rz2 = rz * rz;
    float j00 = fx * rz, j11 = fy * rz;
    float j20 = -fx * p_view.x * rz2, j21 = -fy * p_view.y * rz2;
    float3 mr0(viewmat[0], viewmat[1], viewmat[2]);
    float3 mr1(viewmat[4], viewmat[5], viewmat[6]);
    float3 mr2(viewmat[8], viewmat[9], viewmat[10]);
    float3 t0 = j00 * mr0 + j20 * mr2;
    float3 t1 = j11 * mr1 + j21 * mr2;
    float v00 = cov3d[0], v01 = cov3d[1], v02 = cov3d[2];
    float v11 = cov3d[3], v12 = cov3d[4], v22 = cov3d[5];
    float3 tv0(t0.x * v00 + t0.y * v01 + t0.z * v02,
               t0.x * v01 + t0.y * v11 + t0.z * v12,
               t0.x * v02 + t0.y * v12 + t0.z * v22);
    float3 tv1(t1.x * v00 + t1.y * v01 + t1.z * v02,
               t1.x * v01 + t1.y * v11 + t1.z * v12,
               t1.x * v02 + t1.y * v12 + t1.z * v22);
    return {dot(tv0, t0) + 0.3f, dot(tv0, t1), dot(tv1, t1) + 0.3f};
}

inline bool compute_cov2d_bounds(float3 cov2d, float3 &conic, float &radius) {
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det == 0.f) return false;
    float inv_det = 1.f / det;
    conic.x = cov2d.z * inv_det;
    conic.y = -cov2d.y * inv_det;
    conic.z = cov2d.x * inv_det;
    float b = 0.5f * (cov2d.x + cov2d.z);
    float disc = std::sqrt(std::max(0.1f, b * b - det));
    radius = std::ceil(3.f * std::sqrt(b + disc));
    return true;
}

inline float ndc2pix(float x, float W, float cx) { return 0.5f * W * x + cx - 0.5f; }

inline float2 project_pix(const float *mat, float3 p, uint32_t w, uint32_t h, float2 pp) {
    float4 p_hom = transform_4x4(mat, p);
    float rw = 1.f / (p_hom.w + 1e-6f);
    return {ndc2pix(p_hom.x * rw, (float)(int)w, pp.x), ndc2pix(p_hom.y * rw, (float)(int)h, pp.y)};
}

// Tile-space AABB, inclusive min and exclusive max, clamped to the tile grid
inline void get_tile_bbox(float2 center, float2 radius, int tiles_x, int tiles_y,
                          int &x0, int &y0, int &x1, int &y1) {
    float2 c(center.x / (float)BLOCK_X, center.y / (float)BLOCK_Y);
    float2 r(radius.x / (float)BLOCK_X, radius.y / (float)BLOCK_Y);
    x0 = std::min(std::max(0, (int)(c.x - r.x)), tiles_x);
    x1 = std::min(std::max(0, (int)(c.x + r.x + 1)), tiles_x);
    y0 = std::min(std::max(0, (int)(c.y - r.y)), tiles_y);
    y1 = std::min(std::max(0, (int)(c.y + r.y + 1)), tiles_y);
}

// sh_coeffs_to_color: raw SH color (no +0.5 offset; the rasterizer adds it)
inline void sh_coeffs_to_color(uint32_t degree, float3 viewdir, const float *dc, const float *rest, float *colors) {
    for (int c = 0; c < 3; ++c) colors[c] = SH_C0 * dc[c];
    if (degree < 1) return;
    float x = viewdir.x, y = viewdir.y, z = viewdir.z;
    float xx = x * x, xy = x * y, xz = x * z, yy = y * y, yz = y * z, zz = z * z;
    for (int c = 0; c < 3; ++c) {
        colors[c] += SH_C1 * (-y * rest[0 * 3 + c] + z * rest[1 * 3 + c] - x * rest[2 * 3 + c]);
        if (degree < 2) continue;
        colors[c] += (SH_C2[0] * xy * rest[3 * 3 + c] + SH_C2[1] * yz * rest[4 * 3 + c] +
                      SH_C2[2] * (2.f * zz - xx - yy) * rest[5 * 3 + c] +
                      SH_C2[3] * xz * rest[6 * 3 + c] + SH_C2[4] * (xx - yy) * rest[7 * 3 + c]);
        if (degree < 3) continue;
        colors[c] += (SH_C3[0] * y * (3.f * xx - yy) * rest[8 * 3 + c] +
                      SH_C3[1] * xy * z * rest[9 * 3 + c] +
                      SH_C3[2] * y * (4.f * zz - xx - yy) * rest[10 * 3 + c] +
                      SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy) * rest[11 * 3 + c] +
                      SH_C3[4] * x * (4.f * zz - xx - yy) * rest[12 * 3 + c] +
                      SH_C3[5] * z * (xx - yy) * rest[13 * 3 + c] +
                      SH_C3[6] * x * (xx - 3.f * yy) * rest[14 * 3 + c]);
        if (degree < 4) continue;
        colors[c] += (SH_C4[0] * xy * (xx - yy) * rest[15 * 3 + c] +
                      SH_C4[1] * yz * (3.f * xx - yy) * rest[16 * 3 + c] +
                      SH_C4[2] * xy * (7.f * zz - 1.f) * rest[17 * 3 + c] +
                      SH_C4[3] * yz * (7.f * zz - 3.f) * rest[18 * 3 + c] +
                      SH_C4[4] * (zz * (35.f * zz - 30.f) + 3.f) * rest[19 * 3 + c] +
                      SH_C4[5] * xz * (7.f * zz - 3.f) * rest[20 * 3 + c] +
                      SH_C4[6] * (xx - yy) * (7.f * zz - 1.f) * rest[21 * 3 + c] +
                      SH_C4[7] * xz * (xx - 3.f * yy) * rest[22 * 3 + c] +
                      SH_C4[8] * (xx * (xx - 3.f * yy) - yy * (3.f * xx - yy)) * rest[23 * 3 + c]);
    }
}

// ── Projection VJPs ─────────────────────────────────────────────────────────

// Pixel position gradient back to the world point, through the perspective divide:
// x_pix = W/2 * p_hom.x / p_hom.w + cx - 0.5, so the w row carries the depth term
// (forward motion's looming parallax) that gsplat 0.1 left out.
inline float3 project_pix_vjp(const float *mat, float3 p, uint32_t w, uint32_t h, float2 v_xy) {
    float4 p_hom = transform_4x4(mat, p);
    float rw = 1.f / (p_hom.w + 1e-6f);
    float2 v_ndc(0.5f * (float)w * v_xy.x, 0.5f * (float)h * v_xy.y);
    float4 v_proj(v_ndc.x * rw, v_ndc.y * rw, 0.f, -(v_ndc.x * p_hom.x + v_ndc.y * p_hom.y) * rw * rw);
    return {mat[0] * v_proj.x + mat[4] * v_proj.y + mat[8] * v_proj.z + mat[12] * v_proj.w,
            mat[1] * v_proj.x + mat[5] * v_proj.y + mat[9] * v_proj.z + mat[13] * v_proj.w,
            mat[2] * v_proj.x + mat[6] * v_proj.y + mat[10] * v_proj.z + mat[14] * v_proj.w};
}

inline void cov2d_to_conic_vjp(float3 conic, float3 v_conic, float *v_cov2d) {
    float2x2 X(conic.x, conic.y, conic.y, conic.z);
    float2x2 G(v_conic.x, v_conic.y, v_conic.y, v_conic.z);
    float2x2 v_Sigma = -1.f * (X * G * X);
    v_cov2d[0] = v_Sigma[0][0];
    v_cov2d[1] = v_Sigma[1][0] + v_Sigma[0][1];
    v_cov2d[2] = v_Sigma[1][1];
}

inline void project_cov3d_ewa_vjp(const float *cov3d, const float *viewmat, float fx, float fy,
                                  float tan_fovx, float tan_fovy, float3 v_cov2d, float *v_mean3d,
                                  float *v_cov3d, float3 p_view) {
    float lim_x = 1.3f * tan_fovx, lim_y = 1.3f * tan_fovy;
    p_view.x = p_view.z * std::min(lim_x, std::max(-lim_x, p_view.x / p_view.z));
    p_view.y = p_view.z * std::min(lim_y, std::max(-lim_y, p_view.y / p_view.z));
    float rz = 1.f / p_view.z, rz2 = rz * rz;
    float3x3 W(viewmat[0], viewmat[4], viewmat[8],
               viewmat[1], viewmat[5], viewmat[9],
               viewmat[2], viewmat[6], viewmat[10]);
    float3x3 J(fx * rz, 0.f, 0.f,
               0.f, fy * rz, 0.f,
               -fx * p_view.x * rz2, -fy * p_view.y * rz2, 0.f);
    float3x3 V(cov3d[0], cov3d[1], cov3d[2],
               cov3d[1], cov3d[3], cov3d[4],
               cov3d[2], cov3d[4], cov3d[5]);
    float3x3 v_cov(v_cov2d.x, 0.5f * v_cov2d.y, 0.f,
                   0.5f * v_cov2d.y, v_cov2d.z, 0.f,
                   0.f, 0.f, 0.f);
    float3x3 T = J * W;
    float3x3 Tt = transpose(T);
    float3x3 Vt = transpose(V);
    float3x3 v_V = Tt * v_cov * T;
    float3x3 v_T = v_cov * T * Vt + transpose(v_cov) * T * V;
    v_cov3d[0] = v_V[0][0];
    v_cov3d[1] = v_V[0][1] + v_V[1][0];
    v_cov3d[2] = v_V[0][2] + v_V[2][0];
    v_cov3d[3] = v_V[1][1];
    v_cov3d[4] = v_V[1][2] + v_V[2][1];
    v_cov3d[5] = v_V[2][2];
    float3x3 v_J = v_T * transpose(W);
    float fx_rz2 = fx * rz2, fy_rz2 = fy * rz2, rz3 = rz2 * rz;
    float3 v_t(-fx_rz2 * v_J[2][0],
               -fy_rz2 * v_J[2][1],
               -fx_rz2 * v_J[0][0] + 2.f * fx * p_view.x * rz3 * v_J[2][0] -
                   fy_rz2 * v_J[1][1] + 2.f * fy * p_view.y * rz3 * v_J[2][1]);
    v_mean3d[0] += dot(v_t, W[0]);
    v_mean3d[1] += dot(v_t, W[1]);
    v_mean3d[2] += dot(v_t, W[2]);
}

// Gradient with respect to the stored (unnormalized) quaternion; quat_to_rotmat
// normalizes it, so the result is tangent to the unit sphere. v_R is column-major.
inline float4 quat_to_rotmat_vjp(float4 quat, const float3x3 &v_R) {
    float s = rsqrtf_(quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z);
    float w = quat.x * s, x = quat.y * s, y = quat.z * s, z = quat.w * s;
    float4 v_quat;
    v_quat.x = 2.f * (x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) + z * (v_R[0][1] - v_R[1][0]));
    v_quat.y = 2.f * (-2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
                      z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1]));
    v_quat.z = 2.f * (x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
                      z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2]));
    v_quat.w = 2.f * (x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
                      2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0]));
    // through q / |q|: drop the radial component and scale by 1 / |q|
    float radial = v_quat.x * w + v_quat.y * x + v_quat.z * y + v_quat.w * z;
    return {(v_quat.x - radial * w) * s, (v_quat.y - radial * x) * s, (v_quat.z - radial * y) * s,
            (v_quat.w - radial * z) * s};
}

inline void scale_rot_to_cov3d_vjp(float3 scale, float glob_scale, float4 quat, const float *v_cov3d,
                                   float *v_scale, float *v_quat) {
    float3x3 v_V(v_cov3d[0], 0.5f * v_cov3d[1], 0.5f * v_cov3d[2],
                 0.5f * v_cov3d[1], v_cov3d[3], 0.5f * v_cov3d[4],
                 0.5f * v_cov3d[2], 0.5f * v_cov3d[4], v_cov3d[5]);
    float3x3 R = quat_to_rotmat(quat);
    float3x3 S = scale_to_mat(scale, glob_scale);
    float3x3 M = R * S;
    float3x3 v_M = 2.f * (v_V * M);
    v_scale[0] = dot(R[0], v_M[0]);
    v_scale[1] = dot(R[1], v_M[1]);
    v_scale[2] = dot(R[2], v_M[2]);
    float3x3 v_R = v_M * S;
    float4 q = quat_to_rotmat_vjp(quat, v_R);
    v_quat[0] = q.x; v_quat[1] = q.y; v_quat[2] = q.z; v_quat[3] = q.w;
}

inline void adam_update(float &param, float &ea, float &eas, float grad, float step_size, float beta1,
                        float beta2, float bc2_sqrt, float eps) {
    float m = beta1 * ea + (1.0f - beta1) * grad;
    float v = beta2 * eas + (1.0f - beta2) * grad * grad;
    param -= step_size * m / (std::sqrt(v) / bc2_sqrt + eps);
    ea = m;
    eas = v;
}

// ── Counter-based RNG (pcg_hash / rand_uniform / rand_normal3) ──────────────

inline uint32_t pcg_hash(uint32_t v) {
    uint32_t state = v * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

inline float rand_uniform(uint32_t seed, uint32_t counter) {
    uint32_t h = pcg_hash(seed ^ pcg_hash(counter));
    return ((float)(h >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

inline float3 rand_normal3(uint32_t seed, uint32_t stream) {
    uint32_t c = stream * 4u;
    float u0 = rand_uniform(seed, c), u1 = rand_uniform(seed, c + 1u);
    float u2 = rand_uniform(seed, c + 2u), u3 = rand_uniform(seed, c + 3u);
    float r0 = std::sqrt(-2.0f * std::log(u0)), r1 = std::sqrt(-2.0f * std::log(u2));
    const float tau = 6.28318530717958647692f;
    float t0 = tau * u1, t1 = tau * u3;
    return {r0 * std::cos(t0), r0 * std::sin(t0), r1 * std::cos(t1)};
}

}  // namespace

namespace {

// ── Learned sky (SkyParams / pixel_world_dir / sky_uv / sky_taps) ───────────

struct SkyParams {
    float up[3] = {0.f, 0.f, 0.f}, e1[3] = {0.f, 0.f, 0.f}, e2[3] = {0.f, 0.f, 0.f};
    float fx = 1.f, fy = 1.f, cx = 0.f, cy = 0.f;
    uint32_t tex_w = 1, tex_h = 1;
    bool enabled = false;
};

SkyParams make_sky_params(const PriorStep *prior, float fx, float fy, float cx, float cy) {
    SkyParams sp;
    sp.fx = fx; sp.fy = fy; sp.cx = cx; sp.cy = cy;
    if (prior && prior->sky && prior->sky_tex) {
        std::memcpy(sp.up, prior->sky_frame, 3 * sizeof(float));
        std::memcpy(sp.e1, prior->sky_frame + 3, 3 * sizeof(float));
        std::memcpy(sp.e2, prior->sky_frame + 6, 3 * sizeof(float));
        sp.tex_w = (uint32_t)prior->sky_w;
        sp.tex_h = (uint32_t)prior->sky_h;
        sp.enabled = true;
    }
    return sp;
}

// World direction of the ray through pixel (px, py); viewmat is world-to-camera
inline float3 pixel_world_dir(float px, float py, const float *viewmat, const SkyParams &sky) {
    float dx = (px + 0.5f - sky.cx) / sky.fx;
    float dy = (py + 0.5f - sky.cy) / sky.fy;
    return {viewmat[0] * dx + viewmat[4] * dy + viewmat[8],
            viewmat[1] * dx + viewmat[5] * dy + viewmat[9],
            viewmat[2] * dx + viewmat[6] * dy + viewmat[10]};
}

inline float2 sky_uv(float3 d, const SkyParams &sky) {
    const float pi = 3.14159265358979323846f;
    float3 up(sky.up[0], sky.up[1], sky.up[2]);
    float3 e1(sky.e1[0], sky.e1[1], sky.e1[2]);
    float3 e2(sky.e2[0], sky.e2[1], sky.e2[2]);
    d = normalize(d);
    float el = std::asin(clampf(dot(d, up), -1.f, 1.f));
    float az = std::atan2(dot(d, e2), dot(d, e1));
    float u = (az * (0.5f / pi) + 0.5f) * (float)sky.tex_w - 0.5f;
    float v = (0.5f - el / pi) * (float)sky.tex_h - 0.5f;
    return {u, v};
}

struct SkyTaps {
    uint32_t idx[4];
    float w[4];
};

inline SkyTaps sky_taps(float2 uv, uint32_t tex_w, uint32_t tex_h) {
    float u0f = std::floor(uv.x);
    float fu = uv.x - u0f;
    int u0 = (int)u0f % (int)tex_w;
    if (u0 < 0) u0 += (int)tex_w;
    int u1 = (u0 + 1) % (int)tex_w;
    float vc = clampf(uv.y, 0.f, (float)tex_h - 1.f);
    int v0 = (int)vc;
    int v1 = std::min(v0 + 1, (int)tex_h - 1);
    float fv = vc - (float)v0;
    SkyTaps t;
    t.idx[0] = (uint32_t)(v0 * (int)tex_w + u0); t.w[0] = (1.f - fu) * (1.f - fv);
    t.idx[1] = (uint32_t)(v0 * (int)tex_w + u1); t.w[1] = fu * (1.f - fv);
    t.idx[2] = (uint32_t)(v1 * (int)tex_w + u0); t.w[2] = (1.f - fu) * fv;
    t.idx[3] = (uint32_t)(v1 * (int)tex_w + u1); t.w[3] = fu * fv;
    return t;
}

inline float3 sky_sample(const float *tex, float2 uv, uint32_t tex_w, uint32_t tex_h) {
    SkyTaps t = sky_taps(uv, tex_w, tex_h);
    float3 c(0.f);
    for (int k = 0; k < 4; k++) c = fma3(rd3(tex, t.idx[k]), t.w[k], c);
    return c;
}

// ── Per-frame state (the Metal FusedTensorCache) ────────────────────────────

struct Frame {
    int num_points = 0;
    uint32_t W = 0, H = 0;
    int tiles_x = 0, tiles_y = 0;
    bool aux = false;

    // Per gaussian
    std::vector<float> xys, depths, conics, colors, aabb;
    MTensor radii;  // Int32 (N)

    // Per tile: sorted (front to back) lists [tile_start, tile_end) into the packed arrays
    std::vector<int32_t> tile_start, tile_end;
    std::vector<int32_t> isect_gid;
    std::vector<float> pk_xy_opac, pk_conic, pk_rgb;
    std::vector<uint64_t> bins;
    std::vector<int64_t> bin_offset;
    std::vector<int32_t> tile_count, tile_kept;

    // Per pixel
    MTensor out_img, final_Ts, out_depth, bg_img;
    std::vector<int32_t> final_idx;

    // Training scratch
    MTensor v_rendered, out_adj, v_adj;
    std::vector<float> v_aux, hbuf, deriv;
    std::vector<float> isect_grad;  // 10 per intersection: xy(2) conic(3) rgb(3) opacity depth
    std::vector<int32_t> g_offset, g_isects;
    std::vector<float> v_xy, v_conic, v_rgb, v_opacity, v_depth, v_mean3d, v_scale, v_quat;
};

Frame g_frame;
std::atomic<bool> g_overflow{false};
bool g_overflow_warned = false;

template <typename V> void ensure_size(V &v, size_t n) {
    if (v.size() < n) v.resize(n);
}

void ensure_image(MTensor &t, uint32_t H, uint32_t W, int64_t C) {
    if (!t.defined() || t.size(0) != (int64_t)H || t.size(1) != (int64_t)W ||
        (C > 0 && (t.ndim() < 3 || t.size(2) != C))) {
        std::vector<int64_t> shape = {(int64_t)H, (int64_t)W};
        if (C > 0) shape.push_back(C);
        t = gpu_zeros(shape, DType::Float32);
    }
}

// project_and_sh_forward_kernel
void project_and_sh(Frame &f, int N, const float *means, const float *scales, float glob_scale,
                    const float *quats, const float *viewmat, const float *projmat,
                    float fx, float fy, float cx, float cy, uint32_t W, uint32_t H, float clip_thresh,
                    uint32_t degree, uint32_t degrees_to_use, const float cam_pos[3],
                    const float *features_dc, const float *features_rest) {
    ensure_size(f.xys, (size_t)N * 2);
    ensure_size(f.depths, (size_t)N);
    ensure_size(f.conics, (size_t)N * 3);
    ensure_size(f.colors, (size_t)N * 3);
    ensure_size(f.aabb, (size_t)N * 2);
    if (!f.radii.defined() || f.radii.numel() != N) f.radii = gpu_zeros({std::max(N, 1)}, DType::Int32);
    int32_t *radii = f.radii.data<int32_t>();
    const float3 cpos(cam_pos[0], cam_pos[1], cam_pos[2]);
    const uint32_t num_bases = num_sh_bases(degree);
    const int tiles_x = f.tiles_x, tiles_y = f.tiles_y;

    parallel_for((size_t)N, 4096, [&](size_t b, size_t e) {
        for (size_t idx = b; idx < e; idx++) {
            radii[idx] = 0;
            float3 p_world = rd3(means, idx);
            float3 p_view = transform_4x3(viewmat, p_world);
            if (p_view.z <= clip_thresh) continue;
            float3 scale = exp(rd3(scales, idx));
            float4 quat = rd4(quats, idx);
            float cov3d[6];
            scale_rot_to_cov3d(scale, glob_scale, quat, cov3d);
            float tan_fovx = 0.5f * (float)W / fx, tan_fovy = 0.5f * (float)H / fy;
            float3 cov2d = project_cov3d_ewa(cov3d, viewmat, fx, fy, tan_fovx, tan_fovy, p_view);
            float3 conic;
            float radius;
            if (!compute_cov2d_bounds(cov2d, conic, radius)) continue;
            wr3(f.conics.data(), idx, conic);
            float2 center = project_pix(projmat, p_world, W, H, {cx, cy});
            float aabb_x = std::ceil(3.0f * std::sqrt(cov2d.x));
            float aabb_y = std::ceil(3.0f * std::sqrt(cov2d.z));
            int x0, y0, x1, y1;
            get_tile_bbox(center, {aabb_x, aabb_y}, tiles_x, tiles_y, x0, y0, x1, y1);
            if ((x1 - x0) * (y1 - y0) <= 0) continue;
            f.depths[idx] = p_view.z;
            radii[idx] = (int)radius;
            wr2(f.xys.data(), idx, center);
            f.aabb[2 * idx] = aabb_x;
            f.aabb[2 * idx + 1] = aabb_y;
            float3 viewdir = normalize(p_world - cpos);
            sh_coeffs_to_color(degrees_to_use, viewdir, features_dc + 3 * idx,
                               features_rest + (size_t)(num_bases - 1) * 3 * idx, f.colors.data() + 3 * idx);
        }
    });
}

// scatter_to_prealloc_bins + bitonic_sort_per_tile: per-tile lists sorted front to
// back by (depth, index), capped at MAX_TILE_ELEMS (the nearest are kept), packed.
void bin_and_sort(Frame &f, int N, const float *opacities) {
    const int num_tiles = f.tiles_x * f.tiles_y;
    const int32_t *radii = f.radii.data<int32_t>();
    f.tile_count.assign(num_tiles, 0);
    for (int idx = 0; idx < N; idx++) {
        if (radii[idx] <= 0) continue;
        int x0, y0, x1, y1;
        get_tile_bbox(rd2(f.xys.data(), idx), rd2(f.aabb.data(), idx), f.tiles_x, f.tiles_y, x0, y0, x1, y1);
        for (int ty = y0; ty < y1; ty++)
            for (int tx = x0; tx < x1; tx++) f.tile_count[ty * f.tiles_x + tx]++;
    }
    f.bin_offset.assign(num_tiles + 1, 0);
    for (int t = 0; t < num_tiles; t++) f.bin_offset[t + 1] = f.bin_offset[t] + f.tile_count[t];
    ensure_size(f.bins, (size_t)f.bin_offset[num_tiles]);
    std::vector<int64_t> cursor(f.bin_offset.begin(), f.bin_offset.end() - 1);
    for (int idx = 0; idx < N; idx++) {
        if (radii[idx] <= 0) continue;
        int x0, y0, x1, y1;
        get_tile_bbox(rd2(f.xys.data(), idx), rd2(f.aabb.data(), idx), f.tiles_x, f.tiles_y, x0, y0, x1, y1);
        uint32_t depth_bits;
        std::memcpy(&depth_bits, &f.depths[idx], 4);
        uint64_t key = ((uint64_t)depth_bits << 32) | (uint64_t)(uint32_t)idx;
        for (int ty = y0; ty < y1; ty++)
            for (int tx = x0; tx < x1; tx++) f.bins[cursor[ty * f.tiles_x + tx]++] = key;
    }
    f.tile_kept.assign(num_tiles, 0);
    parallel_for((size_t)num_tiles, 8, [&](size_t b, size_t e) {
        for (size_t t = b; t < e; t++) {
            uint64_t *first = f.bins.data() + f.bin_offset[t], *last = f.bins.data() + f.bin_offset[t + 1];
            int64_t count = last - first;
            if (count > MAX_TILE_ELEMS) {
                std::partial_sort(first, first + MAX_TILE_ELEMS, last);
                g_overflow = true;
            } else {
                std::sort(first, last);
            }
            f.tile_kept[t] = (int32_t)std::min<int64_t>(count, MAX_TILE_ELEMS);
        }
    });
    f.tile_start.assign(num_tiles, 0);
    f.tile_end.assign(num_tiles, 0);
    int64_t total = 0;
    for (int t = 0; t < num_tiles; t++) {
        f.tile_start[t] = (int32_t)total;
        total += f.tile_kept[t];
        f.tile_end[t] = (int32_t)total;
    }
    ensure_size(f.isect_gid, (size_t)std::max<int64_t>(total, 1));
    ensure_size(f.pk_xy_opac, (size_t)std::max<int64_t>(total, 1) * 3);
    ensure_size(f.pk_conic, (size_t)std::max<int64_t>(total, 1) * 3);
    ensure_size(f.pk_rgb, (size_t)std::max<int64_t>(total, 1) * 3);
    parallel_for((size_t)num_tiles, 8, [&](size_t b, size_t e) {
        for (size_t t = b; t < e; t++) {
            const uint64_t *src = f.bins.data() + f.bin_offset[t];
            for (int i = 0; i < f.tile_kept[t]; i++) {
                int32_t g = (int32_t)(src[i] & 0xFFFFFFFFull);
                int64_t k = f.tile_start[t] + i;
                f.isect_gid[k] = g;
                float2 xy = rd2(f.xys.data(), g);
                float opac = 1.f / (1.f + std::exp(-opacities[g]));
                wr3(f.pk_xy_opac.data(), k, {xy.x, xy.y, opac});
                wr3(f.pk_conic.data(), k, rd3(f.conics.data(), g));
                wr3(f.pk_rgb.data(), k, rd3(f.colors.data(), g));
            }
        }
    });
}

inline float gaussian_sigma(float3 conic, float2 delta) {
    return 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
}

// nd_rasterize_forward_kernel / rasterize_forward_aux_kernel
void rasterize_forward(Frame &f, const float *background, const float *sky_tex, const SkyParams &sky,
                       const float *viewmat) {
    const uint32_t W = f.W, H = f.H;
    const int num_tiles = f.tiles_x * f.tiles_y;
    float *out = f.out_img.data<float>();
    float *Ts = f.final_Ts.data<float>();
    float *out_depth = f.aux ? f.out_depth.data<float>() : nullptr;
    float *bg_img = f.aux ? f.bg_img.data<float>() : nullptr;
    const float3 bg_const(background[0], background[1], background[2]);
    parallel_for((size_t)num_tiles, 1, [&](size_t b, size_t e) {
        for (size_t t = b; t < e; t++) {
            const int tx = (int)t % f.tiles_x, ty = (int)t / f.tiles_x;
            const int start = f.tile_start[t], end = f.tile_end[t];
            for (int i = ty * BLOCK_Y; i < std::min((int)H, (ty + 1) * BLOCK_Y); i++) {
                for (int j = tx * BLOCK_X; j < std::min((int)W, (tx + 1) * BLOCK_X); j++) {
                    const float px = (float)j, py = (float)i;
                    float T = 1.f;
                    float3 pix(0.f);
                    float depth = 0.f;
                    int last = start - 1;
                    for (int k = start; k < end; k++) {
                        float3 xyo = rd3(f.pk_xy_opac.data(), k);
                        float sigma = gaussian_sigma(rd3(f.pk_conic.data(), k), {xyo.x - px, xyo.y - py});
                        if (sigma < 0.f || sigma >= 5.55f) continue;
                        float alpha = std::min(0.999f, xyo.z * std::exp(-sigma));
                        if (alpha < 1.f / 255.f) continue;
                        float next_T = T * (1.f - alpha);
                        if (next_T <= 1e-4f) {
                            last = k - 1;
                            break;
                        }
                        float vis = alpha * T;
                        pix = fma3(max(rd3(f.pk_rgb.data(), k) + 0.5f, 0.f), vis, pix);
                        if (out_depth) depth += f.depths[f.isect_gid[k]] * vis;
                        T = next_T;
                        last = k;
                    }
                    const int pid = i * (int)W + j;
                    float3 bg = bg_const;
                    if (f.aux && sky.enabled)
                        bg = sky_sample(sky_tex, sky_uv(pixel_world_dir(px, py, viewmat, sky), sky), sky.tex_w, sky.tex_h);
                    Ts[pid] = T;
                    f.final_idx[pid] = last;
                    if (out_depth) {
                        out_depth[pid] = depth;
                        wr3(bg_img, pid, bg);
                    }
                    wr3(out, pid, saturate(fma3(bg, T, pix)));
                }
            }
        }
    });
}

// The shared forward of render and train: projection, binning, rasterization
void forward(Frame &f, int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
             MTensor &quats, MTensor &viewmat, MTensor &projmat, float fx, float fy, float cx, float cy,
             unsigned img_height, unsigned img_width, const std::tuple<int, int, int> tile_bounds,
             float clip_thresh, unsigned degree, unsigned degrees_to_use, float cam_pos[3],
             MTensor &features_dc, MTensor &features_rest, MTensor &opacities, MTensor &background,
             const PriorStep *prior) {
    if (g_overflow && !g_overflow_warned) {
        std::fprintf(stderr, "WARNING: per-tile overflow (>%d gaussians in a tile). "
                             "The farthest were dropped from overfull tiles.\n", MAX_TILE_ELEMS);
        g_overflow_warned = true;
    }
    f.num_points = num_points;
    f.W = img_width;
    f.H = img_height;
    f.tiles_x = std::get<0>(tile_bounds);
    f.tiles_y = std::get<1>(tile_bounds);
    f.aux = prior && prior->aux;
    ensure_image(f.out_img, img_height, img_width, 3);
    ensure_image(f.final_Ts, img_height, img_width, 0);
    ensure_size(f.final_idx, (size_t)img_height * img_width);
    if (f.aux) {
        ensure_image(f.out_depth, img_height, img_width, 0);
        ensure_image(f.bg_img, img_height, img_width, 3);
    }
    project_and_sh(f, num_points, means3d.data<float>(), scales.data<float>(), glob_scale, quats.data<float>(),
                   viewmat.data<float>(), projmat.data<float>(), fx, fy, cx, cy, img_width, img_height,
                   clip_thresh, degree, degrees_to_use, cam_pos, features_dc.data<float>(),
                   features_rest.data<float>());
    bin_and_sort(f, num_points, opacities.data<float>());
    SkyParams sky = make_sky_params(prior, fx, fy, cx, cy);
    const float *sky_tex = (prior && prior->sky && prior->sky_tex) ? prior->sky_tex->data<float>() : nullptr;
    if (!sky_tex) sky.enabled = false;
    rasterize_forward(f, background.data<float>(), sky_tex, sky, viewmat.data<float>());
}

}  // namespace

namespace {

// ── Loss: L1 + SSIM (ssim_h_fwd / ssim_fused_v_fwd_h_bwd / ssim_v_bwd) ──────
// Zero-padded 11x11 Gaussian SSIM between the render (y) and the ground truth (x).
// The derivative fields exist only where the loss does, inside the image: the
// horizontal backward convolution reads zeros beyond the edges, as the vertical one
// does. Returns the summed per-pixel loss.

void ssim_h_fwd(const float *rendered, const float *gt, uint32_t W, uint32_t H, float *hbuf) {
    parallel_for(H, 8, [&](size_t b, size_t e) {
        for (size_t y = b; y < e; y++) {
            for (uint32_t x = 0; x < W; x++) {
                for (int c = 0; c < 3; c++) {
                    float mu_x = 0, mu_y = 0, sq_x = 0, sq_y = 0, cross = 0;
                    for (int k = 0; k < SSIM_WIN; k++) {
                        int gx = (int)x - SSIM_HALF_WIN + k;
                        float gv = 0.f, rv = 0.f;
                        if (gx >= 0 && gx < (int)W) {
                            size_t idx = (y * W + (size_t)gx) * 3 + c;
                            gv = gt[idx];
                            rv = rendered[idx];
                        }
                        float w = GAUSS_1D[k];
                        mu_x += w * gv;
                        mu_y += w * rv;
                        sq_x += w * gv * gv;
                        sq_y += w * rv * rv;
                        cross += w * gv * rv;
                    }
                    float *o = hbuf + (y * W + x) * 15 + c * 5;
                    o[0] = mu_x; o[1] = mu_y; o[2] = sq_x; o[3] = sq_y; o[4] = cross;
                }
            }
        }
    });
}

double ssim_fused_v_fwd_h_bwd(const float *rendered, const float *gt, const float *hbuf, uint32_t W, uint32_t H,
                              float ssim_weight, float *deriv) {
    std::vector<double> row_loss(H, 0.0);
    parallel_for(H, 4, [&](size_t b, size_t e) {
        const int ext = (int)W + 2 * SSIM_HALF_WIN;
        std::vector<float> f1(ext), f2(ext), f3(ext), ssim_pix(W), l1_pix(W);
        for (size_t y = b; y < e; y++) {
            std::fill(ssim_pix.begin(), ssim_pix.end(), 0.f);
            std::fill(l1_pix.begin(), l1_pix.end(), 0.f);
            for (int c = 0; c < 3; c++) {
                for (int xe = 0; xe < ext; xe++) {
                    const int xp = xe - SSIM_HALF_WIN;
                    if (xp < 0 || xp >= (int)W) {
                        f1[xe] = f2[xe] = f3[xe] = 0.f;
                        continue;
                    }
                    float s[5] = {0, 0, 0, 0, 0};
                    for (int k = 0; k < SSIM_WIN; k++) {
                        int gy = (int)y - SSIM_HALF_WIN + k;
                        if (gy < 0 || gy >= (int)H) continue;
                        const float *h = hbuf + ((size_t)gy * W + xp) * 15 + c * 5;
                        float w = GAUSS_1D[k];
                        for (int q = 0; q < 5; q++) s[q] += w * h[q];
                    }
                    float mu_x = s[0], mu_y = s[1], sq_x = s[2], sq_y = s[3], cross = s[4];
                    float sigma_x_sq = sq_x - mu_x * mu_x, sigma_y_sq = sq_y - mu_y * mu_y;
                    float sigma_xy = cross - mu_x * mu_y;
                    float A = 2.0f * mu_x * mu_y + SSIM_C1, B = 2.0f * sigma_xy + SSIM_C2;
                    float Cd = mu_x * mu_x + mu_y * mu_y + SSIM_C1, D = sigma_x_sq + sigma_y_sq + SSIM_C2;
                    float iCD = 1.0f / (Cd * D);
                    float dmu = 2.0f * B * (mu_x * Cd - A * mu_y) / (Cd * Cd * D);
                    float dsyq = -A * B * iCD / D, dsxy = 2.0f * A * iCD;
                    f1[xe] = dmu - 2.0f * mu_y * dsyq - mu_x * dsxy;
                    f2[xe] = 2.0f * dsyq;
                    f3[xe] = dsxy;
                    size_t i = ((size_t)y * W + xp) * 3 + c;
                    ssim_pix[xp] += (A * B) / (Cd * D);
                    l1_pix[xp] += std::fabs(gt[i] - rendered[i]);
                }
                for (uint32_t x = 0; x < W; x++) {
                    float h1 = 0, h2 = 0, h3 = 0;
                    for (int dx = 0; dx < SSIM_WIN; dx++) {
                        float w = GAUSS_1D[SSIM_WIN - 1 - dx];
                        h1 += w * f1[x + dx];
                        h2 += w * f2[x + dx];
                        h3 += w * f3[x + dx];
                    }
                    float *o = deriv + ((size_t)y * W + x) * 15 + c * 5;
                    o[0] = h1; o[1] = h2; o[2] = h3;
                }
            }
            double sum = 0.0;
            for (uint32_t x = 0; x < W; x++)
                sum += ssim_weight * (1.0f - ssim_pix[x] / 3.0f) + (1.0f - ssim_weight) * l1_pix[x] / 3.0f;
            row_loss[y] = sum;
        }
    });
    double total = 0.0;
    for (double v : row_loss) total += v;
    return total;
}

void ssim_v_bwd(const float *rendered, const float *gt, const float *deriv, uint32_t W, uint32_t H,
                float ssim_weight, float inv_n, float *v_rendered) {
    parallel_for(H, 8, [&](size_t b, size_t e) {
        for (size_t y = b; y < e; y++) {
            for (uint32_t x = 0; x < W; x++) {
                for (int c = 0; c < 3; c++) {
                    float cf1 = 0, cf2 = 0, cf3 = 0;
                    for (int dy = 0; dy < SSIM_WIN; dy++) {
                        int gy = (int)y - SSIM_HALF_WIN + dy;
                        if (gy < 0 || gy >= (int)H) continue;
                        const float *d = deriv + ((size_t)gy * W + x) * 15 + c * 5;
                        float w = GAUSS_1D[SSIM_WIN - 1 - dy];
                        cf1 += w * d[0];
                        cf2 += w * d[1];
                        cf3 += w * d[2];
                    }
                    size_t i = ((size_t)y * W + x) * 3 + c;
                    float rv = rendered[i], gv = gt[i];
                    float v_ssim = cf1 + rv * cf2 + gv * cf3;
                    float v_l1 = (gv > rv) ? -1.0f : ((gv < rv) ? 1.0f : 0.0f);
                    v_rendered[i] = inv_n * (-ssim_weight * v_ssim + (1.0f - ssim_weight) * v_l1);
                }
            }
        }
    });
}

// ── Prior-guided extras ─────────────────────────────────────────────────────

// sample_prior_depth: bilinear that refuses to invent a surface across a depth edge
inline float sample_prior_depth(const float *d, float x, float y, uint32_t w, uint32_t h) {
    x = clampf(x, 0.f, (float)w - 1.f);
    y = clampf(y, 0.f, (float)h - 1.f);
    uint32_t x0 = (uint32_t)x, y0 = (uint32_t)y;
    uint32_t x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
    float fx = x - (float)x0, fy = y - (float)y0;
    float d00 = d[y0 * w + x0], d01 = d[y0 * w + x1];
    float d10 = d[y1 * w + x0], d11 = d[y1 * w + x1];
    float lo = std::min(std::min(d00, d01), std::min(d10, d11));
    float hi = std::max(std::max(d00, d01), std::max(d10, d11));
    if (lo <= 0.f || hi > 1.05f * lo) {
        uint32_t nx = (fx < 0.5f) ? x0 : x1;
        uint32_t ny = (fy < 0.5f) ? y0 : y1;
        return d[ny * w + nx];
    }
    return mixf(mixf(d00, d01, fx), mixf(d10, d11, fx), fy);
}

// prior_loss_kernel: v_aux (dL/dD, dL/dA per pixel), masks v_photo in place, logs terms
void prior_loss(Frame &f, const PriorStep &p, float *v_photo) {
    const uint32_t W = f.W, H = f.H;
    const bool has_prior = p.prior_depth && p.prior_aux && p.prior_w > 0 && p.prior_h > 0;
    const uint32_t pw = has_prior ? (uint32_t)p.prior_w : 1u, ph = has_prior ? (uint32_t)p.prior_h : 1u;
    const float *prior_depth = has_prior ? p.prior_depth->data<float>() : nullptr;
    const uint8_t *prior_aux = has_prior ? p.prior_aux->data<uint8_t>() : nullptr;
    static const uint8_t default_aux[4] = {0, 0, 255, 0};  // keep everything, no sky, zero confidence
    static const float default_depth[1] = {0.f};
    if (!has_prior) { prior_aux = default_aux; prior_depth = default_depth; }
    const bool has_depth = has_prior && p.has_depth;
    const bool has_sky = has_prior && p.has_sky_mask;
    const float huber = std::max(p.huber_delta, 1e-6f);
    const float inv_npix = 1.0f / (float)(W * H);
    ensure_size(f.v_aux, (size_t)W * H * 2);
    const float *Ts = f.final_Ts.data<float>();
    const float *out_depth = f.out_depth.data<float>();
    std::vector<double> terms((size_t)H * 3, 0.0);
    parallel_for(H, 8, [&](size_t b, size_t e) {
        for (size_t gy = b; gy < e; gy++) {
            double ld = 0, ls = 0, lf = 0;
            for (uint32_t gx = 0; gx < W; gx++) {
                size_t pix = gy * W + gx;
                float sx = ((float)gx + 0.5f) * (float)pw / (float)W;
                float sy = ((float)gy + 0.5f) * (float)ph / (float)H;
                uint32_t q = std::min((uint32_t)sy, ph - 1) * pw + std::min((uint32_t)sx, pw - 1);
                float conf = (float)prior_aux[4 * q + 0] * (1.f / 255.f);
                bool is_sky = has_sky && prior_aux[4 * q + 1] >= 128;
                bool keep = prior_aux[4 * q + 2] >= 128;
                float v_d = 0.f, v_a = 0.f;
                if (!keep) {
                    if (p.mask_photometric) wr3(v_photo, pix, float3(0.f));
                } else {
                    float A = 1.f - Ts[pix];
                    if (is_sky) {
                        ls += p.sky_weight * A;
                        v_a = p.sky_weight * inv_npix;
                    } else {
                        if (has_sky && p.fill_weight > 0.f) {
                            lf += p.fill_weight * (1.f - A);
                            v_a -= p.fill_weight * inv_npix;
                        }
                        float D = out_depth[pix];
                        if (has_depth && p.depth_weight > 0.f && conf > 0.f && A > p.min_alpha && D > 0.f) {
                            float dp = sample_prior_depth(prior_depth, sx - 0.5f, sy - 0.5f, pw, ph);
                            if (dp > 0.f) {
                                float r = std::log(D / A) - std::log(dp);
                                float ar = std::fabs(r);
                                float dr;
                                if (ar < huber) {
                                    ld += p.depth_weight * conf * 0.5f * r * r / huber;
                                    dr = r / huber;
                                } else {
                                    ld += p.depth_weight * conf * (ar - 0.5f * huber);
                                    dr = signf(r);
                                }
                                float s = p.depth_weight * conf * dr * inv_npix;
                                v_d = s / D;
                                v_a -= s / A;
                            }
                        }
                    }
                }
                f.v_aux[2 * pix] = v_d;
                f.v_aux[2 * pix + 1] = v_a;
            }
            terms[3 * gy] = ld; terms[3 * gy + 1] = ls; terms[3 * gy + 2] = lf;
        }
    });
    if (p.loss_terms) {
        double t[3] = {0, 0, 0};
        for (uint32_t y = 0; y < H; y++)
            for (int k = 0; k < 3; k++) t[k] += terms[3 * y + k];
        float *lt = p.loss_terms->data<float>();
        for (int k = 0; k < 3; k++) lt[k] += (float)(t[k] * inv_npix);
    }
}

// exposure_apply_kernel
void exposure_apply(const float *img, const float *P, size_t n, float *out) {
    parallel_for(n, 65536, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; i++) {
            float3 c = rd3(img, i);
            wr3(out, i, {dot(float3(P[0], P[1], P[2]), c) + P[9],
                         dot(float3(P[3], P[4], P[5]), c) + P[10],
                         dot(float3(P[6], P[7], P[8]), c) + P[11]});
        }
    });
}

// exposure_backward_kernel + exposure_adam_kernel
void exposure_backward_adam(const float *img, const PriorStep &p, size_t n, const float *v_adj, float *v_img,
                            float beta1, float beta2, float eps) {
    const uint32_t cam = (uint32_t)std::max(0, p.cam_index);
    float *params = p.expo_params->data<float>();
    const float *P = params + 12 * cam;
    double g[12] = {0};
    for (size_t i = 0; i < n; i++) {
        float3 c = rd3(img, i), v = rd3(v_adj, i);
        wr3(v_img, i, {P[0] * v.x + P[3] * v.y + P[6] * v.z,
                       P[1] * v.x + P[4] * v.y + P[7] * v.z,
                       P[2] * v.x + P[5] * v.y + P[8] * v.z});
        g[0] += v.x * c.x; g[1] += v.x * c.y; g[2] += v.x * c.z;
        g[3] += v.y * c.x; g[4] += v.y * c.y; g[5] += v.y * c.z;
        g[6] += v.z * c.x; g[7] += v.z * c.y; g[8] += v.z * c.z;
        g[9] += v.x; g[10] += v.y; g[11] += v.z;
    }
    float *ea = p.expo_exp_avg->data<float>(), *eas = p.expo_exp_avg_sq->data<float>();
    float *grad = p.expo_grad ? p.expo_grad->data<float>() : nullptr;
    for (int k = 0; k < 12; k++) {
        uint32_t i = 12 * cam + k;
        float ident = (k == 0 || k == 4 || k == 8) ? 1.f : 0.f;
        float gk = (float)g[k] + (grad ? grad[k] : 0.f) + p.expo_reg * (params[i] - ident);
        if (grad) grad[k] = 0.f;
        adam_update(params[i], ea[i], eas[i], gk, p.expo_step_size, beta1, beta2, p.expo_bc2_sqrt, eps);
    }
}

// sky_backward_kernel + fused_adam_kernel on the texture
void sky_backward_adam(Frame &f, const PriorStep &p, const SkyParams &sky, const float *viewmat,
                       const float *v_rendered, float beta1, float beta2, float eps) {
    float *grad = p.sky_grad->data<float>();
    const float *Ts = f.final_Ts.data<float>();
    for (uint32_t y = 0; y < f.H; y++) {
        for (uint32_t x = 0; x < f.W; x++) {
            size_t pix = (size_t)y * f.W + x;
            float T = Ts[pix];
            if (T < 1e-4f) continue;
            float3 g = T * rd3(v_rendered, pix);
            SkyTaps taps = sky_taps(sky_uv(pixel_world_dir((float)x, (float)y, viewmat, sky), sky), sky.tex_w, sky.tex_h);
            for (int k = 0; k < 4; k++) {
                if (taps.w[k] == 0.f) continue;
                float *t = grad + 3 * (size_t)taps.idx[k];
                t[0] += taps.w[k] * g.x; t[1] += taps.w[k] * g.y; t[2] += taps.w[k] * g.z;
            }
        }
    }
    float *tex = p.sky_tex->data<float>();
    float *ea = p.sky_exp_avg->data<float>(), *eas = p.sky_exp_avg_sq->data<float>();
    const size_t n = (size_t)p.sky_tex->numel();
    for (size_t i = 0; i < n; i++)
        adam_update(tex[i], ea[i], eas[i], grad[i], p.sky_step_size, beta1, beta2, p.sky_bc2_sqrt, eps);
}

// ── Backward rasterization (rasterize_backward_kernel / _aux_kernel) ────────
// Each tile accumulates its gaussians' gradients over its pixels into the
// gaussians' slots in isect_grad; reduce_isect_grads then sums the slots per
// gaussian. No atomics, and the sums do not depend on thread scheduling.

constexpr int GRAD_STRIDE = 10;  // xy(2) conic(3) rgb(3) opacity depth

void rasterize_backward(Frame &f, const float *background, const float *v_out_img) {
    const uint32_t W = f.W, H = f.H;
    const int num_tiles = f.tiles_x * f.tiles_y;
    const int64_t total = num_tiles ? f.tile_end[num_tiles - 1] : 0;
    ensure_size(f.isect_grad, (size_t)std::max<int64_t>(total, 1) * GRAD_STRIDE);
    const float *Ts = f.final_Ts.data<float>();
    const float *bg_img = f.aux ? f.bg_img.data<float>() : nullptr;
    const float3 bg_const(background[0], background[1], background[2]);
    const bool aux = f.aux;
    parallel_for((size_t)num_tiles, 1, [&](size_t b, size_t e) {
        for (size_t t = b; t < e; t++) {
            const int start = f.tile_start[t], end = f.tile_end[t];
            float *lv = f.isect_grad.data() + (size_t)start * GRAD_STRIDE;
            std::fill(lv, lv + (size_t)(end - start) * GRAD_STRIDE, 0.f);
            if (end == start) continue;
            const int tx = (int)t % f.tiles_x, ty = (int)t / f.tiles_x;
            for (int i = ty * BLOCK_Y; i < std::min((int)H, (ty + 1) * BLOCK_Y); i++) {
                for (int j = tx * BLOCK_X; j < std::min((int)W, (tx + 1) * BLOCK_X); j++) {
                    const int pid = i * (int)W + j;
                    const float px = (float)j, py = (float)i;
                    const float T_final = Ts[pid];
                    float T = T_final;
                    float3 buffer(0.f);
                    float buffer_d = 0.f, buffer_a = 0.f;
                    const int bin_final = f.final_idx[pid];
                    const float3 v_out = rd3(v_out_img, pid);
                    const float v_d = aux ? f.v_aux[2 * pid] : 0.f;
                    const float v_a = aux ? f.v_aux[2 * pid + 1] : 0.f;
                    const float3 T_final_bg = T_final * (aux ? rd3(bg_img, pid) : bg_const);
                    for (int k = std::min(bin_final, end - 1); k >= start; k--) {
                        float3 xyo = rd3(f.pk_xy_opac.data(), k);
                        float3 conic = rd3(f.pk_conic.data(), k);
                        float2 delta(xyo.x - px, xyo.y - py);
                        float sigma = gaussian_sigma(conic, delta);
                        if (sigma < 0.f || sigma >= 5.55f) continue;
                        float alpha = std::min(0.999f, xyo.z * std::exp(-sigma));
                        if (alpha < 1.f / 255.f) continue;
                        float ra = 1.f / (1.f - alpha);
                        T *= ra;
                        const float fac = alpha * T;
                        const float3 raw = rd3(f.pk_rgb.data(), k);
                        const float3 rgb = max(raw + 0.5f, 0.f);
                        float v_alpha = dot(fma3(rgb, T, fma3(-buffer, ra, -ra * T_final_bg)), v_out);
                        float z = 0.f;
                        if (aux) {
                            z = f.depths[f.isect_gid[k]];
                            v_alpha += (z * T - buffer_d * ra) * v_d + (T - buffer_a * ra) * v_a;
                        }
                        buffer = fma3(rgb, fac, buffer);
                        float *g = lv + (size_t)(k - start) * GRAD_STRIDE;
                        if (raw.x + 0.5f >= 0.f) g[5] += fac * v_out.x;
                        if (raw.y + 0.5f >= 0.f) g[6] += fac * v_out.y;
                        if (raw.z + 0.5f >= 0.f) g[7] += fac * v_out.z;
                        if (aux) {
                            buffer_d += z * fac;
                            buffer_a += fac;
                            g[9] += fac * v_d;
                        }
                        if (alpha < 0.999f) {
                            const float v_sigma = -alpha * v_alpha;
                            g[2] += 0.5f * v_sigma * delta.x * delta.x;
                            g[3] += 0.5f * v_sigma * delta.x * delta.y;
                            g[4] += 0.5f * v_sigma * delta.y * delta.y;
                            g[0] += v_sigma * (conic.x * delta.x + conic.y * delta.y);
                            g[1] += v_sigma * (conic.y * delta.x + conic.z * delta.y);
                            g[8] += -v_sigma * (1.f - xyo.z);
                        }
                    }
                }
            }
        }
    });
}

// Per-gaussian sums of the intersection slots (v_xy, v_conic, v_rgb, v_opacity, v_depth)
void reduce_isect_grads(Frame &f, int N) {
    const int num_tiles = f.tiles_x * f.tiles_y;
    const int64_t total = num_tiles ? f.tile_end[num_tiles - 1] : 0;
    f.g_offset.assign((size_t)N + 1, 0);
    for (int64_t k = 0; k < total; k++) f.g_offset[f.isect_gid[k] + 1]++;
    for (int i = 0; i < N; i++) f.g_offset[i + 1] += f.g_offset[i];
    ensure_size(f.g_isects, (size_t)std::max<int64_t>(total, 1));
    {
        std::vector<int32_t> cur(f.g_offset.begin(), f.g_offset.end() - 1);
        for (int64_t k = 0; k < total; k++) f.g_isects[cur[f.isect_gid[k]]++] = (int32_t)k;
    }
    ensure_size(f.v_xy, (size_t)N * 2);
    ensure_size(f.v_conic, (size_t)N * 3);
    ensure_size(f.v_rgb, (size_t)N * 3);
    ensure_size(f.v_opacity, (size_t)N);
    ensure_size(f.v_depth, (size_t)N);
    parallel_for((size_t)N, 4096, [&](size_t b, size_t e) {
        for (size_t g = b; g < e; g++) {
            float s[GRAD_STRIDE] = {0};
            for (int32_t q = f.g_offset[g]; q < f.g_offset[g + 1]; q++) {
                const float *v = f.isect_grad.data() + (size_t)f.g_isects[q] * GRAD_STRIDE;
                for (int c = 0; c < GRAD_STRIDE; c++) s[c] += v[c];
            }
            f.v_xy[2 * g] = s[0]; f.v_xy[2 * g + 1] = s[1];
            f.v_conic[3 * g] = s[2]; f.v_conic[3 * g + 1] = s[3]; f.v_conic[3 * g + 2] = s[4];
            f.v_rgb[3 * g] = s[5]; f.v_rgb[3 * g + 1] = s[6]; f.v_rgb[3 * g + 2] = s[7];
            f.v_opacity[g] = s[8];
            f.v_depth[g] = s[9];
        }
    });
}

// project_and_sh_backward_kernel: v_mean3d / v_scale / v_quat, with the SH backward
// fused into Adam for the visible gaussians (degrees up to 3, like the kernel)
void project_and_sh_backward_adam(Frame &f, int N, const float *means, const float *scales, float glob_scale,
                                  const float *quats, const float *viewmat, const float *projmat,
                                  float fx, float fy, uint32_t degree, uint32_t degrees_to_use,
                                  const float cam_pos[3], float *features_dc, float *features_rest,
                                  float *dc_ea, float *dc_eas, float *rest_ea, float *rest_eas,
                                  float dc_step, float dc_bc2, float rest_step, float rest_bc2,
                                  float beta1, float beta2, float eps) {
    ensure_size(f.v_mean3d, (size_t)N * 3);
    ensure_size(f.v_scale, (size_t)N * 3);
    ensure_size(f.v_quat, (size_t)N * 4);
    const int32_t *radii = f.radii.data<int32_t>();
    const uint32_t W = f.W, H = f.H;
    const float3 cpos(cam_pos[0], cam_pos[1], cam_pos[2]);
    const uint32_t num_bases = num_sh_bases(degree);
    parallel_for((size_t)N, 2048, [&](size_t b, size_t e) {
        for (size_t idx = b; idx < e; idx++) {
            float *vm = f.v_mean3d.data() + 3 * idx;
            float *vs = f.v_scale.data() + 3 * idx;
            float *vq = f.v_quat.data() + 4 * idx;
            if (radii[idx] <= 0) {
                vm[0] = vm[1] = vm[2] = 0.f;
                vs[0] = vs[1] = vs[2] = 0.f;
                vq[0] = vq[1] = vq[2] = vq[3] = 0.f;
                continue;
            }
            float3 p_world = rd3(means, idx);
            float3 v_mean = project_pix_vjp(projmat, p_world, W, H, rd2(f.v_xy.data(), idx));
            v_mean += float3(viewmat[8], viewmat[9], viewmat[10]) * f.v_depth[idx];
            vm[0] = v_mean.x; vm[1] = v_mean.y; vm[2] = v_mean.z;
            float v_cov2d[3];
            cov2d_to_conic_vjp(rd3(f.conics.data(), idx), rd3(f.v_conic.data(), idx), v_cov2d);
            float3 exp_scale = exp(rd3(scales, idx));
            float4 quat = rd4(quats, idx);
            float cov3d[6];
            scale_rot_to_cov3d(exp_scale, glob_scale, quat, cov3d);
            float tan_fovx = 0.5f * (float)W / fx, tan_fovy = 0.5f * (float)H / fy;
            float3 p_view = transform_4x3(viewmat, p_world);
            float v_cov3d[6];
            project_cov3d_ewa_vjp(cov3d, viewmat, fx, fy, tan_fovx, tan_fovy,
                                  {v_cov2d[0], v_cov2d[1], v_cov2d[2]}, vm, v_cov3d, p_view);
            scale_rot_to_cov3d_vjp(exp_scale, glob_scale, quat, v_cov3d, vs, vq);
            vs[0] *= exp_scale.x; vs[1] *= exp_scale.y; vs[2] *= exp_scale.z;

            // SH backward fused with Adam
            float3 viewdir = normalize(p_world - cpos);
            const size_t dc_idx = 3 * idx, rest_idx = (size_t)(num_bases - 1) * 3 * idx;
            const float vc[3] = {f.v_rgb[3 * idx], f.v_rgb[3 * idx + 1], f.v_rgb[3 * idx + 2]};
            for (int c = 0; c < 3; c++)
                adam_update(features_dc[dc_idx + c], dc_ea[dc_idx + c], dc_eas[dc_idx + c], SH_C0 * vc[c],
                            dc_step, beta1, beta2, dc_bc2, eps);
            if (degrees_to_use < 1) continue;
            float x = viewdir.x, y = viewdir.y, z = viewdir.z;
            float xx = x * x, xy = x * y, xz = x * z, yy = y * y, yz = y * z, zz = z * z;
            auto upd = [&](int base, float sh) {
                for (int c = 0; c < 3; c++) {
                    size_t i = rest_idx + (size_t)base * 3 + c;
                    adam_update(features_rest[i], rest_ea[i], rest_eas[i], sh * vc[c], rest_step, beta1, beta2,
                                rest_bc2, eps);
                }
            };
            upd(0, -SH_C1 * y); upd(1, SH_C1 * z); upd(2, -SH_C1 * x);
            if (degrees_to_use < 2) continue;
            upd(3, SH_C2[0] * xy); upd(4, SH_C2[1] * yz); upd(5, SH_C2[2] * (2.f * zz - xx - yy));
            upd(6, SH_C2[3] * xz); upd(7, SH_C2[4] * (xx - yy));
            if (degrees_to_use < 3) continue;
            upd(8, SH_C3[0] * y * (3.f * xx - yy)); upd(9, SH_C3[1] * xy * z);
            upd(10, SH_C3[2] * y * (4.f * zz - xx - yy)); upd(11, SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy));
            upd(12, SH_C3[4] * x * (4.f * zz - xx - yy)); upd(13, SH_C3[5] * z * (xx - yy));
            upd(14, SH_C3[6] * x * (xx - 3.f * yy));
        }
    });
}

// fused_adam_kernel
void adam_dense(float *params, const float *grads, float *ea, float *eas, size_t n, float step_size,
                float beta1, float beta2, float bc2_sqrt, float eps) {
    parallel_for(n, 16384, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; i++) adam_update(params[i], ea[i], eas[i], grads[i], step_size, beta1, beta2, bc2_sqrt, eps);
    });
}

}  // namespace

// ============================================================================
// bindings.h
// ============================================================================

void cleanup_msplat_metal() { g_frame = Frame{}; }

void *msplat_device() { return nullptr; }

MTensor gpu_zeros(std::vector<int64_t> shape, DType dtype) {
    return MTensor(MTensor::HostDevice{}, std::move(shape), dtype);  // zero-filled
}

MTensor gpu_empty(std::vector<int64_t> shape, DType dtype) {
    return MTensor(MTensor::HostDevice{}, std::move(shape), dtype);
}

// Work runs synchronously on the CPU: nothing to commit or wait for.
void msplat_commit() {}
void msplat_gpu_sync() {}

bool msplat_begin_gpu_capture(const char *) {
    std::fprintf(stderr, "msplat: GPU capture needs the Metal backend\n");
    return false;
}
void msplat_end_gpu_capture() {}
void msplat_enable_gpu_timing(bool) {}
void msplat_drain_gpu_times(std::vector<double> &out) { out.clear(); }
void msplat_drain_stage_times(std::vector<double> stage_times[], int max_stages, int &n_stages, const char **) {
    for (int i = 0; i < max_stages; i++) stage_times[i].clear();
    n_stages = 0;
}

MTensor msplat_render(
    int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
    MTensor &quats, MTensor &viewmat, MTensor &projmat,
    float fx, float fy, float cx, float cy,
    unsigned img_height, unsigned img_width,
    const std::tuple<int, int, int> tile_bounds, float clip_thresh,
    unsigned degree, unsigned degrees_to_use, float cam_pos[3],
    MTensor &features_dc, MTensor &features_rest,
    MTensor &opacities, MTensor &background,
    const PriorStep *prior) {
    forward(g_frame, num_points, means3d, scales, glob_scale, quats, viewmat, projmat, fx, fy, cx, cy,
            img_height, img_width, tile_bounds, clip_thresh, degree, degrees_to_use, cam_pos,
            features_dc, features_rest, opacities, background, prior);
    return g_frame.out_img;
}

void msplat_render_aux_outputs(MTensor &depth, MTensor &final_T) {
    depth = g_frame.out_depth;
    final_T = g_frame.final_Ts;
}

std::tuple<MTensor, float> msplat_train_step(
    int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
    MTensor &quats, MTensor &viewmat, MTensor &projmat,
    float fx, float fy, float cx, float cy,
    unsigned img_height, unsigned img_width,
    const std::tuple<int, int, int> tile_bounds, float clip_thresh,
    unsigned degree, unsigned degrees_to_use, float cam_pos[3],
    MTensor &features_dc, MTensor &features_rest,
    MTensor &opacities, MTensor &background,
    MTensor &gt, MTensor &window2d, float ssim_weight,
    float loss_inv_n, int features_rest_bases,
    int num_adam_groups,
    MTensor adam_params[], MTensor adam_exp_avg[], MTensor adam_exp_avg_sq[],
    float adam_step_sizes[], float adam_bc2_sqrts[],
    float adam_beta1, float adam_beta2, float adam_eps,
    MTensor &vis_counts, MTensor &xys_grad_norm, MTensor &max_2d_size,
    float inv_max_dim,
    const PriorStep *prior) {
    (void)window2d;
    (void)features_rest_bases;
    Frame &f = g_frame;
    const bool aux = prior && prior->aux;
    const bool sky = aux && prior->sky && prior->sky_tex && prior->sky_grad;
    const bool exposure = prior && prior->exposure && prior->expo_params;
    const bool scale_cap = prior && prior->log_max_scale_ratio > 0.f;
    const uint32_t W = img_width, H = img_height;
    const size_t npix = (size_t)W * H;

    // Forward
    forward(f, num_points, means3d, scales, glob_scale, quats, viewmat, projmat, fx, fy, cx, cy,
            img_height, img_width, tile_bounds, clip_thresh, degree, degrees_to_use, cam_pos,
            features_dc, features_rest, opacities, background, prior);
    if (aux && prior->loss_terms) prior->loss_terms->zero();
    if (sky) prior->sky_grad->zero();

    // Photometric loss (on the exposure-transformed render when exposure is on)
    ensure_image(f.v_rendered, H, W, 3);
    const float *loss_in = f.out_img.data<float>();
    float *loss_grad = f.v_rendered.data<float>();
    if (exposure) {
        ensure_image(f.out_adj, H, W, 3);
        ensure_image(f.v_adj, H, W, 3);
        exposure_apply(f.out_img.data<float>(), prior->expo_params->data<float>() + 12 * std::max(0, prior->cam_index),
                       npix, f.out_adj.data<float>());
        loss_in = f.out_adj.data<float>();
        loss_grad = f.v_adj.data<float>();
    }
    ensure_size(f.hbuf, npix * 15);
    ensure_size(f.deriv, npix * 15);
    const float *gt_img = gt.data<float>();
    ssim_h_fwd(loss_in, gt_img, W, H, f.hbuf.data());
    double loss_sum = ssim_fused_v_fwd_h_bwd(loss_in, gt_img, f.hbuf.data(), W, H, ssim_weight, f.deriv.data());
    ssim_v_bwd(loss_in, gt_img, f.deriv.data(), W, H, ssim_weight, loss_inv_n, loss_grad);

    // Prior losses (mask loss_grad), exposure backward (loss_grad → v_rendered), sky
    SkyParams sky_params = make_sky_params(sky ? prior : nullptr, fx, fy, cx, cy);
    if (aux) prior_loss(f, *prior, loss_grad);
    if (exposure)
        exposure_backward_adam(f.out_img.data<float>(), *prior, npix, f.v_adj.data<float>(),
                               f.v_rendered.data<float>(), adam_beta1, adam_beta2, adam_eps);
    if (sky)
        sky_backward_adam(f, *prior, sky_params, viewmat.data<float>(), f.v_rendered.data<float>(),
                          adam_beta1, adam_beta2, adam_eps);

    // Backward rasterization and per-gaussian reduction
    rasterize_backward(f, background.data<float>(), f.v_rendered.data<float>());
    reduce_isect_grads(f, num_points);

    // Projection + SH backward with fused SH Adam, then Adam on the other groups
    if (num_adam_groups >= 5) {
        project_and_sh_backward_adam(
            f, num_points, means3d.data<float>(), scales.data<float>(), glob_scale, quats.data<float>(),
            viewmat.data<float>(), projmat.data<float>(), fx, fy, degree, degrees_to_use, cam_pos,
            adam_params[3].data<float>(), adam_params[4].data<float>(),
            adam_exp_avg[3].data<float>(), adam_exp_avg_sq[3].data<float>(),
            adam_exp_avg[4].data<float>(), adam_exp_avg_sq[4].data<float>(),
            adam_step_sizes[3], adam_bc2_sqrts[3], adam_step_sizes[4], adam_bc2_sqrts[4],
            adam_beta1, adam_beta2, adam_eps);
        const float *grads[6] = {f.v_mean3d.data(), f.v_scale.data(), f.v_quat.data(), nullptr, nullptr,
                                 f.v_opacity.data()};
        for (int g = 0; g < num_adam_groups && g < 6; ++g) {
            if (g == 3 || g == 4) continue;
            size_t n = (size_t)adam_params[g].numel();
            if (n == 0) continue;
            adam_dense(adam_params[g].data<float>(), grads[g], adam_exp_avg[g].data<float>(),
                       adam_exp_avg_sq[g].data<float>(), n, adam_step_sizes[g], adam_beta1, adam_beta2,
                       adam_bc2_sqrts[g], adam_eps);
        }
    }

    // Needle cap (scale_ratio_cap_kernel)
    if (scale_cap) {
        float *s = scales.data<float>();
        const float log_max = prior->log_max_scale_ratio;
        parallel_for((size_t)num_points, 16384, [&](size_t b, size_t e) {
            for (size_t i = b; i < e; i++) {
                float a = s[3 * i], bb = s[3 * i + 1], c = s[3 * i + 2];
                float mx = std::max(a, std::max(bb, c)), mn = std::min(a, std::min(bb, c));
                float cap = (a + bb + c - mx - mn) + log_max;
                if (mx <= cap) continue;
                if (a == mx) s[3 * i] = cap;
                else if (bb == mx) s[3 * i + 1] = cap;
                else s[3 * i + 2] = cap;
            }
        });
    }

    // Densification statistics (accumulate_grad_stats_kernel)
    {
        const int32_t *radii = f.radii.data<int32_t>();
        float *vc = vis_counts.data<float>(), *gn = xys_grad_norm.data<float>(), *ms = max_2d_size.data<float>();
        parallel_for((size_t)num_points, 16384, [&](size_t b, size_t e) {
            for (size_t i = b; i < e; i++) {
                if (radii[i] <= 0) continue;
                vc[i] += 1.0f;
                float gx = f.v_xy[2 * i], gy = f.v_xy[2 * i + 1];
                gn[i] += std::sqrt(gx * gx + gy * gy);
                ms[i] = std::max(ms[i], (float)radii[i] * inv_max_dim);
            }
        });
    }

    return std::make_tuple(f.radii, (float)(loss_sum / (double)npix));
}

// GPU-native densification on the CPU: classify → split → dup → cull → compact
int msplat_densify(
    int N, int buf_capacity,
    float grad_thresh, float size_thresh, float screen_thresh, int check_screen,
    float cull_alpha_thresh, float cull_scale_thresh, float cull_screen_size, int check_huge,
    MTensor &xys_grad_norm, MTensor &vis_counts, MTensor &max_2d_size,
    float half_max_dim,
    MTensor &means_buf, MTensor &scales_buf, MTensor &quats_buf,
    MTensor &featuresDc_buf, MTensor &featuresRest_buf, MTensor &opacities_buf,
    int fr_stride,
    MTensor adam_exp_avg_buf[], MTensor adam_exp_avg_sq_buf[],
    MTensor &split_flag, MTensor &dup_flag,
    MTensor &split_prefix, MTensor &dup_prefix,
    MTensor &keep_flag, MTensor &keep_prefix,
    MTensor &block_totals, MTensor &compact_scratch,
    uint32_t seed) {
    (void)split_flag; (void)dup_flag; (void)split_prefix; (void)dup_prefix;
    (void)keep_flag; (void)keep_prefix; (void)block_totals; (void)compact_scratch;
    if (3 * N > buf_capacity) throw std::runtime_error("msplat_densify: 3*N exceeds buf_capacity");
    const float log_size_fac = std::log(1.6f);
    const float *gn = xys_grad_norm.data<float>(), *vc = vis_counts.data<float>(), *ms = max_2d_size.data<float>();
    float *means = means_buf.data<float>(), *scales = scales_buf.data<float>(), *quats = quats_buf.data<float>();
    float *fdc = featuresDc_buf.data<float>(), *frest = featuresRest_buf.data<float>();
    float *opac = opacities_buf.data<float>();

    // Classify (densify_classify_kernel)
    std::vector<uint8_t> split(N), dup(N);
    parallel_for((size_t)N, 16384, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; i++) {
            split[i] = dup[i] = 0;
            if (vc[i] <= 0.0f) continue;
            bool high_grad = (gn[i] / vc[i]) * half_max_dim > grad_thresh;
            float max_scale = std::max(std::max(std::exp(scales[3 * i]), std::exp(scales[3 * i + 1])),
                                       std::exp(scales[3 * i + 2]));
            bool is_large = max_scale > size_thresh;
            bool do_split = is_large;
            if (check_screen && ms[i] > screen_thresh) do_split = true;
            split[i] = (do_split && high_grad) ? 1 : 0;
            dup[i] = (!is_large && high_grad) ? 1 : 0;
        }
    });
    std::vector<int32_t> split_ord(N), dup_ord(N);
    int n_split = 0, n_dup = 0;
    for (int i = 0; i < N; i++) {
        split_ord[i] = split[i] ? n_split++ : -1;
        dup_ord[i] = dup[i] ? n_dup++ : -1;
    }

    const int strides[6] = {3, 3, 4, 3, fr_stride, 1};
    auto zero_state = [&](int child) {
        for (int g = 0; g < 6; g++) {
            std::fill(adam_exp_avg_buf[g].data<float>() + (size_t)child * strides[g],
                      adam_exp_avg_buf[g].data<float>() + (size_t)(child + 1) * strides[g], 0.f);
            std::fill(adam_exp_avg_sq_buf[g].data<float>() + (size_t)child * strides[g],
                      adam_exp_avg_sq_buf[g].data<float>() + (size_t)(child + 1) * strides[g], 0.f);
        }
    };

    // Split children (densify_append_split_kernel)
    parallel_for((size_t)N, 4096, [&](size_t b, size_t e) {
        for (size_t idx = b; idx < e; idx++) {
            if (!split[idx]) continue;
            const int ord = split_ord[idx];
            float qw = quats[idx * 4], qx = quats[idx * 4 + 1], qy = quats[idx * 4 + 2], qz = quats[idx * 4 + 3];
            float qlen = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
            qw /= qlen; qx /= qlen; qy /= qlen; qz /= qlen;
            float sx = std::exp(scales[idx * 3]), sy = std::exp(scales[idx * 3 + 1]), sz = std::exp(scales[idx * 3 + 2]);
            for (int k = 0; k < 2; k++) {
                const int child = N + 2 * ord + k;
                float3 n = rand_normal3(seed, (uint32_t)(ord * 2 + k));
                float r0 = n.x * sx, r1 = n.y * sy, r2 = n.z * sz;
                float v0 = (1 - 2 * (qy * qy + qz * qz)) * r0 + 2 * (qx * qy - qw * qz) * r1 + 2 * (qx * qz + qw * qy) * r2;
                float v1 = 2 * (qx * qy + qw * qz) * r0 + (1 - 2 * (qx * qx + qz * qz)) * r1 + 2 * (qy * qz - qw * qx) * r2;
                float v2 = 2 * (qx * qz - qw * qy) * r0 + 2 * (qy * qz + qw * qx) * r1 + (1 - 2 * (qx * qx + qy * qy)) * r2;
                means[child * 3] = means[idx * 3] + v0;
                means[child * 3 + 1] = means[idx * 3 + 1] + v1;
                means[child * 3 + 2] = means[idx * 3 + 2] + v2;
                for (int j = 0; j < 3; j++) scales[child * 3 + j] = scales[idx * 3 + j] - log_size_fac;
                for (int j = 0; j < 4; j++) quats[child * 4 + j] = quats[idx * 4 + j];
                for (int j = 0; j < 3; j++) fdc[child * 3 + j] = fdc[idx * 3 + j];
                for (int j = 0; j < fr_stride; j++) frest[(size_t)child * fr_stride + j] = frest[idx * fr_stride + j];
                opac[child] = opac[idx];
                zero_state(child);
            }
            for (int j = 0; j < 3; j++) scales[idx * 3 + j] -= log_size_fac;
        }
    });

    // Duplicates (densify_append_dup_kernel), after the splits
    parallel_for((size_t)N, 4096, [&](size_t b, size_t e) {
        for (size_t idx = b; idx < e; idx++) {
            if (!dup[idx]) continue;
            const int dst = N + 2 * n_split + dup_ord[idx];
            for (int j = 0; j < 3; j++) means[dst * 3 + j] = means[idx * 3 + j];
            for (int j = 0; j < 3; j++) scales[dst * 3 + j] = scales[idx * 3 + j];
            for (int j = 0; j < 4; j++) quats[dst * 4 + j] = quats[idx * 4 + j];
            for (int j = 0; j < 3; j++) fdc[dst * 3 + j] = fdc[idx * 3 + j];
            for (int j = 0; j < fr_stride; j++) frest[(size_t)dst * fr_stride + j] = frest[idx * fr_stride + j];
            opac[dst] = opac[idx];
            zero_state(dst);
        }
    });

    // Cull (densify_cull_classify_kernel) and in-place compaction of all 18 buffers
    const int N_new = N + 2 * n_split + n_dup;
    std::vector<uint8_t> keep(N_new);
    parallel_for((size_t)N_new, 16384, [&](size_t b, size_t e) {
        for (size_t idx = b; idx < e; idx++) {
            float o = 1.0f / (1.0f + std::exp(-opac[idx]));
            bool cull = o < cull_alpha_thresh;
            if ((int)idx < N && split[idx]) cull = true;
            if (check_huge) {
                float max_s = std::max(std::max(std::exp(scales[idx * 3]), std::exp(scales[idx * 3 + 1])),
                                       std::exp(scales[idx * 3 + 2]));
                if (max_s > cull_scale_thresh) cull = true;
                if (check_screen && (int)idx < N && ms[idx] > cull_screen_size) cull = true;
            }
            keep[idx] = cull ? 0 : 1;
        }
    });
    float *bufs[18] = {means, scales, quats, fdc, frest, opac};
    for (int g = 0; g < 6; g++) {
        bufs[6 + g] = adam_exp_avg_buf[g].data<float>();
        bufs[12 + g] = adam_exp_avg_sq_buf[g].data<float>();
    }
    int new_count = 0;
    for (int idx = 0; idx < N_new; idx++) {
        if (!keep[idx]) continue;
        if (new_count != idx)
            for (int bi = 0; bi < 18; bi++) {
                const int st = strides[bi % 6];
                std::memmove(bufs[bi] + (size_t)new_count * st, bufs[bi] + (size_t)idx * st, sizeof(float) * st);
            }
        new_count++;
    }
    return new_count;
}

void msplat_opacity_reset(MTensor &opacities, MTensor &exp_avg, MTensor &exp_avg_sq,
                          int num_points, float reset_logit) {
    float *o = opacities.data<float>(), *ea = exp_avg.data<float>(), *eas = exp_avg_sq.data<float>();
    for (int i = 0; i < num_points; i++) {
        o[i] = std::min(o[i], reset_logit);
        ea[i] = 0.f;
        eas[i] = 0.f;
    }
}

void msplat_copy_buffer(MTensor &dst, const MTensor &src, size_t bytes) {
    if (bytes) std::memcpy(dst.data_ptr(), src.data_ptr(), bytes);
}

void msplat_radix_sort(MTensor &keys, MTensor &vals, uint32_t n, int key_bits) {
    if (n == 0) return;
    uint64_t *k = keys.data<uint64_t>();
    int32_t *v = vals.data<int32_t>();
    const uint64_t mask = key_bits >= 64 ? ~0ull : ((1ull << key_bits) - 1ull);
    std::vector<uint32_t> order(n);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return (k[a] & mask) < (k[b] & mask); });
    std::vector<uint64_t> k2(n);
    std::vector<int32_t> v2(n);
    for (uint32_t i = 0; i < n; i++) { k2[i] = k[order[i]]; v2[i] = v[order[i]]; }
    std::memcpy(k, k2.data(), n * sizeof(uint64_t));
    std::memcpy(v, v2.data(), n * sizeof(int32_t));
}

namespace {

// Exact 3-NN through a k-d tree (median splits, index permutation)
struct KdTree {
    const float *pts;
    std::vector<uint32_t> idx;
    std::vector<uint8_t> axis;  // split axis of node [lo, hi) stored at its median
    void build(uint32_t lo, uint32_t hi) {
        if (hi - lo <= 8) return;
        float mn[3] = {INFINITY, INFINITY, INFINITY}, mx[3] = {-INFINITY, -INFINITY, -INFINITY};
        for (uint32_t i = lo; i < hi; i++)
            for (int a = 0; a < 3; a++) {
                mn[a] = std::min(mn[a], pts[3 * idx[i] + a]);
                mx[a] = std::max(mx[a], pts[3 * idx[i] + a]);
            }
        int ax = 0;
        for (int a = 1; a < 3; a++)
            if (mx[a] - mn[a] > mx[ax] - mn[ax]) ax = a;
        uint32_t mid = (lo + hi) / 2;
        std::nth_element(idx.begin() + lo, idx.begin() + mid, idx.begin() + hi,
                         [&](uint32_t p, uint32_t q) { return pts[3 * p + ax] < pts[3 * q + ax]; });
        axis[mid] = (uint8_t)ax;
        build(lo, mid);
        build(mid + 1, hi);
    }
    static void insert3(float d, float best[3]) {
        if (d >= best[2]) return;
        if (d < best[1]) {
            best[2] = best[1];
            if (d < best[0]) { best[1] = best[0]; best[0] = d; }
            else best[1] = d;
        } else {
            best[2] = d;
        }
    }
    void query(uint32_t lo, uint32_t hi, uint32_t self, const float *q, float best[3]) const {
        if (hi - lo <= 8) {
            for (uint32_t i = lo; i < hi; i++) {
                uint32_t p = idx[i];
                if (p == self) continue;
                float dx = pts[3 * p] - q[0], dy = pts[3 * p + 1] - q[1], dz = pts[3 * p + 2] - q[2];
                insert3(dx * dx + dy * dy + dz * dz, best);
            }
            return;
        }
        uint32_t mid = (lo + hi) / 2;
        uint32_t p = idx[mid];
        int ax = axis[mid];
        if (p != self) {
            float dx = pts[3 * p] - q[0], dy = pts[3 * p + 1] - q[1], dz = pts[3 * p + 2] - q[2];
            insert3(dx * dx + dy * dy + dz * dz, best);
        }
        float diff = q[ax] - pts[3 * p + ax];
        if (diff < 0) {
            query(lo, mid, self, q, best);
            if (diff * diff < best[2]) query(mid + 1, hi, self, q, best);
        } else {
            query(mid + 1, hi, self, q, best);
            if (diff * diff < best[2]) query(lo, mid, self, q, best);
        }
    }
};

}  // namespace

void msplat_knn3_mean_dist(MTensor &points, uint32_t n, MTensor &mean_dist) {
    if (n == 0) return;
    KdTree tree;
    tree.pts = points.data<float>();
    tree.idx.resize(n);
    tree.axis.assign(n, 0);
    std::iota(tree.idx.begin(), tree.idx.end(), 0u);
    tree.build(0, n);
    float *out = mean_dist.data<float>();
    parallel_for(n, 2048, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; i++) {
            float best[3] = {INFINITY, INFINITY, INFINITY};
            tree.query(0, n, (uint32_t)i, tree.pts + 3 * i, best);
            float sum = 0.f, cnt = 0.f;
            for (int k = 0; k < 3; k++)
                if (std::isfinite(best[k])) { sum += std::sqrt(best[k]); cnt += 1.f; }
            out[i] = cnt > 0.f ? sum / cnt : 1.f;
        }
    });
}

void msplat_init_gaussians(MTensor &means, MTensor &rgb, uint32_t n, uint32_t seed, float opacity_logit,
                           MTensor &scales, MTensor &quats, MTensor &features_dc, MTensor &opacities) {
    if (n == 0) return;
    MTensor mean_dist = gpu_empty({(int64_t)n}, DType::Float32);
    msplat_knn3_mean_dist(means, n, mean_dist);
    const float *md = mean_dist.data<float>();
    const uint8_t *c = rgb.data<uint8_t>();
    float *s = scales.data<float>(), *q = quats.data<float>(), *dc = features_dc.data<float>(), *o = opacities.data<float>();
    const float tau = 6.28318530717958647692f;
    parallel_for(n, 16384, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; i++) {
            float ls = std::log(std::max(md[i], 1e-7f));
            s[3 * i] = s[3 * i + 1] = s[3 * i + 2] = ls;
            float u = rand_uniform(seed, 3u * (uint32_t)i), v = rand_uniform(seed, 3u * (uint32_t)i + 1u),
                  w = rand_uniform(seed, 3u * (uint32_t)i + 2u);
            q[4 * i + 0] = std::sqrt(1.f - u) * std::sin(tau * v);
            q[4 * i + 1] = std::sqrt(1.f - u) * std::cos(tau * v);
            q[4 * i + 2] = std::sqrt(u) * std::sin(tau * w);
            q[4 * i + 3] = std::sqrt(u) * std::cos(tau * w);
            for (int k = 0; k < 3; k++) dc[3 * i + k] = ((float)c[3 * i + k] * (1.f / 255.f) - 0.5f) / SH_C0;
            o[i] = opacity_logit;
        }
    });
}

// ── Image pipeline, metrics, display ────────────────────────────────────────

void msplat_resize_area(const MTensor &src, bool src_is_u8, int sw, int sh, MTensor &dst, int dw, int dh) {
    if (dw <= 0 || dh <= 0) return;
    const uint8_t *u8 = src_is_u8 ? src.data<uint8_t>() : nullptr;
    const float *fsrc = src_is_u8 ? nullptr : src.data<float>();
    float *out = dst.data<float>();
    const float scale_x = (float)sw / (float)dw, scale_y = (float)sh / (float)dh;
    parallel_for((size_t)dh, 8, [&](size_t b, size_t e) {
        for (size_t gy = b; gy < e; gy++) {
            const float sy0 = (float)gy * scale_y, sy1 = (float)(gy + 1) * scale_y;
            const int iy0 = (int)sy0, iy1 = std::min((int)std::ceil(sy1), sh);
            for (int gx = 0; gx < dw; gx++) {
                const float sx0 = (float)gx * scale_x, sx1 = (float)(gx + 1) * scale_x;
                const int ix0 = (int)sx0, ix1 = std::min((int)std::ceil(sx1), sw);
                float3 sum(0.f);
                float area = 0.f;
                for (int iy = iy0; iy < iy1; iy++) {
                    float wy = std::min((float)(iy + 1), sy1) - std::max((float)iy, sy0);
                    for (int ix = ix0; ix < ix1; ix++) {
                        float wx = std::min((float)(ix + 1), sx1) - std::max((float)ix, sx0);
                        size_t s = (size_t)iy * sw + ix;
                        float3 c = u8 ? float3((float)u8[4 * s], (float)u8[4 * s + 1], (float)u8[4 * s + 2]) * (1.f / 255.f)
                                      : rd3(fsrc, s);
                        sum += (wx * wy) * c;
                        area += wx * wy;
                    }
                }
                wr3(out, gy * dw + gx, sum / area);
            }
        }
    });
}

void msplat_undistort(const MTensor &src, int sw, int sh, const float intr[4], const float dist[5],
                      int roi_x, int roi_y, MTensor &dst, int dw, int dh) {
    if (dw <= 0 || dh <= 0) return;
    const float *in = src.data<float>();
    float *out = dst.data<float>();
    const float fx = intr[0], fy = intr[1], cx = intr[2], cy = intr[3];
    const float k1 = dist[0], k2 = dist[1], p1 = dist[2], p2 = dist[3], k3 = dist[4];
    parallel_for((size_t)dh, 8, [&](size_t b, size_t e) {
        for (size_t gy = b; gy < e; gy++) {
            for (int gx = 0; gx < dw; gx++) {
                float x = ((float)(gx + roi_x) + 0.5f - cx) / fx;
                float y = ((float)((int)gy + roi_y) + 0.5f - cy) / fy;
                float r2 = x * x + y * y;
                float radial = 1.f + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
                float xd = x * radial + 2.f * p1 * x * y + p2 * (r2 + 2.f * x * x);
                float yd = y * radial + p1 * (r2 + 2.f * y * y) + 2.f * p2 * x * y;
                float sx = xd * fx + cx - 0.5f, sy = yd * fy + cy - 0.5f;
                int x0 = (int)std::floor(sx), y0 = (int)std::floor(sy);
                float ax = sx - std::floor(sx), ay = sy - std::floor(sy);
                int xa = std::clamp(x0, 0, sw - 1), xb = std::clamp(x0 + 1, 0, sw - 1);
                int ya = std::clamp(y0, 0, sh - 1), yb = std::clamp(y0 + 1, 0, sh - 1);
                float3 c00 = rd3(in, (size_t)ya * sw + xa), c10 = rd3(in, (size_t)ya * sw + xb);
                float3 c01 = rd3(in, (size_t)yb * sw + xa), c11 = rd3(in, (size_t)yb * sw + xb);
                float3 top = c00 * (1.f - ax) + c10 * ax, bottom = c01 * (1.f - ax) + c11 * ax;
                wr3(out, gy * dw + gx, top * (1.f - ay) + bottom * ay);
            }
        }
    });
}

void msplat_image_metrics(const MTensor &rendered, const MTensor &gt, int h, int w, double out[3]) {
    out[0] = out[1] = out[2] = 0.0;
    if (h <= 0 || w <= 0) return;
    const float *a = rendered.data<float>(), *b = gt.data<float>();
    std::vector<float> hbuf((size_t)h * w * 15);
    parallel_for((size_t)h, 8, [&](size_t yb, size_t ye) {
        for (size_t y = yb; y < ye; y++)
            for (int x = 0; x < w; x++)
                for (int c = 0; c < 3; c++) {
                    float s[5] = {0, 0, 0, 0, 0};
                    for (int k = 0; k < SSIM_WIN; k++) {
                        int sx = std::clamp(x + k - SSIM_HALF_WIN, 0, w - 1);
                        size_t i = (y * w + sx) * 3 + c;
                        float g = GAUSS_1D[k], xa = a[i], yv = b[i];
                        s[0] += g * xa; s[1] += g * yv; s[2] += g * xa * xa; s[3] += g * yv * yv; s[4] += g * xa * yv;
                    }
                    float *o = hbuf.data() + (y * w + x) * 15 + c * 5;
                    for (int q = 0; q < 5; q++) o[q] = s[q];
                }
    });
    std::vector<double> rows((size_t)h * 3, 0.0);
    parallel_for((size_t)h, 8, [&](size_t yb, size_t ye) {
        for (size_t y = yb; y < ye; y++) {
            double ssim = 0, l1 = 0, sq = 0;
            for (int x = 0; x < w; x++)
                for (int c = 0; c < 3; c++) {
                    float s[5] = {0, 0, 0, 0, 0};
                    for (int k = 0; k < SSIM_WIN; k++) {
                        int sy = std::clamp((int)y + k - SSIM_HALF_WIN, 0, h - 1);
                        const float *o = hbuf.data() + ((size_t)sy * w + x) * 15 + c * 5;
                        for (int q = 0; q < 5; q++) s[q] += GAUSS_1D[k] * o[q];
                    }
                    float m12 = s[0] * s[1], m1sq = s[0] * s[0], m2sq = s[1] * s[1];
                    float num = (2.f * m12 + SSIM_C1) * (2.f * (s[4] - m12) + SSIM_C2);
                    float den = (m1sq + m2sq + SSIM_C1) * ((s[2] - m1sq) + (s[3] - m2sq) + SSIM_C2);
                    ssim += num / den;
                    size_t i = (y * w + x) * 3 + c;
                    float d = a[i] - b[i];
                    l1 += std::fabs(d);
                    sq += (double)d * d;
                }
            rows[3 * y] = ssim; rows[3 * y + 1] = l1; rows[3 * y + 2] = sq;
        }
    });
    double ssim = 0, l1 = 0, sq = 0;
    for (int y = 0; y < h; y++) { ssim += rows[3 * y]; l1 += rows[3 * y + 1]; sq += rows[3 * y + 2]; }
    double count = (double)h * w * 3, mse = sq / count;
    out[0] = mse > 0 ? 10.0 * std::log10(1.0 / mse) : INFINITY;
    out[1] = ssim / count;
    out[2] = l1 / count;
}

void msplat_pack_rgba8(const MTensor &img, uint32_t n, uint8_t *out) {
    const float *in = img.data<float>();
    parallel_for(n, 65536, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; i++) {
            float3 c = saturate(rd3(in, i)) * 255.f;
            out[4 * i] = (uint8_t)c.x;
            out[4 * i + 1] = (uint8_t)c.y;
            out[4 * i + 2] = (uint8_t)c.z;
            out[4 * i + 3] = 255;
        }
    });
}

void msplat_finalize_depth(const MTensor &depth_num, const MTensor &final_T, uint32_t n, float inv_scale,
                           MTensor &depth_out, MTensor &alpha_out) {
    const float *d = depth_num.data<float>(), *t = final_T.data<float>();
    float *od = depth_out.data<float>(), *oa = alpha_out.data<float>();
    for (uint32_t i = 0; i < n; i++) {
        float a = 1.f - t[i];
        oa[i] = a;
        od[i] = a > 1e-4f ? d[i] / a * inv_scale : 0.f;
    }
}
