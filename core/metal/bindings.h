#ifndef MSPLAT_BINDINGS_H
#define MSPLAT_BINDINGS_H

#include <tuple>
#include "metal_tensor.hpp"

// Release all cached GPU tensors (call before exit to prevent GPU memory leak)
void cleanup_msplat_metal();

// Returns the Metal device used by the msplat context (void* in C++, id<MTLDevice> in ObjC++)
#ifdef __OBJC__
id<MTLDevice> msplat_device();
#else
void* msplat_device();
#endif

// GPU tensor allocation (callable from C++ — delegates to Metal device)
MTensor gpu_zeros(std::vector<int64_t> shape, DType dtype);
MTensor gpu_empty(std::vector<int64_t> shape, DType dtype);

// Commit current command buffer (non-blocking)
void msplat_commit();

// Synchronize (commit + wait for completion)
void msplat_gpu_sync();

// Capture GPU work to a .gputrace document for Xcode. Needs METAL_CAPTURE_ENABLED=1
// in the environment at launch. Bracket whole iterations and sync on both edges —
// commitAndContinue means a command buffer would otherwise straddle the boundary.
bool msplat_begin_gpu_capture(const char* path);
void msplat_end_gpu_capture();

// GPU timing — non-invasive, uses completion handlers on committed CBs
void msplat_enable_gpu_timing(bool enable);
// Drains accumulated GPU times (ms per CB) into the provided vector. Thread-safe.
void msplat_drain_gpu_times(std::vector<double>& out);
// Drains per-stage GPU times. stage_times must be an array of N_STAGES vectors.
void msplat_drain_stage_times(std::vector<double> stage_times[], int max_stages, int& n_stages,
                              const char** stage_names);

// Optional extras for prior-guided training (depth/sky priors, learned sky, exposure,
// needle cap). A null PriorStep* — the default — runs the original pipeline unchanged.
struct PriorStep {
    // Rasterize expected depth and alpha with a per-pixel background. Required by the
    // learned sky and by every prior loss; when false the default rasterizer runs and
    // the sky and prior-loss fields below are ignored. Forces the monolithic rasterizer.
    bool aux = false;

    // Learned sky: equirect texture (sky_h, sky_w, 3) sampled by view direction and
    // composited behind the gaussians.
    bool sky = false;
    MTensor *sky_tex = nullptr;
    MTensor *sky_grad = nullptr, *sky_exp_avg = nullptr, *sky_exp_avg_sq = nullptr;  // train only
    int sky_w = 0, sky_h = 0;
    float sky_frame[9] = {};              // up, e1 (azimuth 0), e2 (azimuth 90 deg)
    float sky_step_size = 0.f, sky_bc2_sqrt = 1.f;

    // Per-camera priors at their native resolution (train only).
    MTensor *prior_depth = nullptr;       // (prior_h, prior_w) float, scene units, 0 = invalid
    MTensor *prior_aux = nullptr;         // (prior_h, prior_w, 4) uint8: confidence, sky, keep, -
    int prior_w = 0, prior_h = 0;
    bool has_depth = false, has_sky_mask = false;
    float depth_weight = 0.f, sky_weight = 0.f, fill_weight = 0.f;
    float huber_delta = 0.05f, min_alpha = 0.25f;
    bool mask_photometric = true;
    MTensor *loss_terms = nullptr;        // [4] float: depth, sky, fill, - (zeroed every step)

    // Per-image affine exposure compensation (train only).
    bool exposure = false;
    int cam_index = 0;
    MTensor *expo_params = nullptr, *expo_exp_avg = nullptr, *expo_exp_avg_sq = nullptr;  // (num_cams, 12)
    MTensor *expo_grad = nullptr;         // [12], zeroed by the GPU after each update
    float expo_step_size = 0.f, expo_bc2_sqrt = 1.f, expo_reg = 0.f;

    // Needle cap: largest scale <= exp(log_max_scale_ratio) x median scale (train only).
    float log_max_scale_ratio = 0.f;      // <= 0 disables
};

// Render-only forward pass (no loss computation)
// Returns: out_img (H, W, 3) as MTensor
MTensor msplat_render(
    int num_points, MTensor &means3d, MTensor &scales, float glob_scale,
    MTensor &quats, MTensor &viewmat, MTensor &projmat,
    float fx, float fy, float cx, float cy,
    unsigned img_height, unsigned img_width,
    const std::tuple<int, int, int> tile_bounds, float clip_thresh,
    unsigned degree, unsigned degrees_to_use, float cam_pos[3],
    MTensor &features_dc, MTensor &features_rest,
    MTensor &opacities, MTensor &background,
    const PriorStep *prior = nullptr
);

// After msplat_render with prior->aux set: the expected-depth numerator sum_i w_i z_i
// (H, W) and the final transmittance (H, W), so alpha = 1 - final_T and expected
// depth = depth / alpha. Aliases of cached buffers — sync, then copy before the next call.
void msplat_render_aux_outputs(MTensor &depth, MTensor &final_T);

// Fused forward + backward + Adam + grad_stats in one encoder
// Returns: (radii [N], loss_value float)
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
    const PriorStep *prior = nullptr
);

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
    uint32_t seed  // split offsets are drawn in-kernel from this seed
);

// Opacity reset on the GPU: opacity logits clamped to <= reset_logit, Adam moments
// of the opacity group cleared. Encoded after the in-flight step; no host sync.
void msplat_opacity_reset(MTensor &opacities, MTensor &exp_avg, MTensor &exp_avg_sq,
                          int num_points, float reset_logit);

// GPU copy of the first `bytes` of src into dst, ordered after the in-flight work.
void msplat_copy_buffer(MTensor &dst, const MTensor &src, size_t bytes);

// Stable LSD radix sort of (keys: Int64 [n], vals: Int32 [n]) by the low key_bits bits
// of the keys, in place on the GPU. Encoded into the in-flight command buffer.
void msplat_radix_sort(MTensor &keys, MTensor &vals, uint32_t n, int key_bits);

// Exact mean distance from each of n points (n, 3) to its 3 nearest other points,
// written to mean_dist (n). Morton-ordered boxes with distance culling (simple-knn).
void msplat_knn3_mean_dist(MTensor &points, uint32_t n, MTensor &mean_dist);

// Gaussian initialization from points (n, 3) and colors (n, 3 uint8), entirely on the
// GPU: 3-NN isotropic log-scales, uniformly random rotations, DC SH, constant opacity.
void msplat_init_gaussians(MTensor &means, MTensor &rgb, uint32_t n, uint32_t seed, float opacity_logit,
                           MTensor &scales, MTensor &quats, MTensor &features_dc, MTensor &opacities);

#endif
