// Gradient check of the training step against central finite differences, through the
// backend interface (core/metal/bindings.h), so it covers whichever backend msplat_core
// was built with: Metal on macOS, the CPU port elsewhere.
//
// The scenes are smooth on purpose: a few gaussians wider than the image, so neither
// the alpha cutoff nor the tile bounds cross a pixel while a parameter moves, and the
// finite differences see the same function the backward pass differentiates. The
// camera is rotated and translated, and the image sizes are not tile multiples.
//
// msplat_train_step does not return gradients; they are read back through Adam. With
// zero moments, eps = 1e6 and step = K * eps / (1 - beta1), the first update is
// K * g / (1 + 0.03 |g| / eps), i.e. K * g. The loss is recomputed here in double
// precision from the rendered image and the aux outputs.
//
// Exits 0 when every checked gradient is within 5% of its finite difference, plus 0.1%
// of the run's largest gradient (float32 rendering noise in the finite differences).
#include "bindings.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

std::mt19937 rng(1234);
float U(float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); }

const double GAUSS[11] = {0.0010283801, 0.0075987581, 0.0360007721, 0.1093606895, 0.2130055377, 0.2660117249,
                          0.2130055377, 0.1093606895, 0.0360007721, 0.0075987581, 0.0010283801};

struct Options {
    std::string name;
    float ssim_weight = 0.2f;
    bool view_dependent = false;  // nonzero SH rest coefficients (means then not checked)
    bool priors = false;          // aux rasterizer, depth/sky/fill losses, learned sky, exposure
};

struct Scene {
    static constexpr int N = 6, W = 66, H = 46, BASES = 4;  // SH degree 1
    float fx = 60, fy = 58, cx = 33.4f, cy = 22.8f;
    float R[9], t[3], cam_pos[3];
    MTensor means, scales, quats, fdc, frest, opac, bg, viewmat, projmat, gt, window;
    Options opt;
    PriorStep prior;
    MTensor sky_tex, sky_grad, sky_ea, sky_eas, pdepth, paux, terms, expo, expo_ea, expo_eas;

    explicit Scene(const Options &o) : opt(o) {
        rng.seed(1234);  // every run sees the same scene and differs only in its loss terms
        // camera: small yaw/pitch/roll, translated; camera = R world + t
        const float a = 0.2f, b = -0.15f, c = 0.1f;
        const float Rz[9] = {std::cos(c), -std::sin(c), 0, std::sin(c), std::cos(c), 0, 0, 0, 1};
        const float Ry[9] = {std::cos(a), 0, std::sin(a), 0, 1, 0, -std::sin(a), 0, std::cos(a)};
        const float Rx[9] = {1, 0, 0, 0, std::cos(b), -std::sin(b), 0, std::sin(b), std::cos(b)};
        float T1[9] = {};
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++) T1[3 * i + j] += Rz[3 * i + k] * Ry[3 * k + j];
        std::fill(R, R + 9, 0.f);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (int k = 0; k < 3; k++) R[3 * i + j] += T1[3 * i + k] * Rx[3 * k + j];
        t[0] = 0.3f; t[1] = -0.2f; t[2] = 0.5f;
        for (int k = 0; k < 3; k++) cam_pos[k] = -(R[k] * t[0] + R[3 + k] * t[1] + R[6 + k] * t[2]);

        means = gpu_zeros({N, 3}, DType::Float32);
        scales = gpu_zeros({N, 3}, DType::Float32);
        quats = gpu_zeros({N, 4}, DType::Float32);
        fdc = gpu_zeros({N, 3}, DType::Float32);
        frest = gpu_zeros({N, BASES - 1, 3}, DType::Float32);
        opac = gpu_zeros({N, 1}, DType::Float32);
        for (int i = 0; i < N; i++) {
            // off-axis (the perspective-divide gradient grows with the distance from the
            // principal point) and depth-ordered
            const float z = 2.5f + 0.45f * i;
            const float pc[3] = {(i % 2 ? 1.f : -1.f) * U(0.15f, 0.3f) * z, (i % 3 ? 1.f : -1.f) * U(0.1f, 0.22f) * z, z};
            for (int k = 0; k < 3; k++) {
                float v = 0;
                for (int j = 0; j < 3; j++) v += R[3 * j + k] * (pc[j] - t[j]);
                means.data<float>()[3 * i + k] = v;
            }
            for (int k = 0; k < 3; k++) scales.data<float>()[3 * i + k] = std::log(U(32.f, 44.f) * pc[2] / fx);
            float q[4], n = 0;
            for (int k = 0; k < 4; k++) { q[k] = U(-1, 1); n += q[k] * q[k]; }
            for (int k = 0; k < 4; k++) quats.data<float>()[4 * i + k] = 1.3f * q[k] / std::sqrt(n);  // not unit
            for (int k = 0; k < 3; k++) fdc.data<float>()[3 * i + k] = U(-0.6f, 0.6f);
            for (int k = 0; k < (BASES - 1) * 3; k++)
                frest.data<float>()[(BASES - 1) * 3 * i + k] = opt.view_dependent ? U(-0.15f, 0.15f) : 0.f;
            opac.data<float>()[i] = U(-2.f, 0.3f);
        }
        bg = gpu_zeros({3}, DType::Float32);
        bg.data<float>()[0] = 0.2f; bg.data<float>()[1] = 0.3f; bg.data<float>()[2] = 0.45f;

        const float vm[16] = {R[0], R[1], R[2], t[0], R[3], R[4], R[5], t[1], R[6], R[7], R[8], t[2], 0, 0, 0, 1};
        viewmat = gpu_zeros({4, 4}, DType::Float32);
        std::memcpy(viewmat.data_ptr(), vm, sizeof vm);
        // projection as Model::prepareCam builds it
        const float fovX = 2.0f * std::atan(W / (2.0f * fx)), fovY = 2.0f * std::atan(H / (2.0f * fy));
        const float t_p = 0.001f * std::tan(0.5f * fovY), r_p = 0.001f * std::tan(0.5f * fovX);
        const float pm[16] = {0.001f / r_p, 0, 0, 0, 0, 0.001f / t_p, 0, 0, 0, 0, (1000.0f + 0.001f) / (1000.0f - 0.001f),
                              -1000.0f * 0.001f / (1000.0f - 0.001f), 0, 0, 1, 0};
        float pvm[16] = {};
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                for (int k = 0; k < 4; k++) pvm[i * 4 + j] += pm[i * 4 + k] * vm[k * 4 + j];
        projmat = gpu_zeros({4, 4}, DType::Float32);
        std::memcpy(projmat.data_ptr(), pvm, sizeof pvm);

        gt = gpu_zeros({H, W, 3}, DType::Float32);
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                for (int c3 = 0; c3 < 3; c3++)
                    gt.data<float>()[(y * W + x) * 3 + c3] =
                        0.5f + 0.3f * std::sin(0.11f * x + 0.7f * c3) * std::cos(0.07f * y - 0.3f * c3) + U(-0.1f, 0.1f);
        window = gpu_zeros({11, 11}, DType::Float32);
        if (opt.priors) init_priors();
    }

    void init_priors() {
        prior.aux = true;
        const int pw = W / 2 + 3, ph = H / 2 - 1;  // resampled to the render size
        pdepth = gpu_zeros({ph, pw}, DType::Float32);
        paux = gpu_zeros({ph, pw, 4}, DType::UInt8);
        for (int y = 0; y < ph; y++)
            for (int x = 0; x < pw; x++) {
                float d = 3.2f * (1.f + 0.25f * std::sin(0.3f * x) * std::cos(0.2f * y));
                if ((x * 7 + y * 3) % 23 == 0) d = 0.f;  // invalid samples
                pdepth.data<float>()[y * pw + x] = d;
                uint8_t *a = paux.data<uint8_t>() + 4 * (y * pw + x);
                a[0] = (uint8_t)(60 + (x * 13 + y * 7) % 190);  // confidence
                a[1] = (y < ph / 4) ? 255 : 0;                    // sky
                a[2] = 255;                                       // keep
            }
        prior.prior_depth = &pdepth;
        prior.prior_aux = &paux;
        prior.prior_w = pw;
        prior.prior_h = ph;
        prior.has_depth = prior.has_sky_mask = true;
        prior.depth_weight = 0.5f;
        prior.sky_weight = 0.3f;
        prior.fill_weight = 0.1f;
        terms = gpu_zeros({4}, DType::Float32);
        prior.loss_terms = &terms;

        const int sw = 32, sh = 16;
        sky_tex = gpu_zeros({sh, sw, 3}, DType::Float32);
        for (int i = 0; i < sw * sh * 3; i++) sky_tex.data<float>()[i] = U(0.2f, 0.8f);
        sky_grad = gpu_zeros({sh, sw, 3}, DType::Float32);
        sky_ea = gpu_zeros({sh, sw, 3}, DType::Float32);
        sky_eas = gpu_zeros({sh, sw, 3}, DType::Float32);
        prior.sky = true;
        prior.sky_tex = &sky_tex;
        prior.sky_grad = &sky_grad;
        prior.sky_exp_avg = &sky_ea;
        prior.sky_exp_avg_sq = &sky_eas;
        prior.sky_w = sw;
        prior.sky_h = sh;
        const float frame[9] = {0, -1, 0, 0, 0, 1, 1, 0, 0};  // up, azimuth 0, azimuth 90
        std::memcpy(prior.sky_frame, frame, sizeof frame);

        expo = gpu_zeros({2, 12}, DType::Float32);
        for (int k = 0; k < 24; k++) expo.data<float>()[k] = ((k % 12) % 4 == 0 && k % 12 < 9 ? 1.f : 0.f) + U(-0.05f, 0.05f);
        expo_ea = gpu_zeros({2, 12}, DType::Float32);
        expo_eas = gpu_zeros({2, 12}, DType::Float32);
        prior.exposure = true;
        prior.cam_index = 1;
        prior.expo_params = &expo;
        prior.expo_exp_avg = &expo_ea;
        prior.expo_exp_avg_sq = &expo_eas;
    }

    std::tuple<int, int, int> tiles() const { return std::make_tuple((W + 15) / 16, (H + 15) / 16, 1); }

    // One training step whose Adam update moves every parameter by K * grad.
    void gradient_step(float K) {
        const float eps = 1e6f, b1 = 0.9f, b2 = 0.999f, ss = K * eps / (1.f - b1);
        MTensor params[6] = {means, scales, quats, fdc, frest, opac};
        MTensor ea[6], eas[6];
        float sizes[6], bc2[6];
        for (int g = 0; g < 6; g++) {
            ea[g] = gpu_zeros(params[g].shape(), DType::Float32);
            eas[g] = gpu_zeros(params[g].shape(), DType::Float32);
            sizes[g] = ss;
            bc2[g] = 1.f;
        }
        MTensor vis = gpu_zeros({N}, DType::Float32), gn = gpu_zeros({N}, DType::Float32),
                ms = gpu_zeros({N}, DType::Float32);
        if (opt.priors) {
            prior.sky_step_size = prior.expo_step_size = ss;
            prior.sky_bc2_sqrt = prior.expo_bc2_sqrt = 1.f;
        }
        msplat_train_step(N, means, scales, 1.0f, quats, viewmat, projmat, fx, fy, cx, cy, H, W, tiles(), 0.01f, 1,
                          1, cam_pos, fdc, frest, opac, bg, gt, window, opt.ssim_weight, 1.0f / (float)(H * W * 3),
                          BASES - 1, 6, params, ea, eas, sizes, bc2, b1, b2, eps, vis, gn, ms,
                          1.0f / (float)std::max(H, W), opt.priors ? &prior : nullptr);
        msplat_gpu_sync();
    }

    // The training objective in double: photometric loss (on the exposure-adjusted
    // render) plus the prior terms, from the rendered image and the aux outputs.
    double loss() {
        MTensor img = msplat_render(N, means, scales, 1.0f, quats, viewmat, projmat, fx, fy, cx, cy, H, W, tiles(),
                                    0.01f, 1, 1, cam_pos, fdc, frest, opac, bg, opt.priors ? &prior : nullptr);
        msplat_gpu_sync();
        const int n = W * H;
        std::vector<double> r(n * 3);
        for (int i = 0; i < n * 3; i++) r[i] = img.data<float>()[i];
        if (opt.priors) {
            const float *P = expo.data<float>() + 12 * prior.cam_index;
            for (int i = 0; i < n; i++) {
                const double c0 = r[3 * i], c1 = r[3 * i + 1], c2 = r[3 * i + 2];
                for (int k = 0; k < 3; k++) r[3 * i + k] = P[3 * k] * c0 + P[3 * k + 1] * c1 + P[3 * k + 2] * c2 + P[9 + k];
            }
        }
        double total = 0;
        const double C1 = 0.0001, C2 = 0.0009, w = opt.ssim_weight;
        for (int c = 0; c < 3; c++) {
            std::vector<double> x(n), y(n), xx(n), yy(n), xy(n);
            for (int i = 0; i < n; i++) {
                x[i] = gt.data<float>()[3 * i + c];
                y[i] = r[3 * i + c];
                xx[i] = x[i] * x[i]; yy[i] = y[i] * y[i]; xy[i] = x[i] * y[i];
            }
            auto mx = blur(x), my = blur(y), sxx = blur(xx), syy = blur(yy), sxy = blur(xy);
            for (int i = 0; i < n; i++) {
                const double vx = sxx[i] - mx[i] * mx[i], vy = syy[i] - my[i] * my[i], cv = sxy[i] - mx[i] * my[i];
                const double ssim = ((2 * mx[i] * my[i] + C1) * (2 * cv + C2)) /
                                    ((mx[i] * mx[i] + my[i] * my[i] + C1) * (vx + vy + C2));
                total += w * (1.0 - ssim) / 3.0 + (1 - w) * std::fabs(x[i] - y[i]) / 3.0;
            }
        }
        total /= n;
        if (opt.priors) total += prior_terms() / n;
        return total;
    }

    // Zero-padded separable 11-tap Gaussian blur, as the training SSIM
    static std::vector<double> blur(const std::vector<double> &a) {
        std::vector<double> tmp(a.size(), 0.0), out(a.size(), 0.0);
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                for (int k = 0; k < 11; k++) {
                    const int xx = x + k - 5;
                    if (xx >= 0 && xx < W) tmp[y * W + x] += GAUSS[k] * a[y * W + xx];
                }
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                for (int k = 0; k < 11; k++) {
                    const int yy = y + k - 5;
                    if (yy >= 0 && yy < H) out[y * W + x] += GAUSS[k] * tmp[yy * W + x];
                }
        return out;
    }

    // Edge-aware prior depth lookup (sample_prior_depth in the backends)
    static float sample_depth(const float *d, float x, float y, uint32_t w, uint32_t h) {
        x = std::min(std::max(x, 0.f), (float)w - 1.f);
        y = std::min(std::max(y, 0.f), (float)h - 1.f);
        const uint32_t x0 = (uint32_t)x, y0 = (uint32_t)y, x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
        const float fx_ = x - (float)x0, fy_ = y - (float)y0;
        const float d00 = d[y0 * w + x0], d01 = d[y0 * w + x1], d10 = d[y1 * w + x0], d11 = d[y1 * w + x1];
        const float lo = std::min(std::min(d00, d01), std::min(d10, d11));
        const float hi = std::max(std::max(d00, d01), std::max(d10, d11));
        if (lo <= 0.f || hi > 1.05f * lo) return d[((fy_ < 0.5f) ? y0 : y1) * w + ((fx_ < 0.5f) ? x0 : x1)];
        const float top = d00 + (d01 - d00) * fx_, bottom = d10 + (d11 - d10) * fx_;
        return top + (bottom - top) * fy_;
    }

    // Sum over pixels of the depth, sky and fill terms
    double prior_terms() {
        MTensor depth, final_T;
        msplat_render_aux_outputs(depth, final_T);
        const uint32_t pw = prior.prior_w, ph = prior.prior_h;
        double sum = 0;
        for (int gy = 0; gy < H; gy++)
            for (int gx = 0; gx < W; gx++) {
                const int pix = gy * W + gx;
                const float sx = ((float)gx + 0.5f) * (float)pw / (float)W, sy = ((float)gy + 0.5f) * (float)ph / (float)H;
                const uint32_t q = std::min((uint32_t)sy, ph - 1) * pw + std::min((uint32_t)sx, pw - 1);
                const uint8_t *a = paux.data<uint8_t>() + 4 * q;
                const double A = 1.0 - final_T.data<float>()[pix];
                if (a[1] >= 128) {
                    sum += prior.sky_weight * A;
                    continue;
                }
                sum += prior.fill_weight * (1.0 - A);
                const double D = depth.data<float>()[pix], conf = a[0] / 255.0;
                if (conf <= 0 || A <= prior.min_alpha || D <= 0) continue;
                const float dp = sample_depth(pdepth.data<float>(), sx - 0.5f, sy - 0.5f, pw, ph);
                if (dp <= 0) continue;
                const double r = std::log(D / A) - std::log((double)dp), ar = std::fabs(r), hd = prior.huber_delta;
                sum += prior.depth_weight * conf * (ar < hd ? 0.5 * r * r / hd : ar - 0.5 * hd);
            }
        return sum;
    }
};

struct Param {
    std::string name;
    float *ptr;
    bool is_mean;
};

// Returns the number of mismatches
int check(const Options &opt) {
    Scene s(opt);
    std::vector<Param> ps;
    for (int i = 0; i < Scene::N; i++) {
        const std::string g = std::to_string(i);
        for (int k = 0; k < 3; k++) ps.push_back({"mean" + g + "." + std::to_string(k), s.means.data<float>() + 3 * i + k, true});
        for (int k = 0; k < 3; k++) ps.push_back({"scale" + g + "." + std::to_string(k), s.scales.data<float>() + 3 * i + k, false});
        for (int k = 0; k < 4; k++) ps.push_back({"quat" + g + "." + std::to_string(k), s.quats.data<float>() + 4 * i + k, false});
        for (int k = 0; k < 3; k++) ps.push_back({"dc" + g + "." + std::to_string(k), s.fdc.data<float>() + 3 * i + k, false});
        for (int k = 0; k < 9; k++) ps.push_back({"rest" + g + "." + std::to_string(k), s.frest.data<float>() + 9 * i + k, false});
        ps.push_back({"opacity" + g, s.opac.data<float>() + i, false});
    }
    if (opt.priors) {
        for (int k = 0; k < 12; k++) ps.push_back({"exposure" + std::to_string(k), s.expo.data<float>() + 12 + k, false});
        for (int k = 0; k < 32 * 16 * 3; k += 7) ps.push_back({"sky" + std::to_string(k), s.sky_tex.data<float>() + k, false});
    }

    // Analytic gradients: one step that moves each parameter by K * grad, then restore
    std::vector<MTensor> all = {s.means, s.scales, s.quats, s.fdc, s.frest, s.opac};
    if (opt.priors) {
        all.push_back(s.sky_tex);
        all.push_back(s.expo);
    }
    std::vector<std::vector<float>> saved;
    for (auto &t : all) saved.emplace_back(t.data<float>(), t.data<float>() + t.numel());
    std::vector<float> before(ps.size()), analytic(ps.size());
    for (size_t i = 0; i < ps.size(); i++) before[i] = *ps[i].ptr;
    const float K = 1000.f;
    s.gradient_step(K);
    for (size_t i = 0; i < ps.size(); i++) analytic[i] = (before[i] - *ps[i].ptr) / K;
    for (size_t t = 0; t < all.size(); t++) std::memcpy(all[t].data_ptr(), saved[t].data(), saved[t].size() * sizeof(float));

    const float h = 4e-3f;
    std::vector<double> fds(ps.size(), 0.0);
    double gmax = 0;
    for (size_t i = 0; i < ps.size(); i++) {
        if (ps[i].is_mean && opt.view_dependent) continue;  // SH view direction is not differentiated
        const float v = *ps[i].ptr;
        *ps[i].ptr = v + h;
        const double lp = s.loss();
        *ps[i].ptr = v - h;
        const double lm = s.loss();
        *ps[i].ptr = v;
        fds[i] = (lp - lm) / (2.0 * h);
        gmax = std::max(gmax, std::fabs((double)analytic[i]));
    }
    int bad = 0, checked = 0;
    double dot_af = 0, aa = 0, ff = 0;
    for (size_t i = 0; i < ps.size(); i++) {
        if (ps[i].is_mean && opt.view_dependent) continue;
        const double fd = fds[i], a = analytic[i];
        const bool ok = std::fabs(a - fd) <= 0.05 * std::max(std::fabs(a), std::fabs(fd)) + 1e-3 * gmax;
        checked++;
        dot_af += a * fd; aa += a * a; ff += fd * fd;
        if (!ok) {
            bad++;
            std::printf("  %-14s analytic % .6e  finite difference % .6e\n", ps[i].name.c_str(), a, fd);
        }
    }
    std::printf("%-26s %3d/%3d gradients match, cosine %.6f\n", opt.name.c_str(), checked - bad, checked,
                dot_af / std::sqrt(aa * ff + 1e-300));
    return bad;
}

}  // namespace

int main() {
    std::vector<Options> runs = {
        {"L1", 0.f, false, false},
        {"SSIM", 1.f, false, false},
        {"L1 + SSIM (0.2)", 0.2f, false, false},
        {"view-dependent color", 0.2f, true, false},
        {"priors, sky, exposure", 0.2f, false, true},
    };
    int bad = 0;
    for (auto &o : runs) bad += check(o);
    cleanup_msplat_metal();
    std::printf(bad ? "FAILED: %d mismatches\n" : "OK\n", bad);
    return bad ? 1 : 0;
}
