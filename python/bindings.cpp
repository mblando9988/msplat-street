#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/function.h>

#include "model.hpp"
#include "input_data.hpp"
#include "msplat.hpp"
#include "ssim.hpp"
#include "priors.hpp"

#include <filesystem>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <random>

namespace nb = nanobind;
using namespace nb::literals;
namespace fs = std::filesystem;

// ── TrainingConfig ──────────────────────────────────────────────────────────

struct TrainingConfig {
    int iterations = 30000;
    int sh_degree = 3;
    int sh_degree_interval = 1000;
    float ssim_weight = 0.2f;
    int num_downscales = 2;
    int resolution_schedule = 3000;
    int refine_every = 100;
    int warmup_length = 500;
    int reset_alpha_every = 30;
    float densify_grad_thresh = 0.0002f;
    float densify_size_thresh = 0.01f;
    int stop_screen_size_at = 4000;
    float split_screen_size = 0.05f;
    bool keep_crs = false;
    float downscale_factor = 1.0f;
    std::string output = "splat.ply";
    int save_every = -1;
    // Magenta default — high contrast against typical scenes, makes
    // under-reconstructed regions obvious during training.
    std::vector<float> bg_color = {0.6130f, 0.0101f, 0.3984f};

    // Prior-guided training (msplat.street_config() sets these for street captures)
    float depth_weight = 0.0f;
    float depth_weight_final = -1.0f;
    float depth_huber_delta = 0.05f;
    float depth_min_alpha = 0.25f;
    float sky_alpha_weight = 0.0f;
    float fill_weight = 0.0f;
    bool use_masks = false;
    bool learn_sky = false;
    int sky_width = 512;
    int sky_height = 128;
    float sky_lr = 0.01f;
    bool exposure_compensation = false;
    float exposure_lr = 5e-3f;
    float exposure_reg = 1e-2f;
    float max_scale_ratio = 0.0f;

    PriorOptions prior_options() const {
        PriorOptions o;
        o.depthWeight = depth_weight;
        o.depthWeightFinal = depth_weight_final;
        o.depthHuberDelta = depth_huber_delta;
        o.depthMinAlpha = depth_min_alpha;
        o.skyAlphaWeight = sky_alpha_weight;
        o.fillWeight = fill_weight;
        o.useMasks = use_masks;
        o.learnSky = learn_sky;
        o.skyWidth = sky_width;
        o.skyHeight = sky_height;
        o.skyLr = sky_lr;
        o.exposure = exposure_compensation;
        o.exposureLr = exposure_lr;
        o.exposureReg = exposure_reg;
        o.maxScaleRatio = max_scale_ratio;
        return o;
    }
};

// ── TrainingStats ───────────────────────────────────────────────────────────

struct TrainingStats {
    int iteration;
    int splat_count;
    float ms_per_step;
};

// ── Dataset ─────────────────────────────────────────────────────────────────

class Dataset {
public:
    InputData data;
    std::vector<Camera> train_cams;
    std::vector<Camera> test_cams;

    Dataset(const std::string &path, float downscale_factor,
            bool eval_mode, int test_every, const std::string &prior_dir)
    {
        data = inputDataFromX(path);
        if (!prior_dir.empty()) attachPriors(data, prior_dir);

        // Load images (parallel)
        for (auto &cam : data.cameras) {
            cam.loadImage(downscale_factor);
        }

        if (eval_mode) {
            auto split = data.splitTrainTest(test_every);
            train_cams = std::get<0>(split);
            test_cams = std::get<1>(split);
        } else {
            auto t = data.getCameras(false);
            train_cams = std::get<0>(t);
        }
    }

    size_t num_train() const { return train_cams.size(); }
    size_t num_test() const { return test_cams.size(); }

    // Train cameras with prior files attached, by kind
    nb::dict prior_counts() const {
        int any = 0, depth = 0, conf = 0, sky = 0, mask = 0;
        for (auto &c : train_cams) {
            any += c.hasPriorFiles();
            depth += !c.priorDepthPath.empty();
            conf += !c.priorConfidencePath.empty();
            sky += !c.priorSkyPath.empty();
            mask += !c.priorMaskPath.empty();
        }
        nb::dict d;
        d["any"] = any; d["depth"] = depth; d["confidence"] = conf; d["sky"] = sky; d["mask"] = mask;
        return d;
    }

    // Get camera-to-world pose (4x4 row-major) as numpy array
    nb::object camera_pose(int index) {
        if (index < 0 || index >= (int)train_cams.size())
            throw std::runtime_error("Camera index out of range");
        float *buf = new float[16];
        memcpy(buf, train_cams[index].camToWorld, 16 * sizeof(float));
        nb::capsule deleter(buf, [](void *p) noexcept { delete[] static_cast<float*>(p); });
        size_t shape[2] = {4, 4};
        return nb::cast(nb::ndarray<nb::numpy, float>(buf, 2, shape, deleter));
    }
};

// ── GaussianTrainer ─────────────────────────────────────────────────────────

class GaussianTrainer {
public:
    std::unique_ptr<Model> model;
    TrainingConfig config;
    Dataset* dataset_ptr = nullptr;  // non-owning reference
    int current_step = 0;

    // Random camera iterator
    std::vector<size_t> cam_indices;
    size_t cam_iter_pos = 0;
    std::mt19937 rng;

    GaussianTrainer(Dataset &dataset, const TrainingConfig &cfg)
        : config(cfg), dataset_ptr(&dataset)
    {
        model = std::make_unique<Model>(
            dataset.data,
            dataset.train_cams.size(),
            cfg.num_downscales, cfg.resolution_schedule,
            cfg.sh_degree, cfg.sh_degree_interval,
            cfg.refine_every, cfg.warmup_length, cfg.reset_alpha_every,
            cfg.densify_grad_thresh, cfg.densify_size_thresh,
            cfg.stop_screen_size_at, cfg.split_screen_size,
            cfg.iterations, cfg.keep_crs,
            cfg.bg_color.data()
        );
        model->configurePriors(cfg.prior_options(), dataset.train_cams);

        cam_indices.resize(dataset.train_cams.size());
        std::iota(cam_indices.begin(), cam_indices.end(), 0);
        rng.seed(42);
        shuffle_cameras();
    }

    void shuffle_cameras() {
        std::shuffle(cam_indices.begin(), cam_indices.end(), rng);
        cam_iter_pos = 0;
    }

    size_t next_camera() {
        if (cam_iter_pos >= cam_indices.size()) shuffle_cameras();
        return cam_indices[cam_iter_pos++];
    }

    TrainingStats step() {
        current_step++;
        size_t cam_idx = next_camera();
        Camera &cam = dataset_ptr->train_cams[cam_idx];

        int ds = model->getDownscaleFactor(current_step);
        MTensor &gt = cam.getGPUImage(ds);

        auto t0 = std::chrono::high_resolution_clock::now();

        model->fullIteration(cam, current_step, gt, config.ssim_weight);
        model->schedulersStep(current_step);
        model->afterTrain(current_step);
        msplat_commit();

        auto t1 = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0f;

        TrainingStats stats;
        stats.iteration = current_step;
        stats.splat_count = model->means.size(0);
        stats.ms_per_step = ms;
        return stats;
    }

    void train(nb::object callback, int callback_every) {
        while (current_step < config.iterations) {
            TrainingStats stats = step();

            if (callback_every > 0 && stats.iteration % callback_every == 0) {
                callback(stats);
            }
        }
    }

    // Evaluate on test cameras
    nb::dict evaluate() {
        if (dataset_ptr->test_cams.empty()) {
            throw std::runtime_error("No test cameras. Load dataset with eval_mode=True.");
        }

        double sum_psnr = 0, sum_ssim = 0, sum_l1 = 0;
        int n = dataset_ptr->test_cams.size();

        for (int i = 0; i < n; i++) {
            Camera &cam = dataset_ptr->test_cams[i];
            MTensor rgb = model->render(cam, config.iterations);
            msplat_gpu_sync();

            MTensor rgb_cpu = rgb.cpu();
            int ds = model->getDownscaleFactor(config.iterations);
            MTensor gt_cpu = cam.getGPUImage(ds).cpu();

            sum_psnr += psnr(rgb_cpu, gt_cpu);
            sum_ssim += ssim_eval(rgb_cpu, gt_cpu);
            sum_l1 += l1_loss(rgb_cpu, gt_cpu);
        }

        nb::dict result;
        result["psnr"] = sum_psnr / n;
        result["ssim"] = sum_ssim / n;
        result["l1"] = sum_l1 / n;
        result["num_test"] = n;
        result["num_gaussians"] = (int)model->means.size(0);
        return result;
    }

    // Render a single camera view → numpy (H, W, 3) float32
    nb::object render(int cam_idx, bool use_test) {
        auto &cams = use_test ? dataset_ptr->test_cams : dataset_ptr->train_cams;
        if (cam_idx < 0 || cam_idx >= (int)cams.size()) {
            throw std::runtime_error("Camera index out of range");
        }

        Camera &cam = cams[cam_idx];
        MTensor rgb = model->render(cam, current_step);
        msplat_gpu_sync();
        MTensor rgb_cpu = rgb.cpu();

        int h = rgb_cpu.size(0);
        int w = rgb_cpu.size(1);

        // Copy to numpy-owned buffer
        float *buf = new float[h * w * 3];
        memcpy(buf, rgb_cpu.data_ptr(), h * w * 3 * sizeof(float));

        nb::capsule deleter(buf, [](void *p) noexcept { delete[] static_cast<float*>(p); });
        size_t shape[3] = {(size_t)h, (size_t)w, 3};
        return nb::cast(nb::ndarray<nb::numpy, float>(buf, 3, shape, deleter));
    }

    // Render from arbitrary pose → numpy (H, W, 3) float32
    nb::object render_from_pose(nb::ndarray<nb::numpy, float> cam_to_world, int ref_cam_idx) {
        if (cam_to_world.size() != 16)
            throw std::runtime_error("cam_to_world must have 16 elements (4x4 matrix)");
        if (ref_cam_idx < 0 || ref_cam_idx >= (int)dataset_ptr->train_cams.size())
            throw std::runtime_error("ref_cam_idx out of range");

        Camera cam = dataset_ptr->train_cams[ref_cam_idx];
        memcpy(cam.camToWorld, cam_to_world.data(), 16 * sizeof(float));
        cam.cachedViewMat = MTensor();
        cam.cachedProjViewMat = MTensor();

        MTensor rgb = model->render(cam, current_step);
        msplat_gpu_sync();
        MTensor rgb_cpu = rgb.cpu();

        int h = rgb_cpu.size(0);
        int w = rgb_cpu.size(1);
        float *buf = new float[h * w * 3];
        memcpy(buf, rgb_cpu.data_ptr(), h * w * 3 * sizeof(float));

        nb::capsule deleter(buf, [](void *p) noexcept { delete[] static_cast<float*>(p); });
        size_t shape[3] = {(size_t)h, (size_t)w, 3};
        return nb::cast(nb::ndarray<nb::numpy, float>(buf, 3, shape, deleter));
    }

    void export_ply(const std::string &path) {
        model->savePly(path, current_step);
    }

    void export_splat(const std::string &path) {
        model->saveSplat(path);
    }

    void save_checkpoint(const std::string &path) {
        model->saveCheckpoint(path, current_step);
    }

    void load_checkpoint(const std::string &path) {
        current_step = model->loadCheckpoint(path);
    }

    int splat_count() const {
        return model->means.size(0);
    }

    // Expected depth (dataset units, 0 where empty) and accumulated alpha → numpy (H, W) each
    nb::tuple render_depth(int cam_idx, bool use_test) {
        auto &cams = use_test ? dataset_ptr->test_cams : dataset_ptr->train_cams;
        if (cam_idx < 0 || cam_idx >= (int)cams.size())
            throw std::runtime_error("Camera index out of range");
        MTensor depth, alpha;
        model->renderDepth(cams[cam_idx], current_step, depth, alpha);
        return nb::make_tuple(to_numpy_2d(depth), to_numpy_2d(alpha));
    }

    nb::dict prior_losses() {
        float l[3];
        model->lastPriorLosses(l);
        nb::dict d;
        d["depth"] = l[0];
        d["sky"] = l[1];
        d["fill"] = l[2];
        return d;
    }

    void export_sky(const std::string &path) {
        model->saveSky(path);
    }

    static nb::object to_numpy_2d(const MTensor &t) {
        size_t h = (size_t)t.size(0), w = (size_t)t.size(1);
        float *buf = new float[h * w];
        memcpy(buf, t.data_ptr(), h * w * sizeof(float));
        nb::capsule deleter(buf, [](void *p) noexcept { delete[] static_cast<float*>(p); });
        size_t shape[2] = {h, w};
        return nb::cast(nb::ndarray<nb::numpy, float>(buf, 2, shape, deleter));
    }
};

// ── Module definition ───────────────────────────────────────────────────────

NB_MODULE(_core, m) {
    m.doc() = "msplat: Metal-accelerated 3D Gaussian Splatting";

    // TrainingConfig
    nb::class_<TrainingConfig>(m, "TrainingConfig")
        .def(nb::init<>())
        .def("__init__", [](TrainingConfig *cfg,
                int iterations, int sh_degree, int sh_degree_interval,
                float ssim_weight, int num_downscales, int resolution_schedule,
                int refine_every, int warmup_length, int reset_alpha_every,
                float densify_grad_thresh, float densify_size_thresh,
                int stop_screen_size_at, float split_screen_size,
                bool keep_crs, float downscale_factor,
                const std::string &output, int save_every,
                std::vector<float> bg_color,
                float depth_weight, float depth_weight_final,
                float depth_huber_delta, float depth_min_alpha,
                float sky_alpha_weight, float fill_weight, bool use_masks,
                bool learn_sky, int sky_width, int sky_height, float sky_lr,
                bool exposure_compensation, float exposure_lr, float exposure_reg,
                float max_scale_ratio) {
            new (cfg) TrainingConfig();
            cfg->iterations = iterations;
            cfg->sh_degree = sh_degree;
            cfg->sh_degree_interval = sh_degree_interval;
            cfg->ssim_weight = ssim_weight;
            cfg->num_downscales = num_downscales;
            cfg->resolution_schedule = resolution_schedule;
            cfg->refine_every = refine_every;
            cfg->warmup_length = warmup_length;
            cfg->reset_alpha_every = reset_alpha_every;
            cfg->densify_grad_thresh = densify_grad_thresh;
            cfg->densify_size_thresh = densify_size_thresh;
            cfg->stop_screen_size_at = stop_screen_size_at;
            cfg->split_screen_size = split_screen_size;
            cfg->keep_crs = keep_crs;
            cfg->downscale_factor = downscale_factor;
            cfg->output = output;
            cfg->save_every = save_every;
            if (bg_color.size() != 3)
                throw std::invalid_argument("bg_color must have exactly 3 elements [R, G, B]");
            cfg->bg_color = bg_color;
            if (learn_sky && (sky_width < 4 || sky_height < 2))
                throw std::invalid_argument("sky texture must be at least 4x2");
            cfg->depth_weight = depth_weight;
            cfg->depth_weight_final = depth_weight_final;
            cfg->depth_huber_delta = depth_huber_delta;
            cfg->depth_min_alpha = depth_min_alpha;
            cfg->sky_alpha_weight = sky_alpha_weight;
            cfg->fill_weight = fill_weight;
            cfg->use_masks = use_masks;
            cfg->learn_sky = learn_sky;
            cfg->sky_width = sky_width;
            cfg->sky_height = sky_height;
            cfg->sky_lr = sky_lr;
            cfg->exposure_compensation = exposure_compensation;
            cfg->exposure_lr = exposure_lr;
            cfg->exposure_reg = exposure_reg;
            cfg->max_scale_ratio = max_scale_ratio;
        },
            "iterations"_a = 30000,
            "sh_degree"_a = 3,
            "sh_degree_interval"_a = 1000,
            "ssim_weight"_a = 0.2f,
            "num_downscales"_a = 2,
            "resolution_schedule"_a = 3000,
            "refine_every"_a = 100,
            "warmup_length"_a = 500,
            "reset_alpha_every"_a = 30,
            "densify_grad_thresh"_a = 0.0002f,
            "densify_size_thresh"_a = 0.01f,
            "stop_screen_size_at"_a = 4000,
            "split_screen_size"_a = 0.05f,
            "keep_crs"_a = false,
            "downscale_factor"_a = 1.0f,
            "output"_a = "splat.ply",
            "save_every"_a = -1,
            "bg_color"_a = std::vector<float>{0.6130f, 0.0101f, 0.3984f},
            "depth_weight"_a = 0.0f,
            "depth_weight_final"_a = -1.0f,
            "depth_huber_delta"_a = 0.05f,
            "depth_min_alpha"_a = 0.25f,
            "sky_alpha_weight"_a = 0.0f,
            "fill_weight"_a = 0.0f,
            "use_masks"_a = false,
            "learn_sky"_a = false,
            "sky_width"_a = 512,
            "sky_height"_a = 128,
            "sky_lr"_a = 0.01f,
            "exposure_compensation"_a = false,
            "exposure_lr"_a = 5e-3f,
            "exposure_reg"_a = 1e-2f,
            "max_scale_ratio"_a = 0.0f)
        .def_rw("iterations", &TrainingConfig::iterations)
        .def_rw("sh_degree", &TrainingConfig::sh_degree)
        .def_rw("sh_degree_interval", &TrainingConfig::sh_degree_interval)
        .def_rw("ssim_weight", &TrainingConfig::ssim_weight)
        .def_rw("num_downscales", &TrainingConfig::num_downscales)
        .def_rw("resolution_schedule", &TrainingConfig::resolution_schedule)
        .def_rw("refine_every", &TrainingConfig::refine_every)
        .def_rw("warmup_length", &TrainingConfig::warmup_length)
        .def_rw("reset_alpha_every", &TrainingConfig::reset_alpha_every)
        .def_rw("densify_grad_thresh", &TrainingConfig::densify_grad_thresh)
        .def_rw("densify_size_thresh", &TrainingConfig::densify_size_thresh)
        .def_rw("stop_screen_size_at", &TrainingConfig::stop_screen_size_at)
        .def_rw("split_screen_size", &TrainingConfig::split_screen_size)
        .def_rw("keep_crs", &TrainingConfig::keep_crs)
        .def_rw("downscale_factor", &TrainingConfig::downscale_factor)
        .def_rw("output", &TrainingConfig::output)
        .def_rw("save_every", &TrainingConfig::save_every)
        .def_rw("bg_color", &TrainingConfig::bg_color,
            "Background color as [R, G, B] floats in [0, 1]. Default magenta [0.613, 0.010, 0.398].")
        .def_rw("depth_weight", &TrainingConfig::depth_weight,
            "Weight of the log-depth prior loss (0 = off). Needs priors/depth.")
        .def_rw("depth_weight_final", &TrainingConfig::depth_weight_final,
            "Depth prior weight at the last step, log-linear schedule (<0: constant, 0: linear to 0).")
        .def_rw("depth_huber_delta", &TrainingConfig::depth_huber_delta,
            "Huber transition of the depth loss, in log depth (~ relative error).")
        .def_rw("depth_min_alpha", &TrainingConfig::depth_min_alpha,
            "Depth is only supervised where accumulated alpha exceeds this.")
        .def_rw("sky_alpha_weight", &TrainingConfig::sky_alpha_weight,
            "Push sky-mask pixels transparent so the learned sky shows. Needs priors/sky.")
        .def_rw("fill_weight", &TrainingConfig::fill_weight,
            "Push non-sky pixels opaque (cameras with a sky mask).")
        .def_rw("use_masks", &TrainingConfig::use_masks,
            "Drop all gradients where priors/mask is 0 (moving objects).")
        .def_rw("learn_sky", &TrainingConfig::learn_sky,
            "Learn a direction-dependent sky (equirect texture) behind the gaussians.")
        .def_rw("sky_width", &TrainingConfig::sky_width)
        .def_rw("sky_height", &TrainingConfig::sky_height)
        .def_rw("sky_lr", &TrainingConfig::sky_lr)
        .def_rw("exposure_compensation", &TrainingConfig::exposure_compensation,
            "Per-image affine color transform, absorbing auto-exposure/white balance.")
        .def_rw("exposure_lr", &TrainingConfig::exposure_lr)
        .def_rw("exposure_reg", &TrainingConfig::exposure_reg)
        .def_rw("max_scale_ratio", &TrainingConfig::max_scale_ratio,
            "Cap each gaussian's largest/median scale ratio (needle suppression; <=1 = off).");

    // TrainingStats
    nb::class_<TrainingStats>(m, "TrainingStats",
            "Per-step training statistics returned by GaussianTrainer.step().")
        .def_ro("iteration", &TrainingStats::iteration, "Current training iteration.")
        .def_ro("splat_count", &TrainingStats::splat_count, "Number of active Gaussians.")
        .def_ro("ms_per_step", &TrainingStats::ms_per_step, "Wall-clock time for this step in milliseconds.")
        .def("__repr__", [](const TrainingStats &s) {
            return "TrainingStats(iteration=" + std::to_string(s.iteration) +
                   ", splats=" + std::to_string(s.splat_count) +
                   ", ms=" + std::to_string(s.ms_per_step) + ")";
        });

    // Dataset
    nb::class_<Dataset>(m, "Dataset",
            "A loaded dataset of camera images. Auto-detects COLMAP, Nerfstudio, and Polycam formats.")
        .def(nb::init<const std::string &, float, bool, int, const std::string &>(),
            "path"_a, "downscale_factor"_a = 1.0f,
            "eval_mode"_a = false, "test_every"_a = 8, "prior_dir"_a = "",
            "prior_dir: directory of geometric priors (depth/, confidence/, sky/, mask/).\n"
            "Defaults to <path>/priors when present; transforms.json frame keys take precedence.")
        .def_prop_ro("num_train", &Dataset::num_train, "Number of training cameras.")
        .def_prop_ro("num_test", &Dataset::num_test, "Number of test cameras (0 unless eval_mode=True).")
        .def("camera_pose", &Dataset::camera_pose, "index"_a,
            "Get camera-to-world pose (4x4 row-major, OpenGL convention) as numpy array.")
        .def("prior_counts", &Dataset::prior_counts,
            "Number of training cameras with each kind of prior file attached.");

    // GaussianTrainer
    nb::class_<GaussianTrainer>(m, "GaussianTrainer",
            "3D Gaussian Splatting trainer. All computation runs on the Metal GPU.")
        .def(nb::init<Dataset &, const TrainingConfig &>(),
            "dataset"_a, "config"_a, nb::keep_alive<1, 2>())
        .def("step", &GaussianTrainer::step,
            "Run a single training iteration. Returns TrainingStats.")
        .def("train", &GaussianTrainer::train,
            "callback"_a, "callback_every"_a = 100,
            "Run training to completion, calling callback(stats) every callback_every steps.")
        .def("evaluate", &GaussianTrainer::evaluate,
            "Evaluate on held-out test cameras. Returns dict with psnr, ssim, l1 keys.\n"
            "Requires the dataset to have been loaded with eval_mode=True.")
        .def("render", &GaussianTrainer::render,
            "cam_idx"_a, "use_test"_a = false,
            "Render a camera view. Returns a numpy array of shape (H, W, 3), float32, RGB [0,1].")
        .def("render_from_pose", &GaussianTrainer::render_from_pose,
            "cam_to_world"_a, "ref_cam_idx"_a = 0,
            "Render from an arbitrary camera-to-world pose (4x4 row-major, OpenGL convention).\n"
            "Uses intrinsics from ref_cam_idx. Returns numpy (H, W, 3) float32.")
        .def("export_ply", &GaussianTrainer::export_ply, "path"_a,
            "Export the current Gaussians as a PLY file.")
        .def("export_splat", &GaussianTrainer::export_splat, "path"_a,
            "Export the current Gaussians as a .splat file.")
        .def("save_checkpoint", &GaussianTrainer::save_checkpoint, "path"_a,
            "Save a training checkpoint.")
        .def("load_checkpoint", &GaussianTrainer::load_checkpoint, "path"_a,
            "Load a training checkpoint and resume from the saved iteration.")
        .def("render_depth", &GaussianTrainer::render_depth,
            "cam_idx"_a, "use_test"_a = false,
            "Render expected depth (dataset units, 0 where empty) and accumulated alpha.\n"
            "Returns a tuple of two numpy (H, W) float32 arrays.")
        .def("prior_losses", &GaussianTrainer::prior_losses,
            "Mean prior losses of the last step: dict with depth, sky, fill. Syncs the GPU.")
        .def("export_sky", &GaussianTrainer::export_sky, "path"_a,
            "Save the learned sky as an equirect PNG (requires learn_sky=True).")
        .def_prop_ro("splat_count", &GaussianTrainer::splat_count,
            "Current number of active Gaussians.")
        .def_prop_ro("iteration", [](const GaussianTrainer &t) { return t.current_step; },
            "Current training iteration.");

    // Test hooks for the GPU primitives (validated against NumPy in tests/)
    m.def("_gpu_radix_sort", [](nb::ndarray<const uint64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> keys,
                                int key_bits) {
        size_t n = keys.shape(0);
        MTensor k = gpu_empty({(int64_t)std::max<size_t>(n, 1)}, DType::Int64);
        MTensor v = gpu_empty({(int64_t)std::max<size_t>(n, 1)}, DType::Int32);
        memcpy(k.data_ptr(), keys.data(), n * sizeof(uint64_t));
        uint32_t *vp = v.data<uint32_t>();
        for (size_t i = 0; i < n; i++) vp[i] = (uint32_t)i;
        msplat_radix_sort(k, v, (uint32_t)n, key_bits);
        msplat_gpu_sync();
        uint64_t *ko = new uint64_t[std::max<size_t>(n, 1)];
        uint32_t *vo = new uint32_t[std::max<size_t>(n, 1)];
        memcpy(ko, k.data_ptr(), n * sizeof(uint64_t));
        memcpy(vo, v.data_ptr(), n * sizeof(uint32_t));
        nb::capsule kd(ko, [](void *p) noexcept { delete[] static_cast<uint64_t*>(p); });
        nb::capsule vd(vo, [](void *p) noexcept { delete[] static_cast<uint32_t*>(p); });
        size_t shape[1] = {n};
        return nb::make_tuple(nb::ndarray<nb::numpy, uint64_t>(ko, 1, shape, kd),
                              nb::ndarray<nb::numpy, uint32_t>(vo, 1, shape, vd));
    }, "keys"_a, "key_bits"_a = 64,
       "Stable GPU radix sort of uint64 keys. Returns (sorted_keys, order).");
    m.def("_gpu_knn3_mean_dist", [](nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu> points) {
        size_t n = points.shape(0);
        if (n == 0) throw std::invalid_argument("points must not be empty");
        MTensor p = gpu_empty({(int64_t)n, 3}, DType::Float32);
        memcpy(p.data_ptr(), points.data(), n * 3 * sizeof(float));
        MTensor d = gpu_empty({(int64_t)n}, DType::Float32);
        msplat_knn3_mean_dist(p, (uint32_t)n, d);
        msplat_gpu_sync();
        float *out = new float[n];
        memcpy(out, d.data_ptr(), n * sizeof(float));
        nb::capsule del(out, [](void *q) noexcept { delete[] static_cast<float*>(q); });
        size_t shape[1] = {n};
        return nb::ndarray<nb::numpy, float>(out, 1, shape, del);
    }, "points"_a, "Exact mean distance to the 3 nearest other points, computed on the GPU.");

    // Utility
    m.def("sync", &msplat_gpu_sync, "Synchronize GPU (wait for all commands to complete)");
    m.def("cleanup", &cleanup_msplat_metal, "Release all cached GPU resources");
}
