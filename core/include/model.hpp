#ifndef MODEL_H
#define MODEL_H

#include "metal_tensor.hpp"
#include "ssim.hpp"
#include "input_data.hpp"

int numShBases(int degree);

// PSNR, SSIM (11-tap Gaussian, sigma 1.5, clamp-to-edge borders) and mean L1 error of a
// render against its ground truth, both (H, W, 3) GPU tensors. Computed on the GPU; syncs.
struct ImageMetrics { float psnr = 0.f, ssim = 0.f, l1 = 0.f; };
ImageMetrics imageMetrics(const MTensor& rendered, const MTensor& gt);

// Prior-guided training: geometry priors, learned sky, exposure, needle cap. Everything
// is off by default; Model::configurePriors() enables what is set here. See priors.hpp
// for the prior file layout written by msplat-prior.
struct PriorOptions {
    float depthWeight = 0.f;          // log-depth prior loss weight at the first step (0 = off)
    float depthWeightFinal = -1.f;    // weight at the last step, log-linear in between (<0: constant)
    float depthHuberDelta = 0.05f;    // Huber transition in log depth (~ relative depth error)
    float depthMinAlpha = 0.25f;      // only supervise depth where accumulated alpha exceeds this
    float skyAlphaWeight = 0.f;       // push alpha to 0 on sky-mask pixels
    float fillWeight = 0.f;           // push alpha to 1 on non-sky pixels (cameras with a sky mask)
    bool useMasks = false;            // drop all gradients where the keep mask is 0
    bool learnSky = false;            // direction-dependent background (equirect texture)
    int skyWidth = 512, skyHeight = 128;
    float skyLr = 0.01f;
    bool exposure = false;            // per-image affine color compensation
    float exposureLr = 5e-3f;
    float exposureReg = 1e-2f;        // pull toward identity (fixes the global color gauge)
    float maxScaleRatio = 0.f;        // cap largest/median gaussian scale (<= 1: off)
};

struct Model{
  Model(const InputData &inputData, int numCameras,
        int numDownscales, int resolutionSchedule, int shDegree, int shDegreeInterval,
        int refineEvery, int warmupLength, int resetAlphaEvery, float densifyGradThresh, float densifySizeThresh, int stopScreenSizeAt, float splitScreenSize,
        int maxSteps, bool keepCrs,
        const float* bgColor = nullptr);

  ~Model(){ releaseOptimizers(); }

  void setupOptimizers();
  void releaseOptimizers();

  void schedulersStep(int step);
  int getDownscaleFactor(int step);
  void afterTrain(int step);
  void save(const std::string &filename, int step);
  void savePly(const std::string &filename, int step);
  void saveSplat(const std::string &filename);
  int loadPly(const std::string &filename);
  void saveCheckpoint(const std::string &filename, int step);
  int loadCheckpoint(const std::string &filename);
  struct CamSetup {
    float fx, fy, cx, cy;
    int height, width, degree, degreesToUse;
    std::tuple<int,int,int> tileBounds;
    float cam_pos[3];
  };
  CamSetup prepareCam(Camera& cam, int step);
  void fullIteration(Camera& cam, int step, MTensor &gt, float ssimWeight);
  MTensor render(Camera& cam, int step);

  // ── Prior-guided training ──
  // Assigns train indices, loads the cameras' priors and allocates the sky and exposure
  // parameters. Call once after construction with the training cameras (the vector must
  // outlive the model). With default options it only assigns train indices.
  void configurePriors(const PriorOptions &opts, std::vector<Camera> &trainCams);
  bool priorsActive() const;
  float depthWeightAt(int step) const;
  // Expected depth (dataset units, 0 where nothing is rendered) and accumulated alpha,
  // both (H, W), computed on the GPU. Syncs, so both are readable on return.
  void renderDepth(Camera& cam, int step, MTensor &depthOut, MTensor &alphaOut);
  // Mean prior losses of the last training step: depth, sky, fill. Syncs the GPU.
  void lastPriorLosses(float out[3]);
  void saveSky(const std::string &filename);   // equirect PNG of the learned sky

  PriorOptions priorOpts;
  bool priorsConfigured = false;
  int numPriorCameras = 0, numDepthCameras = 0, numSkyCameras = 0, numMaskCameras = 0;
  MTensor skyTex, skyGrad, skyExpAvg, skyExpAvgSq;   // (skyHeight, skyWidth, 3)
  float skyFrame[9] = {};                            // up, e1, e2
  int skySteps = 0;
  MTensor expoParams, expoExpAvg, expoExpAvgSq;      // (numTrainCams, 12)
  MTensor expoGrad;                                  // [12]
  std::vector<int> expoSteps;
  MTensor priorLossTerms;                            // [4]

  MTensor means;
  MTensor scales;
  MTensor quats;
  MTensor featuresDc;
  MTensor featuresRest;
  MTensor opacities;

  static constexpr int N_ADAM_GROUPS = 6;
  MTensor adam_exp_avg[N_ADAM_GROUPS];
  MTensor adam_exp_avg_sq[N_ADAM_GROUPS];
  int adam_step_count = 0;
  float adam_lr[N_ADAM_GROUPS] = {};
  float adam_beta1 = 0.9f, adam_beta2 = 0.999f, adam_eps = 1e-8f;
  float means_lr_init = 0, means_lr_final = 0;

  MTensor means_buf, scales_buf, quats_buf, featuresDc_buf, featuresRest_buf, opacities_buf;
  MTensor adam_exp_avg_buf[N_ADAM_GROUPS], adam_exp_avg_sq_buf[N_ADAM_GROUPS];
  int num_active = 0, buf_capacity = 0;
  void refreshViews();
  void ensureCapacity(int needed);

  MTensor densify_split_flag, densify_dup_flag;
  MTensor densify_split_prefix, densify_dup_prefix;
  MTensor densify_keep_flag, densify_keep_prefix;
  MTensor densify_block_totals;
  MTensor densify_compact_scratch;

  MTensor radii;
  int lastHeight;
  int lastWidth;

  MTensor xysGradNorm;
  MTensor visCounts;
  MTensor max2DSize;

  MTensor backgroundColor;
  MTensor window2d;  // SSIM window (11,11) f32

  int numCameras;
  int numDownscales;
  int resolutionSchedule;
  int shDegree;
  int shDegreeInterval;
  int refineEvery;
  int warmupLength;
  int resetAlphaEvery;
  int stopSplitAt;
  float densifyGradThresh;
  float densifySizeThresh;
  int stopScreenSizeAt;
  float splitScreenSize;
  int maxSteps;
  bool keepCrs;

  float scale;
  float translation[3] = {};
};

#endif
