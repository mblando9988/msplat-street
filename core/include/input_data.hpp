#ifndef INPUT_DATA_H
#define INPUT_DATA_H

#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>
#include "metal_tensor.hpp"
#include "image_sizing.hpp"

// Simple float32 RGB image — replaces cv::Mat
struct Image {
    std::vector<float> data;  // width * height * 3 floats, RGB, [0,1]
    int width = 0, height = 0;

    bool empty() const { return data.empty(); }
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
};

// Image header facts, read without decoding pixels
struct ImageFileInfo {
    int width = 0, height = 0;          // pixels as stored
    int orientation = 1;                // EXIF orientation tag (1 = upright)
    int exifWidth = 0, exifHeight = 0;  // EXIF PixelX/YDimension, 0 when absent
};

// Image decoded into GPU-visible memory: (height, width, 4) RGBA8, alpha unused
struct RGBA8Image {
    MTensor rgba;
    int width = 0, height = 0;
    ImageFileInfo info;
};

struct Camera {
    int width = 0, height = 0;
    float fx = 0, fy = 0, cx = 0, cy = 0;
    float k1 = 0, k2 = 0, k3 = 0, p1 = 0, p2 = 0;
    float camToWorld[16] = {};  // 4x4 row-major, camera-to-world (OpenGL: Y-up, Z-back)
    std::string filePath;

    // Ground truth, GPU-resident only: (height, width, 3) float RGB in [0, 1] after the
    // load-time downscale and undistortion. Coarser levels for the resolution schedule
    // are area-downsampled on the GPU on first use and cached. Copies of a Camera share
    // the buffers.
    MTensor image;
    std::unordered_map<int, MTensor> imagePyramid;
    // How the image was sized on load, from file to training resolution (set by
    // setImage; sizing.width == 0 until then). width/height/fx..cy equal its final values.
    ImageSizing sizing;
    MTensor cachedViewMat, cachedProjViewMat;
    float cachedCamPos[3] = {};
    float cachedFovX = 0, cachedFovY = 0;

    // Optional geometric priors (see priors.hpp). Paths come from transforms.json
    // keys or attachPriors(); imageName is the image path relative to the image root.
    std::string imageName;
    std::string priorDepthPath, priorConfidencePath, priorSkyPath, priorMaskPath;
    int trainIndex = -1;  // slot for per-image parameters, set by Model::configurePriors

    // Native-resolution prior buffers, uploaded by loadPriors()
    MTensor priorDepth;   // (priorH, priorW) float, normalized scene units, 0 = invalid
    MTensor priorAux;     // (priorH, priorW, 4) uint8: confidence, sky, keep, 0
    int priorW = 0, priorH = 0;
    bool priorHasDepth = false, priorHasSky = false, priorHasMask = false;

    // Decode, downscale and undistort the image (updating the intrinsics to match).
    // loadCameraImages() does this for many cameras with parallel decoding.
    void loadImage(float downscaleFactor);
    // Finish loading from an already decoded image, executing planImageSizing():
    // conversion, resampling and undistortion are encoded on the GPU and the intrinsics
    // are updated to match; the decoded buffer can be dropped after. Once per camera.
    void setImage(const RGBA8Image &decoded, float downscaleFactor);
    bool hasImage() const { return image.defined(); }
    MTensor& getGPUImage(int downscaleFactor);
    bool hasDistortion() const { return k1 != 0 || k2 != 0 || k3 != 0 || p1 != 0 || p2 != 0; }
    bool hasPriorFiles() const {
        return !priorDepthPath.empty() || !priorSkyPath.empty() || !priorMaskPath.empty();
    }
    bool hasPriors() const { return priorAux.defined(); }
    // Read the prior files and upload them. depthScale converts dataset units to the
    // normalized scene (InputData::scale); useMask=false ignores the keep mask.
    // Idempotent.
    void loadPriors(float depthScale, bool useMask);
};

// Load the images of all cameras: CPU decoding on a pool of threads (0 = one per core,
// at most 8) feeding the GPU conversion in order, with bounded memory in flight.
void loadCameraImages(std::vector<Camera> &cameras, float downscaleFactor, int numThreads = 0);

struct Points {
    std::vector<float> xyz;     // N*3 flattened
    std::vector<uint8_t> rgb;   // N*3 flattened
    int64_t count = 0;
};

struct InputData {
    std::vector<Camera> cameras;
    float scale = 1.0f;
    float translation[3] = {};
    Points points;
    std::string rootDir;  // dataset directory passed to inputDataFromX

    std::tuple<std::vector<Camera>, Camera*> getCameras(bool validate, const std::string &valImage = "random");
    std::tuple<std::vector<Camera>, std::vector<Camera>> splitTrainTest(int testEvery);
    void saveCameras(const std::string &filename, bool keepCrs) const;
};

// Auto-detect format and load dataset
InputData inputDataFromX(const std::string &path, const std::string &colmapImagePath = "");
// "nerfstudio", "colmap", "polycam", or "" when no supported camera poses are found
std::string detectDatasetFormat(const std::string &path);

#endif
