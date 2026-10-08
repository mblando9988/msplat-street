#ifndef INPUT_DATA_H
#define INPUT_DATA_H

#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>
#include "metal_tensor.hpp"

// Simple float32 RGB image — replaces cv::Mat
struct Image {
    std::vector<float> data;  // width * height * 3 floats, RGB, [0,1]
    int width = 0, height = 0;

    bool empty() const { return data.empty(); }
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
};

struct Camera {
    int width = 0, height = 0;
    float fx = 0, fy = 0, cx = 0, cy = 0;
    float k1 = 0, k2 = 0, k3 = 0, p1 = 0, p2 = 0;
    float camToWorld[16] = {};  // 4x4 row-major, camera-to-world (OpenGL: Y-up, Z-back)
    std::string filePath;

    Image image;
    std::unordered_map<int, Image> imagePyramids;
    std::unordered_map<int, MTensor> mtensorImageCache;
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

    void loadImage(float downscaleFactor);
    Image getImage(int downscaleFactor);
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

#endif
