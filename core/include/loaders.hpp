#ifndef LOADERS_H
#define LOADERS_H

#include "input_data.hpp"

// Format-specific loaders
namespace loaders {
    InputData loadColmap(const std::string &projectRoot, const std::string &imageSourcePath = "");
    InputData loadNerfstudio(const std::string &projectRoot);
    InputData loadPolycam(const std::string &projectRoot);
}

// PLY point cloud reader
Points readPly(const std::string &path);

// COLMAP binary point cloud reader
Points readColmapPoints(const std::string &path);

// Image I/O
Image imreadRGB(const std::string &path);       // returns float32 [0,1] directly
void imwriteRGB(const std::string &path, const Image &img);  // save as PNG

RGBA8Image imreadRGBA8(const std::string &path);  // decoded into GPU-visible memory
ImageFileInfo probeImage(const std::string &path);  // header only: size, orientation, EXIF size

// Pose utilities
void autoScaleAndCenter(InputData &data);

// Gaussian PLY/splat I/O (trained scene export/import)
struct GaussianParams {
    MTensor &means, &scales, &quats, &featuresDc, &featuresRest, &opacities;
    float scale;          // CRS scale factor
    float translation[3]; // CRS translation
    bool keepCrs;
};

void saveGaussianPly(const std::string &path, GaussianParams &p, int step);
void saveGaussianSplat(const std::string &path, GaussianParams &p);

struct LoadedGaussians {
    MTensor means, scales, quats, featuresDc, featuresRest, opacities;
    int step;
};
LoadedGaussians loadGaussianPly(const std::string &path, float scale, const float translation[3], bool keepCrs);

#endif
