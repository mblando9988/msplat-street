#ifndef PRIORS_H
#define PRIORS_H

// Geometric priors for prior-guided training (street view and other sparse,
// forward-moving captures). msplat-prior writes them; the loaders attach them to
// cameras; Camera::loadPriors uploads them at their native resolution.
//
// Layout, per image, found either through per-frame keys in transforms.json
// (depth_file_path, depth_confidence_path, sky_mask_path, mask_path) or under a
// priors/ directory next to the dataset, keyed by the image path relative to the
// image root with its extension replaced:
//   priors/depth/<name>.npy        float32 (H, W) z-depth in dataset units, 0 = invalid
//   priors/confidence/<name>.npy   float (H, W) in [0, 1]
//   priors/sky/<name>.png          8-bit, nonzero = sky
//   priors/mask/<name>.png         8-bit, zero = ignore pixel (moving objects)

#include <cstdint>
#include <string>
#include <vector>
#include "input_data.hpp"

// Minimal .npy reader: little-endian f2/f4/f8, u1/u2, b1; C or Fortran order.
struct NpyArray {
    std::vector<int64_t> shape;
    std::vector<float> data;  // converted to float32, C order
};
NpyArray readNpy(const std::string &path);
std::vector<int64_t> readNpyShape(const std::string &path);  // header only

// Prior maps larger than this (either side) are downsampled on load: they are sampled
// by normalized position, so their resolution need not match the image's.
constexpr int kMaxPriorDim = 1024;

// Fill in missing prior paths from a priors/ directory. An empty priorDir means
// <dataset>/priors when that directory exists. Returns the number of cameras that
// gained at least one prior file.
int attachPriors(InputData &data, const std::string &priorDir = "");

// Nearest-neighbour resample of a single-channel (h, w) buffer.
std::vector<float> resampleNearest(const std::vector<float> &src, int sw, int sh, int dw, int dh);

#endif
