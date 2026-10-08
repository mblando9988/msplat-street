#ifndef SSIM_H
#define SSIM_H

#include <vector>
#include <cmath>
#include "metal_tensor.hpp"

// SSIM window creation. Training uses Metal kernels (ssim_h/v_fwd/bwd) directly and
// evaluation SSIM runs on the GPU too (imageMetrics in model.hpp).

// Create 11x11 Gaussian window for Metal SSIM loss kernel.
// Returns flat float vector (121 elements).
std::vector<float> createSSIMWindow(int windowSize = 11, float sigma = 1.5f);

#endif
