// SSIM window for the training loss kernels. Evaluation SSIM runs on the GPU
// (eval_ssim_h/v_kernel). Ported from https://github.com/Po-Hsun-Su/pytorch-ssim (MIT)

#include "ssim.hpp"
#include <cstring>

std::vector<float> createSSIMWindow(int windowSize, float sigma) {
    // 1D Gaussian
    std::vector<float> g(windowSize);
    float sum = 0;
    for (int i = 0; i < windowSize; i++) {
        float x = (float)(i - windowSize / 2);
        g[i] = std::exp(-(x * x) / (2.0f * sigma * sigma));
        sum += g[i];
    }
    for (int i = 0; i < windowSize; i++) g[i] /= sum;

    // 2D = outer product
    std::vector<float> w(windowSize * windowSize);
    for (int i = 0; i < windowSize; i++)
        for (int j = 0; j < windowSize; j++)
            w[i * windowSize + j] = g[i] * g[j];
    return w;
}
