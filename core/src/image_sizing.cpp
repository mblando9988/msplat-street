#include "image_sizing.hpp"
#include "input_data.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

static std::string dims(int w, int h) { return std::to_string(w) + "x" + std::to_string(h); }

// ── Undistortion crop (Brown-Conrady model) ──────────────────────────────────
// The remap itself runs on the GPU (image_undistort_kernel); only the crop is found here.

// Iteratively invert distortion: normalized distorted → normalized undistorted
static void undistortPoint(float xd, float yd,
    float k1, float k2, float p1, float p2, float k3,
    float &xu, float &yu)
{
    xu = xd;
    yu = yd;
    for (int i = 0; i < 20; i++) {
        float r2 = xu * xu + yu * yu;
        float r4 = r2 * r2;
        float r6 = r4 * r2;
        float radial = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;
        float dx = 2.0f * p1 * xu * yu + p2 * (r2 + 2.0f * xu * xu);
        float dy = p1 * (r2 + 2.0f * yu * yu) + 2.0f * p2 * xu * yu;
        xu = (xd - dx) / radial;
        yu = (yd - dy) / radial;
    }
}

UndistortROI undistortROI(int w, int h,
    float fx, float fy, float cx, float cy,
    float k1, float k2, float p1, float p2, float k3)
{
    // Undistort the centers of the distorted image's border pixels; the innermost
    // position of each edge bounds the region every output pixel of which has a source
    // pixel (alpha=0). The principal point is in pixel-edge coordinates (pixel j's
    // center at j + 0.5); results are in pixel-center coordinates.
    auto undistortPixel = [&](float px, float py, float &ux, float &uy) {
        float xu, yu;
        undistortPoint((px + 0.5f - cx) / fx, (py + 0.5f - cy) / fy, k1, k2, p1, p2, k3, xu, yu);
        ux = xu * fx + cx - 0.5f;
        uy = yu * fy + cy - 0.5f;
    };
    const int nSamples = 200;
    float topMax = -1e9f, bottomMin = 1e9f, leftMax = -1e9f, rightMin = 1e9f;
    for (int i = 0; i < nSamples; i++) {
        float t = (float)i / (nSamples - 1);
        float ux, uy;
        undistortPixel(t * (w - 1), 0.0f, ux, uy);          // top edge
        topMax = std::max(topMax, uy);
        undistortPixel(t * (w - 1), (float)(h - 1), ux, uy);  // bottom edge
        bottomMin = std::min(bottomMin, uy);
        undistortPixel(0.0f, t * (h - 1), ux, uy);          // left edge
        leftMax = std::max(leftMax, ux);
        undistortPixel((float)(w - 1), t * (h - 1), ux, uy);  // right edge
        rightMin = std::min(rightMin, ux);
    }

    UndistortROI roi;
    roi.x = std::max(0, (int)std::ceil(leftMax));
    roi.y = std::max(0, (int)std::ceil(topMax));
    roi.width = std::min(w, (int)std::floor(rightMin)) - roi.x;
    roi.height = std::min(h, (int)std::floor(bottomMin)) - roi.y;
    if (roi.width <= 0 || roi.height <= 0) { roi.x = 0; roi.y = 0; roi.width = w; roi.height = h; }
    return roi;
}

ImageSizing planImageSizing(const Camera &cam, int fileWidth, int fileHeight, float downscaleFactor) {
    if (fileWidth <= 0 || fileHeight <= 0)
        throw std::runtime_error("Image has no pixels: " + cam.filePath);

    ImageSizing s;
    s.file = cam.filePath;
    s.fileWidth = fileWidth;
    s.fileHeight = fileHeight;
    s.metaWidth = std::max(cam.width, 0);
    s.metaHeight = std::max(cam.height, 0);
    float fx = cam.fx, fy = cam.fy, cx = cam.cx, cy = cam.cy;

    // Intrinsics given for another size describe the same camera at that size
    if (s.metaWidth > 0 && s.metaHeight > 0 && (fileWidth != s.metaWidth || fileHeight != s.metaHeight)) {
        s.metaScaleX = (float)fileWidth / (float)s.metaWidth;
        s.metaScaleY = (float)fileHeight / (float)s.metaHeight;
        fx *= s.metaScaleX; cx *= s.metaScaleX;
        fy *= s.metaScaleY; cy *= s.metaScaleY;
    }

    // Load-time downscale to whole pixels; the intrinsics follow the actual ratio, which
    // differs from 1/factor whenever the size does not divide evenly
    s.downscaleFactor = downscaleFactor;
    s.scaledWidth = fileWidth;
    s.scaledHeight = fileHeight;
    if (downscaleFactor > 1.0f) {
        s.scaledWidth = (int)(fileWidth / downscaleFactor);
        s.scaledHeight = (int)(fileHeight / downscaleFactor);
        if (s.scaledWidth < 1 || s.scaledHeight < 1)
            throw std::runtime_error("Downscale factor " + std::to_string(downscaleFactor) + " leaves no pixels of the " +
                                     dims(fileWidth, fileHeight) + " image " + cam.filePath);
        const float rx = (float)s.scaledWidth / (float)fileWidth;
        const float ry = (float)s.scaledHeight / (float)fileHeight;
        fx *= rx; cx *= rx;
        fy *= ry; cy *= ry;
    }
    s.width = s.scaledWidth;
    s.height = s.scaledHeight;

    if (cam.hasDistortion()) {
        // Undistortion crops and remaps the image, which would misalign every prior pixel.
        if (cam.hasPriorFiles())
            throw std::runtime_error("Priors need undistorted images, but " + cam.filePath +
                " has lens distortion. Undistort the dataset first (e.g. colmap image_undistorter).");
        UndistortROI roi = undistortROI(s.scaledWidth, s.scaledHeight, fx, fy, cx, cy,
                                        cam.k1, cam.k2, cam.p1, cam.p2, cam.k3);
        s.undistorted = true;
        s.cropX = roi.x;
        s.cropY = roi.y;
        s.width = roi.width;
        s.height = roi.height;
        cx -= (float)roi.x;
        cy -= (float)roi.y;
    }
    s.fx = fx; s.fy = fy; s.cx = cx; s.cy = cy;
    return s;
}

int scheduleDownscaleFactor(int step, int numDownscales, int resolutionSchedule) {
    if (numDownscales <= 0 || resolutionSchedule <= 0) return 1;
    int remaining = numDownscales - step / resolutionSchedule;
    return 1 << std::min(std::max(remaining, 0), 30);
}

std::vector<ScheduleLevel> resolutionLevels(int numDownscales, int resolutionSchedule, int iterations) {
    std::vector<ScheduleLevel> levels;
    if (iterations < 1) return levels;
    // The factor is piecewise constant and only changes at multiples of the schedule
    int step = 1;
    while (step <= iterations) {
        int f = scheduleDownscaleFactor(step, numDownscales, resolutionSchedule);
        int last = iterations;
        if (f > 1) {
            // first step of the next level: next multiple of resolutionSchedule
            long next = ((long)step / resolutionSchedule + 1) * (long)resolutionSchedule;
            last = (int)std::min<long>(next - 1, iterations);
        }
        levels.push_back({f, step, last});
        step = last + 1;
    }
    return levels;
}
