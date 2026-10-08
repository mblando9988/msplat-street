#ifndef IMAGE_SIZING_H
#define IMAGE_SIZING_H

#include <string>
#include <vector>

struct Camera;

// Every resolution an image passes through, from the file on disk to training.
// planImageSizing() computes it; the loader executes exactly that plan and keeps it on
// the Camera, and the preflight check reports the same plan from image headers
// without decoding a pixel. One source of truth: a report cannot disagree with what
// is trained.
//
// Intrinsics are in pixels with the principal point in pixel-edge coordinates
// (COLMAP's convention: the top-left pixel spans [0, 1) and the rasterizer puts pixel
// j's center at x = j + 0.5), so resampling an axis from n to m pixels scales that
// axis' focal length and principal point by exactly m / n.
struct ImageSizing {
    std::string file;
    int fileWidth = 0, fileHeight = 0;          // pixels stored in the file
    int orientation = 1;                        // EXIF orientation, 1 = upright. Not applied: pixels are used as stored
    int exifWidth = 0, exifHeight = 0;          // EXIF PixelX/YDimension when present, else 0
    int metaWidth = 0, metaHeight = 0;          // size the intrinsics were given for (0 = not given)
    float metaScaleX = 1.f, metaScaleY = 1.f;   // intrinsics rescale from the metadata size to the file size
    float downscaleFactor = 1.f;                // load-time downscale as requested (<= 1: none)
    int scaledWidth = 0, scaledHeight = 0;      // after the load-time downscale
    bool undistorted = false;                   // lens distortion removed on load
    int cropX = 0, cropY = 0;                   // undistortion crop origin, in downscaled pixels
    int width = 0, height = 0;                  // full training resolution: the ground-truth size
    float fx = 0.f, fy = 0.f, cx = 0.f, cy = 0.f;  // intrinsics at width x height
};

// What loading an image of fileWidth x fileHeight does for this camera (its metadata
// size, intrinsics and distortion as read from the dataset) at the given downscale.
// Throws std::runtime_error when the image cannot be loaded (no pixels left, priors on
// a camera with lens distortion).
ImageSizing planImageSizing(const Camera &cam, int fileWidth, int fileHeight, float downscaleFactor);

// Undistortion crop (Brown-Conrady model, alpha=0): the largest rectangle of the
// undistorted image, at unchanged focal lengths, that has no invalid borders. The
// principal point of the cropped image is (cx - x, cy - y).
struct UndistortROI {
    int x = 0, y = 0, width = 0, height = 0;
};
UndistortROI undistortROI(int w, int h,
    float fx, float fy, float cx, float cy,
    float k1, float k2, float p1, float p2, float k3);

// Resolution schedule: training step `step` renders at 1/factor of the full resolution
// (Model::getDownscaleFactor). resolutionSchedule <= 0 means full resolution throughout.
int scheduleDownscaleFactor(int step, int numDownscales, int resolutionSchedule);

struct ScheduleLevel {
    int factor;
    int firstStep, lastStep;  // inclusive
};
// Levels in training order for steps 1..iterations.
std::vector<ScheduleLevel> resolutionLevels(int numDownscales, int resolutionSchedule, int iterations);

// Pixel size of a schedule level, exactly as the GPU pyramid and the renderer size it.
inline int levelSize(int fullSize, int factor) { return factor <= 1 ? fullSize : fullSize / factor; }

#endif
