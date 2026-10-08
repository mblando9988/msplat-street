#include "loaders.hpp"
#include "bindings.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>

#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>

// ── Image loading (CoreGraphics) ─────────────────────────────────────────────

Image imreadRGB(const std::string &path) {
    CFStringRef cfPath = CFStringCreateWithCString(nullptr, path.c_str(), kCFStringEncodingUTF8);
    CFURLRef url = CFURLCreateWithFileSystemPath(nullptr, cfPath, kCFURLPOSIXPathStyle, false);
    CFRelease(cfPath);

    CGImageSourceRef source = CGImageSourceCreateWithURL(url, nullptr);
    CFRelease(url);
    if (!source) {
        throw std::runtime_error("Failed to load image: " + path);
    }

    CGImageRef cgImage = CGImageSourceCreateImageAtIndex(source, 0, nullptr);
    CFRelease(source);
    if (!cgImage) {
        throw std::runtime_error("Failed to decode image: " + path);
    }

    int w = (int)CGImageGetWidth(cgImage);
    int h = (int)CGImageGetHeight(cgImage);

    // Render into RGBA buffer, then extract RGB and convert to float32
    std::vector<uint8_t> rgba(w * h * 4);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(
        rgba.data(), w, h, 8, w * 4, colorSpace,
        kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault
    );
    CGContextDrawImage(ctx, CGRectMake(0, 0, w, h), cgImage);
    CGContextRelease(ctx);
    CGColorSpaceRelease(colorSpace);
    CGImageRelease(cgImage);

    // RGBA → float32 RGB [0,1]
    Image img;
    img.width = w;
    img.height = h;
    img.data.resize(w * h * 3);
    for (int i = 0; i < w * h; i++) {
        img.data[i * 3 + 0] = rgba[i * 4 + 0] / 255.0f;
        img.data[i * 3 + 1] = rgba[i * 4 + 1] / 255.0f;
        img.data[i * 3 + 2] = rgba[i * 4 + 2] / 255.0f;
    }
    return img;
}

// Decode into GPU-visible memory: CoreGraphics draws straight into the shared MTLBuffer,
// so no CPU-side float image ever exists. Safe to call from several threads once the
// Metal context exists (allocation only; nothing is encoded).
RGBA8Image imreadRGBA8(const std::string &path) {
    CFStringRef cfPath = CFStringCreateWithCString(nullptr, path.c_str(), kCFStringEncodingUTF8);
    CFURLRef url = CFURLCreateWithFileSystemPath(nullptr, cfPath, kCFURLPOSIXPathStyle, false);
    CFRelease(cfPath);

    CGImageSourceRef source = CGImageSourceCreateWithURL(url, nullptr);
    CFRelease(url);
    if (!source) {
        throw std::runtime_error("Failed to load image: " + path);
    }

    CGImageRef cgImage = CGImageSourceCreateImageAtIndex(source, 0, nullptr);
    CFRelease(source);
    if (!cgImage) {
        throw std::runtime_error("Failed to decode image: " + path);
    }

    RGBA8Image img;
    img.width = (int)CGImageGetWidth(cgImage);
    img.height = (int)CGImageGetHeight(cgImage);
    img.rgba = gpu_empty({img.height, img.width, 4}, DType::UInt8);

    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(
        img.rgba.data_ptr(), img.width, img.height, 8, (size_t)img.width * 4, colorSpace,
        kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault
    );
    CGColorSpaceRelease(colorSpace);
    if (!ctx) {
        CGImageRelease(cgImage);
        throw std::runtime_error("Failed to decode image: " + path);
    }
    CGContextDrawImage(ctx, CGRectMake(0, 0, img.width, img.height), cgImage);
    CGContextRelease(ctx);
    CGImageRelease(cgImage);
    return img;
}

// ── Image writing (CoreGraphics PNG) ─────────────────────────────────────────

void imwriteRGB(const std::string &path, const Image &img) {
    int w = img.width, h = img.height;

    // float32 RGB → uint8 RGB
    std::vector<uint8_t> rgb8(w * h * 3);
    for (int i = 0; i < w * h * 3; i++) {
        float v = std::clamp(img.data[i] * 255.0f, 0.0f, 255.0f);
        rgb8[i] = (uint8_t)(v + 0.5f);
    }

    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(
        rgb8.data(), w, h, 8, w * 3, colorSpace,
        kCGImageAlphaNone | kCGBitmapByteOrderDefault
    );
    CGImageRef cgImage = CGBitmapContextCreateImage(ctx);
    CGContextRelease(ctx);
    CGColorSpaceRelease(colorSpace);

    CFStringRef cfPath = CFStringCreateWithCString(nullptr, path.c_str(), kCFStringEncodingUTF8);
    CFURLRef url = CFURLCreateWithFileSystemPath(nullptr, cfPath, kCFURLPOSIXPathStyle, false);
    CFRelease(cfPath);

    CGImageDestinationRef dest = CGImageDestinationCreateWithURL(url, CFSTR("public.png"), 1, nullptr);
    CFRelease(url);
    CGImageDestinationAddImage(dest, cgImage, nullptr);
    CGImageDestinationFinalize(dest);

    CFRelease(dest);
    CGImageRelease(cgImage);
}

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
    // Undistort the boundary of the distorted image; the innermost position of each
    // edge bounds the region every output pixel of which has a source pixel (alpha=0).
    const int nSamples = 200;
    float topMax = -1e9f, bottomMin = 1e9f, leftMax = -1e9f, rightMin = 1e9f;
    for (int i = 0; i < nSamples; i++) {
        float t = (float)i / (nSamples - 1);
        float xu, yu;

        // Top edge: all points along y=0
        undistortPoint((t * w - cx) / fx, (0.0f - cy) / fy, k1, k2, p1, p2, k3, xu, yu);
        topMax = std::max(topMax, yu * fy + cy);

        // Bottom edge: all points along y=h-1
        undistortPoint((t * w - cx) / fx, ((float)(h-1) - cy) / fy, k1, k2, p1, p2, k3, xu, yu);
        bottomMin = std::min(bottomMin, yu * fy + cy);

        // Left edge: all points along x=0
        undistortPoint((0.0f - cx) / fx, (t * h - cy) / fy, k1, k2, p1, p2, k3, xu, yu);
        leftMax = std::max(leftMax, xu * fx + cx);

        // Right edge: all points along x=w-1
        undistortPoint(((float)(w-1) - cx) / fx, (t * h - cy) / fy, k1, k2, p1, p2, k3, xu, yu);
        rightMin = std::min(rightMin, xu * fx + cx);
    }

    UndistortROI roi;
    roi.x = std::max(0, (int)std::ceil(leftMax));
    roi.y = std::max(0, (int)std::ceil(topMax));
    roi.width = std::min(w, (int)std::floor(rightMin)) - roi.x;
    roi.height = std::min(h, (int)std::floor(bottomMin)) - roi.y;
    if (roi.width <= 0 || roi.height <= 0) { roi.x = 0; roi.y = 0; roi.width = w; roi.height = h; }
    return roi;
}
