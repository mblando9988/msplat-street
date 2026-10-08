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

static int dictInt(CFDictionaryRef dict, CFStringRef key, int fallback) {
    if (!dict) return fallback;
    CFTypeRef value = CFDictionaryGetValue(dict, key);
    int v = fallback;
    if (value && CFGetTypeID(value) == CFNumberGetTypeID())
        CFNumberGetValue((CFNumberRef)value, kCFNumberIntType, &v);
    return v;
}

// Size, orientation and EXIF size from the container metadata (no pixel decode)
static ImageFileInfo readImageInfo(CGImageSourceRef source) {
    ImageFileInfo info;
    CFDictionaryRef props = CGImageSourceCopyPropertiesAtIndex(source, 0, nullptr);
    if (!props) return info;
    info.width = dictInt(props, kCGImagePropertyPixelWidth, 0);
    info.height = dictInt(props, kCGImagePropertyPixelHeight, 0);
    info.orientation = dictInt(props, kCGImagePropertyOrientation, 1);
    CFTypeRef exif = CFDictionaryGetValue(props, kCGImagePropertyExifDictionary);
    if (exif && CFGetTypeID(exif) == CFDictionaryGetTypeID()) {
        info.exifWidth = dictInt((CFDictionaryRef)exif, kCGImagePropertyExifPixelXDimension, 0);
        info.exifHeight = dictInt((CFDictionaryRef)exif, kCGImagePropertyExifPixelYDimension, 0);
    }
    CFRelease(props);
    return info;
}

static CGImageSourceRef openImageSource(const std::string &path) {
    CFStringRef cfPath = CFStringCreateWithCString(nullptr, path.c_str(), kCFStringEncodingUTF8);
    CFURLRef url = CFURLCreateWithFileSystemPath(nullptr, cfPath, kCFURLPOSIXPathStyle, false);
    CFRelease(cfPath);
    CGImageSourceRef source = CGImageSourceCreateWithURL(url, nullptr);
    CFRelease(url);
    if (!source) {
        throw std::runtime_error("Failed to load image: " + path);
    }
    return source;
}

ImageFileInfo probeImage(const std::string &path) {
    CGImageSourceRef source = openImageSource(path);
    ImageFileInfo info = readImageInfo(source);
    CFRelease(source);
    if (info.width <= 0 || info.height <= 0) {
        throw std::runtime_error("Failed to read image size: " + path);
    }
    return info;
}

// Decode into GPU-visible memory: CoreGraphics draws straight into the shared MTLBuffer,
// so no CPU-side float image ever exists. Safe to call from several threads once the
// Metal context exists (allocation only; nothing is encoded).
RGBA8Image imreadRGBA8(const std::string &path) {
    CGImageSourceRef source = openImageSource(path);
    RGBA8Image img;
    img.info = readImageInfo(source);
    CGImageRef cgImage = CGImageSourceCreateImageAtIndex(source, 0, nullptr);
    CFRelease(source);
    if (!cgImage) {
        throw std::runtime_error("Failed to decode image: " + path);
    }

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
    // The decoded size is authoritative; the header may lack it
    img.info.width = img.width;
    img.info.height = img.height;
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
