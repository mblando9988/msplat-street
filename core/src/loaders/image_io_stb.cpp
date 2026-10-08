// Image I/O without Apple frameworks (CPU backend builds): stb_image decodes,
// stb_image_write encodes, and a small JPEG/EXIF reader stands in for the ImageIO
// properties. Behaves like image_io.cpp: pixels are returned as stored (EXIF
// orientation is reported, not applied) and alpha is composited onto black, as
// CoreGraphics does when it draws into an opaque context.
#include "loaders.hpp"
#include "bindings.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace {

struct ExifInfo {
    int orientation = 1;
    int width = 0, height = 0;  // PixelXDimension / PixelYDimension
};

uint32_t readU(const uint8_t *p, int bytes, bool le) {
    uint32_t v = 0;
    for (int i = 0; i < bytes; i++) v |= (uint32_t)p[le ? i : bytes - 1 - i] << (8 * i);
    return v;
}

// Reads the orientation and pixel-dimension tags of a TIFF-structured EXIF block.
void parseTiff(const uint8_t *t, size_t n, ExifInfo &info) {
    if (n < 8) return;
    bool le;
    if (t[0] == 'I' && t[1] == 'I') le = true;
    else if (t[0] == 'M' && t[1] == 'M') le = false;
    else return;
    if (readU(t + 2, 2, le) != 42) return;

    auto value = [&](const uint8_t *e) -> int {
        uint32_t type = readU(e + 2, 2, le);
        if (type == 3) return (int)readU(e + 8, 2, le);  // SHORT
        if (type == 4) return (int)readU(e + 8, 4, le);  // LONG
        return 0;
    };
    auto walk = [&](uint32_t off, bool ifd0, uint32_t &exif_ifd) {
        if (off + 2 > n) return;
        uint32_t count = readU(t + off, 2, le);
        for (uint32_t i = 0; i < count; i++) {
            size_t e = off + 2 + 12 * (size_t)i;
            if (e + 12 > n) return;
            uint32_t tag = readU(t + e, 2, le);
            if (ifd0 && tag == 0x0112) info.orientation = value(t + e);
            else if (ifd0 && tag == 0x8769) exif_ifd = readU(t + e + 8, 4, le);
            else if (!ifd0 && tag == 0xA002) info.width = value(t + e);
            else if (!ifd0 && tag == 0xA003) info.height = value(t + e);
        }
    };
    uint32_t exif_ifd = 0, unused = 0;
    walk(readU(t + 4, 4, le), true, exif_ifd);
    if (exif_ifd) walk(exif_ifd, false, unused);
    if (info.orientation < 1 || info.orientation > 8) info.orientation = 1;
}

// EXIF from a JPEG's APP1 segment (other formats: none)
ExifInfo readExif(const std::string &path) {
    ExifInfo info;
    std::ifstream f(path, std::ios::binary);
    uint8_t soi[2];
    if (!f.read((char *)soi, 2) || soi[0] != 0xFF || soi[1] != 0xD8) return info;
    for (;;) {
        uint8_t m[4];
        if (!f.read((char *)m, 2) || m[0] != 0xFF) return info;
        while (m[1] == 0xFF)  // fill bytes before a marker
            if (!f.read((char *)m + 1, 1)) return info;
        if (m[1] == 0xD8 || (m[1] >= 0xD0 && m[1] <= 0xD7) || m[1] == 0x01) continue;  // no length
        if (m[1] == 0xDA || m[1] == 0xD9) return info;  // image data starts: no EXIF before it
        if (!f.read((char *)m + 2, 2)) return info;
        size_t len = ((size_t)m[2] << 8) | m[3];
        if (len < 2) return info;
        if (m[1] == 0xE1) {
            std::vector<uint8_t> seg(len - 2);
            if (!f.read((char *)seg.data(), (std::streamsize)seg.size())) return info;
            if (seg.size() > 6 && std::memcmp(seg.data(), "Exif\0\0", 6) == 0) {
                parseTiff(seg.data() + 6, seg.size() - 6, info);
                return info;
            }
        } else {
            f.seekg((std::streamoff)(len - 2), std::ios::cur);
        }
    }
}

[[noreturn]] void decodeError(const std::string &path) {
    const char *reason = stbi_failure_reason();
    std::string msg = "Failed to load image: " + path;
    if (reason) msg += std::string(" (") + reason + ")";
    std::string lower = path;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    for (const char *ext : {".webp", ".heic", ".heif", ".avif", ".tif", ".tiff"})
        if (lower.size() > std::strlen(ext) && lower.compare(lower.size() - std::strlen(ext), std::string::npos, ext) == 0) {
            msg += "; this build reads PNG, JPEG, BMP, TGA, GIF, PSD, HDR and PNM: convert the image to PNG or JPEG";
            break;
        }
    throw std::runtime_error(msg);
}

struct StbPixels {
    std::unique_ptr<stbi_uc, void (*)(void *)> data{nullptr, stbi_image_free};
    int w = 0, h = 0, channels = 0;
};

// RGBA8 as stored, with alpha composited onto black
StbPixels decodeRGBA(const std::string &path) {
    StbPixels px;
    px.data.reset(stbi_load(path.c_str(), &px.w, &px.h, &px.channels, 4));
    if (!px.data) decodeError(path);
    if (px.channels == 2 || px.channels == 4) {
        stbi_uc *p = px.data.get();
        size_t n = (size_t)px.w * px.h;
        for (size_t i = 0; i < n; i++) {
            unsigned a = p[4 * i + 3];
            for (int c = 0; c < 3; c++) p[4 * i + c] = (stbi_uc)((p[4 * i + c] * a + 127) / 255);
            p[4 * i + 3] = 255;
        }
    }
    return px;
}

}  // namespace

Image imreadRGB(const std::string &path) {
    StbPixels px = decodeRGBA(path);
    Image img;
    img.width = px.w;
    img.height = px.h;
    img.data.resize((size_t)px.w * px.h * 3);
    const stbi_uc *p = px.data.get();
    for (size_t i = 0; i < (size_t)px.w * px.h; i++)
        for (int c = 0; c < 3; c++) img.data[i * 3 + c] = p[i * 4 + c] / 255.0f;
    return img;
}

ImageFileInfo probeImage(const std::string &path) {
    ImageFileInfo info;
    int w = 0, h = 0, n = 0;
    if (!stbi_info(path.c_str(), &w, &h, &n) || w <= 0 || h <= 0) {
        std::FILE *f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("Failed to load image: " + path);
        std::fclose(f);
        decodeError(path);
    }
    info.width = w;
    info.height = h;
    ExifInfo exif = readExif(path);
    info.orientation = exif.orientation;
    info.exifWidth = exif.width;
    info.exifHeight = exif.height;
    return info;
}

// Decodes into a device buffer (host memory in the CPU backend)
RGBA8Image imreadRGBA8(const std::string &path) {
    StbPixels px = decodeRGBA(path);
    RGBA8Image img;
    ExifInfo exif = readExif(path);
    img.info.orientation = exif.orientation;
    img.info.exifWidth = exif.width;
    img.info.exifHeight = exif.height;
    img.width = px.w;
    img.height = px.h;
    img.info.width = px.w;
    img.info.height = px.h;
    img.rgba = gpu_empty({img.height, img.width, 4}, DType::UInt8);
    std::memcpy(img.rgba.data_ptr(), px.data.get(), (size_t)px.w * px.h * 4);
    return img;
}

void imwriteRGB(const std::string &path, const Image &img) {
    int w = img.width, h = img.height;
    std::vector<uint8_t> rgb8((size_t)w * h * 3);
    for (size_t i = 0; i < rgb8.size(); i++) {
        float v = std::clamp(img.data[i] * 255.0f, 0.0f, 255.0f);
        rgb8[i] = (uint8_t)(v + 0.5f);
    }
    if (!stbi_write_png(path.c_str(), w, h, 3, rgb8.data(), w * 3))
        throw std::runtime_error("Failed to write image: " + path);
}
