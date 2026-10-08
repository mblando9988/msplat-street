#include "priors.hpp"
#include "loaders.hpp"
#include "msplat.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;

// ── .npy reader ─────────────────────────────────────────────────────────────

static float halfToFloat(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {  // subnormal: renormalize
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000 | (mant << 13);  // inf / nan
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

NpyArray readNpy(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open .npy file: " + path);

    char magic[6];
    f.read(magic, 6);
    if (!f || memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error("Not a .npy file: " + path);
    uint8_t version[2];
    f.read(reinterpret_cast<char*>(version), 2);
    uint32_t headerLen = 0;
    if (version[0] == 1) {
        uint16_t hl;
        f.read(reinterpret_cast<char*>(&hl), 2);
        headerLen = hl;
    } else {
        f.read(reinterpret_cast<char*>(&headerLen), 4);
    }
    std::string header(headerLen, '\0');
    f.read(header.data(), headerLen);
    if (!f) throw std::runtime_error("Truncated .npy header: " + path);

    auto field = [&](const std::string &key) -> std::string {
        size_t k = header.find("'" + key + "'");
        if (k == std::string::npos) throw std::runtime_error("Malformed .npy header (" + key + "): " + path);
        size_t colon = header.find(':', k);
        return header.substr(colon + 1);
    };

    std::string descrField = field("descr");
    size_t q0 = descrField.find('\''), q1 = descrField.find('\'', q0 + 1);
    std::string descr = descrField.substr(q0 + 1, q1 - q0 - 1);
    bool fortran = field("fortran_order").find("True") < field("fortran_order").find(',');

    std::string shapeField = field("shape");
    size_t p0 = shapeField.find('('), p1 = shapeField.find(')');
    std::string dims = shapeField.substr(p0 + 1, p1 - p0 - 1);
    NpyArray arr;
    int64_t count = 1;
    size_t pos = 0;
    while (pos < dims.size()) {
        size_t comma = dims.find(',', pos);
        std::string tok = dims.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        tok.erase(std::remove_if(tok.begin(), tok.end(), ::isspace), tok.end());
        if (!tok.empty()) {
            int64_t d = std::stoll(tok);
            arr.shape.push_back(d);
            count *= d;
        }
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }

    // Endianness: '<' little, '|' not applicable, '=' native (little on Apple Silicon)
    char endian = descr.empty() ? '<' : descr[0];
    std::string kind = descr.substr(1);
    if (endian == '>') throw std::runtime_error("Big-endian .npy not supported: " + path);

    arr.data.resize(count);
    auto readAll = [&](size_t elemSize, auto convert) {
        std::vector<char> raw(count * elemSize);
        f.read(raw.data(), raw.size());
        if (!f) throw std::runtime_error("Truncated .npy data: " + path);
        for (int64_t i = 0; i < count; i++) arr.data[i] = convert(raw.data() + i * elemSize);
    };
    if (kind == "f4") {
        readAll(4, [](const char *p) { float v; memcpy(&v, p, 4); return v; });
    } else if (kind == "f8") {
        readAll(8, [](const char *p) { double v; memcpy(&v, p, 8); return (float)v; });
    } else if (kind == "f2") {
        readAll(2, [](const char *p) { uint16_t v; memcpy(&v, p, 2); return halfToFloat(v); });
    } else if (kind == "u1" || kind == "b1") {
        readAll(1, [](const char *p) { return (float)(uint8_t)*p; });
    } else if (kind == "u2") {
        readAll(2, [](const char *p) { uint16_t v; memcpy(&v, p, 2); return (float)v; });
    } else {
        throw std::runtime_error("Unsupported .npy dtype '" + descr + "': " + path);
    }

    if (fortran && arr.shape.size() == 2) {
        int64_t h = arr.shape[0], w = arr.shape[1];
        std::vector<float> c(count);
        for (int64_t y = 0; y < h; y++)
            for (int64_t x = 0; x < w; x++)
                c[y * w + x] = arr.data[x * h + y];
        arr.data.swap(c);
    } else if (fortran && arr.shape.size() > 2) {
        throw std::runtime_error("Fortran-order .npy with more than 2 dims not supported: " + path);
    }
    return arr;
}

std::vector<float> resampleNearest(const std::vector<float> &src, int sw, int sh, int dw, int dh) {
    if (sw == dw && sh == dh) return src;
    std::vector<float> dst((size_t)dw * dh);
    for (int y = 0; y < dh; y++) {
        int sy = std::min(sh - 1, (int)(((float)y + 0.5f) * (float)sh / (float)dh));
        for (int x = 0; x < dw; x++) {
            int sx = std::min(sw - 1, (int)(((float)x + 0.5f) * (float)sw / (float)dw));
            dst[(size_t)y * dw + x] = src[(size_t)sy * sw + sx];
        }
    }
    return dst;
}

// ── priors/ directory lookup ────────────────────────────────────────────────

int attachPriors(InputData &data, const std::string &priorDir) {
    fs::path dir = priorDir.empty() ? fs::path(data.rootDir) / "priors" : fs::path(priorDir);
    if (priorDir.empty() && (data.rootDir.empty() || !fs::is_directory(dir))) return 0;
    if (!fs::is_directory(dir)) throw std::runtime_error("Prior directory not found: " + dir.string());

    int attached = 0;
    for (auto &cam : data.cameras) {
        // Key by the image path relative to the image root (so cam0/0001.jpg and
        // cam1/0001.jpg stay distinct), falling back to the bare file stem.
        fs::path rel(cam.imageName.empty() ? fs::path(cam.filePath).filename().string() : cam.imageName);
        std::vector<fs::path> keys = {rel.parent_path() / rel.stem()};
        if (rel.has_parent_path()) keys.push_back(rel.stem());

        auto find = [&](const char *sub, std::initializer_list<const char*> exts) -> std::string {
            for (auto &k : keys)
                for (auto ext : exts) {
                    fs::path p = dir / sub / k;
                    p += ext;
                    if (fs::exists(p)) return p.string();
                }
            return "";
        };
        bool before = cam.hasPriorFiles();
        if (cam.priorDepthPath.empty()) cam.priorDepthPath = find("depth", {".npy"});
        if (cam.priorConfidencePath.empty()) cam.priorConfidencePath = find("confidence", {".npy"});
        if (cam.priorSkyPath.empty()) cam.priorSkyPath = find("sky", {".png", ".jpg"});
        if (cam.priorMaskPath.empty()) cam.priorMaskPath = find("mask", {".png", ".jpg"});
        if (!before && cam.hasPriorFiles()) attached++;
    }
    return attached;
}

// ── Camera::loadPriors ──────────────────────────────────────────────────────

// Priors are sampled by normalized pixel position, so they need not match the image
// resolution. Larger ones are downsampled to bound GPU memory per camera.
static constexpr int kMaxPriorDim = 1024;

void Camera::loadPriors(float depthScale, bool useMask) {
    if (priorAux.defined() || !hasPriorFiles()) return;
    if (priorDepthPath.empty() && priorSkyPath.empty() && !useMask) return;

    int w = 0, h = 0;
    auto fit = [&](int sw, int sh) {
        float s = std::min(1.0f, (float)kMaxPriorDim / (float)std::max(sw, sh));
        w = std::max(1, (int)std::lround(sw * s));
        h = std::max(1, (int)std::lround(sh * s));
    };
    auto dims2d = [&](const NpyArray &a, const std::string &path, int &sw, int &sh) {
        bool ok = a.shape.size() == 2 || (a.shape.size() == 3 && a.shape[2] == 1);
        if (!ok) throw std::runtime_error("Prior must be a 2-D (H, W) array: " + path);
        sh = (int)a.shape[0];
        sw = (int)a.shape[1];
    };

    std::vector<float> depth, conf, sky, keep;
    if (!priorDepthPath.empty()) {
        NpyArray a = readNpy(priorDepthPath);
        int sw, sh;
        dims2d(a, priorDepthPath, sw, sh);
        fit(sw, sh);
        depth = resampleNearest(a.data, sw, sh, w, h);
    }
    if (!priorConfidencePath.empty() && !depth.empty()) {
        NpyArray a = readNpy(priorConfidencePath);
        int sw, sh;
        dims2d(a, priorConfidencePath, sw, sh);
        conf = resampleNearest(a.data, sw, sh, w, h);
    }
    auto loadMask = [&](const std::string &path) {
        Image m = imreadRGB(path);
        std::vector<float> g((size_t)m.width * m.height);
        for (size_t i = 0; i < g.size(); i++)
            g[i] = (m.data[3 * i] + m.data[3 * i + 1] + m.data[3 * i + 2]) * (1.0f / 3.0f);
        if (w == 0) fit(m.width, m.height);
        return resampleNearest(g, m.width, m.height, w, h);
    };
    if (!priorSkyPath.empty()) sky = loadMask(priorSkyPath);
    if (!priorMaskPath.empty() && useMask) keep = loadMask(priorMaskPath);

    priorW = w;
    priorH = h;
    priorDepth = gpu_zeros({h, w}, DType::Float32);
    priorAux = gpu_zeros({h, w, 4}, DType::UInt8);
    float *dp = priorDepth.data<float>();
    uint8_t *ap = priorAux.data<uint8_t>();
    for (size_t i = 0; i < (size_t)w * h; i++) {
        float d = depth.empty() ? 0.f : depth[i];
        if (!std::isfinite(d) || d <= 0.f) d = 0.f;
        float c = 0.f;
        if (d > 0.f) {
            c = conf.empty() ? 1.f : conf[i];
            c = std::isfinite(c) ? std::clamp(c, 0.f, 1.f) : 0.f;
        }
        dp[i] = d * depthScale;
        ap[4 * i + 0] = (uint8_t)std::lround(c * 255.f);
        ap[4 * i + 1] = (!sky.empty() && sky[i] > 0.5f) ? 255 : 0;
        ap[4 * i + 2] = (keep.empty() || keep[i] > 0.5f) ? 255 : 0;
        ap[4 * i + 3] = 0;
    }
    priorHasDepth = !depth.empty();
    priorHasSky = !sky.empty();
    priorHasMask = !keep.empty();
}
