#include "input_data.hpp"
#include "loaders.hpp"
#include "msplat.hpp"
#include "priors.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ── Image loading ───────────────────────────────────────────────────────────

void Camera::loadImage(float downscaleFactor) {
    setImage(imreadRGBA8(filePath), downscaleFactor);
}

void Camera::setImage(const RGBA8Image &decoded, float downscaleFactor) {
    if (image.defined())
        throw std::logic_error("Image already loaded (intrinsics were adapted to it): " + filePath);
    ImageSizing s = planImageSizing(*this, decoded.width, decoded.height, downscaleFactor);
    s.orientation = decoded.info.orientation;
    s.exifWidth = decoded.info.exifWidth;
    s.exifHeight = decoded.info.exifHeight;

    // RGBA8 → float RGB, area-resampled to the downscaled size in the same pass
    MTensor level = gpu_empty({s.scaledHeight, s.scaledWidth, 3}, DType::Float32);
    msplat_resize_area(decoded.rgba, true, s.fileWidth, s.fileHeight, level, s.scaledWidth, s.scaledHeight);

    if (s.undistorted) {
        MTensor undist = gpu_empty({s.height, s.width, 3}, DType::Float32);
        // intrinsics of the downscaled image, before the crop
        const float intr[4] = {s.fx, s.fy, s.cx + (float)s.cropX, s.cy + (float)s.cropY};
        const float dist[5] = {k1, k2, p1, p2, k3};
        msplat_undistort(level, s.scaledWidth, s.scaledHeight, intr, dist, s.cropX, s.cropY,
                         undist, s.width, s.height);
        level = std::move(undist);
        k1 = k2 = k3 = p1 = p2 = 0;
    }

    width = s.width; height = s.height;
    fx = s.fx; fy = s.fy; cx = s.cx; cy = s.cy;
    sizing = s;
    image = std::move(level);
    imagePyramid.clear();
}

MTensor& Camera::getGPUImage(int downscaleFactor) {
    if (!image.defined()) throw std::runtime_error("Image not loaded: " + filePath);
    if (downscaleFactor <= 1) return image;

    auto it = imagePyramid.find(downscaleFactor);
    if (it != imagePyramid.end()) return it->second;

    int w = (int)image.size(1), h = (int)image.size(0);
    int newW = levelSize(w, downscaleFactor);
    int newH = levelSize(h, downscaleFactor);
    MTensor scaled = gpu_empty({newH, newW, 3}, DType::Float32);
    msplat_resize_area(image, false, w, h, scaled, newW, newH);
    return imagePyramid.emplace(downscaleFactor, std::move(scaled)).first->second;
}

void loadCameraImages(std::vector<Camera> &cameras, float downscaleFactor, int numThreads) {
    const size_t n = cameras.size();
    if (n == 0) return;
    msplat_device();  // create the Metal context here, before any worker allocates
    if (numThreads <= 0) numThreads = (int)std::min(8u, std::max(1u, std::thread::hardware_concurrency()));
    numThreads = (int)std::min<size_t>((size_t)numThreads, n);

    // Workers decode into GPU-visible RGBA8 buffers, at most `window` images ahead of the
    // consumer; this thread encodes the GPU conversion in camera order and commits so
    // the RGBA8 buffers are released as the GPU finishes with them.
    const size_t window = (size_t)numThreads + 2;
    std::vector<RGBA8Image> decoded(n);
    std::vector<std::exception_ptr> errors(n);
    std::vector<char> ready(n, 0);
    std::mutex m;
    std::condition_variable cv;
    size_t next = 0, consumed = 0;
    bool stop = false;

    auto worker = [&] {
        for (;;) {
            size_t i;
            {
                std::unique_lock<std::mutex> lock(m);
                cv.wait(lock, [&] { return stop || next >= n || next < consumed + window; });
                if (stop || next >= n) return;
                i = next++;
            }
            RGBA8Image img;
            std::exception_ptr err;
            try { img = imreadRGBA8(cameras[i].filePath); }
            catch (...) { err = std::current_exception(); }
            {
                std::lock_guard<std::mutex> lock(m);
                decoded[i] = std::move(img);
                errors[i] = err;
                ready[i] = 1;
            }
            cv.notify_all();
        }
    };

    std::vector<std::thread> threads;
    struct Joiner {
        std::vector<std::thread> &threads; std::mutex &m; std::condition_variable &cv; bool &stop;
        ~Joiner() {
            { std::lock_guard<std::mutex> lock(m); stop = true; }
            cv.notify_all();
            for (auto &t : threads) if (t.joinable()) t.join();
        }
    } joiner{threads, m, cv, stop};
    for (int t = 0; t < numThreads; t++) threads.emplace_back(worker);

    for (size_t i = 0; i < n; i++) {
        RGBA8Image img;
        {
            std::unique_lock<std::mutex> lock(m);
            cv.wait(lock, [&] { return ready[i] != 0; });
            if (errors[i]) std::rethrow_exception(errors[i]);
            img = std::move(decoded[i]);
        }
        cameras[i].setImage(img, downscaleFactor);
        img = RGBA8Image();   // the command buffer holds the buffer until the GPU is done
        msplat_commit();
        {
            std::lock_guard<std::mutex> lock(m);
            consumed = i + 1;
        }
        cv.notify_all();
    }
}

// ── Scale & center ──────────────────────────────────────────────────────────

void autoScaleAndCenter(InputData &data) {
    if (data.cameras.empty()) return;

    // Compute mean camera position
    float mean[3] = {};
    for (auto &cam : data.cameras) {
        mean[0] += cam.camToWorld[3];   // column 3 of row 0
        mean[1] += cam.camToWorld[7];   // column 3 of row 1
        mean[2] += cam.camToWorld[11];  // column 3 of row 2
    }
    int n = (int)data.cameras.size();
    mean[0] /= n; mean[1] /= n; mean[2] /= n;

    data.translation[0] = mean[0];
    data.translation[1] = mean[1];
    data.translation[2] = mean[2];

    // Center camera poses
    for (auto &cam : data.cameras) {
        cam.camToWorld[3]  -= mean[0];
        cam.camToWorld[7]  -= mean[1];
        cam.camToWorld[11] -= mean[2];
    }

    // Compute scale from max absolute camera position
    float maxAbs = 0;
    for (auto &cam : data.cameras) {
        maxAbs = std::max(maxAbs, std::abs(cam.camToWorld[3]));
        maxAbs = std::max(maxAbs, std::abs(cam.camToWorld[7]));
        maxAbs = std::max(maxAbs, std::abs(cam.camToWorld[11]));
    }
    data.scale = (maxAbs > 0) ? (1.0f / maxAbs) : 1.0f;

    // Apply scale to camera positions
    for (auto &cam : data.cameras) {
        cam.camToWorld[3]  *= data.scale;
        cam.camToWorld[7]  *= data.scale;
        cam.camToWorld[11] *= data.scale;
    }

    // Apply to point cloud
    for (int64_t i = 0; i < data.points.count; i++) {
        data.points.xyz[i*3+0] = (data.points.xyz[i*3+0] - mean[0]) * data.scale;
        data.points.xyz[i*3+1] = (data.points.xyz[i*3+1] - mean[1]) * data.scale;
        data.points.xyz[i*3+2] = (data.points.xyz[i*3+2] - mean[2]) * data.scale;
    }
}

// ── Train/test split ────────────────────────────────────────────────────────

std::tuple<std::vector<Camera>, Camera*> InputData::getCameras(bool validate, const std::string &valImage) {
    if (!validate) return {cameras, nullptr};

    // Find validation camera
    int valIdx = -1;
    if (valImage == "random") {
        std::mt19937 rng(42);
        valIdx = rng() % cameras.size();
    } else {
        for (int i = 0; i < (int)cameras.size(); i++) {
            if (cameras[i].filePath.find(valImage) != std::string::npos) { valIdx = i; break; }
        }
    }
    if (valIdx < 0) valIdx = 0;

    Camera *valCam = &cameras[valIdx];
    std::vector<Camera> train;
    for (int i = 0; i < (int)cameras.size(); i++)
        if (i != valIdx) train.push_back(cameras[i]);

    return {train, valCam};
}

std::tuple<std::vector<Camera>, std::vector<Camera>> InputData::splitTrainTest(int testEvery) {
    std::vector<Camera> train, test;
    for (int i = 0; i < (int)cameras.size(); i++) {
        if (i % testEvery == 0)
            test.push_back(cameras[i]);
        else
            train.push_back(cameras[i]);
    }
    return {train, test};
}

// ── Save cameras ────────────────────────────────────────────────────────────

void InputData::saveCameras(const std::string &filename, bool keepCrs) const {
    json arr = json::array();
    for (auto &cam : cameras) {
        json c;
        c["file_path"] = fs::path(cam.filePath).filename().string();
        c["width"] = cam.width;
        c["height"] = cam.height;
        c["fx"] = cam.fx; c["fy"] = cam.fy;
        c["cx"] = cam.cx; c["cy"] = cam.cy;

        // Extract rotation and translation from camToWorld
        float R[9], T[3];
        // Undo OpenGL flip (negate columns 1,2 back to OpenCV convention)
        R[0] =  cam.camToWorld[0]; R[1] = -cam.camToWorld[1]; R[2] = -cam.camToWorld[2];
        R[3] =  cam.camToWorld[4]; R[4] = -cam.camToWorld[5]; R[5] = -cam.camToWorld[6];
        R[6] =  cam.camToWorld[8]; R[7] = -cam.camToWorld[9]; R[8] = -cam.camToWorld[10];
        T[0] =  cam.camToWorld[3]; T[1] =  cam.camToWorld[7]; T[2] =  cam.camToWorld[11];

        if (keepCrs) {
            T[0] = T[0] / scale + translation[0];
            T[1] = T[1] / scale + translation[1];
            T[2] = T[2] / scale + translation[2];
        }

        c["rotation"] = {{R[0],R[1],R[2]},{R[3],R[4],R[5]},{R[6],R[7],R[8]}};
        c["translation"] = {T[0], T[1], T[2]};
        arr.push_back(c);
    }

    std::ofstream f(filename);
    f << arr.dump(2);
}

// ── Format dispatcher ───────────────────────────────────────────────────────

std::string detectDatasetFormat(const std::string &path) {
    fs::path root(path);
    // Nerfstudio: transforms.json
    if (fs::exists(root / "transforms.json")) return "nerfstudio";
    // COLMAP: cameras.bin (direct or in sparse/0/)
    if (fs::exists(root / "cameras.bin") || fs::exists(root / "sparse" / "0" / "cameras.bin")) return "colmap";
    // Polycam: keyframes/ directory or cameras.json
    if (fs::exists(root / "keyframes" / "corrected_cameras") || fs::exists(root / "cameras.json")) return "polycam";
    return "";
}

static InputData loadByFormat(const std::string &path, const std::string &colmapImagePath) {
    std::string format = detectDatasetFormat(path);
    if (format == "nerfstudio") return loaders::loadNerfstudio(path);
    if (format == "colmap") return loaders::loadColmap(path, colmapImagePath);
    if (format == "polycam") return loaders::loadPolycam(path);
    throw std::runtime_error("Unrecognized dataset format in: " + path +
        "\nSupported: COLMAP (cameras.bin), Nerfstudio (transforms.json), Polycam (keyframes/)");
}

InputData inputDataFromX(const std::string &path, const std::string &colmapImagePath) {
    InputData data = loadByFormat(path, colmapImagePath);
    data.rootDir = path;
    attachPriors(data);  // <dataset>/priors, when present
    return data;
}
