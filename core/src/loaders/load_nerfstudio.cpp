#include "loaders.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;
using json = nlohmann::json;

// Try adding common image extensions if file doesn't exist
static std::string resolveImagePath(const std::string &path) {
    if (fs::exists(path)) return path;
    for (auto ext : {".png", ".jpg", ".jpeg", ".JPG"})
        if (fs::exists(path + ext)) return path + ext;
    return path;
}

// Relative frame paths ("images/x.png", "./images/x.png", "../x.png") are relative to
// the transforms.json directory, per the Nerfstudio format. Paths that only exist
// relative to the working directory still resolve, as they did before 1.2.
static std::string resolveFramePath(const fs::path &root, const std::string &fp) {
    fs::path p(fp);
    if (p.is_absolute()) return resolveImagePath(fp);
    std::string underRoot = resolveImagePath((root / p).lexically_normal().string());
    if (fs::exists(underRoot)) return underRoot;
    std::string underCwd = resolveImagePath(fp);
    return fs::exists(underCwd) ? underCwd : underRoot;
}

// Prior file keys written by msplat-prior. Absent keys leave the path empty.
static std::string priorPath(const json &frame, const char *key, const fs::path &root) {
    if (!frame.contains(key) || !frame[key].is_string()) return "";
    fs::path p(frame[key].get<std::string>());
    return (p.is_absolute() ? p : (root / p).lexically_normal()).string();
}

InputData loaders::loadNerfstudio(const std::string &projectRoot) {
    std::ifstream f((fs::path(projectRoot) / "transforms.json").string());
    json j = json::parse(f);

    // Global defaults (overridden per-frame if present)
    int gW = j.value("w", 0), gH = j.value("h", 0);
    float gFx = j.value("fl_x", 0.0f), gFy = j.value("fl_y", 0.0f);
    float gCx = j.value("cx", 0.0f), gCy = j.value("cy", 0.0f);
    float gK1 = j.value("k1", 0.0f), gK2 = j.value("k2", 0.0f), gK3 = j.value("k3", 0.0f);
    float gP1 = j.value("p1", 0.0f), gP2 = j.value("p2", 0.0f);

    InputData data;

    for (auto &frame : j["frames"]) {
        Camera cam;
        cam.width  = frame.value("w", gW);  cam.height = frame.value("h", gH);
        cam.fx = frame.value("fl_x", gFx);  cam.fy = frame.value("fl_y", gFy);
        cam.cx = frame.value("cx", gCx);     cam.cy = frame.value("cy", gCy);
        cam.k1 = frame.value("k1", gK1);     cam.k2 = frame.value("k2", gK2);
        cam.k3 = frame.value("k3", gK3);
        cam.p1 = frame.value("p1", gP1);     cam.p2 = frame.value("p2", gP2);

        fs::path root(projectRoot);
        std::string fp = frame["file_path"].get<std::string>();
        cam.filePath = resolveFramePath(root, fp);
        cam.imageName = fs::path(fp).lexically_normal().string();

        // Optional priors (msplat-prior); mask_path follows the Nerfstudio convention
        // (zero = ignore pixel).
        cam.priorDepthPath = priorPath(frame, "prior_depth_path", root);
        cam.priorConfidencePath = priorPath(frame, "prior_confidence_path", root);
        cam.priorSkyPath = priorPath(frame, "prior_sky_mask_path", root);
        cam.priorMaskPath = priorPath(frame, "mask_path", root);

        // transform_matrix is 4x4 c2w (OpenGL convention)
        auto &tm = frame["transform_matrix"];
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
                cam.camToWorld[r*4+c] = tm[r][c].get<float>();

        data.cameras.push_back(cam);
    }

    std::sort(data.cameras.begin(), data.cameras.end(),
        [](const Camera &a, const Camera &b) { return a.filePath < b.filePath; });

    // Point cloud
    if (j.contains("ply_file_path")) {
        std::string p = j["ply_file_path"].get<std::string>();
        if (p[0] != '/') p = (fs::path(projectRoot) / p).string();
        if (fs::exists(p)) data.points = readPly(p);
    }
    if (data.points.count == 0) {
        for (auto p : {"sparse/0/points3D.ply", "points3D.ply"}) {
            auto path = (fs::path(projectRoot) / p).string();
            if (fs::exists(path)) { data.points = readPly(path); break; }
        }
    }

    autoScaleAndCenter(data);
    return data;
}
