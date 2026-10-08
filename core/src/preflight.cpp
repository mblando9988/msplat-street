// Dataset checks and sizing reports (see preflight.hpp). Issue codes are part of the
// report format: add new ones freely, but do not rename or repurpose existing ones.

#include "preflight.hpp"
#include "image_sizing.hpp"
#include "loaders.hpp"
#include "priors.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <map>
#include <set>
#include <unistd.h>

#ifndef APP_VERSION
#define APP_VERSION "unknown"
#endif

using json = nlohmann::ordered_json;
namespace fs = std::filesystem;

namespace {

const char *const ERR = "error";
const char *const WARN = "warning";
const char *const INFO = "info";

std::string strf(const char *format, ...) {
    va_list args, copy;
    va_start(args, format);
    va_copy(copy, args);
    const int n = vsnprintf(nullptr, 0, format, copy);
    va_end(copy);
    std::string out(n > 0 ? (size_t)n : 0, '\0');
    if (n > 0) vsnprintf(out.data(), (size_t)n + 1, format, args);
    va_end(args);
    return out;
}

std::string wxh(int w, int h) { return std::to_string(w) + "x" + std::to_string(h); }
json size2(int w, int h) { return json::array({w, h}); }
double round6(double v) { return std::round(v * 1e6) / 1e6; }

int severityRank(const std::string &s) { return s == ERR ? 0 : s == WARN ? 1 : 2; }

// Findings, merged by (severity, code, message): the same finding on many images is
// one entry with the list of images.
class IssueList {
  public:
    void add(const char *severity, const char *code, const std::string &message, const std::string &image = "") {
        std::string key = std::string(severity) + '\x1f' + code + '\x1f' + message;
        auto it = index_.find(key);
        if (it == index_.end()) {
            it = index_.emplace(key, items_.size()).first;
            items_.push_back({severity, code, message, {}});
        }
        if (!image.empty()) items_[it->second].images.push_back(image);
    }
    int count(const char *severity) const {
        int n = 0;
        for (auto &i : items_) n += i.severity == severity;
        return n;
    }
    json toJson() const {
        std::vector<const Item*> sorted;
        for (auto &i : items_) sorted.push_back(&i);
        std::stable_sort(sorted.begin(), sorted.end(), [](const Item *a, const Item *b) {
            return severityRank(a->severity) < severityRank(b->severity);
        });
        json out = json::array();
        for (auto *i : sorted)
            out.push_back({{"severity", i->severity}, {"code", i->code}, {"message", i->message},
                           {"count", i->images.size()}, {"images", i->images}});
        return out;
    }

  private:
    struct Item {
        std::string severity, code, message;
        std::vector<std::string> images;
    };
    std::vector<Item> items_;
    std::map<std::string, size_t> index_;
};

const char *orientationName(int o) {
    switch (o) {
        case 2: return "mirrored horizontally";
        case 3: return "rotated 180 degrees";
        case 4: return "mirrored vertically";
        case 5: return "mirrored and rotated 90 degrees";
        case 6: return "rotated 90 degrees clockwise";
        case 7: return "mirrored and rotated 270 degrees";
        case 8: return "rotated 90 degrees counter-clockwise";
        default: return "upright";
    }
}

// Findings about one image's sizing; shared by the preflight and the loaded report.
void checkSizing(const ImageSizing &s, const std::string &name, IssueList &issues, std::vector<std::string> &codes) {
    auto add = [&](const char *severity, const char *code, const std::string &message) {
        issues.add(severity, code, message, name);
        codes.push_back(code);
    };
    const std::string file = wxh(s.fileWidth, s.fileHeight);

    if (s.metaWidth > 0 && s.metaHeight > 0 && (s.fileWidth != s.metaWidth || s.fileHeight != s.metaHeight)) {
        const std::string meta = wxh(s.metaWidth, s.metaHeight);
        if (std::fabs(s.metaScaleX - s.metaScaleY) > 0.01f * std::max(s.metaScaleX, s.metaScaleY))
            add(ERR, "META_ASPECT_MISMATCH", strf(
                "camera metadata says %s but the file is %s: the intrinsics would be rescaled x%.4f horizontally "
                "and x%.4f vertically, distorting the camera (image rotated by EXIF orientation, or cropped?)",
                meta.c_str(), file.c_str(), s.metaScaleX, s.metaScaleY));
        else
            add(WARN, "META_SIZE_MISMATCH", strf(
                "intrinsics were given for %s but the file is %s: rescaled x%.4f to match "
                "(poses estimated on a different copy of the images?)",
                meta.c_str(), file.c_str(), s.metaScaleX));
    }

    if (s.orientation > 1 && s.orientation <= 8) {
        const bool swapped = s.orientation >= 5 && s.fileWidth != s.fileHeight &&
                             s.metaWidth == s.fileHeight && s.metaHeight == s.fileWidth;
        if (swapped)
            add(ERR, "ORIENTATION_SWAPPED", strf(
                "EXIF orientation %d (%s): the camera metadata matches the rotated %s image, but msplat trains on "
                "the %s pixels as stored. Bake the rotation into the pixels, or reset the tag "
                "(exiftool -Orientation=1 -n) and estimate poses again",
                s.orientation, orientationName(s.orientation), wxh(s.fileHeight, s.fileWidth).c_str(), file.c_str()));
        else
            add(WARN, "ORIENTATION_NOT_APPLIED", strf(
                "EXIF orientation %d (%s) is ignored: training uses the %s pixels as stored, as COLMAP does. "
                "Fine if the poses were estimated the same way",
                s.orientation, orientationName(s.orientation), file.c_str()));
    }

    if (s.exifWidth > 0 && s.exifHeight > 0) {
        const bool same = (s.exifWidth == s.fileWidth && s.exifHeight == s.fileHeight) ||
                          (s.exifWidth == s.fileHeight && s.exifHeight == s.fileWidth);
        if (!same) {
            const float r = (float)std::max(s.fileWidth, s.fileHeight) / (float)std::max(s.exifWidth, s.exifHeight);
            const std::string exif = wxh(s.exifWidth, s.exifHeight);
            if (r > 1.01f)
                add(WARN, "UPSCALED_FILE", strf(
                    "EXIF records %s but the file is %s: the photo was upscaled x%.2f before msplat. Upscaling adds "
                    "no detail, only time and memory; train on the original, or set downscale_factor %.2f",
                    exif.c_str(), file.c_str(), r, r));
            else if (r < 0.99f)
                add(INFO, "RESIZED_FILE", strf("EXIF records %s; the file is %s (resized x%.3f before msplat)",
                                               exif.c_str(), file.c_str(), r));
        }
    }

    if (s.downscaleFactor > 1.f) {
        const float exactW = s.fileWidth / s.downscaleFactor, exactH = s.fileHeight / s.downscaleFactor;
        if (exactW != std::floor(exactW) || exactH != std::floor(exactH))
            add(INFO, "DOWNSCALE_ROUNDED", strf(
                "%s / %g = %.2fx%.2f, trained at %s (x%.5f, x%.5f); the intrinsics use these exact ratios",
                file.c_str(), s.downscaleFactor, exactW, exactH, wxh(s.scaledWidth, s.scaledHeight).c_str(),
                (double)s.scaledWidth / s.fileWidth, (double)s.scaledHeight / s.fileHeight));
    }

    if (s.undistorted) {
        const double kept = 100.0 * s.width * s.height / ((double)s.scaledWidth * s.scaledHeight);
        add(INFO, "UNDISTORTED", strf(
            "lens distortion removed on load: cropped to %s at (%d, %d) of the %s image (%.0f%% of the pixels kept)",
            wxh(s.width, s.height).c_str(), s.cropX, s.cropY, wxh(s.scaledWidth, s.scaledHeight).c_str(), kept));
    }

    const std::string train = wxh(s.width, s.height);
    if (!(s.fx > 0.f) || !(s.fy > 0.f)) {
        add(ERR, "INVALID_INTRINSICS", strf("focal length must be positive (fx=%.2f, fy=%.2f at %s)",
                                            s.fx, s.fy, train.c_str()));
    } else {
        if (s.cx < 0.f || s.cx > s.width || s.cy < 0.f || s.cy > s.height) {
            const float kx = 2.f * s.cx / s.width, ky = 2.f * s.cy / s.height;
            std::string hint;
            if (kx > 1.1f && std::fabs(kx - ky) < 0.05f * std::max(kx, ky))
                hint = strf(": they fit a %.2fx larger image (%s), so the metadata size is missing or wrong", kx,
                            wxh((int)std::lround(s.width * kx), (int)std::lround(s.height * ky)).c_str());
            add(ERR, "PRINCIPAL_POINT_OUTSIDE", strf("principal point (%.1f, %.1f) lies outside the %s training image%s",
                                                     s.cx, s.cy, train.c_str(), hint.c_str()));
        } else if (std::fabs(s.cx / s.width - 0.5f) > 0.15f || std::fabs(s.cy / s.height - 0.5f) > 0.15f) {
            add(WARN, "PRINCIPAL_POINT_OFF_CENTER", strf(
                "principal point at %.0f%%, %.0f%% of the %s image, far from the center "
                "(intrinsics for another size or crop?)",
                100.f * s.cx / s.width, 100.f * s.cy / s.height, train.c_str()));
        }
        const double hfov = 2.0 * std::atan(0.5 * s.width / s.fx) * 180.0 / M_PI;
        if (hfov < 8.0 || hfov > 150.0)
            add(WARN, "UNUSUAL_FIELD_OF_VIEW", strf(
                "horizontal field of view %.1f degrees (fx=%.1f for %d px): check the focal length and the "
                "metadata size", hfov, s.fx, s.width));
    }

    if (std::min(s.width, s.height) < 128)
        add(WARN, "SMALL_IMAGE", strf("trains at only %s px", train.c_str()));
    else if ((double)s.width * s.height > 24e6)
        add(INFO, "LARGE_IMAGE", strf("trains at %s (%.0f MP): consider downscale_factor for speed and memory",
                                      train.c_str(), (double)s.width * s.height / 1e6));
}

json sizingJson(const ImageSizing &s, const std::vector<ScheduleLevel> &levels) {
    json j;
    j["file_size"] = size2(s.fileWidth, s.fileHeight);
    j["orientation"] = s.orientation;
    j["exif_size"] = s.exifWidth > 0 && s.exifHeight > 0 ? size2(s.exifWidth, s.exifHeight) : json();
    j["metadata_size"] = s.metaWidth > 0 && s.metaHeight > 0 ? size2(s.metaWidth, s.metaHeight) : json();
    j["intrinsics_rescale"] = json::array({round6(s.metaScaleX), round6(s.metaScaleY)});
    j["downscale_factor"] = round6(s.downscaleFactor);
    j["downscaled_size"] = size2(s.scaledWidth, s.scaledHeight);
    j["pixel_scale"] = json::array({round6((double)s.scaledWidth / s.fileWidth),
                                    round6((double)s.scaledHeight / s.fileHeight)});
    j["undistort_crop"] = s.undistorted
        ? json{{"x", s.cropX}, {"y", s.cropY}, {"size", size2(s.width, s.height)}} : json();
    j["train_size"] = size2(s.width, s.height);
    j["intrinsics"] = {{"fx", round6(s.fx)}, {"fy", round6(s.fy)}, {"cx", round6(s.cx)}, {"cy", round6(s.cy)}};
    if (!levels.empty()) {
        json sizes = json::array();
        for (auto &l : levels) sizes.push_back(size2(levelSize(s.width, l.factor), levelSize(s.height, l.factor)));
        j["schedule_sizes"] = sizes;
    }
    return j;
}

struct Entry {
    std::string name, file, split;
    bool planned = false;              // sizing available
    ImageSizing sizing;
    std::vector<std::string> codes;    // issues of this image
    json priors;                       // preflight only
};

std::string groupKey(const ImageSizing &s) {
    return strf("%d %d %d %d %d %d %d %d %d %d %d", s.fileWidth, s.fileHeight, s.metaWidth, s.metaHeight,
                s.scaledWidth, s.scaledHeight, (int)s.undistorted, s.cropX, s.cropY, s.width, s.height);
}

// Size groups, schedule and dataset-level sizing findings
void addSizingSections(json &report, const std::vector<Entry> &entries, const std::vector<ScheduleLevel> &levels,
                       int numDownscales, int resolutionSchedule, int iterations, IssueList &issues) {
    std::map<std::string, size_t> groupIndex;
    std::vector<std::vector<const Entry*>> groups;
    for (auto &e : entries) {
        if (!e.planned) continue;
        auto key = groupKey(e.sizing);
        auto it = groupIndex.find(key);
        if (it == groupIndex.end()) {
            it = groupIndex.emplace(key, groups.size()).first;
            groups.emplace_back();
        }
        groups[it->second].push_back(&e);
    }
    std::stable_sort(groups.begin(), groups.end(), [](auto &a, auto &b) { return a.size() > b.size(); });

    const int finalFactor = levels.empty() ? 1 : levels.back().factor;
    json gj = json::array();
    std::set<std::pair<int, int>> trainSizes;
    for (auto &g : groups) {
        const ImageSizing &s = g.front()->sizing;
        trainSizes.insert({s.width, s.height});
        json j;
        j["count"] = g.size();
        json examples = json::array();
        for (size_t i = 0; i < g.size() && i < 3; i++) examples.push_back(g[i]->name);
        j["examples"] = examples;
        json sj = sizingJson(s, levels);
        sj.erase("intrinsics");
        for (auto it = sj.begin(); it != sj.end(); ++it) j[it.key()] = it.value();
        j["final_size"] = size2(levelSize(s.width, finalFactor), levelSize(s.height, finalFactor));
        gj.push_back(j);

        if (!levels.empty()) {
            const int coarse = levels.front().factor;
            const int cw = levelSize(s.width, coarse), ch = levelSize(s.height, coarse);
            if (coarse > 1 && std::min(cw, ch) < 32)
                for (auto *e : g)
                    issues.add(WARN, "SCHEDULE_LEVEL_TINY", strf(
                        "the first %d steps train at 1/%d resolution, only %s px: lower num_downscales",
                        levels.front().lastStep, coarse, wxh(cw, ch).c_str()), e->name);
        }
    }
    report["sizes"] = {{"distinct_train_sizes", trainSizes.size()}, {"groups", gj}};

    if (trainSizes.size() > 1) {
        std::string list;
        for (auto &g : groups) {
            if (!list.empty()) list += ", ";
            list += strf("%zu x %s", g.size(), wxh(g.front()->sizing.width, g.front()->sizing.height).c_str());
        }
        issues.add(WARN, "MIXED_RESOLUTIONS", strf(
            "%zu different training sizes (%s): supported, but often a sign of mixed sources", trainSizes.size(),
            list.c_str()));
    }

    if (iterations > 0) {
        json lj = json::array();
        for (auto &l : levels)
            lj.push_back({{"factor", l.factor}, {"steps", json::array({l.firstStep, l.lastStep})}});
        report["schedule"] = {{"num_downscales", numDownscales}, {"resolution_schedule", resolutionSchedule},
                              {"iterations", iterations}, {"final_factor", finalFactor}, {"levels", lj}};
        if (finalFactor > 1 && !groups.empty()) {
            const ImageSizing &s = groups.front().front()->sizing;
            issues.add(WARN, "EVAL_AT_REDUCED_RESOLUTION", strf(
                "training ends at 1/%d resolution (iterations %d < num_downscales %d x resolution_schedule %d): "
                "the last steps, renders and evaluation use %s instead of %s",
                finalFactor, iterations, numDownscales, resolutionSchedule,
                wxh(levelSize(s.width, finalFactor), levelSize(s.height, finalFactor)).c_str(),
                wxh(s.width, s.height).c_str()));
        }
    }
}

void finishReport(json &report, const std::vector<Entry> &entries, const std::vector<ScheduleLevel> &levels,
                  const IssueList &issues, bool strict) {
    const int errors = issues.count(ERR), warnings = issues.count(WARN), infos = issues.count(INFO);
    report["ok"] = errors == 0 && (!strict || warnings == 0);
    report["strict"] = strict;
    report["counts"] = {{"error", errors}, {"warning", warnings}, {"info", infos}};
    report["issues"] = issues.toJson();
    json images = json::array();
    for (auto &e : entries) {
        json j;
        j["name"] = e.name;
        j["file"] = e.file;
        j["split"] = e.split;
        if (e.planned) {
            json sj = sizingJson(e.sizing, levels);
            for (auto it = sj.begin(); it != sj.end(); ++it) j[it.key()] = it.value();
        }
        if (!e.priors.is_null()) j["priors"] = e.priors;
        j["issues"] = e.codes;
        images.push_back(j);
    }
    report["images"] = images;
}

std::string displayName(const Camera &cam) {
    return cam.imageName.empty() ? fs::path(cam.filePath).filename().string() : cam.imageName;
}

bool isImageFile(const fs::path &p) {
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".heic" || ext == ".tif" || ext == ".tiff";
}

// Size of one prior map, checked against the image it supervises
json checkPriorMap(const char *kind, const std::string &path, bool npy, const ImageSizing &s,
                   const std::string &name, IssueList &issues, std::vector<std::string> &codes, int dims[2]) {
    dims[0] = dims[1] = 0;
    auto add = [&](const char *severity, const char *code, const std::string &message) {
        issues.add(severity, code, message, name);
        codes.push_back(code);
    };
    if (path.empty()) return json();
    if (!fs::exists(path)) {
        add(ERR, "PRIOR_MISSING", strf("%s prior not found: %s", kind, path.c_str()));
        return {{"path", path}, {"size", nullptr}};
    }
    int pw = 0, ph = 0;
    try {
        if (npy) {
            std::vector<int64_t> shape = readNpyShape(path);
            if (!(shape.size() == 2 || (shape.size() == 3 && shape[2] == 1)))
                throw std::runtime_error(strf("must be a 2-D (H, W) array, has %zu dimensions", shape.size()));
            ph = (int)shape[0];
            pw = (int)shape[1];
        } else {
            ImageFileInfo info = probeImage(path);
            pw = info.width;
            ph = info.height;
        }
    } catch (const std::exception &e) {
        add(ERR, "PRIOR_INVALID", strf("%s prior %s: %s", kind, path.c_str(), e.what()));
        return {{"path", path}, {"size", nullptr}};
    }
    dims[0] = pw;
    dims[1] = ph;
    if (pw > 0 && ph > 0 && s.width > 0 && s.height > 0) {
        const double pa = (double)pw / ph, ia = (double)s.width / s.height;
        if (std::fabs(pa / ia - 1.0) > 0.02) {
            add(ERR, "PRIOR_ASPECT_MISMATCH", strf(
                "%s prior is %s but the image trains at %s (aspect %.3f vs %.3f): priors are sampled by normalized "
                "position, so it would be misaligned (cropped or padded when it was computed?)",
                kind, wxh(pw, ph).c_str(), wxh(s.width, s.height).c_str(), pa, ia));
        } else {
            const double up = (double)s.width / pw;
            if (up > 4.0)
                add(INFO, "PRIOR_UPSAMPLED", strf("%s prior %s is sampled at %s (%.1fx coarser than the image)",
                                                  kind, wxh(pw, ph).c_str(), wxh(s.width, s.height).c_str(), up));
            if (std::max(pw, ph) > kMaxPriorDim)
                add(INFO, "PRIOR_DOWNSAMPLED", strf("%s prior %s is downsampled to at most %d px on load",
                                                    kind, wxh(pw, ph).c_str(), kMaxPriorDim));
        }
    }
    return {{"path", path}, {"size", size2(pw, ph)}};
}

}  // namespace

std::string preflightDataset(const std::string &path, const PreflightOptions &o) {
    json report;
    report["kind"] = "preflight";
    report["msplat_version"] = APP_VERSION;
    IssueList issues;
    std::vector<Entry> entries;

    const fs::path root = fs::absolute(fs::path(path)).lexically_normal();
    json options = {
        {"downscale_factor", round6(o.downscaleFactor)}, {"num_downscales", o.numDownscales},
        {"resolution_schedule", o.resolutionSchedule}, {"iterations", o.iterations},
        {"eval_mode", o.evalMode}, {"test_every", o.testEvery},
        {"prior_dir", o.priorDir}, {"colmap_image_path", o.colmapImagePath}, {"output", o.output},
        {"require_depth", o.requireDepth}, {"require_sky_masks", o.requireSkyMasks},
        {"require_masks", o.requireMasks}};
    report["dataset"] = {{"path", root.string()}};
    report["options"] = options;

    // Options
    if (o.downscaleFactor < 0.f)
        issues.add(ERR, "BAD_OPTION", strf("downscale_factor must be positive, got %g", o.downscaleFactor));
    else if (o.downscaleFactor > 0.f && o.downscaleFactor < 1.f)
        issues.add(WARN, "DOWNSCALE_IGNORED", strf(
            "downscale_factor %g is below 1: images load at their file resolution (msplat never upsamples)",
            o.downscaleFactor));
    if (o.iterations < 1) issues.add(ERR, "BAD_OPTION", strf("iterations must be at least 1, got %d", o.iterations));
    if (o.numDownscales < 0)
        issues.add(ERR, "BAD_OPTION", strf("num_downscales must not be negative, got %d", o.numDownscales));
    if (o.numDownscales > 0 && o.resolutionSchedule <= 0)
        issues.add(INFO, "SCHEDULE_DISABLED", "resolution_schedule <= 0: training uses full resolution throughout");
    if (o.evalMode && o.testEvery <= 0)
        issues.add(ERR, "BAD_OPTION", strf("test_every must be at least 1 in eval mode, got %d", o.testEvery));

    auto done = [&]() {
        std::vector<ScheduleLevel> none;
        finishReport(report, entries, none, issues, o.strict);
        return report.dump(2);
    };

    if (!fs::exists(root)) {
        issues.add(ERR, "DATASET_NOT_FOUND", "dataset directory not found: " + root.string());
        return done();
    }
    if (!fs::is_directory(root)) {
        issues.add(ERR, "DATASET_NOT_A_DIRECTORY", "dataset path is not a directory: " + root.string());
        return done();
    }

    const std::string format = detectDatasetFormat(root.string());
    report["dataset"]["format"] = format.empty() ? json() : json(format);
    if (format.empty()) {
        int nImages = 0;
        for (auto dir : {root / "images", root})
            if (fs::is_directory(dir)) {
                for (auto &f : fs::directory_iterator(dir)) nImages += f.is_regular_file() && isImageFile(f.path());
                if (nImages) break;
            }
        issues.add(ERR, "NO_POSES", nImages > 0
            ? strf("%d images but no camera poses: run COLMAP (writes sparse/0/cameras.bin) or msplat-prior "
                   "(writes transforms.json) first", nImages)
            : "no camera poses and no images: expected transforms.json (Nerfstudio), sparse/0/cameras.bin "
              "(COLMAP) or a Polycam export, with the photos in images/");
        return done();
    }

    InputData data;
    try {
        data = inputDataFromX(root.string(), o.colmapImagePath);
    } catch (const std::exception &e) {
        issues.add(ERR, "DATASET_UNREADABLE", strf("cannot read the %s dataset: %s", format.c_str(), e.what()));
        return done();
    }
    if (!o.priorDir.empty()) {
        try {
            attachPriors(data, o.priorDir);
        } catch (const std::exception &e) {
            issues.add(ERR, "PRIOR_DIR_NOT_FOUND", e.what());
        }
    }

    const int n = (int)data.cameras.size();
    std::set<std::string> imageDirs;
    for (auto &cam : data.cameras) imageDirs.insert(fs::path(cam.filePath).parent_path().string());
    int nTest = 0;
    if (o.evalMode && o.testEvery > 0)
        for (int i = 0; i < n; i++) nTest += i % o.testEvery == 0;
    report["dataset"]["image_dirs"] = std::vector<std::string>(imageDirs.begin(), imageDirs.end());
    report["dataset"]["cameras"] = n;
    report["dataset"]["train"] = n - nTest;
    report["dataset"]["test"] = nTest;
    report["dataset"]["points"] = data.points.count;
    if (n == 0) issues.add(ERR, "NO_CAMERAS", "the dataset lists no cameras");
    if (data.points.count == 0)
        issues.add(ERR, "NO_POINTS", "no initial point cloud (COLMAP points3D.bin, points3D.ply, or ply_file_path "
                                     "in transforms.json): training starts from these points");
    if (o.evalMode && n > 0 && n - nTest == 0)
        issues.add(ERR, "NO_TRAIN_CAMERAS", strf("test_every %d leaves no training cameras", o.testEvery));

    // Duplicates
    std::map<std::string, int> fileCount;
    for (auto &cam : data.cameras) fileCount[fs::absolute(cam.filePath).lexically_normal().string()]++;
    for (auto &[file, count] : fileCount)
        if (count > 1) issues.add(WARN, "DUPLICATE_IMAGE", strf("%d cameras use the same image file", count), file);

    const std::vector<ScheduleLevel> levels =
        resolutionLevels(o.numDownscales, o.resolutionSchedule, std::max(o.iterations, 0));

    // Images: header-only probe, then the loader's own sizing plan
    int withPriors = 0, withDepth = 0, withConf = 0, withSky = 0, withMask = 0, trainCams = 0;
    int trainWithDepth = 0, trainWithSky = 0, trainWithMask = 0;
    std::set<std::string> usedPriorFiles;
    for (int i = 0; i < n; i++) {
        const Camera &cam = data.cameras[i];
        Entry e;
        e.name = displayName(cam);
        e.file = fs::absolute(cam.filePath).lexically_normal().string();
        e.split = (o.evalMode && o.testEvery > 0 && i % o.testEvery == 0) ? "test" : "train";
        const bool isTrain = e.split == "train";
        trainCams += isTrain;

        if (!fs::exists(e.file)) {
            issues.add(ERR, "IMAGE_MISSING", "image file not found", e.file);
            e.codes.push_back("IMAGE_MISSING");
            entries.push_back(std::move(e));
            continue;
        }
        ImageFileInfo info;
        try {
            info = probeImage(e.file);
        } catch (const std::exception &ex) {
            issues.add(ERR, "IMAGE_UNREADABLE", ex.what(), e.name);
            e.codes.push_back("IMAGE_UNREADABLE");
            entries.push_back(std::move(e));
            continue;
        }
        // Conditions the loader refuses, reported by code; the sizes are still planned
        // (without the priors) so the report shows them
        Camera planned = cam;
        if (cam.hasDistortion() && cam.hasPriorFiles()) {
            issues.add(ERR, "PRIORS_NEED_UNDISTORTED_IMAGES",
                       "priors are attached but the camera has lens distortion, which the loader removes by "
                       "remapping and cropping: undistort the dataset first (e.g. colmap image_undistorter) and "
                       "compute the priors on the undistorted images", e.name);
            e.codes.push_back("PRIORS_NEED_UNDISTORTED_IMAGES");
            planned.priorDepthPath.clear();
            planned.priorSkyPath.clear();
            planned.priorMaskPath.clear();
        }
        if (o.downscaleFactor > 1.f &&
            ((int)(info.width / o.downscaleFactor) < 1 || (int)(info.height / o.downscaleFactor) < 1)) {
            issues.add(ERR, "DOWNSCALE_TOO_LARGE", strf("downscale_factor %g leaves no pixels of a %s image",
                                                        o.downscaleFactor, wxh(info.width, info.height).c_str()),
                       e.name);
            e.codes.push_back("DOWNSCALE_TOO_LARGE");
            entries.push_back(std::move(e));
            continue;
        }
        try {
            e.sizing = planImageSizing(planned, info.width, info.height, o.downscaleFactor);
            e.sizing.orientation = info.orientation;
            e.sizing.exifWidth = info.exifWidth;
            e.sizing.exifHeight = info.exifHeight;
            e.planned = true;
        } catch (const std::exception &ex) {
            issues.add(ERR, "LOAD_PLAN_FAILED", ex.what(), e.name);
            e.codes.push_back("LOAD_PLAN_FAILED");
        }
        if (e.planned) checkSizing(e.sizing, e.name, issues, e.codes);

        if (cam.hasPriorFiles() || !cam.priorConfidencePath.empty()) {
            withPriors++;
            int dDepth[2], dConf[2], dSky[2], dMask[2];
            json pj;
            pj["depth"] = checkPriorMap("depth", cam.priorDepthPath, true, e.sizing, e.name, issues, e.codes, dDepth);
            pj["confidence"] = checkPriorMap("confidence", cam.priorConfidencePath, true, e.sizing, e.name, issues,
                                             e.codes, dConf);
            pj["sky"] = checkPriorMap("sky mask", cam.priorSkyPath, false, e.sizing, e.name, issues, e.codes, dSky);
            pj["mask"] = checkPriorMap("mask", cam.priorMaskPath, false, e.sizing, e.name, issues, e.codes, dMask);
            if (dDepth[0] > 0 && dConf[0] > 0 && (dDepth[0] != dConf[0] || dDepth[1] != dConf[1])) {
                issues.add(WARN, "PRIOR_SHAPE_MISMATCH", strf(
                    "confidence %s differs from depth %s: resampled to the depth grid on load",
                    wxh(dConf[0], dConf[1]).c_str(), wxh(dDepth[0], dDepth[1]).c_str()), e.name);
                e.codes.push_back("PRIOR_SHAPE_MISMATCH");
            }
            e.priors = pj;
            withDepth += !cam.priorDepthPath.empty();
            withConf += !cam.priorConfidencePath.empty();
            withSky += !cam.priorSkyPath.empty();
            withMask += !cam.priorMaskPath.empty();
            if (isTrain) {
                trainWithDepth += !cam.priorDepthPath.empty();
                trainWithSky += !cam.priorSkyPath.empty();
                trainWithMask += !cam.priorMaskPath.empty();
            }
            for (auto *p : {&cam.priorDepthPath, &cam.priorConfidencePath, &cam.priorSkyPath, &cam.priorMaskPath})
                if (!p->empty()) usedPriorFiles.insert(fs::absolute(*p).lexically_normal().string());
        }
        entries.push_back(std::move(e));
    }

    addSizingSections(report, entries, levels, o.numDownscales, o.resolutionSchedule, o.iterations, issues);

    // Priors: coverage and files no camera picked up
    const fs::path priorDir = o.priorDir.empty() ? root / "priors" : fs::absolute(o.priorDir).lexically_normal();
    json pr = {{"dir", fs::is_directory(priorDir) ? json(priorDir.string()) : json()},
               {"cameras_with_priors", withPriors}, {"depth", withDepth}, {"confidence", withConf},
               {"sky", withSky}, {"mask", withMask}};
    auto requireKind = [&](bool required, int have, const char *kind, const char *code, const char *why) {
        if (!required) return;
        if (have == 0)
            issues.add(ERR, code, strf("the run uses %s (%s) but no training camera has them: run msplat-prior or "
                                       "pass prior_dir", kind, why));
        else if (have < trainCams)
            issues.add(WARN, "PRIORS_PARTIAL", strf("only %d of %d training cameras have %s; the rest train "
                                                    "without them", have, trainCams, kind));
    };
    requireKind(o.requireDepth, trainWithDepth, "depth priors", "NO_DEPTH_PRIORS", "depth_weight > 0");
    requireKind(o.requireSkyMasks, trainWithSky, "sky masks", "NO_SKY_MASKS", "sky_alpha_weight or fill_weight > 0");
    requireKind(o.requireMasks, trainWithMask, "masks", "NO_MASKS", "use_masks");
    if (fs::is_directory(priorDir)) {
        std::vector<std::string> orphans;
        for (auto sub : {"depth", "confidence", "sky", "mask"}) {
            fs::path d = priorDir / sub;
            if (!fs::is_directory(d)) continue;
            for (auto &f : fs::recursive_directory_iterator(d)) {
                if (!f.is_regular_file()) continue;
                std::string ext = f.path().extension().string();
                if (ext != ".npy" && ext != ".png" && ext != ".jpg") continue;
                std::string abs = fs::absolute(f.path()).lexically_normal().string();
                if (!usedPriorFiles.count(abs)) orphans.push_back(abs);
            }
        }
        pr["unmatched_files"] = orphans.size();
        if (!orphans.empty()) {
            std::string expect;
            if (n > 0) {
                fs::path rel(displayName(data.cameras[0]));
                expect = rel.has_parent_path()
                    ? strf(" (for %s: depth/%s.npy or depth/%s.npy)", rel.string().c_str(),
                           (rel.parent_path() / rel.stem()).string().c_str(), rel.stem().string().c_str())
                    : strf(" (for %s: depth/%s.npy)", rel.string().c_str(), rel.stem().string().c_str());
            }
            for (size_t k = 0; k < orphans.size(); k++)
                issues.add(WARN, "UNMATCHED_PRIORS", strf(
                    "%zu prior file%s in %s match%s no image: name them after the image path without its "
                    "extension%s", orphans.size(), orphans.size() == 1 ? "" : "s", priorDir.string().c_str(),
                    orphans.size() == 1 ? "es" : "", expect.c_str()), orphans[k]);
        }
    }
    report["priors"] = pr;

    // Output location
    if (!o.output.empty()) {
        const fs::path out = fs::absolute(fs::path(o.output)).lexically_normal();
        const fs::path dir = out.parent_path();
        fs::path existing = dir;
        while (!existing.empty() && !fs::exists(existing) && existing != existing.parent_path())
            existing = existing.parent_path();
        const bool dirExists = fs::is_directory(dir);
        const bool writable = fs::is_directory(existing) && access(existing.c_str(), W_OK) == 0;
        report["output"] = {{"path", out.string()}, {"directory_exists", dirExists}, {"writable", writable},
                            {"exists", fs::exists(out)}};
        if (fs::is_directory(out))
            issues.add(ERR, "OUTPUT_IS_DIRECTORY", "output must be a file path, not a directory: " + out.string());
        else if (!writable)
            issues.add(ERR, "OUTPUT_NOT_WRITABLE", "cannot write to " + (dirExists ? dir : existing).string());
        else if (!dirExists)
            issues.add(INFO, "OUTPUT_DIR_CREATED", "output directory will be created: " + dir.string());
        if (fs::is_regular_file(out)) issues.add(INFO, "OUTPUT_EXISTS", "will be overwritten: " + out.string());
        const std::string ext = out.extension().string();
        if (ext != ".ply" && ext != ".splat")
            issues.add(WARN, "OUTPUT_EXTENSION", "output should end in .ply or .splat: " + out.string());
    }

    finishReport(report, entries, levels, issues, o.strict);
    return report.dump(2);
}

std::string sizingReport(const std::vector<Camera> &train, const std::vector<Camera> &test,
                         const ScheduleOptions &schedule) {
    json report;
    report["kind"] = "loaded";
    report["msplat_version"] = APP_VERSION;
    report["dataset"] = {{"cameras", train.size() + test.size()}, {"train", train.size()}, {"test", test.size()}};
    IssueList issues;
    std::vector<Entry> entries;
    auto collect = [&](const std::vector<Camera> &cams, const char *split) {
        for (auto &cam : cams) {
            Entry e;
            e.name = displayName(cam);
            e.file = fs::absolute(cam.filePath).lexically_normal().string();
            e.split = split;
            e.planned = cam.sizing.width > 0;
            e.sizing = cam.sizing;
            if (e.planned) checkSizing(e.sizing, e.name, issues, e.codes);
            entries.push_back(std::move(e));
        }
    };
    collect(train, "train");
    collect(test, "test");
    const std::vector<ScheduleLevel> levels =
        resolutionLevels(schedule.numDownscales, schedule.resolutionSchedule, schedule.iterations);
    addSizingSections(report, entries, levels, schedule.numDownscales, schedule.resolutionSchedule,
                      schedule.iterations, issues);
    finishReport(report, entries, levels, issues, false);
    return report.dump(2);
}

// ── Human-readable rendering ────────────────────────────────────────────────

namespace {

std::string sizeText(const json &j) {
    if (!j.is_array() || j.size() != 2) return "?";
    return wxh(j[0].get<int>(), j[1].get<int>());
}

// "4946x3286 file -> intrinsics from 4946x3286 -> /4: 1236x821 (x0.24990, x0.24985) -> train 1236x821"
std::string sizingChain(const json &g) {
    std::string out = sizeText(g["file_size"]) + " file";
    if (g["metadata_size"].is_array() && g["metadata_size"] != g["file_size"]) {
        const json &r = g["intrinsics_rescale"];
        const double rx = r[0].get<double>(), ry = r[1].get<double>();
        out += std::fabs(rx - ry) > 1e-4
            ? strf(" (intrinsics given for %s, rescaled x%.4f horizontally, x%.4f vertically)",
                   sizeText(g["metadata_size"]).c_str(), rx, ry)
            : strf(" (intrinsics given for %s, rescaled x%.4f)", sizeText(g["metadata_size"]).c_str(), rx);
    }
    if (g["downscale_factor"].get<double>() > 1.0) {
        const json &ps = g["pixel_scale"];
        out += strf(" -> downscale /%g -> %s (x%.5f, x%.5f)", g["downscale_factor"].get<double>(),
                    sizeText(g["downscaled_size"]).c_str(), ps[0].get<double>(), ps[1].get<double>());
    }
    if (g["undistort_crop"].is_object())
        out += strf(" -> undistorted, crop at (%d, %d)", g["undistort_crop"]["x"].get<int>(),
                    g["undistort_crop"]["y"].get<int>());
    out += " -> trains at " + sizeText(g["train_size"]);
    return out;
}

}  // namespace

std::string formatReport(const std::string &reportJson, bool verbose) {
    json r = json::parse(reportJson);
    std::string out;
    const std::string kind = r.value("kind", "");
    const json &ds = r["dataset"];
    const std::string root = ds.contains("path") ? ds["path"].get<std::string>() + "/" : "";

    if (kind == "preflight") {
        out += "msplat preflight: " + ds.value("path", "") + "\n";
        if (ds.contains("cameras"))
            out += strf("  %s dataset, %d cameras (%d train, %d test), %lld initial points\n",
                        ds.value("format", "").c_str(), ds.value("cameras", 0), ds.value("train", 0),
                        ds.value("test", 0), (long long)ds.value("points", (int64_t)0));
        if (ds.contains("image_dirs"))
            for (auto &d : ds["image_dirs"]) out += "  images: " + d.get<std::string>() + "\n";
    } else {
        out += strf("Loaded images: %d train, %d test\n", ds.value("train", 0), ds.value("test", 0));
    }

    if (r.contains("sizes")) {
        out += "\nImage sizes (file -> training):\n";
        if (r["sizes"]["groups"].empty()) out += "  (no image could be sized)\n";
        const json &sched = r.contains("schedule") ? r["schedule"] : json();
        for (auto &g : r["sizes"]["groups"]) {
            out += strf("  %4d x %s\n", g["count"].get<int>(), sizingChain(g).c_str());
            if (sched.is_object() && g.contains("schedule_sizes")) {
                std::string line;
                const json &levels = sched["levels"];
                for (size_t k = 0; k < levels.size(); k++) {
                    if (!line.empty()) line += " -> ";
                    line += strf("%s (1/%d, steps %d-%d)", sizeText(g["schedule_sizes"][k]).c_str(),
                                 levels[k]["factor"].get<int>(), levels[k]["steps"][0].get<int>(),
                                 levels[k]["steps"][1].get<int>());
                }
                out += "         schedule: " + line + "\n";
                out += "         renders and evaluation after training: " + sizeText(g["final_size"]) + "\n";
            }
            std::string ex;
            for (auto &e : g["examples"]) ex += (ex.empty() ? "" : ", ") + e.get<std::string>();
            out += "         e.g. " + ex + "\n";
        }
    }

    if (r.contains("priors")) {
        const json &p = r["priors"];
        if (p.value("cameras_with_priors", 0) > 0)
            out += strf("\nPriors: %d cameras (depth %d, confidence %d, sky %d, mask %d)%s\n",
                        p.value("cameras_with_priors", 0), p.value("depth", 0), p.value("confidence", 0),
                        p.value("sky", 0), p.value("mask", 0),
                        p["dir"].is_string() ? (" from " + p["dir"].get<std::string>()).c_str() : "");
        else
            out += "\nPriors: none\n";
    }
    if (r.contains("output") && r["output"].is_object())
        out += "Output: " + r["output"].value("path", "") +
               (r["output"].value("directory_exists", true) ? "" : " (directory will be created)") + "\n";

    if (verbose && r.contains("images")) {
        out += "\nImages:\n";
        for (auto &im : r["images"]) {
            std::string line = "  " + im.value("name", "") + " [" + im.value("split", "") + "]";
            if (im.contains("train_size")) {
                const json &k = im["intrinsics"];
                line += strf(": %s -> %s  fx=%.2f fy=%.2f cx=%.2f cy=%.2f", sizeText(im["file_size"]).c_str(),
                             sizeText(im["train_size"]).c_str(), k["fx"].get<double>(), k["fy"].get<double>(),
                             k["cx"].get<double>(), k["cy"].get<double>());
            }
            if (!im["issues"].empty()) {
                std::string codes;
                for (auto &c : im["issues"]) codes += (codes.empty() ? "" : ",") + c.get<std::string>();
                line += "  [" + codes + "]";
            }
            out += line + "\n";
        }
    }

    const json &counts = r["counts"];
    auto plural = [](int n, const char *word) { return strf("%d %s%s", n, word, n == 1 ? "" : "s"); };
    out += strf("\nIssues: %s, %s, %d info\n", plural(counts.value("error", 0), "error").c_str(),
                plural(counts.value("warning", 0), "warning").c_str(), counts.value("info", 0));
    for (auto &i : r["issues"]) {
        const std::string sev = i.value("severity", "");
        out += strf("  %-8s %s", sev.c_str(), i.value("code", "").c_str());
        const int count = i.value("count", 0);
        if (count > 1) out += strf(" (%d images)", count);
        out += ": " + i.value("message", "") + "\n";
        if (count > 0) {
            std::string ex;
            int k = 0;
            for (auto &im : i["images"]) {
                if (k++ == 3) { ex += ", ..."; break; }
                std::string name = im.get<std::string>();
                if (!root.empty() && name.compare(0, root.size(), root) == 0) name = name.substr(root.size());
                ex += (ex.empty() ? "" : ", ") + name;
            }
            out += "           " + ex + "\n";
        }
    }
    if (kind == "preflight")
        out += r.value("ok", false) ? "\nOK: ready to train\n"
                                    : (r.value("strict", false) && counts.value("error", 0) == 0
                                           ? "\nFAILED (strict): fix or accept the warnings\n"
                                           : "\nFAILED: fix the errors before training\n");
    return out;
}
