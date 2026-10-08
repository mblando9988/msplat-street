#ifndef PREFLIGHT_H
#define PREFLIGHT_H

// Dataset checks and sizing reports for scripted and agent-driven runs.
//
// Reports are JSON text with stable keys and issue codes (docs/agents.md), so any front
// end can act on them; formatReport() renders one for people. Every image size in a
// report comes from planImageSizing() (image_sizing.hpp), the function the loader
// executes, so the stated sizes are the trained sizes.

#include <string>
#include <vector>
#include "input_data.hpp"

struct PreflightOptions {
    std::string colmapImagePath;   // COLMAP image directory override ("" = <dataset>/images)
    std::string priorDir;          // "" = <dataset>/priors when present
    std::string output;            // planned output file; "" skips the output checks
    float downscaleFactor = 1.f;
    int numDownscales = 2;
    int resolutionSchedule = 3000;
    int iterations = 30000;
    bool evalMode = false;
    int testEvery = 8;
    // Prior kinds the run trains with (depth weight, sky/fill weights, use_masks): none
    // found is an error, some cameras without them a warning
    bool requireDepth = false, requireSkyMasks = false, requireMasks = false;
    bool strict = false;           // warnings fail the check too
};

// Check a dataset before any heavy work: format and paths, every image's header (size,
// EXIF orientation, EXIF size), the sizing from file to training resolution and the
// resolution schedule, the priors, the point cloud and the output location. Dataset
// problems never throw; they become issues in the report ("ok": false on errors).
std::string preflightDataset(const std::string &path, const PreflightOptions &opts);

// The same report for loaded cameras: the sizes training actually uses.
struct ScheduleOptions {
    int numDownscales = 0;
    int resolutionSchedule = 3000;
    int iterations = 0;            // 0: no schedule section
};
std::string sizingReport(const std::vector<Camera> &train, const std::vector<Camera> &test,
                         const ScheduleOptions &schedule);

// Human-readable text for either report; verbose lists every image.
std::string formatReport(const std::string &reportJson, bool verbose = false);

#endif
