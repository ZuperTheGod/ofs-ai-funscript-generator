#include "OFS_CVTracker.h"
#include "OFS_Util.h"

#include "subprocess.h"
#include "nlohmann/json.hpp"

#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>

namespace
{
    std::string CVEnvDir() noexcept
    {
        return (Util::PathFromString(Util::Prefpath("cv_tracker_env"))).u8string();
    }

    // Runs one subprocess to completion, streaming its combined stdout/stderr line-by-line to
    // onProgress. Returns the process exit code, or -1 if it couldn't even be started. Same
    // pattern as OFS_AudioSeparator's RunAndStream - duplicated rather than shared since the two
    // callers live in separate optional features with no other coupling.
    int RunAndStream(const std::vector<std::string>& argv, const OFS_CVTracker::ProgressFn& onProgress) noexcept
    {
        std::vector<const char*> args;
        args.reserve(argv.size() + 1);
        for (auto& a : argv) args.push_back(a.c_str());
        args.push_back(nullptr);

        struct subprocess_s proc;
        int options = subprocess_option_combined_stdout_stderr
                    | subprocess_option_enable_async
                    | subprocess_option_inherit_environment
                    | subprocess_option_no_window;
        if (subprocess_create(args.data(), options, &proc) != 0) {
            if (onProgress) onProgress("FAILED to start process.");
            return -1;
        }

        std::array<char, 4096> buf;
        std::string lineBuf;
        while (subprocess_alive(&proc)) {
            unsigned n = subprocess_read_stdout(&proc, buf.data(), (unsigned)buf.size() - 1);
            if (n == 0) { SDL_Delay(10); continue; }
            buf[n] = '\0';
            lineBuf.append(buf.data(), n);

            size_t pos;
            while ((pos = lineBuf.find('\n')) != std::string::npos) {
                std::string line = lineBuf.substr(0, pos);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                lineBuf.erase(0, pos + 1);
                if (onProgress && !line.empty()) onProgress(line);
            }
        }
        unsigned n;
        while ((n = subprocess_read_stdout(&proc, buf.data(), (unsigned)buf.size() - 1)) > 0) {
            buf[n] = '\0';
            lineBuf.append(buf.data(), n);
        }
        if (!lineBuf.empty() && onProgress) onProgress(lineBuf);

        int returnCode = -1;
        subprocess_join(&proc, &returnCode);
        subprocess_destroy(&proc);
        return returnCode;
    }

    // Embedded rather than shipped as a data file: this is the entire "product", written out
    // fresh to Prefpath("tmp") on every Track()/Bootstrap() call so it can never go stale
    // relative to the OFS build that's running it (see OFS_AudioSeparator.h's similar reasoning
    // for the venv name it reuses). Farneback optical flow on a cropped/downscaled ROI patch,
    // tracking the ROI's own position (not just raw velocity) so the output is directly a
    // position curve rather than something that needs integrating downstream.
    constexpr const char* CVTrackerScript = R"PYSCRIPT(
import sys, json
import numpy as np
import cv2

def main():
    with open(sys.argv[1], "r") as f:
        cfg = json.load(f)

    cap = cv2.VideoCapture(cfg["video_path"])
    if not cap.isOpened():
        raise RuntimeError("could not open video: " + cfg["video_path"])

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise RuntimeError("could not read video dimensions")

    roi_w = max(4.0, cfg["roi_w"] * width)
    roi_h = max(4.0, cfg["roi_h"] * height)
    cx = cfg["roi_x"] * width + roi_w / 2.0
    cy = cfg["roi_y"] * height + roi_h / 2.0

    start_ms = float(cfg["start_ms"])
    end_ms = cfg["end_ms"]
    if end_ms is None or end_ms < 0:
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        end_ms = (frame_count / fps) * 1000.0 if frame_count > 0 else 1e18

    cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)

    analysis_fps = max(1.0, float(cfg["analysis_fps"]))
    step_ms = 1000.0 / analysis_fps

    target_px = max(8, int(cfg["image_pixel_target"]))
    mag_min = float(cfg["magnitude_min"])
    mag_max = float(cfg["magnitude_max"])
    ang_min = float(cfg["angle_min_deg"])
    ang_max = float(cfg["angle_max_deg"])

    prev_gray = None
    next_sample_ms = start_ms
    points = []

    def crop_gray(frame):
        x0 = int(max(0, cx - roi_w / 2.0))
        y0 = int(max(0, cy - roi_h / 2.0))
        x1 = int(min(width, x0 + roi_w))
        y1 = int(min(height, y0 + roi_h))
        patch = frame[y0:y1, x0:x1]
        if patch.size == 0:
            return None
        scale = target_px / max(patch.shape[0], patch.shape[1])
        work_w = max(2, int(patch.shape[1] * scale))
        work_h = max(2, int(patch.shape[0] * scale))
        small = cv2.resize(patch, (work_w, work_h), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), scale

    while True:
        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_ms > end_ms:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if pos_ms + 1.0 < next_sample_ms:
            continue
        next_sample_ms = pos_ms + step_ms

        result = crop_gray(frame)
        if result is None:
            continue
        gray, scale = result

        if prev_gray is not None and prev_gray.shape == gray.shape:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                cfg["pyr_scale"], cfg["levels"], cfg["winsize"],
                cfg["iterations"], cfg["poly_n"], cfg["poly_sigma"], 0)
            fx = flow[..., 0]
            fy = flow[..., 1]
            mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)

            def angle_band(lo, hi):
                lo = lo % 360.0
                hi = hi % 360.0
                if lo <= hi:
                    return (ang >= lo) & (ang <= hi)
                return (ang >= lo) | (ang <= hi)

            mask = (mag >= mag_min) & (mag <= mag_max) & (angle_band(ang_min, ang_max) | angle_band(ang_min + 180.0, ang_max + 180.0))

            if np.any(mask):
                dx = float(np.mean(fx[mask]))
                dy = float(np.mean(fy[mask]))
            else:
                dx = 0.0
                dy = 0.0

            # same scope as the initial cx/cy assignment above (not inside crop_gray), so a
            # plain reassignment is visible to crop_gray's closure on the next iteration
            cx = cx + dx / scale
            cy = cy + dy / scale
            half_w = roi_w / 2.0
            half_h = roi_h / 2.0
            cx = min(max(cx, half_w), width - half_w)
            cy = min(max(cy, half_h), height - half_h)

        prev_gray = gray
        points.append({"t": pos_ms, "y": cy})

    cap.release()

    if len(points) < 2:
        print(json.dumps({"success": False, "error": "not enough tracked points (ROI may have left the frame, or the clip is too short)"}))
        return

    ys = [p["y"] for p in points]
    lo, hi = min(ys), max(ys)
    span = hi - lo
    invert = bool(cfg.get("invert", False))
    out_points = []
    for p in points:
        if span > 1e-6:
            v = (p["y"] - lo) / span * 100.0
        else:
            v = 50.0
        if invert:
            v = 100.0 - v
        out_points.append({"t": p["t"], "v": v})

    print(json.dumps({"success": True, "width": width, "height": height, "points": out_points}))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)
)PYSCRIPT";
}

std::string OFS_CVTracker::VenvPythonPath() noexcept
{
#if defined(WIN32)
    return CVEnvDir() + "\\Scripts\\python.exe";
#else
    return CVEnvDir() + "/bin/python";
#endif
}

bool OFS_CVTracker::IsInstalled() noexcept
{
    return Util::FileExists(VenvPythonPath());
}

bool OFS_CVTracker::Bootstrap(const ProgressFn& onProgress) noexcept
{
    auto envDir = CVEnvDir();
    auto venvPython = VenvPythonPath();

    if (onProgress) onProgress("Creating virtual environment at " + envDir + " ...");
    int rc = RunAndStream({ "python", "-m", "venv", envDir }, onProgress);
    if (rc != 0 || !Util::FileExists(venvPython)) {
        if (onProgress) onProgress("FAILED to create the virtual environment. Is Python installed and on PATH?");
        return false;
    }

    if (onProgress) onProgress("Upgrading pip ...");
    rc = RunAndStream({ venvPython, "-m", "pip", "install", "--upgrade", "pip" }, onProgress);
    if (rc != 0) {
        if (onProgress) onProgress("FAILED to upgrade pip.");
        return false;
    }

    if (onProgress) onProgress("Installing numpy + opencv-python-headless ...");
    rc = RunAndStream({ venvPython, "-m", "pip", "install", "numpy", "opencv-python-headless" }, onProgress);
    if (rc != 0) {
        if (onProgress) onProgress("FAILED to install numpy/opencv.");
        return false;
    }

    if (onProgress) onProgress("Computer vision tracking set up successfully.");
    return true;
}

bool OFS_CVTracker::Track(
    const std::string& pythonPath,
    const std::string& videoPath,
    const CVTrackerSettings& settings,
    std::vector<CVTrackPoint>& outPoints,
    const ProgressFn& onProgress) noexcept
{
    outPoints.clear();

    auto tmpDir = Util::Prefpath("tmp");
    if (!Util::CreateDirectories(Util::PathFromString(tmpDir))) {
        if (onProgress) onProgress("FAILED to create tmp directory.");
        return false;
    }

    auto scriptPath = (Util::PathFromString(tmpDir) / "cv_tracker.py").u8string();
    if (Util::WriteFile(scriptPath.c_str(), CVTrackerScript, strlen(CVTrackerScript)) == 0) {
        if (onProgress) onProgress("FAILED to write tracker script.");
        return false;
    }

    nlohmann::json cfg;
    cfg["video_path"] = videoPath;
    cfg["roi_x"] = settings.RoiX;
    cfg["roi_y"] = settings.RoiY;
    cfg["roi_w"] = settings.RoiW;
    cfg["roi_h"] = settings.RoiH;
    cfg["start_ms"] = settings.StartTimeMs;
    cfg["end_ms"] = settings.EndTimeMs;
    cfg["analysis_fps"] = settings.AnalysisFps;
    cfg["magnitude_min"] = settings.MagnitudeMin;
    cfg["magnitude_max"] = settings.MagnitudeMax;
    cfg["angle_min_deg"] = settings.AngleMinDeg;
    cfg["angle_max_deg"] = settings.AngleMaxDeg;
    cfg["image_pixel_target"] = settings.ImagePixelTarget;
    cfg["pyr_scale"] = settings.PyrScale;
    cfg["levels"] = settings.Levels;
    cfg["winsize"] = settings.WinSize;
    cfg["iterations"] = settings.Iterations;
    cfg["poly_n"] = settings.PolyN;
    cfg["poly_sigma"] = settings.PolySigma;
    cfg["invert"] = settings.Invert;

    auto configText = cfg.dump();
    auto configPath = (Util::PathFromString(tmpDir) / "cv_tracker_config.json").u8string();
    if (Util::WriteFile(configPath.c_str(), configText.data(), configText.size()) == 0) {
        if (onProgress) onProgress("FAILED to write tracker config.");
        return false;
    }

    std::string lastLine;
    int rc = RunAndStream({ pythonPath, scriptPath, configPath },
        [&](const std::string& line) noexcept {
            lastLine = line;
            if (onProgress) onProgress(line);
        });

    if (rc != 0 && lastLine.empty()) {
        if (onProgress) onProgress("FAILED to run tracker script.");
        return false;
    }

    bool parseOk = false;
    auto result = Util::ParseJson(lastLine, &parseOk);
    if (!parseOk || !result.value("success", false)) {
        if (onProgress) onProgress("Tracker failed: " + (parseOk ? result.value("error", "unknown error") : std::string("could not parse tracker output")));
        return false;
    }

    outPoints.reserve(result["points"].size());
    for (auto& p : result["points"]) {
        CVTrackPoint pt;
        pt.timeMs = p.value("t", 0.f);
        pt.value = p.value("v", 0.f);
        outPoints.push_back(pt);
    }

    return !outPoints.empty();
}
