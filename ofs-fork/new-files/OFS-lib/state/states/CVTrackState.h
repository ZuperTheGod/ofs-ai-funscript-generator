#pragma once
#include "OFS_StateHandle.h"
#include "OFS_BinarySerialization.h"

#include <string>
#include <cstdint>

struct CVTrackerSettings
{
    // ROI in normalized [0,1] video-pixel coordinates, top-left origin - as produced by
    // ROISelectedEvent.
    float RoiX = 0.f, RoiY = 0.f, RoiW = 0.1f, RoiH = 0.1f;

    // Analysis range in milliseconds. EndTimeMs < 0 means "to the end of the video".
    float StartTimeMs = 0.f;
    float EndTimeMs = -1.f;

    // How many analysis samples per second of video - the ROI patch is diffed against the
    // previous *analyzed* frame, not every decoded frame, to keep long videos tractable.
    float AnalysisFps = 30.f;

    // Optical-flow vectors are only counted toward the ROI's tracked motion if their magnitude
    // (in working-resolution pixels/frame) and angle (degrees, measured like Scripter3's) both
    // fall in range - this is what rejects sensor/background noise vs. the tracked motion.
    float MagnitudeMin = 1.f;
    float MagnitudeMax = 60.f;
    float AngleMinDeg = 45.f;
    float AngleMaxDeg = 135.f;

    // The ROI crop is downscaled so its longer side is about this many pixels before running
    // optical flow on it - keeps per-frame cost roughly constant regardless of ROI/video size.
    int32_t ImagePixelTarget = 64;

    // cv2.calcOpticalFlowFarneback parameters - same names/defaults ballpark as Scripter3 exposes.
    float PyrScale = 0.5f;
    int32_t Levels = 3;
    int32_t WinSize = 15;
    int32_t Iterations = 3;
    int32_t PolyN = 5;
    float PolySigma = 1.2f;

    bool Invert = false;

    // Simplification applied to the raw per-frame curve when turning it into funscript actions:
    // a point is kept only at a local extremum whose swing exceeds MinAmplitude and which is at
    // least MinPointGapMs after the previously kept point.
    float MinAmplitude = 5.f;
    float MinPointGapMs = 60.f;
};

struct CVTrackPoint
{
    float timeMs = 0.f;
    float value = 0.f; // 0..100
};

// Project-scoped: the ROI/settings picked here are meaningful only for the video they were
// picked on, same tier as AudioClassState.
struct CVTrackState
{
    static constexpr auto StateName = "CVTrackState";

    std::string Filename;
    bool HasROI = false;
    CVTrackerSettings Settings;

    inline static CVTrackState& State(uint32_t stateHandle) noexcept
    {
        return OFS_ProjectState<CVTrackState>(stateHandle).Get();
    }

    inline static CVTrackState& StaticStateSlow() noexcept
    {
        uint32_t handle = OFS_ProjectState<CVTrackState>::Register(StateName);
        return State(handle);
    }
};

// App-scoped: whether the machine-wide tracker venv is installed is a property of this
// computer, not of any particular funscript project - same tier as AudioSeparatorState.
struct CVTrackerInstallState
{
    static constexpr auto StateName = "CVTrackerInstallState";

    bool Installed = false;
    // empty = use the default bundled venv under Prefpath("cv_tracker_env")
    std::string CustomPythonPath;

    inline static CVTrackerInstallState& State(uint32_t stateHandle) noexcept
    {
        return OFS_AppState<CVTrackerInstallState>(stateHandle).Get();
    }

    inline static CVTrackerInstallState& StaticStateSlow() noexcept
    {
        uint32_t handle = OFS_AppState<CVTrackerInstallState>::Register(StateName);
        return State(handle);
    }
};

REFL_TYPE(CVTrackerSettings)
    REFL_FIELD(RoiX)
    REFL_FIELD(RoiY)
    REFL_FIELD(RoiW)
    REFL_FIELD(RoiH)
    REFL_FIELD(StartTimeMs)
    REFL_FIELD(EndTimeMs)
    REFL_FIELD(AnalysisFps)
    REFL_FIELD(MagnitudeMin)
    REFL_FIELD(MagnitudeMax)
    REFL_FIELD(AngleMinDeg)
    REFL_FIELD(AngleMaxDeg)
    REFL_FIELD(ImagePixelTarget)
    REFL_FIELD(PyrScale)
    REFL_FIELD(Levels)
    REFL_FIELD(WinSize)
    REFL_FIELD(Iterations)
    REFL_FIELD(PolyN)
    REFL_FIELD(PolySigma)
    REFL_FIELD(Invert)
    REFL_FIELD(MinAmplitude)
    REFL_FIELD(MinPointGapMs)
REFL_END

REFL_TYPE(CVTrackState)
    REFL_FIELD(Filename)
    REFL_FIELD(HasROI)
    REFL_FIELD(Settings)
REFL_END

REFL_TYPE(CVTrackerInstallState)
    REFL_FIELD(Installed)
    REFL_FIELD(CustomPythonPath)
REFL_END
