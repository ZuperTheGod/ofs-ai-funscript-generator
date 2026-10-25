#pragma once

#include "state/states/CVTrackState.h"

#include <string>
#include <vector>
#include <functional>
#include <cstdint>

// Bootstraps and drives a local Python + OpenCV install to track a user-picked region of
// interest (ROI) across a video using dense (Farneback) optical flow, producing a raw
// time/position curve - the same idea as Scripter3's "Computer Vision ROI Tracker"
// (https://playground.handyfeeling.com/scripter3), reimplemented from scratch since that tool
// is closed-source. Entirely optional/additive, same venv-per-feature pattern as
// OFS_AudioSeparator: nothing here is required for the rest of OFS to work.
// CVTrackerSettings/CVTrackPoint live in state/states/CVTrackState.h alongside the project
// state that persists them, mirroring AudioClassSettings/AudioClassState.
class OFS_CVTracker
{
    public:
    using ProgressFn = std::function<void(const std::string& line)>;

    // Path to the bundled venv's python interpreter (not necessarily existing yet).
    static std::string VenvPythonPath() noexcept;

    // Whether the bundled venv + opencv-python install already exists.
    static bool IsInstalled() noexcept;

    // Creates the venv and installs numpy + opencv-python-headless into it. Safe to call from a
    // background thread - blocks for the duration of the subprocesses. Streams combined
    // stdout/stderr line-by-line to onProgress. Returns false and stops early on the first
    // failing step.
    static bool Bootstrap(const ProgressFn& onProgress) noexcept;

    // Runs the ROI tracker on videoPath with the given settings, filling outPoints on success.
    // Blocking - intended to be called from a background thread.
    static bool Track(
        const std::string& pythonPath,
        const std::string& videoPath,
        const CVTrackerSettings& settings,
        std::vector<CVTrackPoint>& outPoints,
        const ProgressFn& onProgress) noexcept;
};
