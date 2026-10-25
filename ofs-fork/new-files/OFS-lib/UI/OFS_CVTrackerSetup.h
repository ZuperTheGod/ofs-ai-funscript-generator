#pragma once
#include "imgui.h"

// One-time setup modal for the CV tracker (venv + numpy/opencv install). Mirrors
// OFS_AudioSeparatorSetup - see that header for why this lives in OFS-lib alongside the feature
// it belongs to rather than in src/.
struct OFS_CVTrackerSetup
{
    static constexpr const char* WindowId = "###CV_TRACKER_SETUP";
    static void DrawModal() noexcept;
};
