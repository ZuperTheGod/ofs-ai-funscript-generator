#pragma once
#include "imgui.h"

// One-time setup modal for the audio separator (venv + GPU PyTorch + audio-separator/BS-RoFormer
// install). Mirrors OFS_DownloadFfmpeg's modal pattern in spirit, but lives in OFS-lib
// alongside OFS_AudioSeparator itself rather than in src/, since the app layer (src/) links
// against OFS-lib and not the other way around - keeping this here avoids a layering violation
// with the menu code in OFS_ScriptTimeline.cpp that opens it.
struct OFS_AudioSeparatorSetup
{
    static constexpr const char* WindowId = "###AUDIO_SEPARATOR_SETUP";
    static void DrawModal() noexcept;
};
