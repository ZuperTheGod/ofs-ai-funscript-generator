#pragma once

#include <string>
#include <vector>
#include <atomic>

#include "state/states/AudioClassState.h"

// Decodes a video's audio track and splits it into timestamped segments of
// Voice / Moan / Clap / Thump / Beat using DSP heuristics (onset detection,
// bandpass envelopes, autocorrelation-based pitch & tempo).
// No ML model / network access involved - everything runs locally.
class OFS_AudioClassifier
{
    public:
    // Runs on whatever thread calls it - intended to be called from a background thread,
    // mirroring OFS_Waveform::GenerateAndLoadFlac.
    static bool ClassifyAudio(
        const std::string& ffmpegPath,
        const std::string& videoPath,
        const std::string& tmpFlacPath,
        const AudioClassSettings& settings,
        std::vector<AudioSegment>& outSegments) noexcept;
};
