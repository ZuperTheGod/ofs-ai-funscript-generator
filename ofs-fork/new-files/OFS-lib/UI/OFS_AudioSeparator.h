#pragma once

#include <string>
#include <functional>

// Bootstraps and drives a local audio-separator (https://github.com/nomadkaraoke/python-audio-separator)
// install to split a video's audio into vocals vs. everything else, so the DSP classifier in
// OFS_AudioClassifier can be run separately on each clean stem instead of the tangled full mix.
// Uses the package's default model, a BS-RoFormer checkpoint (bs_roformer_ep_317_sdr_12.9755) -
// a transformer-based separator that clearly beats Demucs on vocal-isolation SDR benchmarks.
// Entirely optional/additive: if this isn't installed, callers should fall back to classifying
// the original mixed audio directly.
class OFS_AudioSeparator
{
    public:
    using ProgressFn = std::function<void(const std::string& line)>;

    // Path to the bundled venv's python interpreter (not necessarily existing yet).
    static std::string VenvPythonPath() noexcept;

    // Path to the bundled venv's audio-separator console script (not necessarily existing yet).
    static std::string VenvSeparatorExePath() noexcept;

    // Whether the bundled venv + audio-separator install already exists.
    static bool IsInstalled() noexcept;

    // Creates the venv and installs GPU-capable PyTorch + audio-separator into it. Safe to call
    // from a background thread - this blocks for the duration of several subprocesses (a
    // multi-minute download). Streams each subprocess's combined stdout/stderr line-by-line to
    // onProgress. Returns false and stops early on the first failing step.
    static bool Bootstrap(const ProgressFn& onProgress) noexcept;

    // Runs audio-separator (default BS-RoFormer model) on inputAudioPath, writing into outDir.
    // On success fills in outVocalsPath/outInstrumentalPath with the resulting flac paths.
    static bool Separate(
        const std::string& pythonPath,
        const std::string& inputAudioPath,
        const std::string& outDir,
        std::string& outVocalsPath,
        std::string& outInstrumentalPath) noexcept;
};
