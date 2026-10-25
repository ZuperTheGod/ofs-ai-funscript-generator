#pragma once
#include "OFS_StateHandle.h"
#include "OFS_BinarySerialization.h"

#include <vector>
#include <string>
#include <array>
#include <cstdint>

enum class AudioClass : uint8_t
{
    Voice = 0,
    Moan = 1,
    Clap = 2,
    Thump = 3,
    Beat = 4,
    Count = 5
};

inline const char* AudioClassName(AudioClass c) noexcept
{
    switch (c) {
        case AudioClass::Voice: return "Voice";
        case AudioClass::Moan: return "Moan";
        case AudioClass::Clap: return "Clap";
        case AudioClass::Thump: return "Thump";
        case AudioClass::Beat: return "Beat";
        default: return "Unknown";
    }
}

// Voice/Moan come from the isolated vocals stem, Clap/Thump/Beat from the isolated
// instrumental stem - used to route a segment to the right waveform lane when separated
// audio is available.
inline bool IsVocalClass(AudioClass c) noexcept
{
    return c == AudioClass::Voice || c == AudioClass::Moan;
}

struct AudioSegment
{
    float startTime = 0.f;
    float endTime = 0.f;
    uint8_t classId = 0;

    inline AudioClass Class() const noexcept { return (AudioClass)classId; }
};

struct AudioClassSettings
{
    // peak-picking sensitivity for onset/transient detection (mean + Sensitivity * stddev)
    float Sensitivity = 1.6f;
    // low-pass cutoff in Hz separating "bass" energy from the rest
    float BassCutoffHz = 150.f;
    // high-pass cutoff in Hz used for the "presence" band (claps)
    float PresenceCutoffHz = 2500.f;
    // segments shorter than this get merged into a neighbour or dropped
    float MinSegmentSeconds = 0.12f;
    // RMS below this (0..1) counts as silence
    float SilenceThreshold = 0.02f;
};

struct AudioClassState
{
    static constexpr auto StateName = "AudioClassState";
    std::string Filename;
    std::vector<AudioSegment> Segments;
    AudioClassSettings Settings;

    std::array<ImColor, (size_t)AudioClass::Count> Colors = { {
        ImColor(IM_COL32(66, 135, 245, 255)),  // Voice - blue
        ImColor(IM_COL32(235, 64, 174, 255)),  // Moan - pink
        ImColor(IM_COL32(245, 166, 35, 255)),  // Clap - orange
        ImColor(IM_COL32(180, 90, 20, 255)),   // Thump - brown/orange
        ImColor(IM_COL32(140, 60, 235, 255)),  // Beat - purple
    } };

    inline static AudioClassState& State(uint32_t stateHandle) noexcept
    {
        return OFS_ProjectState<AudioClassState>(stateHandle).Get();
    }

    inline static AudioClassState& StaticStateSlow() noexcept
    {
        uint32_t handle = OFS_ProjectState<AudioClassState>::Register(StateName);
        return State(handle);
    }
};

REFL_TYPE(AudioSegment)
    REFL_FIELD(startTime)
    REFL_FIELD(endTime)
    REFL_FIELD(classId)
REFL_END

REFL_TYPE(AudioClassSettings)
    REFL_FIELD(Sensitivity)
    REFL_FIELD(BassCutoffHz)
    REFL_FIELD(PresenceCutoffHz)
    REFL_FIELD(MinSegmentSeconds)
    REFL_FIELD(SilenceThreshold)
REFL_END

REFL_TYPE(AudioClassState)
    REFL_FIELD(Filename)
    REFL_FIELD(Segments)
    REFL_FIELD(Settings)
    REFL_FIELD(Colors)
REFL_END
