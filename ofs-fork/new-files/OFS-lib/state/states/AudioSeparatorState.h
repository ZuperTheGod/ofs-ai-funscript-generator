#pragma once
#include "OFS_StateHandle.h"

#include <string>

// App-scoped (not project-scoped) - whether the machine-wide separator venv is installed is a
// property of this computer, not of any particular funscript project, same tier as the
// keybindings/overlay state.
struct AudioSeparatorState
{
    static constexpr auto StateName = "AudioSeparatorState";

    bool Installed = false;
    // empty = use the default bundled venv under Prefpath("demucs_env")
    std::string CustomPythonPath;

    inline static AudioSeparatorState& State(uint32_t stateHandle) noexcept
    {
        return OFS_AppState<AudioSeparatorState>(stateHandle).Get();
    }

    inline static AudioSeparatorState& StaticStateSlow() noexcept
    {
        uint32_t handle = OFS_AppState<AudioSeparatorState>::Register(StateName);
        return State(handle);
    }
};

REFL_TYPE(AudioSeparatorState)
    REFL_FIELD(Installed)
    REFL_FIELD(CustomPythonPath)
REFL_END
