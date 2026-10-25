#include "OFS_CVTrackerSetup.h"
#include "OFS_CVTracker.h"
#include "state/states/CVTrackState.h"

#include "SDL_thread.h"
#include "SDL_mutex.h"

#include <vector>
#include <string>

namespace
{
    SDL_mutex* logMutex = nullptr;
    std::vector<std::string> logLines;
    bool installing = false;
    bool installFinished = false;
    bool installSucceeded = false;

    inline SDL_mutex* Mutex() noexcept
    {
        if (!logMutex) logMutex = SDL_CreateMutex();
        return logMutex;
    }

    void AppendLog(const std::string& line) noexcept
    {
        SDL_LockMutex(Mutex());
        logLines.push_back(line);
        SDL_UnlockMutex(Mutex());
    }

    int InstallThread(void*) noexcept
    {
        bool ok = OFS_CVTracker::Bootstrap([](const std::string& line) noexcept {
            AppendLog(line);
        });
        installSucceeded = ok;
        installFinished = true;
        return 0;
    }
}

void OFS_CVTrackerSetup::DrawModal() noexcept
{
    bool isOpen = true;
    ImGui::SetNextWindowSize(ImVec2(640.f, 420.f), ImGuiCond_FirstUseEver);
    if (ImGui::BeginPopupModal(WindowId, installing ? NULL : &isOpen, ImGuiWindowFlags_None)) {
        if (!installing && !installFinished) {
            ImGui::TextWrapped(
                "This installs a local Python environment with numpy and OpenCV, used to track "
                "a region of interest across the video via optical flow and turn its motion into "
                "funscript actions.");
            ImGui::Spacing();
            ImGui::TextWrapped(
                "One-time download, a few hundred MB - requires Python 3.10+ already installed "
                "and on PATH, and a working internet connection. Safe to close this and try "
                "again later.");
            ImGui::Spacing();
            if (ImGui::Button("Install now", ImVec2(-1.f, 0.f))) {
                installing = true;
                installFinished = false;
                SDL_LockMutex(Mutex());
                logLines.clear();
                SDL_UnlockMutex(Mutex());
                auto handle = SDL_CreateThread(InstallThread, "OFS_CVTrackerInstall", nullptr);
                SDL_DetachThread(handle);
            }
        }

        if (installing || installFinished) {
            ImGui::BeginChild("##InstallLog", ImVec2(-1.f, 300.f), true);
            SDL_LockMutex(Mutex());
            for (auto& line : logLines) {
                ImGui::TextUnformatted(line.c_str());
            }
            SDL_UnlockMutex(Mutex());
            if (installing) {
                ImGui::SetScrollHereY(1.f);
            }
            ImGui::EndChild();
        }

        if (installing && installFinished) {
            installing = false;
            if (installSucceeded) {
                auto& state = CVTrackerInstallState::StaticStateSlow();
                state.Installed = true;
            }
        }

        if (installFinished && !installing) {
            if (installSucceeded) {
                ImGui::TextColored(ImColor(IM_COL32(80, 220, 100, 255)).Value, "Done! Computer vision tracking is ready to use.");
            }
            else {
                ImGui::TextColored(ImColor(IM_COL32(255, 80, 80, 255)).Value, "Setup failed - see the log above for details.");
            }
            if (ImGui::Button("Close", ImVec2(-1.f, 0.f))) {
                isOpen = false;
            }
        }

        ImGui::EndPopup();
    }

    if (!isOpen) {
        installFinished = false;
        installSucceeded = false;
    }
}
