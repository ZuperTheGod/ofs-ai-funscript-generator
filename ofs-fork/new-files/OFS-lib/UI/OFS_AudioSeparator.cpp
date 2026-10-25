#include "OFS_AudioSeparator.h"
#include "OFS_Util.h"

#include "subprocess.h"
#include "SDL_timer.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>

namespace
{
    // kept as "demucs_env" so machines that already bootstrapped it (multi-GB PyTorch download)
    // don't have to redo that just because the separator package installed into it changed.
    std::string SeparatorEnvDir() noexcept
    {
        return (Util::PathFromString(Util::Prefpath("demucs_env"))).u8string();
    }

    // Finds the first file directly inside dir whose name contains needle (e.g. "(Vocals)").
    // audio-separator names output files "<input>_(StemName)_model_<model>.<ext>" - matching on
    // the stem marker rather than the full name keeps us decoupled from the exact model string.
    std::string FindOutputContaining(const std::string& dir, const char* needle) noexcept
    {
        std::error_code ec;
        for (auto& entry : std::filesystem::directory_iterator(Util::PathFromString(dir), ec)) {
            if (!entry.is_regular_file()) continue;
            auto name = entry.path().filename().u8string();
            if (name.find(needle) != std::string::npos) {
                return entry.path().u8string();
            }
        }
        return {};
    }

    // Runs one subprocess to completion, streaming its combined stdout/stderr line-by-line to
    // onProgress. Returns the process exit code, or -1 if it couldn't even be started.
    int RunAndStream(const std::vector<std::string>& argv, const OFS_AudioSeparator::ProgressFn& onProgress) noexcept
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
        // drain whatever is left after the process exited
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
}

std::string OFS_AudioSeparator::VenvPythonPath() noexcept
{
#if defined(WIN32)
    return SeparatorEnvDir() + "\\Scripts\\python.exe";
#else
    return SeparatorEnvDir() + "/bin/python";
#endif
}

std::string OFS_AudioSeparator::VenvSeparatorExePath() noexcept
{
#if defined(WIN32)
    return SeparatorEnvDir() + "\\Scripts\\audio-separator.exe";
#else
    return SeparatorEnvDir() + "/bin/audio-separator";
#endif
}

bool OFS_AudioSeparator::IsInstalled() noexcept
{
    return Util::FileExists(VenvPythonPath()) && Util::FileExists(VenvSeparatorExePath());
}

bool OFS_AudioSeparator::Bootstrap(const ProgressFn& onProgress) noexcept
{
    auto envDir = SeparatorEnvDir();
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

    if (onProgress) onProgress("Installing PyTorch (this is the big download, several GB, may take a while) ...");
    rc = RunAndStream({ venvPython, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu126" }, onProgress);
    if (rc != 0) {
        if (onProgress) onProgress("FAILED to install PyTorch.");
        return false;
    }

    if (onProgress) onProgress("Installing audio-separator (BS-RoFormer vocal model) ...");
    rc = RunAndStream({ venvPython, "-m", "pip", "install", "audio-separator[gpu]" }, onProgress);
    if (rc != 0 || !Util::FileExists(VenvSeparatorExePath())) {
        if (onProgress) onProgress("FAILED to install audio-separator.");
        return false;
    }

    if (onProgress) onProgress("Audio separation set up successfully.");
    return true;
}

bool OFS_AudioSeparator::Separate(
    const std::string& pythonPath,
    const std::string& inputAudioPath,
    const std::string& outDir,
    std::string& outVocalsPath,
    std::string& outInstrumentalPath) noexcept
{
    if (!Util::CreateDirectories(Util::PathFromString(outDir))) {
        return false;
    }

    // the separator exe lives alongside whatever python.exe was passed in (same venv Scripts/bin
    // dir), so a custom python path (AudioSeparatorState::CustomPythonPath) still resolves correctly
    auto separatorExe = (Util::PathFromString(pythonPath).parent_path() /
#if defined(WIN32)
        "audio-separator.exe"
#else
        "audio-separator"
#endif
    ).u8string();

    int rc = RunAndStream({
        separatorExe,
        inputAudioPath,
        "--output_dir", outDir,
        "--output_format", "FLAC"
    }, nullptr);
    if (rc != 0) return false;

    outVocalsPath = FindOutputContaining(outDir, "(Vocals)");
    outInstrumentalPath = FindOutputContaining(outDir, "(Instrumental)");

    return !outVocalsPath.empty() && !outInstrumentalPath.empty()
        && Util::FileExists(outVocalsPath) && Util::FileExists(outInstrumentalPath);
}
