#include "OFS_JoiMaker.h"

#include "OpenFunscripter.h"
#include "OFS_Util.h"
#include "UI/OFS_JoiExplain.h"
#include "imgui.h"
#include "imgui_stdlib.h"

#include <SDL.h>
#include <SDL_thread.h>
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#endif

static std::wstring widen(const std::string& s) noexcept
{
#ifdef _WIN32
	int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
	std::wstring w(n > 0 ? n - 1 : 0, L'\0');
	if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
	return w;
#else
	return std::wstring(s.begin(), s.end());
#endif
}

// Windows argument quoting: backslashes right before the closing quote must be doubled, or a path ending in "\"
// swallows the quote and glues the next arguments onto it.
static std::string quote(const std::string& s) noexcept
{
	size_t trailing = 0;
	while (trailing < s.size() && s[s.size() - 1 - trailing] == '\\') trailing += 1;
	return "\"" + s + std::string(trailing, '\\') + "\"";
}

// a saved script file (not a folder, which is what an unsaved track's path resolves to)
static bool isSavedScript(const std::string& path) noexcept
{
	std::error_code ec;
	auto p = Util::PathFromString(path);
	return !path.empty() && std::filesystem::is_regular_file(p, ec) && !ec;
}

std::string OFS_JoiMaker::python() const noexcept
{
	return (Util::PathFromString(settings.joiMakerDir) / ".venv" / "Scripts" / "python.exe").u8string();
}

bool OFS_JoiMaker::isPublic() const noexcept
{
	return Util::FileExists((Util::PathFromString(settings.joiMakerDir) / "funscript_ai" / "generate.py").u8string());
}

bool OFS_JoiMaker::canTrain() const noexcept
{
	return Util::FileExists((Util::PathFromString(settings.joiMakerDir) / "tests" / "add_correction.py").u8string());
}

static std::string findGenerator() noexcept
{
	// the portable release puts OFS in <generator>\OFS-AI\ : look next to / above the exe first
	char* base = SDL_GetBasePath();
	std::filesystem::path exeDir = base ? Util::PathFromString(base) : std::filesystem::path();
	if (base) SDL_free(base);
	for (auto d : { exeDir, exeDir.parent_path(), exeDir.parent_path().parent_path() }) {
		if (!d.empty() && Util::FileExists((d / "funscript_ai" / "generate.py").u8string())) return d.u8string();
	}
	return std::string();   // not found: the user sets the folder in the panel (it is remembered)
}

std::string OFS_JoiMaker::outBase(const std::string& videoPath) const noexcept
{
	auto dir = Util::PathFromString(settings.joiMakerDir) / "work" / "_ofs";
	std::error_code ec;
	std::filesystem::create_directories(dir, ec);
	return (dir / Util::PathFromString(videoPath).stem()).u8string();
}

void OFS_JoiMaker::loadSettings() noexcept
{
	settingsLoaded = true;
	settings.joiMakerDir = findGenerator();
	auto text = Util::ReadFileString(Util::Prefpath("joi_maker_panel.json").c_str());
	if (text.empty()) return;
	bool ok = false;
	auto js = Util::ParseJson(text, &ok);
	if (!ok) return;
	settings.joiMakerDir = js.value("joiMakerDir", settings.joiMakerDir);
	settings.model = js.value("model", settings.model);
	settings.method = js.value("method", settings.method);
	settings.actThr = js.value("actThr", settings.actThr);
	settings.anchorConf = js.value("anchorConf", settings.anchorConf);
	settings.sound = js.value("sound", settings.sound);
	settings.rest = js.value("rest", settings.rest);
	settings.slowHoleFactor = js.value("slowHoleFactor", settings.slowHoleFactor);
	settings.beatLock = js.value("beatLock", settings.beatLock);
}

void OFS_JoiMaker::saveSettings() noexcept
{
	nlohmann::json js = {
		{"joiMakerDir", settings.joiMakerDir}, {"model", settings.model}, {"method", settings.method},
		{"actThr", settings.actThr}, {"anchorConf", settings.anchorConf}, {"sound", settings.sound},
		{"rest", settings.rest}, {"slowHoleFactor", settings.slowHoleFactor}, {"beatLock", settings.beatLock}};
	auto s = js.dump(1);
	Util::WriteFile(Util::Prefpath("joi_maker_panel.json").c_str(), s.data(), s.size());
}

void OFS_JoiMaker::run(const std::string& args, const std::string& scriptOut, const std::string& explainOut) noexcept
{
	if (running) return;
	running = true;
	finished = false;
	pendingScript = scriptOut;
	pendingExplain = explainOut;
	{
		std::lock_guard<std::mutex> lk(logMutex);
		log = "> " + args + "\n";
	}
	std::string dir = settings.joiMakerDir;
	// the portable release brings its own ffmpeg (bin\) and model cache (torch_home\)
	std::string env = "set PYTHONUNBUFFERED=1&& set PYTHONIOENCODING=utf-8&& ";
	if (Util::DirectoryExists((Util::PathFromString(dir) / "bin").u8string())) env += "set \"PATH=" + dir + "\\bin;%PATH%\"&& ";
	if (Util::DirectoryExists((Util::PathFromString(dir) / "torch_home").u8string())) env += "set \"TORCH_HOME=" + dir + "\\torch_home\"&& ";
	std::string cmdline = "cd /d " + quote(dir) + " && " + env + quote(python()) + " " + args + " 2>&1";
	std::thread([this, cmdline]() {
		auto append = [this](const char* data, size_t n) {
			std::lock_guard<std::mutex> lk(logMutex);
			log.append(data, n);
			if (log.size() > 200000) log.erase(0, log.size() - 150000);
		};
		auto failed = [this]() {
			std::lock_guard<std::mutex> lk(logMutex);
			log += "could not start the generator (is the folder right and setup done?)\n";
			lastExit = -1;
		};
#ifdef _WIN32
		// CreateProcess with CREATE_NO_WINDOW instead of _wpopen: no terminal window pops up, and python /
		// ffmpeg started by the generator share the hidden console instead of opening their own.
		SECURITY_ATTRIBUTES sa = { sizeof(sa), nullptr, TRUE };
		HANDLE readEnd = nullptr, writeEnd = nullptr;
		if (!CreatePipe(&readEnd, &writeEnd, &sa, 0)) { failed(); finished = true; running = false; return; }
		SetHandleInformation(readEnd, HANDLE_FLAG_INHERIT, 0);
		HANDLE nul = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);
		STARTUPINFOW si = {};
		si.cb = sizeof(si);
		si.dwFlags = STARTF_USESTDHANDLES;
		si.hStdInput = nul;
		si.hStdOutput = writeEnd;
		si.hStdError = writeEnd;
		PROCESS_INFORMATION pi = {};
		std::wstring cmd = L"cmd.exe /d /s /c \"" + widen(cmdline) + L"\"";
		BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
		CloseHandle(writeEnd);              // only the child holds the write end now: ReadFile ends when it exits
		if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
		if (!ok) failed();
		else {
			char buf[4096];
			DWORD got = 0;
			while (ReadFile(readEnd, buf, sizeof(buf), &got, nullptr) && got > 0) append(buf, got);
			WaitForSingleObject(pi.hProcess, INFINITE);
			DWORD code = 1;
			GetExitCodeProcess(pi.hProcess, &code);
			lastExit = (int)code;
			CloseHandle(pi.hThread);
			CloseHandle(pi.hProcess);
		}
		CloseHandle(readEnd);
#else
		FILE* p = popen(cmdline.c_str(), "r");
		if (!p) failed();
		else {
			char buf[1024];
			while (fgets(buf, sizeof(buf), p)) append(buf, strlen(buf));
			lastExit = pclose(p);
		}
#endif
		finished = true;
		running = false;
	}).detach();
}

void OFS_JoiMaker::Update() noexcept
{
	if (!finished) return;
	finished = false;
	auto app = OpenFunscripter::ptr;
	if (!pendingScript.empty() && lastExit == 0 && Util::FileExists(pendingScript)) {
		// replace an AI track from an earlier run, then add the new one next to the user's script
		auto& scripts = app->LoadedFunscripts();
		for (int i = (int)scripts.size() - 1; i >= 0; i -= 1) {
			auto abs = app->LoadedProject->MakePathAbsolute(scripts[i]->RelativePath());
			if (Util::PathFromString(abs) == Util::PathFromString(pendingScript) && scripts.size() > 1)
				app->LoadedProject->RemoveFunscript(i);
		}
		app->LoadedProject->AddFunscript(pendingScript);
		std::lock_guard<std::mutex> lk(logMutex);
		log += "\nLoaded the AI script as a new track.\n";
	}
	if (!pendingExplain.empty() && Util::FileExists(pendingExplain)) {
		OFS_JoiExplain::Get().Load(pendingExplain);
		OFS_JoiExplain::Get().Show = true;
	}
	pendingScript.clear();
	pendingExplain.clear();
	if (!pendingFlags.empty() && Util::FileExists(pendingFlags)) {
		OFS_JoiExplain::Get().LoadFlags(pendingFlags);
		std::lock_guard<std::mutex> lk(logMutex);
		log += "\nDirection check loaded: marked stretches are shown on the timeline (red = likely flipped, orange = check).\n";
	}
	pendingFlags.clear();
}

void OFS_JoiMaker::Show(const std::string& videoPath, const std::string& activeScriptPath) noexcept
{
	if (!settingsLoaded) loadSettings();
	if (videoPath != lastVideo) {
		// another video: show its last AI explanation if there is one
		lastVideo = videoPath;
		musicMode = false;
		auto ex = outBase(videoPath) + ".ai.explain.json";
		if (!videoPath.empty() && Util::FileExists(ex)) OFS_JoiExplain::Get().Load(ex);
		else OFS_JoiExplain::Get().Clear();
	}
	if (!ShowWindow) return;
	if (!ImGui::Begin("AI Funscript Generator###JoiMaker", &ShowWindow)) {
		ImGui::End();
		return;
	}
	bool changed = false;
	bool haveVideo = !videoPath.empty();
	ImGui::TextWrapped("Video: %s", haveVideo ? Util::PathFromString(videoPath).filename().u8string().c_str() : "(open a video first)");

	if (ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
		changed |= ImGui::Combo("Method", &settings.method, "AI (learned from your scripts)\0Motion tracking only (good for loops)\0");
		ImGui::BeginDisabled(settings.method != 0);
		changed |= ImGui::Combo("Model", &settings.model, "v4 (v2 + v4, default)\0v3 (v2 + v3, previous)\0");
		changed |= ImGui::SliderFloat("Scripting threshold", &settings.actThr, 0.05f, 0.7f, "%.2f");
		if (ImGui::IsItemHovered()) ImGui::SetTooltip("How sure the AI must be that a scene is stroking before it scripts it.\nLower = scripts more (more off-screen), higher = stops more readily.");
		changed |= ImGui::SliderFloat("Follow visible strokes", &settings.anchorConf, 0.3f, 1.0f, "%.2f");
		if (ImGui::IsItemHovered()) ImGui::SetTooltip("Where the motion is at least this clear, strokes are pulled onto the visible ones.\n1.00 = off.");
		changed |= ImGui::SliderFloat("Fill gaps wider than x rhythm", &settings.slowHoleFactor, 0.f, 4.f, "%.1f");
		if (ImGui::IsItemHovered()) ImGui::SetTooltip("Carry the rhythm through stretches where the spacing suddenly gets this much wider (camera away).\n0 = off.");
		changed |= ImGui::Checkbox("Use thrust sounds (videos without music)", &settings.sound);
		changed |= ImGui::Checkbox("Rest points (every stretch starts and ends at 0)", &settings.rest);
		changed |= ImGui::Checkbox("Beat lock (HMV: INs on the beat where the strokes follow the music)", &settings.beatLock);
		if (ImGui::IsItemHovered()) ImGui::SetTooltip("Only where most strokes already land on beats; everywhere else nothing changes.\nOnly on videos with music.");
		ImGui::Checkbox("MUSIC MODE for this video (every IN on a beat)", &musicMode);
		if (ImGui::IsItemHovered()) ImGui::SetTooltip("Only for videos where you know the thrusts follow the music (PMV / cock hero style).\nEvery IN goes onto a beat, every OUT halfway between, at the pace the AI found.\nWrong on normal videos - switches itself off when you open another video.");
		ImGui::EndDisabled();
		if (ImGui::SmallButton("Reset to defaults")) { settings = Settings{ settings.joiMakerDir }; changed = true; }
		ImGui::SameLine();
		ImGui::SetNextItemWidth(-1.f);
		changed |= ImGui::InputText("##jmdir", &settings.joiMakerDir);
		if (ImGui::IsItemHovered()) ImGui::SetTooltip("Generator folder (where '1 - Setup' was run)");
	}
	if (changed) saveSettings();

	ImGui::Separator();
	ImGui::BeginDisabled(!haveVideo || running);
	if (ImGui::Button(running ? "Generating..." : "Generate AI script for this video", ImVec2(-1.f, 0.f))) {
		auto base = outBase(videoPath);
		std::string scriptOut = base + ".ai.funscript", explainOut = base + ".ai.explain.json";
		char cfg[512];
		snprintf(cfg, sizeof(cfg), "{\\\"act_thr\\\": %.3f, \\\"anchor_conf\\\": %.3f, \\\"sound\\\": %s, \\\"rest\\\": %s, \\\"slow_hole_factor\\\": %.2f, \\\"beat_lock\\\": %s, \\\"beat_lock_share\\\": 0.7, \\\"beat_lock_near_ms\\\": 60, \\\"music_mode\\\": %s}",
			settings.actThr, settings.anchorConf, settings.sound ? "true" : "false", settings.rest ? "true" : "false",
			settings.slowHoleFactor, settings.beatLock ? "true" : "false", musicMode ? "true" : "false");
		std::string args;
		if (isPublic()) {
			// the public generator (github: ofs-ai-funscript-generator)
			args = "-m funscript_ai.generate";
			if (settings.method == 1) args += " --motion-only";
			else {
				if (settings.model == 1) args += " --model v3";
				args += std::string(" --settings \"") + cfg + "\" --explain " + quote(explainOut);
			}
		}
		else {
			args = "-m joi_maker --force-generate --no-voice --no-scenes --no-voice-over";
			if (settings.method == 1) args += " --method og";
			else {
				if (settings.model == 1) args += " --thrust-model v3";
				args += std::string(" --post-cfg \"") + cfg + "\" --explain " + quote(explainOut);
			}
		}
		args += " --out " + quote(scriptOut) + " " + quote(videoPath);
		if (settings.method == 1) OFS_JoiExplain::Get().Clear();
		run(args, scriptOut, settings.method == 0 ? explainOut : std::string());
	}
	ImGui::EndDisabled();

	ImGui::BeginDisabled(activeScriptPath.empty() || !haveVideo || running);
	if (ImGui::Button("Check direction of the selected script (save it first)", ImVec2(-1.f, 0.f))) {
		auto flagsOut = outBase(videoPath) + ".direction.json";
		if (!isSavedScript(activeScriptPath)) {
			std::lock_guard<std::mutex> lk(logMutex);
			log = "The selected track has no saved .funscript file yet.\n"
				"Write it to a file first (File > Export... > Export active script), then select that track and\n"
				"press Check direction again.\n";
		}
		else {
			run(std::string(isPublic() ? "-m funscript_ai.direction_check " : "-m joi_maker.direction_check ") + quote(activeScriptPath) + " " + quote(videoPath) + " --out " + quote(flagsOut),
				std::string(), std::string());
			pendingFlags = flagsOut;
		}
	}
	ImGui::EndDisabled();
	if (ImGui::IsItemHovered()) ImGui::SetTooltip("Marks stretches that may be upside down (100 at OUT): red = likely flipped\n(AI and motion both disagree), orange = check (only one clue). Catches ~3 of 4 flips in tests.");
	{
		auto& fl = OFS_JoiExplain::Get().Flags;
		if (!fl.empty()) {
			ImGui::Text("%d stretches to look at (click to jump):", (int)fl.size());
			if (ImGui::BeginChild("##flaglist", ImVec2(0.f, ImGui::GetFontSize() * std::min(8.f, (float)fl.size() + 1.f)), true)) {
				for (int i = 0; i < (int)fl.size(); i += 1) {
					auto& f = fl[i];
					char label[160];
					snprintf(label, sizeof(label), "%s  %d:%02d - %d:%02d##fl%d", f.likely ? "LIKELY FLIPPED" : "check         ",
						(int)f.start / 60, (int)f.start % 60, (int)f.end / 60, (int)f.end % 60, i);
					if (ImGui::Selectable(label)) OpenFunscripter::ptr->player->SetPositionExact(f.start);
					if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", f.clues.c_str());
				}
			}
			ImGui::EndChild();
			if (ImGui::SmallButton("Clear direction marks")) fl.clear();
		}
	}

	if (canTrain()) {   // only with the full training setup, not the public generator
		ImGui::BeginDisabled(activeScriptPath.empty() || running);
		if (ImGui::Button("Send the selected script to training (after fixing + saving it)", ImVec2(-1.f, 0.f))) {
			if (!isSavedScript(activeScriptPath)) {
				std::lock_guard<std::mutex> lk(logMutex);
				log = "The selected track has no saved .funscript file yet.\n"
					"Write it to a file first (File > Export... > Export active script), then send it to training.\n";
			}
			else run("tests/add_correction.py " + quote(activeScriptPath), std::string(), std::string());
		}
		ImGui::EndDisabled();
		if (ImGui::IsItemHovered()) ImGui::SetTooltip("Files the selected track as a correction: %s", activeScriptPath.c_str());
	}

	if (ImGui::CollapsingHeader("Why view", ImGuiTreeNodeFlags_DefaultOpen)) {
		auto& ex = OFS_JoiExplain::Get();
		ImGui::Checkbox("Show on the timeline", &ex.Show);
		ImGui::SameLine();
		if (ImGui::SmallButton("Load last explanation") && haveVideo) ex.Load(outBase(videoPath) + ".ai.explain.json");
		ImGui::TextWrapped("%s", ex.Summary().c_str());
		ex.ShowLegend();
	}
	if (ImGui::CollapsingHeader("Output", ImGuiTreeNodeFlags_DefaultOpen)) {
		std::lock_guard<std::mutex> lk(logMutex);
		ImGui::BeginChild("##jmlog", ImVec2(0.f, 0.f), true, ImGuiWindowFlags_HorizontalScrollbar);
		ImGui::TextUnformatted(log.c_str());
		if (running) ImGui::SetScrollHereY(1.f);
		ImGui::EndChild();
	}
	ImGui::End();
}
