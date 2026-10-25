#pragma once
// AI Funscript Generator panel: run the funscript AI (github: ofs-ai-funscript-generator) on the open video with every
// setting exposed, load the result as an extra script track plus its "why" view, check a script's direction, and
// (with the full training setup) send a fixed script to training.
#include <string>
#include <vector>
#include <mutex>
#include <atomic>

class OFS_JoiMaker
{
public:
	bool ShowWindow = false;
	void Show(const std::string& videoPath, const std::string& activeScriptPath) noexcept;
	void Update() noexcept;   // main thread: picks up finished runs

	struct Settings {
		std::string joiMakerDir;      // generator folder (found automatically next to the exe, or set in the panel)
		int model = 0;            // 0 = v4 (default), 1 = v3
		int method = 0;           // 0 = learned AI, 1 = motion tracking only (og)
		float actThr = 0.2f;      // how sure before scripting a scene
		float anchorConf = 0.65f; // pull onto clearly visible strokes (1.0 = off)
		bool sound = true;        // thrust sounds (only on videos without music)
		bool rest = true;         // start / end every stretch at 0
		float slowHoleFactor = 2.2f;  // fill gaps that are this much wider than the rhythm (0 = off)
		bool beatLock = false;        // HMVs: snap INs onto the beat where the strokes already follow it
	};
	Settings settings;

private:
	std::mutex logMutex;
	std::string log;
	std::atomic<bool> running = false;
	std::atomic<bool> finished = false;
	int lastExit = 0;
	std::string pendingScript, pendingExplain, pendingVideo, pendingFlags;
	bool settingsLoaded = false;
	bool musicMode = false;        // per video, never saved: switched off whenever another video opens
	std::string lastVideo;

	void loadSettings() noexcept;
	void saveSettings() noexcept;
	void run(const std::string& args, const std::string& scriptOut, const std::string& explainOut) noexcept;
	std::string python() const noexcept;
	bool isPublic() const noexcept;      // folder holds the public generator (funscript_ai)
	bool canTrain() const noexcept;      // folder holds the full training setup (tests/add_correction.py)
	std::string outBase(const std::string& videoPath) const noexcept;
};
