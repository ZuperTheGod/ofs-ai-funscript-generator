#pragma once
// AI Funscript Generator "why" view: what the AI saw and where each generated stroke came from, drawn as a band along the bottom
// of the script timeline. Data comes from `python -m funscript_ai.generate --explain <file>`.
#include "imgui.h"

#include <string>
#include <vector>

class OFS_JoiExplain
{
public:
	static OFS_JoiExplain& Get() noexcept;

	bool Load(const std::string& explainJsonPath) noexcept;
	void Clear() noexcept { loaded = false; points.clear(); }
	bool Loaded() const noexcept { return loaded; }

	// "Check direction" result (python -m funscript_ai.direction_check): stretches that may be upside down
	struct Flag { float start, end; bool likely; std::string clues; };
	std::vector<Flag> Flags;
	std::string FlagsFor;          // the script the check was run on
	bool LoadFlags(const std::string& jsonPath) noexcept;

	bool Show = true;
	static constexpr float BandHeight = 34.f;

	void Draw(ImDrawList* drawList, ImVec2 canvasPos, ImVec2 canvasSize, float offsetTime, float visibleTime) noexcept;
	void ShowLegend() const noexcept;
	std::string Summary() const noexcept;

	struct Point { float at; float pos; int source; };
	static constexpr const char* SourceNames[] = { "visible motion", "AI model", "rhythm fill", "thrust sound", "rest point" };
	static ImU32 SourceColor(int source) noexcept;

private:
	bool loaded = false;
	std::string path;
	float rate = 20.f;
	float actThr = 0.3f;
	bool music = false, soundUsed = false;
	std::vector<float> pIn, pOut, pActive, halfCycle;
	std::vector<Point> points;
	std::vector<std::pair<float, float>> holes, flips;
	std::string models;
	int counts[5] = {};
};
