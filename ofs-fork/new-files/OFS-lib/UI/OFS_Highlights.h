#pragma once
// Highlights: tagged, coloured sections on the script timeline, made to label videos for training the funscript AI.
//   right-click the timeline -> "Start highlight here" pins a line, "End highlight here" fills the section
//   click a highlight's tag strip (top of the timeline) -> tag editor: searchable tag gallery, new tags are kept
// Saved next to the video as "<video>.labels.json" (read by the training tools); the tag library is kept in
// OFS's settings folder so tags stay available across videos.
#include "imgui.h"

#include <string>
#include <vector>

struct Highlight
{
	float startTime = 0.f;
	float endTime = 0.f;
	ImColor color;
	std::vector<std::string> tags;
};

class OFS_Highlights
{
public:
	static OFS_Highlights& Get() noexcept;

	void VideoLoaded(const std::string& videoPath) noexcept;

	// timeline hooks
	void Draw(ImDrawList* drawList, ImVec2 canvasPos, ImVec2 canvasSize, float offsetTime, float visibleTime) noexcept;
	// returns true when the click landed on a tag strip (the timeline must not handle it then)
	bool HandleClick(ImVec2 mousePos, ImVec2 canvasPos, ImVec2 canvasSize, float offsetTime, float visibleTime) noexcept;
	void ContextMenuItems(float clickTime) noexcept;
	void ShowEditor() noexcept;   // call once per frame at window level

	static constexpr float TagStripHeight = 16.f;

private:
	std::string videoPath;
	std::string sidecarPath;
	std::vector<Highlight> items;
	float pendingStart = -1.f;
	ImColor pendingColor;

	std::vector<std::string> library;     // every tag ever used, most recent first
	bool libraryLoaded = false;

	int editIdx = -1;                     // highlight whose tags are being edited
	bool openEditor = false;
	char search[128] = {};

	int indexAt(float time) const noexcept;
	ImColor pickColor(int neighbourA, int neighbourB) const noexcept;
	void save() noexcept;
	void load() noexcept;
	void loadLibrary() noexcept;
	void saveLibrary() noexcept;
	void addTag(Highlight& h, const std::string& tag) noexcept;

	// highlights are mirrored into OFS's chapters, so saving the funscript writes them into it as chapters
	std::vector<std::pair<float, float>> syncedRanges;
	bool needsChapterSync = false;
	void syncChapters() noexcept;
};
