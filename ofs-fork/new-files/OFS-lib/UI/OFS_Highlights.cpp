#include "OFS_Highlights.h"

#include "OFS_Util.h"
#include "OFS_EventSystem.h"
#include "state/states/ChapterState.h"
#include "imgui_internal.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <random>

// distinct, readable on the dark timeline
static const ImU32 Palette[] = {
	IM_COL32(0xE6, 0x4B, 0x3C, 0xFF), IM_COL32(0x2E, 0xCC, 0x71, 0xFF), IM_COL32(0x34, 0x98, 0xDB, 0xFF),
	IM_COL32(0xF1, 0xC4, 0x0F, 0xFF), IM_COL32(0x9B, 0x59, 0xB6, 0xFF), IM_COL32(0x1A, 0xBC, 0x9C, 0xFF),
	IM_COL32(0xE6, 0x7E, 0x22, 0xFF), IM_COL32(0xEC, 0x40, 0x7A, 0xFF), IM_COL32(0x8B, 0xC3, 0x4A, 0xFF),
	IM_COL32(0x00, 0xBC, 0xD4, 0xFF), IM_COL32(0xFF, 0x57, 0x22, 0xFF), IM_COL32(0x79, 0x86, 0xCB, 0xFF),
};
static constexpr int PaletteSize = sizeof(Palette) / sizeof(Palette[0]);

static std::string lower(std::string s) noexcept
{
	std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
	return s;
}

static std::string trim(const std::string& s) noexcept
{
	auto a = s.find_first_not_of(" \t\r\n");
	if (a == std::string::npos) return std::string();
	auto b = s.find_last_not_of(" \t\r\n");
	return s.substr(a, b - a + 1);
}

static std::string hexColor(const ImColor& c) noexcept
{
	char buf[16];
	snprintf(buf, sizeof(buf), "#%02X%02X%02X", (int)(c.Value.x * 255.f + 0.5f), (int)(c.Value.y * 255.f + 0.5f),
		(int)(c.Value.z * 255.f + 0.5f));
	return buf;
}

static ImColor fromHex(const std::string& s, ImColor fallback) noexcept
{
	unsigned r, g, b;
	if (s.size() == 7 && sscanf(s.c_str(), "#%02x%02x%02x", &r, &g, &b) == 3) return ImColor((int)r, (int)g, (int)b, 255);
	return fallback;
}

static float colorDistance(const ImColor& a, const ImColor& b) noexcept
{
	float dr = a.Value.x - b.Value.x, dg = a.Value.y - b.Value.y, db = a.Value.z - b.Value.z;
	return std::sqrt(dr * dr + dg * dg + db * db);
}

static std::string fmtTime(float t) noexcept
{
	char buf[32];
	int s = (int)std::max(t, 0.f);
	snprintf(buf, sizeof(buf), "%d:%02d", s / 60, s % 60);
	return buf;
}

OFS_Highlights& OFS_Highlights::Get() noexcept
{
	static OFS_Highlights instance;
	return instance;
}

void OFS_Highlights::VideoLoaded(const std::string& path) noexcept
{
	if (path == videoPath) return;
	videoPath = path;
	items.clear();
	pendingStart = -1.f;
	editIdx = -1;
	auto p = Util::PathFromString(videoPath);
	p.replace_extension(".labels.json");
	sidecarPath = p.u8string();
	syncedRanges.clear();
	load();
	loadLibrary();
	needsChapterSync = !items.empty();   // done on the next draw, once the project's chapters are loaded
}

int OFS_Highlights::indexAt(float time) const noexcept
{
	for (int i = 0; i < (int)items.size(); i += 1) {
		if (time >= items[i].startTime && time <= items[i].endTime) return i;
	}
	return -1;
}

ImColor OFS_Highlights::pickColor(int a, int b) const noexcept
{
	static std::mt19937 rng{ std::random_device{}() };
	std::vector<ImColor> avoid;
	if (a >= 0 && a < (int)items.size()) avoid.push_back(items[a].color);
	if (b >= 0 && b < (int)items.size()) avoid.push_back(items[b].color);
	for (int tries = 0; tries < 50; tries += 1) {
		ImColor c(Palette[rng() % PaletteSize]);
		bool ok = true;
		for (auto& x : avoid) ok = ok && colorDistance(c, x) > 0.35f;
		if (ok) return c;
	}
	return ImColor(Palette[rng() % PaletteSize]);
}

void OFS_Highlights::addTag(Highlight& h, const std::string& raw) noexcept
{
	auto tag = trim(raw);
	if (tag.empty()) return;
	for (auto& t : h.tags) if (lower(t) == lower(tag)) return;
	h.tags.push_back(tag);
	auto it = std::find_if(library.begin(), library.end(), [&](const std::string& t) { return lower(t) == lower(tag); });
	if (it != library.end()) library.erase(it);
	library.insert(library.begin(), tag);
	saveLibrary();
}

void OFS_Highlights::save() noexcept
{
	if (sidecarPath.empty()) return;
	std::sort(items.begin(), items.end(), [](const Highlight& x, const Highlight& y) { return x.startTime < y.startTime; });
	// one label per line, "start", "end", "name" first: the OFS "Stroke Labels" Lua panel reads the same file
	std::string out = "{\"version\": 2, \"labels\": [\n";
	for (int i = 0; i < (int)items.size(); i += 1) {
		auto& h = items[i];
		std::string name, tags;
		for (int k = 0; k < (int)h.tags.size(); k += 1) {
			nlohmann::json js = h.tags[k];
			name += (k ? ", " : "") + h.tags[k];
			tags += (k ? ", " : "") + js.dump();
		}
		nlohmann::json jname = name;
		char buf[96];
		snprintf(buf, sizeof(buf), "  {\"start\": %.3f, \"end\": %.3f, \"name\": ", h.startTime, h.endTime);
		out += buf + jname.dump() + ", \"tags\": [" + tags + "], \"color\": \"" + hexColor(h.color) + "\"}";
		out += (i + 1 < (int)items.size()) ? ",\n" : "\n";
	}
	out += "]}\n";
	syncChapters();
	if (items.empty()) {
		std::error_code ec;
		std::filesystem::remove(Util::PathFromString(sidecarPath), ec);
		return;
	}
	Util::WriteFile(sidecarPath.c_str(), out.data(), out.size());
}

void OFS_Highlights::syncChapters() noexcept
{
	auto& cs = ChapterState::StaticStateSlow();
	auto same = [](float a, float b) { return std::fabs(a - b) < 0.02f; };
	// drop the chapters made from highlights before (and copies that came back in with a saved funscript)
	cs.chapters.erase(std::remove_if(cs.chapters.begin(), cs.chapters.end(), [&](const Chapter& c) {
		for (auto& r : syncedRanges) if (same(c.startTime, r.first) && same(c.endTime, r.second)) return true;
		for (auto& h : items) if (same(c.startTime, h.startTime) && same(c.endTime, h.endTime)) return true;
		return false;
	}), cs.chapters.end());
	syncedRanges.clear();
	for (auto& h : items) {
		Chapter c;
		c.startTime = h.startTime;
		c.endTime = h.endTime;
		for (int k = 0; k < (int)h.tags.size(); k += 1) c.name += (k ? ", " : "") + h.tags[k];
		if (c.name.empty()) c.name = "highlight";
		c.color = h.color;
		cs.chapters.push_back(std::move(c));
		syncedRanges.emplace_back(h.startTime, h.endTime);
	}
	std::sort(cs.chapters.begin(), cs.chapters.end(),
		[](const Chapter& a, const Chapter& b) noexcept { return a.startTime < b.startTime; });
	EV::Enqueue<ChapterStateChanged>();
}

void OFS_Highlights::load() noexcept
{
	auto text = Util::ReadFileString(sidecarPath.c_str());
	if (text.empty()) return;
	bool ok = false;
	auto js = Util::ParseJson(text, &ok);
	if (!ok || !js.contains("labels") || !js["labels"].is_array()) return;
	for (auto& l : js["labels"]) {
		Highlight h;
		h.startTime = l.value("start", 0.f);
		h.endTime = l.value("end", -1.f);
		if (h.endTime < h.startTime) continue;              // an unfinished label from the Lua panel
		if (l.contains("tags") && l["tags"].is_array()) {
			for (auto& t : l["tags"]) if (t.is_string()) h.tags.push_back(t.get<std::string>());
		}
		else {
			// labels from the Lua panel: "riding, she moves"
			std::string name = l.value("name", std::string());
			size_t a = 0;
			while (a <= name.size()) {
				size_t b = name.find(',', a);
				if (b == std::string::npos) b = name.size();
				auto t = trim(name.substr(a, b - a));
				if (!t.empty()) h.tags.push_back(t);
				a = b + 1;
			}
		}
		h.color = fromHex(l.value("color", std::string()), ImColor(Palette[items.size() % PaletteSize]));
		items.push_back(std::move(h));
	}
}

void OFS_Highlights::loadLibrary() noexcept
{
	if (libraryLoaded) return;
	libraryLoaded = true;
	auto text = Util::ReadFileString(Util::Prefpath("highlight_tags.json").c_str());
	bool ok = false;
	auto js = text.empty() ? nlohmann::json() : Util::ParseJson(text, &ok);
	if (ok && js.is_array()) {
		for (auto& t : js) if (t.is_string()) library.push_back(t.get<std::string>());
	}
	if (library.empty()) {
		// starter tags = the words the training already understands; anything else you type is kept as well
		library = { "blowjob", "handjob", "footjob", "titjob", "riding", "doggy", "missionary", "anal", "standing",
			"other sex", "he moves", "she moves", "both move", "no sex", "kissing", "talking", "teasing", "posing",
			"story", "half speed", "double speed" };
		saveLibrary();
	}
}

void OFS_Highlights::saveLibrary() noexcept
{
	nlohmann::json js = library;
	auto s = js.dump(1);
	Util::WriteFile(Util::Prefpath("highlight_tags.json").c_str(), s.data(), s.size());
}

void OFS_Highlights::Draw(ImDrawList* dl, ImVec2 pos, ImVec2 size, float offsetTime, float visibleTime) noexcept
{
	if (videoPath.empty()) return;
	if (needsChapterSync) {
		needsChapterSync = false;
		syncChapters();
	}
	const float viewEnd = offsetTime + visibleTime;
	auto xOf = [&](float t) { return pos.x + size.x * ((t - offsetTime) / visibleTime); };
	for (int i = 0; i < (int)items.size(); i += 1) {
		auto& h = items[i];
		if (h.endTime < offsetTime || h.startTime > viewEnd) continue;
		float x1 = std::max(xOf(h.startTime), pos.x), x2 = std::min(xOf(h.endTime), pos.x + size.x);
		ImColor fill = h.color;
		fill.Value.w = 0.22f;
		dl->AddRectFilled(ImVec2(x1, pos.y), ImVec2(x2, pos.y + size.y), fill);
		ImColor strip = h.color;
		strip.Value.w = i == editIdx ? 1.f : 0.85f;
		dl->AddRectFilled(ImVec2(x1, pos.y), ImVec2(x2, pos.y + TagStripHeight), strip);
		std::string label;
		for (int k = 0; k < (int)h.tags.size(); k += 1) label += (k ? ", " : "") + h.tags[k];
		if (label.empty()) label = "(click to add tags)";
		dl->PushClipRect(ImVec2(x1, pos.y), ImVec2(x2, pos.y + TagStripHeight), true);
		dl->AddText(ImVec2(x1 + 4.f, pos.y + 1.f), IM_COL32(0, 0, 0, 255), label.c_str());
		dl->PopClipRect();
	}
	if (pendingStart >= 0.f && pendingStart >= offsetTime && pendingStart <= viewEnd) {
		float x = xOf(pendingStart);
		dl->AddLine(ImVec2(x, pos.y), ImVec2(x, pos.y + size.y), pendingColor, 3.f);
		dl->AddText(ImVec2(x + 4.f, pos.y + 1.f), pendingColor, "highlight start");
	}
}

bool OFS_Highlights::HandleClick(ImVec2 m, ImVec2 pos, ImVec2 size, float offsetTime, float visibleTime) noexcept
{
	if (m.y < pos.y || m.y > pos.y + TagStripHeight || m.x < pos.x || m.x > pos.x + size.x) return false;
	float t = offsetTime + visibleTime * ((m.x - pos.x) / size.x);
	int i = indexAt(t);
	if (i < 0) return false;
	editIdx = i;
	openEditor = true;
	search[0] = 0;
	return true;
}

void OFS_Highlights::ContextMenuItems(float t) noexcept
{
	if (videoPath.empty()) return;
	int at = indexAt(t);
	if (pendingStart < 0.f) {
		if (ImGui::MenuItem(("Start highlight at " + fmtTime(t)).c_str())) {
			pendingStart = t;
			int before = -1;
			for (int i = 0; i < (int)items.size(); i += 1) if (items[i].endTime <= t) before = i;
			pendingColor = pickColor(before, before + 1);
		}
	}
	else {
		if (ImGui::MenuItem(("End highlight at " + fmtTime(t) + " (started " + fmtTime(pendingStart) + ")").c_str())) {
			Highlight h;
			h.startTime = std::min(pendingStart, t);
			h.endTime = std::max(pendingStart, t);
			h.color = pendingColor;
			pendingStart = -1.f;
			if (h.endTime - h.startTime >= 0.2f) {
				items.push_back(h);
				save();
				for (int i = 0; i < (int)items.size(); i += 1) {
					if (items[i].startTime == h.startTime && items[i].endTime == h.endTime) editIdx = i;
				}
				openEditor = true;      // straight to tagging
				search[0] = 0;
			}
		}
		if (ImGui::MenuItem("Cancel highlight start")) pendingStart = -1.f;
	}
	if (at >= 0) {
		if (ImGui::MenuItem("Edit highlight tags...")) {
			editIdx = at;
			openEditor = true;
			search[0] = 0;
		}
		if (ImGui::MenuItem("Change highlight color")) {
			items[at].color = pickColor(at - 1, at + 1);
			save();
		}
		if (ImGui::MenuItem("Delete highlight")) {
			items.erase(items.begin() + at);
			editIdx = -1;
			save();
		}
	}
	ImGui::Separator();
}

void OFS_Highlights::ShowEditor() noexcept
{
	if (openEditor) {
		ImGui::OpenPopup("Highlight tags###HighlightTags");
		openEditor = false;
	}
	ImGui::SetNextWindowSize(ImVec2(ImGui::GetFontSize() * 26.f, ImGui::GetFontSize() * 28.f), ImGuiCond_Appearing);
	if (!ImGui::BeginPopupModal("Highlight tags###HighlightTags", nullptr, ImGuiWindowFlags_None)) return;
	if (editIdx < 0 || editIdx >= (int)items.size()) {
		ImGui::CloseCurrentPopup();
		ImGui::EndPopup();
		return;
	}
	auto& h = items[editIdx];
	ImGui::Text("Highlight %s - %s", fmtTime(h.startTime).c_str(), fmtTime(h.endTime).c_str());
	ImGui::SameLine();
	if (ImGui::ColorEdit3("##hlcolor", &h.color.Value.x, ImGuiColorEditFlags_NoInputs)) save();
	ImGui::SameLine();
	if (ImGui::SmallButton("new color")) {
		h.color = pickColor(editIdx - 1, editIdx + 1);
		save();
	}

	ImGui::Separator();
	ImGui::TextUnformatted("Tags on this highlight (click to remove):");
	int removeIdx = -1;
	for (int k = 0; k < (int)h.tags.size(); k += 1) {
		if (k) ImGui::SameLine();
		ImGui::PushID(k);
		ImGui::PushStyleColor(ImGuiCol_Button, (ImU32)h.color);
		ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 0, 0, 255));
		if (ImGui::SmallButton((h.tags[k] + "  x").c_str())) removeIdx = k;
		ImGui::PopStyleColor(2);
		ImGui::PopID();
		if (ImGui::GetItemRectMax().x > ImGui::GetWindowPos().x + ImGui::GetWindowWidth() - ImGui::GetFontSize() * 8.f)
			ImGui::NewLine();
	}
	if (h.tags.empty()) ImGui::TextDisabled("(none yet)");
	if (removeIdx >= 0) {
		h.tags.erase(h.tags.begin() + removeIdx);
		save();
	}

	ImGui::Separator();
	if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
	ImGui::SetNextItemWidth(-1.f);
	bool enter = ImGui::InputTextWithHint("##tagsearch", "type to search, Enter adds it", search, sizeof(search),
		ImGuiInputTextFlags_EnterReturnsTrue);
	std::string q = lower(trim(search));
	bool exact = false;
	std::vector<std::string> shown;
	for (auto& t : library) {
		auto lt = lower(t);
		if (lt == q) exact = true;
		if (q.empty() || lt.find(q) != std::string::npos) shown.push_back(t);
	}
	if (enter && !q.empty()) {
		// Enter: the best match if there is one, otherwise a brand new tag
		addTag(h, exact || shown.empty() ? std::string(search) : shown.front());
		search[0] = 0;
		save();
		ImGui::SetKeyboardFocusHere(-1);
	}
	if (!q.empty() && !exact) {
		if (ImGui::Button(("+ add new tag \"" + trim(search) + "\"").c_str())) {
			addTag(h, search);
			search[0] = 0;
			save();
		}
	}
	ImGui::TextUnformatted(q.empty() ? "Tag gallery:" : "Matching tags:");
	ImGui::BeginChild("##taggallery", ImVec2(0.f, -ImGui::GetFrameHeightWithSpacing() * 1.3f), true);
	float rowEnd = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
	for (int k = 0; k < (int)shown.size(); k += 1) {
		bool has = false;
		for (auto& t : h.tags) has = has || lower(t) == lower(shown[k]);
		ImGui::PushID(k + 1000);
		if (has) ImGui::BeginDisabled();
		float w = ImGui::CalcTextSize(shown[k].c_str()).x + ImGui::GetStyle().FramePadding.x * 2.f;
		if (k > 0) {
			float next = ImGui::GetItemRectMax().x + ImGui::GetStyle().ItemSpacing.x + w;
			if (next < rowEnd) ImGui::SameLine();
		}
		if (ImGui::Button(shown[k].c_str())) {
			addTag(h, shown[k]);
			save();
		}
		if (has) ImGui::EndDisabled();
		ImGui::PopID();
	}
	if (shown.empty()) ImGui::TextDisabled("no tag like that yet - press Enter to add it");
	ImGui::EndChild();
	if (ImGui::Button("Done") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
		editIdx = -1;
		ImGui::CloseCurrentPopup();
	}
	ImGui::EndPopup();
}
