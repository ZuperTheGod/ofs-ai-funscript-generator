#include "OFS_JoiExplain.h"

#include "OFS_Util.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

OFS_JoiExplain& OFS_JoiExplain::Get() noexcept
{
	static OFS_JoiExplain instance;
	return instance;
}

ImU32 OFS_JoiExplain::SourceColor(int s) noexcept
{
	switch (s) {
		case 0: return IM_COL32(0x4C, 0xE0, 0x6A, 0xFF);   // visible motion: green
		case 1: return IM_COL32(0x4A, 0x9B, 0xFF, 0xFF);   // AI model: blue
		case 2: return IM_COL32(0xFF, 0xD2, 0x3F, 0xFF);   // rhythm fill: yellow
		case 3: return IM_COL32(0xFF, 0x8C, 0x2E, 0xFF);   // thrust sound: orange
		default: return IM_COL32(0x9A, 0x9A, 0x9A, 0xFF);  // rest point: grey
	}
}

static int sourceIndex(const std::string& s) noexcept
{
	if (s == "visible") return 0;
	if (s == "model") return 1;
	if (s == "filled") return 2;
	if (s == "sound") return 3;
	return 4;
}

bool OFS_JoiExplain::Load(const std::string& p) noexcept
{
	Clear();
	auto text = Util::ReadFileString(p.c_str());
	if (text.empty()) return false;
	bool ok = false;
	auto js = Util::ParseJson(text, &ok);
	if (!ok) return false;
	auto floats = [&](const char* key, std::vector<float>& out) {
		out.clear();
		if (js.contains(key) && js[key].is_array()) for (auto& v : js[key]) out.push_back(v.is_number() ? v.get<float>() : 0.f);
	};
	floats("p_in", pIn);
	floats("p_out", pOut);
	floats("p_active", pActive);
	floats("half_cycle_ms", halfCycle);
	rate = js.value("rate", 20.f);
	actThr = js.value("act_thr", 0.3f);
	music = js.value("music", false);
	soundUsed = js.value("sound_used", false);
	models.clear();
	if (js.contains("models") && js["models"].is_array())
		for (auto& m : js["models"]) if (m.is_string()) models += (models.empty() ? "" : " + ") + m.get<std::string>();
	std::fill(std::begin(counts), std::end(counts), 0);
	if (js.contains("points") && js["points"].is_array()) {
		for (auto& q : js["points"]) {
			Point pt{ q.value("at", 0.f) / 1000.f, q.value("pos", 0.f), sourceIndex(q.value("source", std::string())) };
			counts[pt.source] += 1;
			points.push_back(pt);
		}
	}
	auto ranges = [&](const char* key, std::vector<std::pair<float, float>>& out) {
		out.clear();
		if (js.contains(key) && js[key].is_array())
			for (auto& r : js[key]) if (r.is_array() && r.size() >= 2) out.emplace_back(r[0].get<float>() / 1000.f, r[1].get<float>() / 1000.f);
	};
	ranges("holes_ms", holes);
	ranges("sound_flips_ms", flips);
	path = p;
	loaded = !pActive.empty();
	return loaded;
}

bool OFS_JoiExplain::LoadFlags(const std::string& p) noexcept
{
	Flags.clear();
	auto text = Util::ReadFileString(p.c_str());
	bool ok = false;
	auto js = text.empty() ? nlohmann::json() : Util::ParseJson(text, &ok);
	if (!ok || !js.contains("flags")) return false;
	FlagsFor = js.value("script", std::string());
	for (auto& f : js["flags"]) {
		Flag fl{ f.value("start_s", 0.f), f.value("end_s", 0.f), f.value("level", std::string()) == "likely flipped", "" };
		if (f.contains("clues") && f["clues"].is_array())
			for (auto& c : f["clues"]) if (c.is_string()) fl.clues += (fl.clues.empty() ? "" : "; ") + c.get<std::string>();
		Flags.push_back(fl);
	}
	return true;
}

std::string OFS_JoiExplain::Summary() const noexcept
{
	if (!loaded) return "no AI explanation loaded";
	int total = 0;
	for (int c : counts) total += c;
	char buf[512];
	snprintf(buf, sizeof(buf), "%s | %d strokes: %d visible motion, %d AI model, %d rhythm fill, %d thrust sound, %d rest | "
		"music: %s | scripting threshold %.2f", models.c_str(), total, counts[0], counts[1], counts[2], counts[3],
		counts[4], music ? "yes (sound not used for direction)" : "no", actThr);
	return buf;
}

void OFS_JoiExplain::ShowLegend() const noexcept
{
	for (int s = 0; s < 5; s += 1) {
		if (s) ImGui::SameLine();
		ImGui::ColorButton(SourceNames[s], ImColor(SourceColor(s)), ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoInputs,
			ImVec2(ImGui::GetFontSize(), ImGui::GetFontSize()));
		ImGui::SameLine();
		ImGui::TextUnformatted(SourceNames[s]);
	}
	ImGui::TextDisabled("grey area = how sure the AI is that this is a stroking scene (line = threshold); "
		"white line = which way it leans (up = IN)");
}

void OFS_JoiExplain::Draw(ImDrawList* dl, ImVec2 pos, ImVec2 size, float offsetTime, float visibleTime) noexcept
{
	// direction-check flags: red = likely flipped, orange = check (drawn even without an AI explanation)
	for (auto& f : Flags) {
		if (f.end < offsetTime || f.start > offsetTime + visibleTime) continue;
		float x1 = std::max(pos.x + size.x * ((f.start - offsetTime) / visibleTime), pos.x);
		float x2 = std::min(pos.x + size.x * ((f.end - offsetTime) / visibleTime), pos.x + size.x);
		ImU32 c = f.likely ? IM_COL32(230, 40, 40, 70) : IM_COL32(255, 150, 30, 55);
		dl->AddRectFilled(ImVec2(x1, pos.y), ImVec2(x2, pos.y + size.y), c);
		dl->AddRect(ImVec2(x1, pos.y), ImVec2(x2, pos.y + size.y), f.likely ? IM_COL32(230, 40, 40, 220) : IM_COL32(255, 150, 30, 200), 0.f, 0, 2.f);
		dl->PushClipRect(ImVec2(x1, pos.y), ImVec2(x2, pos.y + size.y), true);
		dl->AddText(ImVec2(x1 + 4.f, pos.y + 18.f), IM_COL32(255, 255, 255, 230), f.likely ? "LIKELY FLIPPED" : "check direction");
		dl->PopClipRect();
	}
	if (!loaded || !Show || pActive.empty()) return;
	const float h = std::min(BandHeight, size.y * 0.45f);
	const ImVec2 b0(pos.x, pos.y + size.y - h), b1(pos.x + size.x, pos.y + size.y);
	dl->AddRectFilled(b0, b1, IM_COL32(0, 0, 0, 150));
	auto xOf = [&](float t) { return pos.x + size.x * ((t - offsetTime) / visibleTime); };
	const int n = (int)pActive.size();
	const int i0 = std::max(0, (int)std::floor(offsetTime * rate) - 1);
	const int i1 = std::min(n - 1, (int)std::ceil((offsetTime + visibleTime) * rate) + 1);
	const int stride = std::max(1, (int)((i1 - i0) / std::max(size.x, 1.f)));
	// rhythm fill and sound-flipped stretches behind everything
	for (auto& r : holes) {
		if (r.second < offsetTime || r.first > offsetTime + visibleTime) continue;
		dl->AddRectFilled(ImVec2(std::max(xOf(r.first), b0.x), b0.y), ImVec2(std::min(xOf(r.second), b1.x), b1.y), IM_COL32(0xFF, 0xD2, 0x3F, 50));
	}
	for (auto& r : flips) {
		if (r.second < offsetTime || r.first > offsetTime + visibleTime) continue;
		dl->AddRectFilled(ImVec2(std::max(xOf(r.first), b0.x), b0.y), ImVec2(std::min(xOf(r.second), b1.x), b1.y), IM_COL32(0xFF, 0x8C, 0x2E, 50));
	}
	// how sure the AI is that this is a scene you'd script
	for (int i = i0; i + stride <= i1; i += stride) {
		float xa = xOf(i / rate), xb = xOf((i + stride) / rate);
		float a = pActive[i];
		dl->AddRectFilled(ImVec2(xa, b1.y - h * a), ImVec2(xb, b1.y), a >= actThr ? IM_COL32(200, 200, 200, 70) : IM_COL32(230, 60, 60, 60));
	}
	float yThr = b1.y - h * actThr;
	dl->AddLine(ImVec2(b0.x, yThr), ImVec2(b1.x, yThr), IM_COL32(255, 255, 255, 60), 1.f);
	// which way it leans: p_in - p_out
	if ((int)pIn.size() == n && (int)pOut.size() == n) {
		ImVec2 prev;
		for (int i = i0; i <= i1; i += stride) {
			float d = std::clamp(pIn[i] - pOut[i], -1.f, 1.f);
			ImVec2 p(xOf(i / rate), b0.y + h * 0.5f - d * h * 0.45f);
			if (i > i0) dl->AddLine(prev, p, IM_COL32(255, 255, 255, 170), 1.f);
			prev = p;
		}
	}
	// every generated stroke, coloured by what decided it
	for (auto& pt : points) {
		if (pt.at < offsetTime || pt.at > offsetTime + visibleTime) continue;
		float x = xOf(pt.at);
		float y = pt.pos > 50.f ? b0.y + 3.f : b1.y - 3.f;
		dl->AddCircleFilled(ImVec2(x, y), 3.f, SourceColor(pt.source));
	}
	// hover: explain the moment under the mouse
	auto m = ImGui::GetMousePos();
	if (ImGui::IsWindowHovered() && m.x >= b0.x && m.x <= b1.x && m.y >= b0.y && m.y <= b1.y) {
		float t = offsetTime + visibleTime * ((m.x - pos.x) / size.x);
		int i = std::clamp((int)(t * rate), 0, n - 1);
		const Point* best = nullptr;
		for (auto& pt : points) if (!best || std::fabs(pt.at - t) < std::fabs(best->at - t)) best = &pt;
		ImGui::BeginTooltip();
		ImGui::Text("%d:%05.2f", (int)t / 60, std::fmod(t, 60.f));
		ImGui::Text("scene you'd script: %.0f%% (threshold %.0f%%)", pActive[i] * 100.f, actThr * 100.f);
		if ((int)pIn.size() == n) ImGui::Text("leans %s (IN %.2f / OUT %.2f)", pIn[i] >= pOut[i] ? "IN" : "OUT", pIn[i], pOut[i]);
		if (i < (int)halfCycle.size()) ImGui::Text("expected stroke: %.0f ms in, %.0f ms out", halfCycle[i], halfCycle[i]);
		if (best && std::fabs(best->at - t) < 1.f)
			ImGui::Text("nearest stroke (%s) decided by: %s", best->pos > 50.f ? "IN" : "OUT", SourceNames[best->source]);
		ImGui::Text("music: %s", music ? "yes - sound not used for direction" : (soundUsed ? "no - thrust sounds used" : "no"));
		ImGui::EndTooltip();
	}
}
