#include "OFS_AudioClassifier.h"
#include "OFS_Util.h"

#include "dr_flac.h"
#include "subprocess.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <algorithm>

namespace
{
    constexpr int32_t FrameSize = 512; // 32ms @ 16kHz
    constexpr int32_t HopSize = 256;   // 16ms @ 16kHz

    inline float Clamp01(float v) noexcept { return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); }

    // Normalized autocorrelation is highest near lag 0 for almost any smooth/tonal signal,
    // regardless of its true periodicity. Taking a plain max over [minLag,maxLag] therefore
    // false-positives on signals whose real period lies outside that range (their correlation
    // curve is simply still decaying/rising at the window edge). Requiring a genuine local
    // peak - a lag that's higher than *both* neighbours - filters that out and only accepts
    // lags where the signal is actually repeating at that specific rate.
    // `scratch` must have capacity for at least (maxLag - minLag + 3) floats.
    template<typename CorrFunc>
    inline float FindPeakCorrelation(CorrFunc&& corrAt, float energy0, int32_t minLag, int32_t maxLag, std::vector<float>& scratch) noexcept
    {
        if (energy0 <= 1e-8f || maxLag <= minLag + 1) return 0.f;
        const int32_t count = maxLag - minLag + 3; // [minLag-1, maxLag+1]
        if ((int32_t)scratch.size() < count) scratch.resize(count);
        for (int32_t lag = minLag - 1; lag <= maxLag + 1; ++lag) {
            scratch[lag - (minLag - 1)] = corrAt(lag) / energy0;
        }
        float best = 0.f;
        for (int32_t lag = minLag; lag <= maxLag; ++lag) {
            int32_t idx = lag - (minLag - 1);
            float c = scratch[idx];
            if (c > scratch[idx - 1] && c >= scratch[idx + 1]) {
                best = Util::Max(best, c);
            }
        }
        return Clamp01(best);
    }

    // RBJ cookbook biquad, transposed direct form II
    struct Biquad
    {
        float b0 = 0.f, b1 = 0.f, b2 = 0.f, a1 = 0.f, a2 = 0.f;
        float z1 = 0.f, z2 = 0.f;

        void LowPass(float sampleRate, float cutoff, float q) noexcept
        {
            float w0 = 2.f * 3.14159265f * cutoff / sampleRate;
            float cosw0 = cosf(w0);
            float sinw0 = sinf(w0);
            float alpha = sinw0 / (2.f * q);
            float a0 = 1.f + alpha;
            b0 = ((1.f - cosw0) / 2.f) / a0;
            b1 = (1.f - cosw0) / a0;
            b2 = b0;
            a1 = (-2.f * cosw0) / a0;
            a2 = (1.f - alpha) / a0;
        }

        void HighPass(float sampleRate, float cutoff, float q) noexcept
        {
            float w0 = 2.f * 3.14159265f * cutoff / sampleRate;
            float cosw0 = cosf(w0);
            float sinw0 = sinf(w0);
            float alpha = sinw0 / (2.f * q);
            float a0 = 1.f + alpha;
            b0 = ((1.f + cosw0) / 2.f) / a0;
            b1 = -(1.f + cosw0) / a0;
            b2 = b0;
            a1 = (-2.f * cosw0) / a0;
            a2 = (1.f - alpha) / a0;
        }

        inline float Process(float x) noexcept
        {
            float y = b0 * x + z1;
            z1 = b1 * x - a1 * y + z2;
            z2 = b2 * x - a2 * y;
            return y;
        }
    };

    enum class RawClass : uint8_t
    {
        Silence,
        Beat,
        Clap,
        Thump,
        Voiced,
        Unclassified
    };

    struct Run
    {
        int32_t startFrame;
        int32_t endFrame;
        RawClass cls;
    };
}

bool OFS_AudioClassifier::ClassifyAudio(
    const std::string& ffmpegPath,
    const std::string& videoPath,
    const std::string& tmpFlacPath,
    const AudioClassSettings& settings,
    std::vector<AudioSegment>& outSegments) noexcept
{
    outSegments.clear();

    // 1) decode audio to mono 16kHz flac via the bundled ffmpeg (same subprocess pattern as OFS_Waveform)
    {
        std::array<const char*, 13> args = {
            ffmpegPath.c_str(),
            "-y",
            "-loglevel", "quiet",
            "-i", videoPath.c_str(),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            tmpFlacPath.c_str(),
            nullptr
        };
        struct subprocess_s proc;
        if (subprocess_create(args.data(), subprocess_option_no_window, &proc) != 0) {
            return false;
        }
        if (proc.stdout_file) { fclose(proc.stdout_file); proc.stdout_file = nullptr; }
        if (proc.stderr_file) { fclose(proc.stderr_file); proc.stderr_file = nullptr; }
        int returnCode = 0;
        subprocess_join(&proc, &returnCode);
        subprocess_destroy(&proc);
        if (returnCode != 0) return false;
    }

    // 2) decode PCM
    drflac* flac = drflac_open_file(tmpFlacPath.c_str(), NULL);
    if (!flac) return false;

    std::vector<float> pcm;
    pcm.resize((size_t)flac->totalPCMFrameCount);
    {
        std::vector<drflac_int16> chunk;
        chunk.resize(65536);
        size_t written = 0;
        drflac_uint64 readCount;
        while ((readCount = drflac_read_pcm_frames_s16(flac, chunk.size(), chunk.data())) > 0) {
            if (written + readCount > pcm.size()) pcm.resize(written + readCount);
            for (drflac_uint64 i = 0; i < readCount; ++i) {
                pcm[written++] = chunk[i] / 32768.f;
            }
        }
        pcm.resize(written);
    }
    const float sampleRate = (float)flac->sampleRate;
    drflac_close(flac);

    if (pcm.empty() || sampleRate <= 0.f) return false;

    const int32_t sampleCount = (int32_t)pcm.size();

    // 3) bass / presence envelope filters, run once over the whole buffer (O(N))
    Biquad bassFilter, presenceFilter;
    bassFilter.LowPass(sampleRate, settings.BassCutoffHz, 0.707f);
    presenceFilter.HighPass(sampleRate, settings.PresenceCutoffHz, 0.707f);

    std::vector<float> bassSignal(sampleCount);
    std::vector<float> presenceSignal(sampleCount);
    for (int32_t i = 0; i < sampleCount; ++i) {
        bassSignal[i] = bassFilter.Process(pcm[i]);
        presenceSignal[i] = presenceFilter.Process(pcm[i]);
    }

    // 4) per-frame broadband / band-limited energy
    const int32_t frameCount = Util::Max(0, (sampleCount - FrameSize) / HopSize + 1);
    if (frameCount <= 0) return true; // too short to say anything - not an error

    std::vector<float> rms(frameCount, 0.f);
    std::vector<float> bassEnergy(frameCount, 0.f);
    std::vector<float> presenceEnergy(frameCount, 0.f);

    for (int32_t f = 0; f < frameCount; ++f) {
        int32_t start = f * HopSize;
        float sumSq = 0.f, bassSq = 0.f, presSq = 0.f;
        for (int32_t i = 0; i < FrameSize; ++i) {
            int32_t idx = start + i;
            float s = pcm[idx];
            sumSq += s * s;
            float b = bassSignal[idx];
            bassSq += b * b;
            float p = presenceSignal[idx];
            presSq += p * p;
        }
        rms[f] = sqrtf(sumSq / FrameSize);
        bassEnergy[f] = sqrtf(bassSq / FrameSize);
        presenceEnergy[f] = sqrtf(presSq / FrameSize);
    }

    // 5) onset strength (positive RMS derivative) + adaptive-threshold peak picking
    std::vector<float> onsetStrength(frameCount, 0.f);
    for (int32_t f = 1; f < frameCount; ++f) {
        onsetStrength[f] = Util::Max(0.f, rms[f] - rms[f - 1]);
    }

    // Median + MAD (median absolute deviation) instead of mean/stddev: a single loud onset
    // (e.g. music kicking in after silence) otherwise inflates the trailing baseline for the
    // following second and can mask smaller transients (claps/thumps) right after it.
    const int32_t adaptiveWindow = Util::Max(4, (int32_t)(1.f * sampleRate / HopSize));
    std::vector<bool> isOnsetPeak(frameCount, false);
    std::vector<float> windowScratch;
    std::vector<float> devScratch;
    for (int32_t f = 0; f < frameCount; ++f) {
        int32_t winStart = Util::Max(0, f - adaptiveWindow);
        int32_t n = f - winStart;

        float median = 0.f;
        float mad = 0.f;
        if (n > 0) {
            windowScratch.assign(onsetStrength.begin() + winStart, onsetStrength.begin() + f);
            size_t mid = windowScratch.size() / 2;
            std::nth_element(windowScratch.begin(), windowScratch.begin() + mid, windowScratch.end());
            median = windowScratch[mid];

            devScratch.resize(windowScratch.size());
            for (size_t i = 0; i < windowScratch.size(); ++i) devScratch[i] = fabsf(windowScratch[i] - median);
            std::nth_element(devScratch.begin(), devScratch.begin() + mid, devScratch.end());
            mad = devScratch[mid];
        }

        float threshold = median + settings.Sensitivity * (mad * 1.4826f + 1e-4f);
        bool isLocalMax = (f == 0 || onsetStrength[f] >= onsetStrength[f - 1])
                        && (f == frameCount - 1 || onsetStrength[f] >= onsetStrength[f + 1]);
        isOnsetPeak[f] = isLocalMax && onsetStrength[f] > threshold && onsetStrength[f] > 0.001f;
    }

    // 6) voicing/pitch confidence via decimated autocorrelation.
    //    Only evaluated for non-silent, non-transient frames, every other frame (held between).
    std::vector<float> voicedConfidence(frameCount, 0.f);
    {
        constexpr int32_t Decimation = 4;
        const int32_t decWindow = FrameSize / Decimation;
        std::vector<float> decimated(decWindow);
        std::vector<float> corrScratch;

        const float decRate = sampleRate / Decimation;
        const int32_t minLag = Util::Max(2, (int32_t)(decRate / 400.f));
        const int32_t maxLag = Util::Min(decWindow - 2, (int32_t)(decRate / 70.f));

        float lastConfidence = 0.f;
        for (int32_t f = 0; f < frameCount; ++f) {
            if (rms[f] < settings.SilenceThreshold || isOnsetPeak[f]) {
                voicedConfidence[f] = 0.f;
                lastConfidence = 0.f;
                continue;
            }
            if (f % 2 != 0) {
                voicedConfidence[f] = lastConfidence;
                continue;
            }

            int32_t start = f * HopSize;
            for (int32_t i = 0; i < decWindow; ++i) {
                int32_t base = start + i * Decimation;
                float sum = 0.f;
                for (int32_t k = 0; k < Decimation; ++k) {
                    sum += pcm[Util::Min(base + k, sampleCount - 1)];
                }
                decimated[i] = sum / Decimation;
            }

            float energy0 = 0.f;
            for (int32_t i = 0; i < decWindow; ++i) energy0 += decimated[i] * decimated[i];

            auto corrAt = [&](int32_t lag) noexcept -> float {
                float corr = 0.f;
                for (int32_t i = 0; i < decWindow - lag; ++i) {
                    corr += decimated[i] * decimated[i + lag];
                }
                return corr;
            };
            float confidence = FindPeakCorrelation(corrAt, energy0, minLag, maxLag, corrScratch);
            voicedConfidence[f] = confidence;
            lastConfidence = confidence;
        }
    }

    // 7) sustained bass periodicity (tempo) - catches 808/bass loops that don't have sharp onsets.
    //    Evaluated on the bass-energy frame series over a trailing ~4s window, ~once per second.
    std::vector<float> bassPeriodicity(frameCount, 0.f);
    {
        const float frameRate = sampleRate / (float)HopSize;
        const int32_t tempoWindow = Util::Max(1, (int32_t)(4.f * frameRate));
        const int32_t minLagFrames = Util::Max(2, (int32_t)(60.f / 180.f * frameRate));
        const int32_t maxLagFrames = Util::Max(minLagFrames + 2, (int32_t)(60.f / 60.f * frameRate));
        const int32_t evalStride = Util::Max(1, (int32_t)(0.5f * frameRate));
        std::vector<float> corrScratch;

        float lastScore = 0.f;
        for (int32_t f = 0; f < frameCount; ++f) {
            if (f % evalStride != 0) {
                bassPeriodicity[f] = lastScore;
                continue;
            }
            int32_t winStart = Util::Max(0, f - tempoWindow);
            int32_t winLen = f - winStart;
            if (winLen < (maxLagFrames + 1) * 2) { bassPeriodicity[f] = lastScore; continue; }

            float mean = 0.f;
            for (int32_t i = winStart; i < f; ++i) mean += bassEnergy[i];
            mean /= winLen;

            float energy0 = 0.f;
            for (int32_t i = winStart; i < f; ++i) { float d = bassEnergy[i] - mean; energy0 += d * d; }

            auto corrAt = [&](int32_t lag) noexcept -> float {
                float corr = 0.f;
                for (int32_t i = winStart; i < f - lag; ++i) {
                    corr += (bassEnergy[i] - mean) * (bassEnergy[i + lag] - mean);
                }
                return corr;
            };
            float score = FindPeakCorrelation(corrAt, energy0, minLagFrames, maxLagFrames, corrScratch);
            bassPeriodicity[f] = score;
            lastScore = score;
        }
    }

    // 8) per-frame raw classification (priority order)
    std::vector<RawClass> raw(frameCount, RawClass::Silence);
    constexpr float VoicingThreshold = 0.35f;
    constexpr float PeriodicityThreshold = 0.5f;
    for (int32_t f = 0; f < frameCount; ++f) {
        if (rms[f] < settings.SilenceThreshold) { raw[f] = RawClass::Silence; continue; }

        if (isOnsetPeak[f]) {
            float total = bassEnergy[f] + presenceEnergy[f] + 1e-6f;
            float bassRatio = bassEnergy[f] / total;
            float presenceRatio = presenceEnergy[f] / total;
            if (bassRatio > 0.6f) raw[f] = RawClass::Beat;
            else if (presenceRatio > 0.55f) raw[f] = RawClass::Clap;
            else raw[f] = RawClass::Thump;
            continue;
        }

        if (voicedConfidence[f] > VoicingThreshold) { raw[f] = RawClass::Voiced; continue; }
        if (bassPeriodicity[f] > PeriodicityThreshold) { raw[f] = RawClass::Beat; continue; }
        // sustained bass/sub-drone with no sharp attack and no rhythmic pulsing still counts as Beat
        if (bassEnergy[f] > 0.7f * rms[f]) { raw[f] = RawClass::Beat; continue; }
        raw[f] = RawClass::Unclassified;
    }

    // 9) merge adjacent same-raw-class frames into runs
    std::vector<Run> runs;
    for (int32_t f = 0; f < frameCount; ++f) {
        if (!runs.empty() && runs.back().cls == raw[f]) {
            runs.back().endFrame = f;
        } else {
            runs.push_back({ f, f, raw[f] });
        }
    }

    // Voiced speech/moans naturally have brief dips (consonants, vibrato, breaths) that can flicker
    // into a one-or-two-frame onset/unclassified blip and split what's really one continuous voiced
    // run in two. Bridge short non-voiced runs sandwiched between voiced ones before anything downstream
    // (duration-based Voice/Moan classification, min-segment filtering) judges them individually.
    constexpr int32_t MaxBridgeFrames = 8;
    for (size_t i = 1; i + 1 < runs.size(); ++i) {
        auto& run = runs[i];
        if (run.cls == RawClass::Voiced) continue;
        if (runs[i - 1].cls != RawClass::Voiced || runs[i + 1].cls != RawClass::Voiced) continue;
        if (run.endFrame - run.startFrame + 1 > MaxBridgeFrames) continue;
        run.cls = RawClass::Voiced;
    }
    {
        std::vector<Run> bridged;
        for (auto& run : runs) {
            if (!bridged.empty() && bridged.back().cls == run.cls) {
                bridged.back().endFrame = run.endFrame;
            } else {
                bridged.push_back(run);
            }
        }
        runs = std::move(bridged);
    }

    auto frameTime = [&](int32_t f) noexcept -> float {
        return (f * HopSize) / sampleRate;
    };

    // 10) resolve Voiced runs into Voice/Moan: duration + envelope "choppiness"
    //     (speech has many syllable-driven dips per second, a sustained moan has few).
    auto classifyVoicedRun = [&](const Run& run) noexcept -> AudioClass {
        float duration = frameTime(run.endFrame + 1) - frameTime(run.startFrame);
        int32_t dips = 0;
        for (int32_t f = Util::Max(run.startFrame + 1, 1); f < run.endFrame; ++f) {
            if (rms[f] < rms[f - 1] && rms[f] < rms[f + 1]) dips += 1;
        }
        float dipsPerSecond = duration > 0.01f ? dips / duration : 0.f;
        if (duration > 0.6f && dipsPerSecond < 3.f) return AudioClass::Moan;
        return AudioClass::Voice;
    };

    // 11) build segments - silence/unclassified runs are gaps, everything else becomes a segment
    std::vector<AudioSegment> segments;
    for (auto& run : runs) {
        if (run.cls == RawClass::Silence || run.cls == RawClass::Unclassified) continue;

        AudioSegment seg;
        seg.startTime = frameTime(run.startFrame);
        seg.endTime = frameTime(run.endFrame + 1);

        AudioClass cls;
        switch (run.cls) {
            case RawClass::Beat: cls = AudioClass::Beat; break;
            case RawClass::Clap: cls = AudioClass::Clap; break;
            case RawClass::Thump: cls = AudioClass::Thump; break;
            case RawClass::Voiced: cls = classifyVoicedRun(run); break;
            default: continue;
        }
        seg.classId = (uint8_t)cls;
        segments.push_back(seg);
    }

    // 12) coalesce adjacent same-class segments first - independent of their individual length.
    //     Two short blips of the same class right next to each other (e.g. a noisy transient that
    //     produced two separate one-frame onset peaks) should become one segment before either one
    //     is judged against MinSegmentSeconds; testing each in isolation would drop both.
    constexpr float CoalesceGap = 0.1f;
    std::vector<AudioSegment> coalesced;
    for (auto& seg : segments) {
        if (!coalesced.empty() && coalesced.back().classId == seg.classId
            && seg.startTime - coalesced.back().endTime < CoalesceGap) {
            coalesced.back().endTime = seg.endTime;
        } else {
            coalesced.push_back(seg);
        }
    }

    // 13) transient hits (Clap/Thump/Beat) are detected from a single onset frame, which is
    //     shorter than any real percussive sound's audible decay tail. Give short transient
    //     segments a bit of extra length so they don't get thrown out as isolated blips below,
    //     capped so they can't eat into the next segment.
    constexpr float TransientTailPad = 0.1f;
    for (size_t i = 0; i < coalesced.size(); ++i) {
        auto& seg = coalesced[i];
        bool isTransient = seg.classId == (uint8_t)AudioClass::Clap
                         || seg.classId == (uint8_t)AudioClass::Thump
                         || seg.classId == (uint8_t)AudioClass::Beat;
        if (!isTransient || (seg.endTime - seg.startTime) >= settings.MinSegmentSeconds) continue;

        float maxEnd = (i + 1 < coalesced.size()) ? coalesced[i + 1].startTime : (seg.endTime + TransientTailPad);
        seg.endTime = Util::Min(seg.endTime + TransientTailPad, maxEnd);
    }

    // 14) anything still shorter than MinSegmentSeconds is an isolated blip - drop it
    std::vector<AudioSegment> result;
    result.reserve(coalesced.size());
    for (auto& seg : coalesced) {
        if (seg.endTime - seg.startTime < settings.MinSegmentSeconds) continue;
        result.push_back(seg);
    }

    outSegments = std::move(result);
    return true;
}
