"""Learned thrust timing: a model trained on the user's own hand-made scripts.

Inputs (60 steps/s): camera-corrected motion per screen cell (velocity and its
running position), camera affine + cut signal, five percussive onset bands +
loudness + how harmonic (musical) the sound is, and beat/downbeat position.
Outputs per step: IN event, OUT event, "the user would be scripting here", and
the user's normalised position (0 = out, 1 = in).

A non-causal dilated temporal conv net sees ~8.5 s either side of every step,
which is what lets it carry a rhythm through a camera pan and snap back to
thrust sounds or visible strokes when they return.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import gpu

RATE = 60.0                  # model steps per second
AUDIO_TRACKS = ("full", "perc", "low", "mid", "high", "rms", "harm_share")


# ---------------------------------------------------------------- features
def audio_tracks(wav: str | Path, hop_s: float = 0.01) -> dict:
    """Onset tracks at hop_s: spectral flux of full band and of the percussive
    part (HPSS) in low/mid/high bands, loudness, and harmonic share (music vs bodies)."""
    import librosa
    import soundfile as sf
    y, sr = sf.read(str(wav), dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    hop = int(round(hop_s * sr))
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    H, P = librosa.decompose.hpss(S, kernel_size=(17, 17))

    def flux(M, lo, hi):
        band = (freqs >= lo) & (freqs < hi)
        L = np.log1p(100 * M[band])
        d = np.maximum(np.diff(L, axis=1, prepend=L[:, :1]), 0).mean(0)
        return d.astype(np.float32)

    return dict(
        full=flux(S, 30, 11000), perc=flux(P, 30, 11000),
        low=flux(P, 40, 300), mid=flux(P, 300, 2000), high=flux(P, 2000, 9000),
        rms=np.sqrt((S ** 2).mean(0)).astype(np.float32),
        harm_share=(H.sum(0) / (H.sum(0) + P.sum(0) + 1e-9)).astype(np.float32),
        hop_s=np.float32(hop / sr))


def _resample(x: np.ndarray, src_rate: float, n: int, how: str = "linear") -> np.ndarray:
    """(T, ...) at src_rate -> (n, ...) at RATE. 'max' keeps short onsets."""
    t_dst = np.arange(n) / RATE
    if how == "max":
        from scipy.ndimage import maximum_filter1d
        k = max(int(round(src_rate / RATE)), 1)
        x = maximum_filter1d(x, size=k + 1, axis=0, mode="nearest")
    t_src = np.arange(len(x)) / src_rate
    flat = x.reshape(len(x), -1)
    out = np.stack([np.interp(t_dst, t_src, flat[:, j]) for j in range(flat.shape[1])], 1)
    return out.reshape((n,) + x.shape[1:])


def _local_z(x: np.ndarray, w: int) -> np.ndarray:
    from scipy.ndimage import uniform_filter1d
    mu = uniform_filter1d(x, w, axis=0, mode="nearest")
    sd = np.sqrt(np.maximum(uniform_filter1d(x * x, w, axis=0, mode="nearest") - mu * mu, 1e-8))
    return (x - mu) / (sd + 1e-3)


def build_features(flow: dict, audio: dict, beats: np.ndarray, downbeats: np.ndarray) -> np.ndarray:
    """-> (T, D) float32 at RATE. `flow` = motion.extract_flow_cells output,
    `audio` = study/pipeline audio tracks at their own hop."""
    from scipy.signal import butter, sosfiltfilt
    from . import motion
    fps = float(flow["fps"])
    cells = flow["cells"].astype(np.float64)
    T_src = len(cells)
    dur = T_src / fps
    n = int(dur * RATE)
    cuts = motion.cut_frames(flow["hist_diff"], fps).astype(np.float64)

    X = cells.reshape(T_src, -1)
    scale = np.percentile(np.abs(X), 95) + 1e-6
    X = X / scale
    # running position per cell, reset at cuts and high-passed so it does not drift
    seg = np.cumsum(cuts)
    P = np.cumsum(X, 0)
    for sid in np.unique(seg):
        m = seg == sid
        P[m] -= P[m][0]
    if T_src > 30:
        P = sosfiltfilt(butter(2, 0.3, btype="high", fs=fps, output="sos"), P, axis=0)
    aff = flow["affine"].astype(np.float64)
    aff = aff / (np.percentile(np.abs(aff), 95, axis=0) + 1e-6)
    hd = flow["hist_diff"].astype(np.float64)[:, None]
    vis = np.concatenate([X, np.clip(P, -20, 20) / 4.0, aff, hd, cuts[:, None]], 1)
    vis = _resample(vis, fps, n)
    cut_imp = _resample(cuts[:, None], fps, n, how="max")
    vis[:, -1:] = cut_imp

    hop = float(audio["hop_s"])
    A = np.stack([np.asarray(audio[k], np.float64) for k in AUDIO_TRACKS], 1)
    A[:, :6] = np.log1p(np.maximum(A[:, :6], 0) * 10)
    Az = _local_z(A, int(3.0 / hop))
    Az[:, 6] = A[:, 6]                           # harmonic share stays absolute (music vs bodies)
    Az = _resample(Az, 1.0 / hop, n, how="max")
    if audio.get("no_audio") is not None and int(np.asarray(audio["no_audio"])):
        Az[:] = 0.0

    t = np.arange(n) / RATE
    bf = np.zeros((n, 4))
    if len(beats) >= 4:
        for col, ev in ((0, beats), (1, downbeats)):
            if len(ev):
                j = np.clip(np.searchsorted(ev, t), 1, len(ev) - 1)
                d = np.minimum(np.abs(t - ev[j - 1]), np.abs(ev[j] - t))
                bf[:, col] = np.exp(-0.5 * (d / 0.03) ** 2)
        i = np.clip(np.searchsorted(beats, t) - 1, 0, len(beats) - 2)
        ph = np.clip((t - beats[i]) / np.maximum(beats[i + 1] - beats[i], 1e-3), 0, 1)
        inside = (t >= beats[0]) & (t <= beats[-1])
        bf[:, 2] = np.where(inside, np.sin(2 * np.pi * ph), 0)
        bf[:, 3] = np.where(inside, np.cos(2 * np.pi * ph), 0)
    F = np.concatenate([vis, np.clip(Az, -6, 6), bf], 1)
    return np.nan_to_num(F).astype(np.float32)


def n_vis_cells(flow_cells_shape) -> int:
    gy, gx = flow_cells_shape[1:3]
    return gy * gx * 2


# ---------------------------------------------------------------- targets
def build_targets(at_ms: np.ndarray, pos: np.ndarray, n: int, max_gap_ms: float = 1500.0) -> dict:
    """User script -> per-step targets at RATE.
    in/out: 1 at the step nearest each IN (top) / OUT (bottom) turn.
    active: inside a run of turns closer than max_gap_ms.
    level: position normalised to the local out..in pair (0..1)."""
    from . import funscript_io as fio
    tt, tp, tk = fio.extremes(at_ms, pos, min_depth=3.0)
    ev_in, ev_out, act = np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32)
    idx = np.round(tt / 1000.0 * RATE).astype(int)
    ok = (idx >= 0) & (idx < n)
    ev_in[idx[ok & (tk == 1)]] = 1
    ev_out[idx[ok & (tk == -1)]] = 1
    for a, b in zip(tt[:-1], tt[1:]):
        if b - a <= max_gap_ms:
            act[int(a / 1000 * RATE):int(b / 1000 * RATE) + 1] = 1
    t = np.arange(n) / RATE * 1000
    level = np.full(n, 0.5, np.float32)
    if len(tt) >= 2:
        u = np.interp(t, at_ms, pos)
        tops = np.interp(t, tt[tk == 1], tp[tk == 1]) if np.sum(tk == 1) else np.full(n, 100.0)
        bots = np.interp(t, tt[tk == -1], tp[tk == -1]) if np.sum(tk == -1) else np.zeros(n)
        level = np.clip((u - bots) / np.maximum(tops - bots, 1.0), 0, 1).astype(np.float32)
    return dict(ev_in=ev_in, ev_out=ev_out, active=act, level=level)


# ---------------------------------------------------------------- model
def make_model(d_in: int, width: int = 128, dilations=(1, 2, 4, 8, 16, 32, 64, 128), kernel: int = 5,
               dropout: float = 0.1):
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, dil):
            super().__init__()
            pad = dil * (kernel - 1) // 2
            self.conv = nn.Conv1d(width, width, kernel, padding=pad, dilation=dil)
            self.norm = nn.GroupNorm(8, width)
            self.mix = nn.Conv1d(width, width, 1)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            h = torch.nn.functional.gelu(self.norm(self.conv(x)))
            return x + self.drop(self.mix(h))

    class ThrustNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Conv1d(d_in, width, 1), nn.GELU(), nn.Conv1d(width, width, 3, padding=1))
            self.blocks = nn.Sequential(*[Block(d) for d in dilations])
            self.head = nn.Conv1d(width, 4, 1)      # in, out, active (logits), level

        def forward(self, x):                        # x (B, T, D)
            h = self.blocks(self.inp(x.transpose(1, 2)))
            return self.head(h).transpose(1, 2)      # (B, T, 4)

    return ThrustNet()


# ---------------------------------------------------------------- decoding
def decode(p_in: np.ndarray, p_out: np.ndarray, p_act: np.ndarray, thr: float = 0.3,
           act_thr: float = 0.5, min_gap_s: float = 0.07) -> tuple[np.ndarray, np.ndarray]:
    """Per-step probabilities -> alternating IN/OUT points (at_ms, pos 100/0).
    Peaks above thr inside active stretches; two peaks of the same kind in a row
    keep only the stronger one, so the script always alternates."""
    from scipy.signal import find_peaks
    dist = max(int(min_gap_s * RATE), 1)
    ev = []
    for p, kind in ((p_in, 1), (p_out, -1)):
        pk, prop = find_peaks(p, height=thr, distance=dist)
        for i, h in zip(pk, prop["peak_heights"]):
            if p_act[i] >= act_thr:
                # sub-step refinement: parabola through the peak
                di = 0.0
                if 0 < i < len(p) - 1:
                    den = p[i - 1] - 2 * p[i] + p[i + 1]
                    if den < 0:
                        di = float(np.clip(0.5 * (p[i - 1] - p[i + 1]) / den, -0.5, 0.5))
                ev.append(((i + di) / RATE * 1000.0, kind, float(h)))
    ev.sort()
    out = []
    for e in ev:
        if out and out[-1][1] == e[1]:
            if e[2] > out[-1][2]:
                out[-1] = e
            continue
        if out and e[0] - out[-1][0] < min_gap_s * 1000:
            if e[2] > out[-1][2]:
                out[-1] = e
            continue
        out.append(e)
    at = np.array([e[0] for e in out])
    pos = np.array([100.0 if e[1] == 1 else 0.0 for e in out])
    return at, pos


def load_checkpoint(path: str | Path, device: str = "cuda"):
    import torch
    ck = torch.load(path, map_location=device, weights_only=False)
    model = make_model(ck["d_in"], **ck.get("arch", {})).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck


def predict(model, F: np.ndarray, device: str = "cuda", chunk: int = 60 * int(RATE), ctx: int = 12 * int(RATE)):
    """Run the net over a whole video in overlapping chunks -> (T, 4) with
    sigmoid applied to the three event/active heads."""
    import torch
    T = len(F)
    out = np.zeros((T, 4), np.float32)
    with torch.inference_mode():
        s = 0
        while s < T:
            a, b = max(s - ctx, 0), min(s + chunk + ctx, T)
            x = torch.from_numpy(F[a:b][None]).to(device)
            y = model(x)[0].float().cpu().numpy()
            e = min(s + chunk, T)
            out[s:e] = y[s - a:e - a]
            s = e
    out[:, :3] = 1 / (1 + np.exp(-out[:, :3]))
    return out


DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "thrust" / "thrust_v1.pt"


def hybrid(og_at, og_pos, og_active, og_fps, P, thr: float, act_thr: float = 0.3, cut_ms=()):
    """OG motion turns keep their precise timing; the model decides which way up each
    shot is (vote of p_in - p_out at every OG turn between two cuts), and model events fill the
    stretches where the OG tracker is inactive (camera away, beat-only, off-screen).
    Measured on 50 held-out videos: right way up 86% (OG 70%, model alone 76%),
    F1@100 0.414 (OG 0.386), tempo 0.635 (OG 0.646), off-screen points 28% (OG 1%)."""
    og_at, og_pos = np.asarray(og_at, float), np.asarray(og_pos, float)
    k_og = np.where(og_pos > 50, 1, -1)
    kind = k_og.copy()
    if len(og_at):
        step = np.clip(np.round(og_at / 1000 * RATE).astype(int), 0, len(P) - 1)
        lo, hi = np.clip(step - 2, 0, len(P) - 1), np.clip(step + 2, 0, len(P) - 1)
        pin = np.array([P[a:b + 1, 0].max() for a, b in zip(lo, hi)])
        pout = np.array([P[a:b + 1, 1].max() for a, b in zip(lo, hi)])
        sid = np.searchsorted(np.sort(np.asarray(cut_ms, float)), og_at)
        for s in np.unique(sid):
            m = sid == s
            if np.sum((pin[m] - pout[m]) * k_og[m]) < 0:
                kind[m] = -k_og[m]
    ev = [(t, int(k), 1.0) for t, k in zip(og_at, kind)]
    mt, mp = decode(P[:, 0], P[:, 1], P[:, 2], thr=thr, act_thr=act_thr)
    if len(mt):
        fi = np.clip(np.round(mt / 1000 * og_fps).astype(int), 0, len(og_active) - 1)
        blind = ~np.asarray(og_active, bool)[fi]
        ev += [(t, 1 if p > 50 else -1, 0.5) for t, p in zip(mt[blind], mp[blind])]
    ev.sort()
    out = []
    for e in ev:
        if out and out[-1][1] == e[1]:
            if e[2] > out[-1][2]:
                out[-1] = e
            continue
        out.append(e)
    flipped = float(np.mean(kind != k_og)) if len(og_at) else 0.0
    return (np.array([e[0] for e in out]), np.array([100.0 if e[1] == 1 else 0.0 for e in out]),
            dict(flipped_share=flipped, filled=int(sum(1 for e in out if e[2] < 1.0))))


def script_from_video(video, work_dir, flow: dict, checkpoint: str | Path = DEFAULT_CHECKPOINT,
                      has_audio: bool = True, og: tuple | None = None, log=print):
    """Learned in/out points for a whole video -> (at_ms, pos 100/0, info).
    Sound features and full-mix beats are computed exactly as in training and
    cached in work_dir; `flow` is the pipeline's extract_flow_cells result.
    With `og` = (at_ms, pos, active, fps, cut_ms, conf) from the OG motion tracker the result is the
    hybrid (OG timing, model direction per shot), then thrust-sound direction (no-music videos only),
    then rhythm continuation through gaps (rhythm_fill.py); without it, the model alone."""
    from . import media
    from .audio_beats import detect_beats
    work_dir = Path(work_dir)
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"no thrust model at {checkpoint} (run '1 - Setup (run once).bat' to download the models)")
    audio_npz, beats_json = work_dir / "thrust_audio.npz", work_dir / "beats_fullmix.json"
    beats = downbeats = np.array([])
    mel_npz = work_dir / "mel_flux.npz"
    flux = None
    if has_audio:
        from . import rhythm_fill
        wav = work_dir / "audio_22k_mono.wav"
        if not audio_npz.exists() or not beats_json.exists() or not mel_npz.exists():
            media.extract_audio(video, wav, sr=22050, channels=1)
        if not audio_npz.exists():
            np.savez_compressed(audio_npz, **audio_tracks(wav))
        if not mel_npz.exists():
            np.savez_compressed(mel_npz, flux=rhythm_fill.mel_flux(wav))
        beats, downbeats = detect_beats(wav, beats_json)
        wav.unlink(missing_ok=True)
        audio = dict(np.load(audio_npz))
        flux = np.load(mel_npz)["flux"]
    else:
        n = int(len(flow["cells"]) / flow["fps"] / 0.01) + 1
        z = np.zeros(n, np.float32)
        audio = dict(full=z, perc=z, low=z, mid=z, high=z, rms=z, harm_share=z, hop_s=np.float32(0.01),
                     no_audio=np.int8(1))
    F = build_features(flow, audio, np.asarray(beats), np.asarray(downbeats))
    model, ck = load_checkpoint(checkpoint)
    P = predict(model, F)
    thr = float(ck.get("thr", 0.3))
    act_thr = float(ck.get("act_thr", 0.3))
    extra = {}
    if og is not None:
        from . import rhythm_fill as rf
        o_at, o_pos, o_active, o_fps, o_cuts, o_conf = og
        at, pos, extra = hybrid(o_at, o_pos, o_active, o_fps, P, thr=thr, act_thr=act_thr, cut_ms=o_cuts)
        # thrust sounds overrule the direction of a shot, but only in videos without music and only when strong
        # (held-out: neutral; test video A: matches the user's notes where the sound is strong)
        duration_s = len(flow["cells"]) / flow["fps"]
        extra["music"] = rf.has_music(beats, duration_s)
        if flux is not None and not extra["music"]:
            pos, flips = rf.sound_direction(at, pos, rf.impact_track(flux), cut_ms=o_cuts, seg_s=1e9, margin=1.0)
            extra["sound_flips"] = flips
        # continue the rhythm through gaps: tracker inactive, its rhythm fell apart, or turns dropped out
        bad = rf.unreliable(o_conf, o_fps, np.asarray(o_at, float), conf_thr=0.4, cv_thr=0.4)
        gaps = rf.gaps_from(o_active, o_fps, np.asarray(o_at, float), max_turn_gap_ms=1200.0, bad=bad)
        at, pos, filled = rf.fill(at, pos, gaps, P)
        extra["gaps_filled"] = filled
    else:
        at, pos = decode(P[:, 0], P[:, 1], P[:, 2], thr=thr, act_thr=act_thr)
    try:
        import torch
        del model
        torch.cuda.empty_cache()
    except Exception:
        pass
    active = P[:, 2] >= float(ck.get("act_thr", 0.3))
    how = "hybrid" if og is not None else "model only"
    detail = (f", direction flipped vs OG on {extra.get('flipped_share', 0):.0%} of turns, "
              f"{len(extra.get('sound_flips', []))} shots turned by thrust sound"
              f"{' (music: sound not used)' if extra.get('music') else ''}, "
              f"{len(extra.get('gaps_filled', []))} gaps carried through with the rhythm") if og is not None else ""
    log(f"  learned strokes ({how}, {checkpoint.name}, step {ck.get('step')}): {len(at)} points{detail}")
    return at, pos, dict(active=active, rate=RATE, checkpoint=str(checkpoint), **extra)


# ---------------------------------------------------------------- v2: fine visual input
FINE_FLOW = (18, 32)
FINE_GRAY = (36, 64)
N_MEL = 64


def mel_features(flux: np.ndarray, n: int, hop_s: float = 0.01) -> np.ndarray:
    """64-band onset flux (T100, 64) -> locally z-scored, clipped, max-resampled to RATE (n, 64)."""
    if flux is None or len(flux) == 0:
        return np.zeros((n, N_MEL), np.float32)
    z = _local_z(flux.astype(np.float64), int(3.0 / hop_s))
    return np.clip(_resample(z, 1.0 / hop_s, n, how="max"), -6, 6).astype(np.float32)


def fine_frames(fine: dict) -> tuple[np.ndarray, np.ndarray]:
    """fine_v1.npz -> (flow (T,2,18,32) float16 scaled, gray (T,1,36,64) uint8)."""
    flow = fine["flow"].astype(np.float32)
    scale = np.percentile(np.abs(flow), 95) + 1e-6
    flow = np.clip(flow / scale, -8, 8).transpose(0, 3, 1, 2).astype(np.float16)
    gray = fine["gray"][:, None]
    T = min(len(flow), len(gray))
    return flow[:T], gray[:T]


def make_model_v2(d_seq: int, width: int = 160, emb: int = 128, dilations=(1, 2, 4, 8, 16, 32, 64, 128),
                  kernel: int = 5, dropout: float = 0.1):
    """Per-frame CNN over fine flow + greyscale (+ its frame difference) -> embedding at 30 fps,
    upsampled to RATE and joined with the v1 sequence features + mel onsets, then the TCN."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class FrameEnc(nn.Module):
        def __init__(self):
            super().__init__()
            self.g = nn.Sequential(nn.Conv2d(2, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 16, 3, stride=2, padding=1),
                                   nn.GELU())                                     # 36x64 -> 18x32
            self.body = nn.Sequential(
                nn.Conv2d(16 + 2, 48, 3, padding=1), nn.GELU(),
                nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),   # 9x16
                nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.GroupNorm(8, 96), nn.GELU(),   # 5x8
                nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU())  # 3x4
            self.out = nn.Linear(128 * 3 * 4, emb)

        def forward(self, flow, gray, gdiff):          # (N,2,18,32) (N,1,36,64) (N,1,36,64)
            h = self.g(torch.cat([gray, gdiff], 1))
            h = self.body(torch.cat([h, flow], 1))
            h = F.adaptive_avg_pool2d(h, (3, 4)).flatten(1)
            return self.out(h)

    class Block(nn.Module):
        def __init__(self, dil):
            super().__init__()
            self.conv = nn.Conv1d(width, width, kernel, padding=dil * (kernel - 1) // 2, dilation=dil)
            self.norm = nn.GroupNorm(8, width)
            self.mix = nn.Conv1d(width, width, 1)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            return x + self.drop(self.mix(F.gelu(self.norm(self.conv(x)))))

    class ThrustNetV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = FrameEnc()
            self.inp = nn.Sequential(nn.Conv1d(d_seq + emb, width, 1), nn.GELU(), nn.Conv1d(width, width, 3, padding=1))
            self.blocks = nn.Sequential(*[Block(d) for d in dilations])
            self.head = nn.Conv1d(width, 4, 1)

        def forward(self, seq, flow, gray):
            # seq (B,T,D) at RATE; flow (B,Tf,2,18,32) fp; gray (B,Tf,1,36,64) in [0,1]; Tf = T/2 (30 fps)
            B, Tf = flow.shape[:2]
            gd = torch.cat([torch.zeros_like(gray[:, :1]), gray[:, 1:] - gray[:, :-1]], 1)
            e = self.enc(flow.flatten(0, 1), gray.flatten(0, 1), gd.flatten(0, 1)).view(B, Tf, -1)
            e = F.interpolate(e.transpose(1, 2), size=seq.shape[1], mode="linear", align_corners=False)
            h = self.inp(torch.cat([seq.transpose(1, 2), e], 1))
            return self.head(self.blocks(h)).transpose(1, 2)

    return ThrustNetV2()


def predict_v2(model, seq: np.ndarray, flow: np.ndarray, gray: np.ndarray, device: str = "cuda",
               chunk_s: float = 40.0, ctx_s: float = 10.0):
    """Whole-video inference for v2 in overlapping chunks -> (T, 4), sigmoid on the first three."""
    import torch
    T = len(seq)
    out = np.zeros((T, 4), np.float32)
    ch, cx = int(chunk_s * RATE) // 2 * 2, int(ctx_s * RATE) // 2 * 2
    with torch.inference_mode():
        s = 0
        while s < T:
            a, b = max(s - cx, 0), min(s + ch + cx, T)
            a -= a % 2
            fa, fb = a // 2, min((b + 1) // 2, len(flow))
            x = torch.from_numpy(seq[a:b][None]).to(device)
            fl = torch.from_numpy(flow[fa:fb][None].astype(np.float32)).to(device)
            gr = torch.from_numpy(gray[fa:fb][None].astype(np.float32) / 255.0).to(device)
            with gpu.amp():                    # bfloat16 only where the card has it (not on Pascal / Turing)
                y = model(x, fl, gr)[0].float().cpu().numpy()
            e = min(s + ch, T)
            out[s:e] = y[s - a:e - a]
            s = e
    out[:, :3] = 1 / (1 + np.exp(-out[:, :3]))
    return out
