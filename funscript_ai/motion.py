"""Visual stroke signal from optical flow.

1. RAFT-small optical flow on the GPU at low resolution. A robust affine
   global-motion model (pan + zoom + rotation + shear) is fitted per frame and
   subtracted, so camera moves and PMV zoom/shake effects don't masquerade as
   strokes. The residual is average-pooled to a grid of cells. The affine
   parameters are kept: their zoom/shake pulses are the "visual beat".
   Cuts are scored by colour-histogram change (robust to fast motion and to
   limited animation, unlike pixel difference).
2. Per sliding window, the first principal component of the band-passed cell
   flows is the dominant repetitive motion; windows are overlap-added with
   their sign aligned to what came before. Periodicity (autocorrelation peak)
   decides whether a window is a real rhythmic stroke or just movement.
3. Integrate to a position, find direction reversals, normalise by the local
   median stroke size and keep reversals that are a real share of a stroke.
The global sign of step 2 is arbitrary; polarity is resolved separately.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt

from . import gpu, media, funscript_io as fio

GRID = (9, 12)  # rows, cols
HIST_BINS = 32
CACHE_VERSION = 2


def _size8(info: media.VideoInfo, width: int) -> tuple[int, int]:
    h = int(round(width * info.height / info.width / 8)) * 8
    return width, max(h, 64)


def _remove_global_motion(flow, A, iters: int = 2, keep_q: float = 0.6):
    """flow (n,2,h,w); A (hw,3) = [1,x,y]. Weighted least squares with the
    largest residuals (the moving subject) dropped each pass."""
    import torch
    n = flow.shape[0]
    f = flow.flatten(2).transpose(1, 2)  # (n,hw,2)
    wgt = torch.ones(n, 1, f.shape[1], device=flow.device)
    eye = torch.eye(3, device=flow.device) * 1e-4
    for it in range(iters + 1):
        Aw = A.T[None] * wgt  # (n,3,hw)
        coef = torch.linalg.solve(Aw @ A + eye, Aw @ f)  # (n,3,2)
        res = f - A @ coef  # (n,hw,2)
        if it < iters:
            mag = res.norm(dim=2)
            thr = torch.quantile(mag, keep_q, dim=1, keepdim=True)
            wgt = (mag <= thr).float()[:, None, :]
    return res.transpose(1, 2).reshape(flow.shape), coef.reshape(n, 6)


def _color_hist(x, bins: int = HIST_BINS):
    """x (n,3,h,w) in [-1,1] -> (n, 3*bins) normalised per-channel histograms."""
    import torch
    n, c, h, w = x.shape
    q = ((x + 1) * 0.5 * bins).long().clamp_(0, bins - 1).flatten(2)
    q = q + (torch.arange(c, device=x.device) * bins)[None, :, None]
    out = torch.zeros(n, c * bins, device=x.device)
    out.scatter_add_(1, q.flatten(1), torch.ones(1, device=x.device).expand(n, c * h * w))
    return out / (h * w)


def extract_flow_cells(video, cache_npz, fps: float = 30.0, width: int = 320, device: str = "cuda",
                       updates: int = 8, batch: int = 48, log=print, raft: str = "small") -> dict:
    """Return dict: cells (T,gy,gx,2) residual px/frame, affine (T,6), hist_diff (T,),
    fps. Index k is the motion arriving at frame k (time k / fps); row 0 is zeros."""
    cache_npz = Path(cache_npz)
    if cache_npz.exists():
        d = np.load(cache_npz)
        if int(d.get("version", 0)) == CACHE_VERSION and media.covers(video, len(d["cells"]), float(d["fps"])):
            return {k: d[k] for k in ("cells", "affine", "hist_diff")} | {"fps": float(d["fps"])}
        log(f"    cached motion in {cache_npz.name} is outdated or does not cover the whole video - measuring again")
    import torch
    import torch.nn.functional as F
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights, raft_large, Raft_Large_Weights

    info = media.probe(video)
    w, h = _size8(info, width)
    gy, gx = GRID
    # raft="large": the bigger, more accurate RAFT (tests/tracking_large.py); small is the default everywhere
    model = (raft_large(weights=Raft_Large_Weights.DEFAULT) if raft == "large"
             else raft_small(weights=Raft_Small_Weights.DEFAULT)).to(device).eval()
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, device=device),
                            torch.linspace(-1, 1, w, device=device), indexing="ij")
    A = torch.stack([torch.ones_like(xs), xs, ys], -1).reshape(-1, 3)
    cells_out = [np.zeros((1, gy, gx, 2), np.float32)]
    aff_out = [np.zeros((1, 6), np.float32)]
    hd_out = [np.zeros(1, np.float32)]
    prev, prev_hist = None, None
    done, t0, next_log = 0, time.perf_counter(), 120.0
    with torch.inference_mode():
        for chunk in media.iter_frames(video, fps, w, h, pix="rgb24", batch=batch):
            x = torch.from_numpy(chunk.copy()).to(device).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
            hist = _color_hist(x)
            seq = x if prev is None else torch.cat([prev, x], 0)
            hseq = hist if prev_hist is None else torch.cat([prev_hist, hist], 0)
            if seq.shape[0] >= 2:
                a, b = seq[:-1].contiguous(), seq[1:].contiguous()
                flow = gpu.raft_flow(model, a, b, updates)   # halves the batch on small cards
                resid, coef = _remove_global_motion(flow, A)
                cells = F.adaptive_avg_pool2d(resid, (gy, gx)).permute(0, 2, 3, 1)
                cells_out.append(cells.float().cpu().numpy())
                aff_out.append(coef.float().cpu().numpy())
                hd_out.append((0.5 * (hseq[1:] - hseq[:-1]).abs().sum(1) / 3).float().cpu().numpy())
            prev, prev_hist = x[-1:], hist[-1:]
            done += chunk.shape[0]
            if done / fps >= next_log:
                el = time.perf_counter() - t0
                log(f"    flow {done / fps:7.0f}s of {info.duration:.0f}s video  ({done / fps / el:.1f}x realtime)")
                next_log += 120.0
    out = {"cells": np.concatenate(cells_out), "affine": np.concatenate(aff_out),
           "hist_diff": np.concatenate(hd_out), "fps": fps}
    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_npz, version=CACHE_VERSION, size=np.array([w, h]), **out)
    return out


def cut_frames(hist_diff: np.ndarray, fps: float, thresh: float = 0.30, min_gap_s: float = 0.25) -> np.ndarray:
    """Hard cuts: colour-histogram change above `thresh` that is also the local
    maximum within +-min_gap_s (so a flash or a dissolve counts once)."""
    n = max(int(min_gap_s * fps) * 2 + 1, 3)
    from scipy.ndimage import maximum_filter1d
    return (hist_diff >= thresh) & (hist_diff >= maximum_filter1d(hist_diff, n))


def dominant_velocity(cells: np.ndarray, cuts: np.ndarray, fps: float, win_s: float = 4.0,
                      hop_s: float = 0.5, band=(0.4, 7.0)):
    """Return (velocity (T,), periodicity (T,) 0-1, share (T,) PC1 variance share)."""
    T = len(cells)
    X = cells.reshape(T, -1).astype(np.float64)
    ci = np.nonzero(cuts)[0]
    for off in (-1, 0, 1):
        X[np.clip(ci + off, 0, T - 1)] = 0.0
    sos = butter(2, [band[0], min(band[1], fps * 0.45)], btype="band", fs=fps, output="sos")
    Xf = sosfiltfilt(sos, X, axis=0)
    win, hop = max(int(win_s * fps), 8), max(int(hop_s * fps), 1)
    lag_lo, lag_hi = max(int(0.12 * fps), 2), int(2.5 * fps)
    v, wsum = np.zeros(T), np.full(T, 1e-9)
    period, share = np.zeros(T), np.zeros(T)
    taper = np.hanning(win) + 1e-3
    starts = list(range(0, max(T - win, 0) + 1, hop))
    if starts and starts[-1] != max(T - win, 0):
        starts.append(max(T - win, 0))
    for s in starts:
        seg = Xf[s:s + win]
        seg = seg - seg.mean(0)
        _, S, Vt = np.linalg.svd(seg, full_matrices=False)
        proj = seg @ Vt[0]
        L = len(proj)
        tp = taper[:L]
        ov = wsum[s:s + L] > 1e-6
        if ov.sum() > 4 and np.dot(v[s:s + L][ov] / wsum[s:s + L][ov], proj[ov]) < 0:
            proj = -proj
        p0 = proj - proj.mean()
        ac = np.correlate(p0, p0, "full")[L - 1:]
        ac = ac / max(ac[0], 1e-12)
        per = float(ac[lag_lo:min(lag_hi, L - 1)].max()) if L - 1 > lag_lo else 0.0
        v[s:s + L] += proj * tp
        period[s:s + L] += max(per, 0.0) * tp
        share[s:s + L] += float(S[0] ** 2 / max((S ** 2).sum(), 1e-12)) * tp
        wsum[s:s + L] += tp
    return v / wsum, period / wsum, share / wsum


def velocity_to_script(v: np.ndarray, periodicity: np.ndarray, cuts: np.ndarray, fps: float,
                       min_periodicity: float = 0.35, rel_depth: float = 0.35, min_rel_amp: float = 0.12):
    """Return (at_ms, pos, info). Turning points = velocity reversals kept only
    when the move since the last kept one is >= rel_depth of the local median
    stroke; active = periodic enough and strokes not negligible vs the video."""
    T = len(v)
    pos = np.cumsum(v)
    seg_id = np.cumsum(cuts)
    for sid in np.unique(seg_id):
        m = seg_id == sid
        pos[m] -= pos[m].mean()
    if T > 30:
        pos = sosfiltfilt(butter(2, 0.25, btype="high", fs=fps, output="sos"), pos)
    t_ms = np.arange(T) / fps * 1000.0
    # raw reversals -> local median stroke size envelope
    sgn = np.sign(np.diff(pos))
    raw = np.nonzero(sgn[1:] * sgn[:-1] < 0)[0] + 1
    if len(raw) < 4:
        return np.array([]), np.array([]), dict(active_share=0.0)
    d = np.abs(np.diff(pos[raw]))
    big = d >= 0.25 * np.median(d)  # ignore jitter when measuring stroke size
    mids = (raw[1:] + raw[:-1]) / 2
    env_pts = median_filter(np.where(big, d, np.nan)[big], size=9, mode="nearest") if big.sum() > 0 else d
    env = np.interp(np.arange(T), mids[big] if big.sum() > 0 else mids, env_pts)
    env = np.maximum(env, 1e-6)
    norm = pos / env * 100.0
    et, _, ek = fio.extremes(t_ms, norm, min_depth=rel_depth * 100.0)
    idx = np.clip(np.round(et / 1000.0 * fps).astype(int), 0, T - 1)
    active = (periodicity >= min_periodicity) & (env >= min_rel_amp * np.percentile(env, 90))
    keep = active[idx]
    et, ek = et[keep], ek[keep]
    return et, np.where(ek == 1, 100.0, 0.0), dict(active_share=float(active.mean()),
                                                   cuts=int(cuts.sum()))


def _zero_crossings(v: np.ndarray, fps: float):
    """Sub-frame velocity sign changes -> (t_ms, kind, frame_idx). kind +1 = the
    position peaked (velocity went + to -)."""
    s = np.signbit(v)
    idx = np.nonzero(s[1:] != s[:-1])[0]
    frac = v[idx] / (v[idx] - v[idx + 1] + 1e-12)
    # v[k] is the motion from frame k-1 to k, i.e. centred at frame k - 0.5
    return (idx + frac - 0.5) / fps * 1000.0, np.where(v[idx] > 0, 1, -1), idx


def _snap_to_extremes(t_ms, kind, vb, f0t, fps, frac):
    """Move each narrowband turn to the true peak/trough of the broadband
    position within +-frac of the local stroke period (parabolic sub-frame
    refinement), keeping times strictly increasing."""
    T = len(vb)
    pos = np.cumsum(vb)
    if T > 30:
        pos = sosfiltfilt(butter(2, 0.25, btype="high", fs=fps, output="sos"), pos)
    out, last = t_ms.copy(), -np.inf
    for i, (t, k) in enumerate(zip(t_ms, kind)):
        c = t / 1000.0 * fps
        ci = int(np.clip(round(c), 0, T - 1))
        r = max(int(frac * fps / max(f0t[ci], 0.2)), 1)
        lo, hi = max(int(np.floor(c)) - r, 0), min(int(np.ceil(c)) + r + 1, T)
        if hi - lo < 1:
            continue
        j = lo + int(np.argmax(pos[lo:hi] * k))
        dj = 0.0
        if 0 < j < T - 1:
            y0, y1, y2 = pos[j - 1] * k, pos[j] * k, pos[j + 1] * k
            den = y0 - 2 * y1 + y2
            if den < 0:
                dj = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
        nt = (j + dj) / fps * 1000.0
        if nt <= last:
            nt = t if t > last else last + 1.0
        out[i], last = nt, nt
    return out


def narrowband_strokes(cells: np.ndarray, cuts: np.ndarray, fps: float, win_s: float = 4.0,
                       hop_s: float = 0.5, band=(0.4, 7.5), harmonic_ratio: float = 0.5,
                       min_conf: float = 0.15, min_rel_amp: float = 0.15, snap_frac: float = 0.0,
                       strong_conf: float | None = None, abs_floor_px: float = 0.0,
                       orient: str = "chain", vy_min_share: float = 0.15):
    """Spectrum-guided strokes. Per window: PC1 of the cell flows, its dominant
    frequency f0 (optionally stepping up to 2*f0 when that harmonic carries at
    least `harmonic_ratio` of the peak power), a zero-phase band-pass around f0,
    overlap-added with sign alignment. Every reversal of the result is a turn,
    so the stroke rate follows f0 by construction.

    conf = share of in-band power near f0 (how clearly one rhythm dominates).
    Returns (at_ms, pos, info) with per-frame f0/conf/amp tracks in info."""
    T = len(cells)
    X = cells.reshape(T, -1).astype(np.float64)
    ci = np.nonzero(cuts)[0]
    for off in (-1, 0, 1):
        X[np.clip(ci + off, 0, T - 1)] = 0.0
    hi = min(band[1], fps * 0.45)
    Xf = sosfiltfilt(butter(2, [band[0], hi], btype="band", fs=fps, output="sos"), X, axis=0)
    win, hop = max(int(win_s * fps), 16), max(int(hop_s * fps), 1)
    nfft = 4096
    freqs = np.fft.rfftfreq(nfft, 1 / fps)
    bmask = (freqs >= band[0]) & (freqs <= hi)
    v, vb, wsum = np.zeros(T), np.zeros(T), np.full(T, 1e-9)
    f0t, conft, ampt = np.zeros(T), np.zeros(T), np.zeros(T)
    hann = np.hanning(win)
    starts = list(range(0, max(T - win, 0) + 1, hop))
    if starts and starts[-1] != max(T - win, 0):
        starts.append(max(T - win, 0))
    for s in starts:
        seg = Xf[s:s + win]
        seg = seg - seg.mean(0)
        L = len(seg)
        if L < 16:
            continue
        _, _, Vt = np.linalg.svd(seg, full_matrices=False)
        proj = seg @ Vt[0]
        P = np.abs(np.fft.rfft(proj * hann[:L], nfft)) ** 2
        Pb = np.where(bmask, P, 0.0)
        k = int(np.argmax(Pb))
        f0 = freqs[k]
        if harmonic_ratio > 0 and 2 * f0 <= hi:
            k2 = int(np.argmin(np.abs(freqs - 2 * f0)))
            lo2 = max(k2 - 4, 0)
            j = lo2 + int(np.argmax(Pb[lo2:k2 + 5]))
            if Pb[j] >= harmonic_ratio * Pb[k]:
                k, f0 = j, freqs[j]
        near = np.abs(freqs - f0) <= max(0.25, 0.15 * f0)
        conf = float(Pb[near].sum() / max(Pb.sum(), 1e-12))
        sos = butter(2, [max(0.6 * f0, 0.2), min(1.5 * f0, fps * 0.45)], btype="band", fs=fps, output="sos")
        nb = sosfiltfilt(sos, proj)
        tp = hann[:L] + 1e-3
        oriented = False
        if orient == "vy":
            # Physical orientation per window: point PC1 so its vertical (screen-down) loadings sum positive, i.e.
            # moving down on screen = position up. Chaining signs window-to-window drifted inside compilations
            # (tests/exp_polarity.py: 65 sign flips in 166 confident windows on one test video, 44/238 on another) and was
            # inverted for a whole POV clip (corr -0.74 vs local vertical motion +0.75).
            ly = Vt[0][1::2]  # columns are per cell [vx, vy]
            if abs(ly.sum()) >= vy_min_share * (np.abs(Vt[0]).sum() + 1e-12):
                if ly.sum() < 0:
                    nb, proj = -nb, -proj
                oriented = True
        ov = wsum[s:s + L] > 1e-6
        if not oriented and ov.sum() > 4 and np.dot(v[s:s + L][ov], nb[ov]) < 0:
            nb, proj = -nb, -proj
        v[s:s + L] += nb * tp
        vb[s:s + L] += proj * tp
        f0t[s:s + L] += f0 * tp
        conft[s:s + L] += conf * tp
        ampt[s:s + L] += float(np.sqrt(np.mean(nb ** 2))) * tp
        wsum[s:s + L] += tp
    v, vb, f0t, conft, ampt = v / wsum, vb / wsum, f0t / wsum, conft / wsum, ampt / wsum
    t_ms, kind, idx = _zero_crossings(v, fps)
    if snap_frac > 0 and len(t_ms):
        t_ms = _snap_to_extremes(t_ms, kind, vb, f0t, fps, snap_frac)
    big_enough = ampt >= min_rel_amp * np.percentile(ampt, 90)
    if strong_conf is not None:
        # a clearly dominant rhythm counts even when small next to the video's biggest scenes
        big_enough |= (conft >= strong_conf) & (ampt >= abs_floor_px)
    active = (conft >= min_conf) & big_enough
    keep = active[idx]
    return (t_ms[keep], np.where(kind[keep] == 1, 100.0, 0.0),
            dict(active_share=float(active.mean()), active=active, v=v, vb=vb, f0=f0t, conf=conft, amp=ampt))


def script_from_video(video, work_dir, fps: float = 30.0, log=print):
    data = extract_flow_cells(video, Path(work_dir) / f"flow_v{CACHE_VERSION}_{int(fps)}fps.npz", fps=fps, log=log)
    cuts = cut_frames(data["hist_diff"], data["fps"])
    v, per, share = dominant_velocity(data["cells"], cuts, data["fps"])
    at, pos, info = velocity_to_script(v, per, cuts, data["fps"])
    return at, pos, info, dict(data=data, cuts=cuts, v=v, periodicity=per, share=share)
