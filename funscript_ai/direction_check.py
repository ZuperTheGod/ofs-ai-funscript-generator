"""Check direction: which stretches of a script may be upside down (100 at OUT instead of IN)?

Three independent clues, 4 s window by window:
  scene  -- inside one camera shot every real IN looks alike and every OUT looks alike (user, 2026-10-24). A stretch
            whose INs look like the shot's OUTs (and its OUTs like the INs) is inconsistent with the rest of that shot.
            Uses the 36x64 picture the program already keeps (fine_v1.npz). Cannot judge a shot that is flipped as a
            whole -- the other two clues cover that.
  ai     -- the learned method's own script goes the opposite way
  motion -- the motion tracker's script goes the opposite way

    python -m funscript_ai.direction_check <script.funscript> <video> [--out flags.json]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import funscript_io as fio

WIN_MS = 4000.0


def _turns(at, pos):
    t, p, k = fio.extremes(np.asarray(at, float), np.asarray(pos, float), 3.0)
    return t, k


def scene_scores(at, pos, gray, fps: float, cut_frames, min_shot_s: float = 6.0):
    """-> list of (window_start_ms, score): score < 0 = this window's INs look like the shot's OUTs.
    The IN-vs-OUT look of the shot is learned from the rest of the shot (the window itself left out)."""
    t, k = _turns(at, pos)
    if len(t) < 8:
        return []
    g = gray.reshape(len(gray), -1).astype(np.float32)
    cuts = np.r_[0, np.nonzero(cut_frames)[0], len(g)]
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if (b - a) / fps < min_shot_s:
            continue
        m = (t >= a / fps * 1000) & (t < b / fps * 1000)
        if m.sum() < 8:
            continue
        idx = np.clip(np.round(t[m] / 1000 * fps).astype(int), a, b - 1)
        f = g[idx] - g[a:b].mean(0)
        f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-6
        kk, tt = k[m], t[m]
        for w0 in np.arange(a / fps * 1000, b / fps * 1000 - WIN_MS / 2, WIN_MS):
            inw = (tt >= w0) & (tt < w0 + WIN_MS)
            rest = ~inw
            if inw.sum() < 4 or (rest & (kk == 1)).sum() < 3 or (rest & (kk == -1)).sum() < 3:
                continue
            d = f[rest & (kk == 1)].mean(0) - f[rest & (kk == -1)].mean(0)
            nd = np.linalg.norm(d)
            if nd < 1e-6:
                continue
            proj = f[inw] @ (d / nd)
            sep = np.std(f[rest] @ (d / nd)) + 1e-6
            score = float(np.mean(proj * kk[inw]) / sep)     # + = INs look like the shot's INs
            out.append((float(w0), score))
    return out


def window_verdicts(ref_at, ref_pos, at, pos):
    """feel-score verdicts of (at, pos) against the user's script per 4 s window: {start_ms: same/reversed/lost}."""
    from .feel_score import feel_windows
    if len(at) < 4:
        return {}
    return {float(w0): v for w0, _, v in feel_windows(ref_at, ref_pos, at, pos)}


def flags(user_at, user_pos, gray, fps, cut_frames, ai=None, motion=None, scene_thr: float = -0.5):
    """Merge the clues into stretches to look at: [{start_s, end_s, level, clues}]."""
    sc = dict(scene_scores(user_at, user_pos, gray, fps, cut_frames))
    va = window_verdicts(user_at, user_pos, *ai) if ai is not None else {}
    vm = window_verdicts(user_at, user_pos, *motion) if motion is not None else {}
    wins = sorted(set(va) | set(vm) | set(sc))
    marks = []
    for w in wins:
        s = next((v for k, v in sc.items() if abs(k - w) < WIN_MS / 2), None)
        clues = []
        if s is not None and s < scene_thr:
            clues.append("scene: these INs look like this shot's OUTs")
        if va.get(w) == "reversed":
            clues.append("AI goes the other way")
        if vm.get(w) == "reversed":
            clues.append("motion tracker goes the other way")
        if len(clues) >= 2:
            marks.append((w, "likely flipped", clues))
        elif clues and (clues[0].startswith("scene") or clues[0].startswith("AI")):
            marks.append((w, "check", clues))
    merged = []
    for w, lvl, clues in sorted(marks):
        if merged and w <= merged[-1]["end_ms"] + WIN_MS * 0.1:
            # overlapping / touching marks become one; "likely flipped" wins
            merged[-1]["end_ms"] = max(merged[-1]["end_ms"], w + WIN_MS)
            merged[-1]["clues"] = sorted(set(merged[-1]["clues"]) | set(clues))
            if lvl == "likely flipped":
                merged[-1]["level"] = lvl
        else:
            merged.append(dict(start_ms=w, end_ms=w + WIN_MS, level=lvl, clues=clues))
    return [dict(start_s=m["start_ms"] / 1000, end_s=m["end_ms"] / 1000, level=m["level"], clues=m["clues"])
            for m in merged]


def main(argv=None):
    import argparse
    import warnings
    warnings.filterwarnings("ignore")
    from . import motion as mo, paths, thrust_v2
    ap = argparse.ArgumentParser(description="Which stretches of a script may be upside down?")
    ap.add_argument("script", type=Path)
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    wd = paths.work_dir_for(a.video)
    wd.mkdir(parents=True, exist_ok=True)
    print("checking direction ...", flush=True)
    flow = mo.extract_flow_cells(a.video, wd / "flow_v2_30fps.npz", log=lambda *x: None)
    fps = float(flow["fps"])
    cuts = mo.cut_frames(flow["hist_diff"], fps)
    oa, op, oi = mo.narrowband_strokes(flow["cells"], cuts, fps, harmonic_ratio=0.0, min_conf=0.1, strong_conf=0.4,
                                        abs_floor_px=0.1, orient="vy")
    gray = thrust_v2.extract_fine(a.video, wd / "fine_v1.npz")["gray"]
    from . import media
    has_audio = media.probe(a.video).has_audio
    ai_at, ai_pos, _ = thrust_v2.script_from_video(a.video, wd, flow, oi, np.nonzero(cuts)[0] / fps * 1000,
                                                    has_audio=has_audio, og_at=oa, og_pos=op, log=lambda *x: None)
    ua, up, _ = fio.load(a.script)
    fl = flags(ua, up, gray, fps, cuts, ai=(ai_at, ai_pos), motion=(oa, op))
    tot = sum(f["end_s"] - f["start_s"] for f in fl)
    print(f"{len(fl)} stretches to look at ({tot:.0f} s):", flush=True)
    for f in fl:
        print(f"  {int(f['start_s'] // 60)}:{f['start_s'] % 60:04.1f} - {int(f['end_s'] // 60)}:{f['end_s'] % 60:04.1f}"
              f"  {f['level']}: {'; '.join(f['clues'])}", flush=True)
    out = a.out or a.script.with_suffix(".direction.json")
    out.write_text(json.dumps(dict(script=str(a.script), video=str(a.video), flags=fl), indent=1), encoding="utf-8")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
