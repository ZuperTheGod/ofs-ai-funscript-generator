"""Make a funscript from a video.

    python -m funscript_ai.generate <video or folder> [...] [--out file.funscript] [--settings JSON]
                                    [--model v4|v3] [--explain why.json] [--points points.txt] [--motion-only]

Output (unless --out): <video>.funscript next to the video, or <video>.ai.funscript when the video already has a
script that this program did not make -- your own scripts are never overwritten.
--settings takes the post-processing options, e.g. {"act_thr": 0.2, "music_mode": true} (see README).
--points writes "seconds position" lines (used by the OFS Lua extension).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

from . import funscript_io as fio, media, motion, paths, thrust_v2

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".wmv", ".flv", ".mpg", ".mpeg", ".ts"}
CREATOR = "ofs-ai-funscript-generator"


def made_by_us(script: Path) -> bool:
    try:
        meta = json.loads(script.read_text(encoding="utf-8-sig")).get("metadata") or {}
        return isinstance(meta, dict) and meta.get("creator") == CREATOR
    except (OSError, ValueError):
        return False


def output_path(video: Path) -> Path:
    own = video.with_suffix(".funscript")
    if not own.exists() or made_by_us(own):
        return own
    return video.with_name(video.stem + ".ai.funscript")


def write_explain(path: Path, video: Path, duration_s: float, at, pos, info: dict) -> None:
    """Why each stroke was made (for the OFS "why" view)."""
    ex, P = info["explain"], info["P"]
    step = 3                                         # model runs at 60 steps/s -> 20 values/s here
    anchors = np.asarray(ex["anchors_ms"], float)
    rest = set(int(round(t)) for t in ex["rest_ms"])
    flips = [(a * 1000, b * 1000) for a, b, *_ in info.get("sound_flips", [])]
    src = []
    for t in at:
        if int(round(t)) in rest:
            src.append("rest")
        elif any(a <= t <= b for a, b in ex["holes_ms"]):
            src.append("filled")
        elif len(anchors) and np.min(np.abs(anchors - t)) <= 60:
            src.append("visible")
        elif any(a <= t <= b for a, b in flips):
            src.append("sound")
        else:
            src.append("model")
    out = dict(version=1, video=str(video), duration_s=float(duration_s), models=info.get("members"),
               settings={k: v for k, v in (info.get("cfg") or {}).items() if isinstance(v, (int, float, bool, str))},
               music=bool(info.get("music")), sound_used=ex["sound_used"], act_thr=ex["act_thr"], rate=20,
               p_in=np.round(P[::step, 0], 3).tolist(), p_out=np.round(P[::step, 1], 3).tolist(),
               p_active=np.round(P[::step, 2], 3).tolist(),
               half_cycle_ms=np.round(np.asarray(ex["half_cycle_ms"])[::step], 1).tolist(),
               holes_ms=ex["holes_ms"], sound_flips_ms=flips, beat_locked_ms=ex.get("beat_locked_ms", []),
               points=[dict(at=float(t), pos=float(p), source=s) for t, p, s in zip(at, pos, src)])
    path.write_text(json.dumps(out), encoding="utf-8")


def generate(video: Path, out: Path | None = None, settings: dict | None = None, model: str | None = None,
             explain: Path | None = None, points: Path | None = None, motion_only: bool = False, log=print) -> Path:
    t0 = time.perf_counter()
    info = media.probe(video)
    log(f"=== {video.name}\n  {info.duration / 60:.1f} min, {info.width}x{info.height} @ {info.fps:.0f} fps, "
        f"audio: {info.has_audio}")
    wd = paths.work_dir_for(video)
    wd.mkdir(parents=True, exist_ok=True)
    flow = motion.extract_flow_cells(video, wd / "flow_v2_30fps.npz", fps=30.0, log=log)
    fps = float(flow["fps"])
    cuts = motion.cut_frames(flow["hist_diff"], fps)
    cut_ms = np.nonzero(cuts)[0] / fps * 1000
    log(f"  motion measured: {int(cuts.sum())} camera cuts [{time.perf_counter() - t0:.0f}s]")
    o_at, o_pos, o_info = motion.narrowband_strokes(flow["cells"], cuts, fps, harmonic_ratio=0.0, min_conf=0.1,
                                                    snap_frac=0.0, strong_conf=0.4, abs_floor_px=0.1, orient="vy",
                                                    vy_min_share=0.15)
    if motion_only:
        at, pos, how = o_at, o_pos, "motion tracking only"
    else:
        at, pos, l_info = thrust_v2.script_from_video(video, wd, flow, o_info, cut_ms, model or thrust_v2.DEFAULT_CHECKPOINT,
                                                      has_audio=info.has_audio, og_at=o_at, og_pos=o_pos, cfg=settings,
                                                      log=log)
        how = f"AI ({' + '.join(l_info.get('members') or [])})"
        if explain:
            write_explain(Path(explain), video, info.duration, at, pos, l_info)
    target = Path(out) if out else output_path(video)
    meta = dict(creator=CREATOR, title=video.stem, duration=int(info.duration),
                notes=f"generated by {CREATOR}: {how}; settings {json.dumps(settings or {})}")
    fio.save(target, at, pos, metadata=meta)
    if points:
        Path(points).write_text("\n".join(f"{t / 1000:.3f} {int(round(p))}" for t, p in zip(at, pos)), encoding="utf-8")
    log(f"  wrote {target}  ({len(at)} points) [{time.perf_counter() - t0:.0f}s total]")
    return target


def iter_videos(items):
    for it in items:
        p = Path(it)
        if p.is_dir():
            yield from sorted(q for q in p.rglob("*") if q.suffix.lower() in VIDEO_EXTS)
        elif p.suffix.lower() in VIDEO_EXTS:
            yield p


def main(argv=None):
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from . import gpu
    gpu.limit_memory()
    ap = argparse.ArgumentParser(prog="funscript_ai.generate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", type=Path, default=None, help="output file (one video only)")
    ap.add_argument("--settings", default=None, help='post-processing settings as JSON, e.g. {"act_thr": 0.2}')
    ap.add_argument("--model", default=None, choices=["v4", "v3"], help="model pair (default v4)")
    ap.add_argument("--explain", type=Path, default=None, help="write why each stroke was made (JSON)")
    ap.add_argument("--points", type=Path, default=None, help="also write 'seconds position' lines")
    ap.add_argument("--motion-only", action="store_true", help="plain motion tracking, no AI (good for loops)")
    a = ap.parse_args(argv)
    settings = json.loads(a.settings) if a.settings else None
    videos = list(iter_videos(a.paths))
    if not videos:
        print("No videos found in:", *a.paths)
        return 1
    failed = 0
    for v in videos:
        try:
            generate(v, a.out if len(videos) == 1 else None, settings, a.model, a.explain, a.points, a.motion_only)
        except Exception:  # noqa: BLE001 - keep going with the other videos
            failed += 1
            print(f"\n!!! failed on {v}")
            traceback.print_exc()
    print(f"\nDone: {len(videos) - failed} ok, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
