"""ffmpeg/ffprobe helpers. Video is decoded on the GPU (NVDEC + scale_cuda)
so long files don't pin the CPU; falls back to CPU decode with a thread cap
for codecs NVDEC can't handle."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

CPU_THREADS = "2"


@dataclass
class VideoInfo:
    path: Path
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    vcodec: str


def probe(path: str | Path) -> VideoInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    j = json.loads(out.stdout)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    num, den = (v.get("r_frame_rate") or "30/1").split("/")
    return VideoInfo(
        path=Path(path), duration=float(j["format"]["duration"]),
        width=int(v["width"]), height=int(v["height"]),
        fps=float(num) / float(den or 1),
        has_audio=any(s["codec_type"] == "audio" for s in j["streams"]),
        vcodec=v.get("codec_name", ""))


def extract_audio(path: str | Path, out_wav: str | Path, sr: int = 44100, channels: int = 1) -> Path:
    out_wav = Path(out_wav)
    if out_wav.exists() and out_wav.stat().st_mtime >= Path(path).stat().st_mtime:
        return out_wav
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_wav.with_suffix(".part.wav")
    subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-threads", CPU_THREADS, "-i", str(path),
                    "-vn", "-ac", str(channels), "-ar", str(sr), "-y", str(tmp)], check=True)
    tmp.replace(out_wav)
    return out_wav


def analysis_size(info: VideoInfo, width: int = 320) -> tuple[int, int]:
    h = int(round(width * info.height / info.width / 2)) * 2
    return width, max(h, 2)


def _frame_cmd(path, fps, w, h, pix, start, duration, gpu):
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-threads", CPU_THREADS]
    if gpu:
        cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(path)]
    if gpu:
        vf = f"fps={fps},scale_cuda={w}:{h},hwdownload,format=nv12,format={pix}"
    else:
        vf = f"fps={fps},scale={w}:{h}:flags=area,format={pix}"
    return cmd + ["-vf", vf, "-an", "-f", "rawvideo", "pipe:1"]


def covers(path: str | Path, n_frames: int, fps: float, slack_s: float = 2.0) -> bool:
    """Does a cache with n_frames at fps cover the whole video? (A run cut short must never be reused.)"""
    try:
        return n_frames / fps >= probe(path).duration - slack_s
    except Exception:  # noqa: BLE001 - video gone / unreadable: trust the cache
        return True


def iter_frames(path: str | Path, fps: float, w: int, h: int, pix: str = "gray",
                start: float | None = None, duration: float | None = None,
                batch: int = 256) -> Iterator[np.ndarray]:
    """Yield uint8 batches shaped (n, h, w) for gray or (n, h, w, 3) for rgb24.
    Frame k of the stream is at time start + k / fps.
    The GPU decoder can stop early without an error -- e.g. while OpenFunscripter plays the same video on it
    (user, 2026-10-23: the AI script stopped after a minute). Then the rest is decoded on the CPU, from the frame
    where the GPU stopped, so callers always get the whole video."""
    ch = 3 if pix == "rgb24" else 1
    fsize = w * h * ch
    try:
        want = ((duration if duration else probe(path).duration - (start or 0.0))) * fps
    except Exception:  # noqa: BLE001 - unknown length: no completeness check
        want = None
    produced = 0
    for gpu in (True, False):
        s0 = (start or 0.0) + produced / fps
        d0 = (duration - produced / fps) if duration else None
        proc = subprocess.Popen(_frame_cmd(path, fps, w, h, pix, s0 if s0 > 0 else None, d0, gpu),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=fsize * batch)
        before = produced
        try:
            while True:
                buf = proc.stdout.read(fsize * batch)
                n = len(buf) // fsize
                if n == 0:
                    break
                arr = np.frombuffer(buf[: n * fsize], dtype=np.uint8)
                produced += n
                yield arr.reshape((n, h, w, 3) if ch == 3 else (n, h, w))
                if n < batch:
                    break
        finally:
            proc.stdout.close()
            err = proc.stderr.read().decode(errors="replace")
            proc.wait()
        complete = want is None or produced >= want - 2 * fps
        if gpu and produced > 0 and not complete:
            print(f"    (GPU video decoder stopped at {produced / fps:.0f}s of {want / fps:.0f}s - "
                  f"decoding the rest on the CPU)", flush=True)
            continue
        if produced > before or not gpu:
            if proc.returncode not in (0, None) and produced == 0:
                raise RuntimeError(f"ffmpeg decode failed: {err.strip()[-500:]}")
            return
        # GPU path produced nothing -> retry on CPU
