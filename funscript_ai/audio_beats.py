"""Music beat + downbeat tracking with beat_this (GPU), cached as JSON."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_MODEL = None


def detect_beats(wav_path: str | Path, cache_json: str | Path, device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    """Return (beats_s, downbeats_s)."""
    cache_json = Path(cache_json)
    if cache_json.exists():
        d = json.loads(cache_json.read_text())
        return np.array(d["beats"]), np.array(d["downbeats"])
    global _MODEL
    if _MODEL is None:
        from beat_this.inference import File2Beats
        _MODEL = File2Beats(checkpoint_path="final0", device=device, dbn=False)
    beats, downbeats = _MODEL(str(wav_path))
    beats, downbeats = np.asarray(beats, float), np.asarray(downbeats, float)
    cache_json.parent.mkdir(parents=True, exist_ok=True)
    cache_json.write_text(json.dumps({"beats": beats.round(4).tolist(), "downbeats": downbeats.round(4).tolist()}))
    return beats, downbeats


def local_bpm(beats: np.ndarray) -> np.ndarray:
    """BPM at each beat from the median of the surrounding 8 intervals."""
    if len(beats) < 3:
        return np.zeros(len(beats))
    ibi = np.diff(beats)
    pad = np.pad(ibi, 4, mode="edge")
    med = np.array([np.median(pad[i:i + 8]) for i in range(len(ibi))])
    return 60.0 / np.append(med, med[-1])
