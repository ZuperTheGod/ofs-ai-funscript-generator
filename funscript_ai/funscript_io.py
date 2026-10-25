"""Funscript read/write plus the stroke turning-point extraction shared by the
generator and the scorer.

Funscript on disk: {"version": "1.0", "inverted": false, "range": 100,
"actions": [{"at": <ms int>, "pos": <0-100 int>}, ...]}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (at_ms float64, pos float64, raw_json). Sorted by time,
    duplicate timestamps collapsed (last wins), honours `inverted`."""
    data = Path(path).read_bytes()
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:          # some tools save funscripts in the Windows codepage
        text = data.decode("cp1252", errors="replace")
    raw = json.loads(text)
    acts = raw.get("actions") or []
    at = np.array([a["at"] for a in acts], dtype=np.float64)
    pos = np.array([a["pos"] for a in acts], dtype=np.float64)
    order = np.argsort(at, kind="stable")
    at, pos = at[order], pos[order]
    if len(at):
        keep = np.append(np.diff(at) > 0, True)
        at, pos = at[keep], pos[keep]
    if raw.get("inverted"):
        pos = 100.0 - pos
    return at, np.clip(pos, 0, 100), raw


def save(path: str | Path, at_ms, pos, metadata: dict | None = None) -> Path:
    at_ms = np.round(np.asarray(at_ms, dtype=np.float64)).astype(np.int64)
    pos = np.clip(np.round(np.asarray(pos, dtype=np.float64)), 0, 100).astype(np.int64)
    order = np.argsort(at_ms, kind="stable")
    at_ms, pos = at_ms[order], pos[order]
    keep = np.append(np.diff(at_ms) > 0, True) if len(at_ms) else np.array([], bool)
    actions = [{"at": int(a), "pos": int(p)} for a, p in zip(at_ms[keep], pos[keep])]
    doc = {"version": "1.0", "inverted": False, "range": 100, "actions": actions}
    if metadata:
        doc["metadata"] = metadata
    path = Path(path)
    path.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    return path


def extremes(at: np.ndarray, pos: np.ndarray, min_depth: float = 8.0):
    """Zigzag reduction of any script (sparse hand-made or dense per-frame) to
    its turning points. A point is only confirmed as a top/bottom once the
    curve has reversed at least `min_depth` from it, so jitter on dense
    scripts is ignored. On a plateau the first sample is used (the moment the
    stroke arrived).

    Returns (at_ms, pos, kind) where kind is +1 for a top, -1 for a bottom.
    """
    n = len(at)
    t_out, k_out = [], []
    if n < 2:
        return np.array([]), np.array([]), np.array([], dtype=np.int8)
    hi = lo = 0
    trend = 0  # +1 rising (hunting a top), -1 falling (hunting a bottom)
    cand = 0
    for i in range(1, n):
        p = pos[i]
        if trend == 0:
            if p > pos[hi]:
                hi = i
            if p < pos[lo]:
                lo = i
            if pos[hi] - pos[lo] >= min_depth:
                if hi > lo:
                    t_out.append(lo); k_out.append(-1); trend, cand = 1, hi
                else:
                    t_out.append(hi); k_out.append(1); trend, cand = -1, lo
        elif trend == 1:
            if p > pos[cand]:
                cand = i
            elif pos[cand] - p >= min_depth:
                t_out.append(cand); k_out.append(1); trend, cand = -1, i
        else:
            if p < pos[cand]:
                cand = i
            elif p - pos[cand] >= min_depth:
                t_out.append(cand); k_out.append(-1); trend, cand = 1, i
    if trend != 0:
        t_out.append(cand); k_out.append(trend)
    idx = np.array(t_out, dtype=np.int64)
    return at[idx], pos[idx], np.array(k_out, dtype=np.int8)
