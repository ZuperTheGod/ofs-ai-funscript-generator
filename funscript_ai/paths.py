"""Where caches go: <repo>/work/<video name>_<hash>/ (motion, picture and sound features, AI predictions)."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = ROOT / "work"


def work_dir_for(video, root: Path = WORK_ROOT) -> Path:
    video = Path(video)
    h = hashlib.sha1(str(video.resolve()).lower().encode("utf-8")).hexdigest()[:8]
    return root / f"{video.stem[:40].strip()}_{h}"
