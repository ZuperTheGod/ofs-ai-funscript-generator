"""Runner for the OFS Lua extension. Lua's os.execute cannot pass non-English file names safely, so the extension writes
a UTF-8 request file and starts:  python -m funscript_ai.ofs_request <request.json>

request: {"video": ..., "settings": {...}, "model": "v4", "motion_only": false, "task": "generate" | "check",
          "script": "<funscript for task=check>", "out_dir": "<folder for results>"}
results in out_dir: log.txt (progress), points.txt ("seconds position" lines), flags.txt ("start end level" lines for
task=check), done.txt (written last: "ok" or "failed").
"""
from __future__ import annotations

import json
import sys
import traceback
import warnings
from pathlib import Path


def main():
    warnings.filterwarnings("ignore")
    req_path = Path(sys.argv[1])
    req = json.loads(req_path.read_text(encoding="utf-8"))
    out = Path(req["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for f in ("done.txt", "points.txt", "flags.txt"):
        (out / f).unlink(missing_ok=True)
    log_f = open(out / "log.txt", "w", encoding="utf-8", buffering=1)

    def log(*a):
        print(*a, file=log_f, flush=True)

    sys.stdout = sys.stderr = log_f
    ok = False
    try:
        video = Path(req["video"])
        if req.get("task") == "check":
            from . import direction_check
            flags_json = out / "flags.json"
            direction_check.main([req["script"], str(video), "--out", str(flags_json)])
            fl = json.loads(flags_json.read_text(encoding="utf-8"))["flags"]
            (out / "flags.txt").write_text("\n".join(f"{f['start_s']:.2f} {f['end_s']:.2f} {f['level'].replace(' ', '_')}"
                                                     for f in fl), encoding="utf-8")
        else:
            from . import generate
            generate.generate(video, out / "ai.funscript", req.get("settings") or None, req.get("model") or None,
                              explain=out / "explain.json", points=out / "points.txt",
                              motion_only=bool(req.get("motion_only")), log=log)
        ok = True
    except Exception:  # noqa: BLE001 - reported to the extension through log.txt
        traceback.print_exc(file=log_f)
    finally:
        (out / "done.txt").write_text("ok" if ok else "failed", encoding="utf-8")
        log_f.close()


if __name__ == "__main__":
    main()
