"""Build the release files into dist/:
  models.zip                          -- the trained models (setup downloads this when installing from source)
  AI-Funscript-Generator-Windows.zip  -- ready to use: generator + models + uv + ffmpeg + RAFT/beat weights + OFS-AI

    python tools/build_release.py --ofs-build <OFS-AI bin folder> --ffmpeg <folder with ffmpeg.exe/ffprobe.exe/LICENSE>
                                  --uv <uv.exe> --weights <torch hub checkpoints folder>
"""
import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
REPO_FILES = ["funscript_ai", "ofs-extension", "1 - Setup (run once).bat", "2 - Make funscripts.bat",
              "3 - Install OFS extension.bat", "requirements.txt", "README.md", "LICENSE", "run_ofs_request.py"]
MODELS = ["thrust_v2.pt", "thrust_v3.pt", "thrust_v4.pt"]
OFS_FILES = ["OpenFunscripter.exe", "mpv-2.dll", "data"]            # nothing else from the build folder
WEIGHTS = ["raft_small_C_T_V2-01064c6d.pth", "beat_this-final0.ckpt"]


def add(z: zipfile.ZipFile, src: Path, arc: str):
    if src.is_dir():
        for p in sorted(src.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                z.write(p, f"{arc}/{p.relative_to(src).as_posix()}")
    else:
        z.write(src, arc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ofs-build", type=Path, required=True)
    ap.add_argument("--ffmpeg", type=Path, required=True)
    ap.add_argument("--uv", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    a = ap.parse_args()
    DIST.mkdir(exist_ok=True)
    with zipfile.ZipFile(DIST / "models.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for m in MODELS:
            add(z, ROOT / "models" / "thrust" / m, f"models/thrust/{m}")
    top = "AI-Funscript-Generator"
    with zipfile.ZipFile(DIST / "AI-Funscript-Generator-Windows.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in REPO_FILES:
            add(z, ROOT / f, f"{top}/{f}")
        for m in MODELS:
            add(z, ROOT / "models" / "thrust" / m, f"{top}/models/thrust/{m}")
        add(z, a.uv, f"{top}/bin/uv.exe")
        for f in ("ffmpeg.exe", "ffprobe.exe"):
            add(z, a.ffmpeg / f, f"{top}/bin/{f}")
        if (a.ffmpeg.parent / "LICENSE").exists():
            add(z, a.ffmpeg.parent / "LICENSE", f"{top}/bin/FFMPEG-LICENSE.txt")
        z.writestr(f"{top}/bin/README.txt",
                   "uv (MIT/Apache-2.0, https://github.com/astral-sh/uv) and FFmpeg (GPL, https://ffmpeg.org; this build:\n"
                   "https://www.gyan.dev/ffmpeg/builds/, source: https://github.com/FFmpeg/FFmpeg) are included unmodified.\n")
        for w in WEIGHTS:
            add(z, a.weights / w, f"{top}/torch_home/hub/checkpoints/{w}")
        for f in OFS_FILES:
            add(z, a.ofs_build / f, f"{top}/OFS-AI/{f}")
        ofs_license = a.ofs_build.parents[1] / "LICENSE"       # <OFS source>/bin/RelWithDebInfo -> <OFS source>
        if ofs_license.exists():
            add(z, ofs_license, f"{top}/OFS-AI/LICENSE.txt")
        z.writestr(f"{top}/OFS-AI/README.txt",
                   "OFS-AI = OpenFunscripter (GPL-3.0, https://github.com/OpenFunscripter/OFS) with the AI Funscript\n"
                   "Generator built in. Source changes: ofs-fork/ in https://github.com/ZuperTheGod/ofs-ai-funscript-generator\n"
                   "Start OpenFunscripter.exe here; menu 'AI Generator' -> 'AI Funscript Generator panel'.\n"
                   "Run '1 - Setup (run once).bat' in the folder above first.\n")
    for f in DIST.iterdir():
        print(f"{f.name}: {f.stat().st_size / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
