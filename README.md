# AI Funscript Generator for OpenFunscripter

Makes a funscript (IN = 100, OUT = 0) from a video, and plugs into
[OpenFunscripter (OFS)](https://github.com/OpenFunscripter/OFS) so you can generate, check and fix scripts in one place.

It combines three things:

- **Motion tracking**: optical flow (RAFT) measures what moves on screen, frame by frame.
- **An AI model trained on hand-made scripts**: it reads the motion, a small picture of every frame and the sound
  (impacts, beats), and decides where the strokes are and which way is IN. It was trained on a few hundred
  hand-made scripts and then on corrections of its own output.
- **A rhythm tracker**: places alternating IN/OUT points at the measured tempo, keeps the rhythm going when the camera
  looks away, and starts/ends every stretch at 0.

The output is a starting point you fix up in OFS, not a finished expert script. On test videos it never saw,
it matches the direction of the human script in roughly half of all 4-second windows and stays in step with clearly
visible strokes. It is weakest on very fast strokes and on 2D / sprite animation.

> Windows + an NVIDIA graphics card (GTX 900 series or newer) are required.

**Video guide (3 min):** [install and use it, step by step](../../releases/latest/download/AI-Funscript-Generator-tutorial.mp4)

---

## Install (easy way)

1. Download **`AI-Funscript-Generator-Windows.zip`** from the [latest release](../../releases/latest) and unzip it
   anywhere (a path without special characters is best).
2. Run **`1 - Setup (run once).bat`**. It downloads Python and the GPU libraries (~3 GB) into that folder only, and
   picks the right build for your graphics card. Nothing is installed elsewhere; delete the folder to remove it.
3. Choose how you want to use it:
   - **Drag videos** (or a folder) onto **`2 - Make funscripts.bat`**. A funscript appears next to each video
     (`<video>.ai.funscript` if the video already has a script - yours is never overwritten).
   - **Inside OpenFunscripter**, two options:
     - **OFS extension** (works with the official OFS 3): run **`3 - Install OFS extension.bat`**, then in OFS open
       *Extensions* → enable **AI Funscript Generator** → *Show window*.
     - **OFS-AI** (a build of OFS with the generator built in, in the `OFS-AI` folder of the zip): adds the timeline
       views below. Start `OFS-AI\OpenFunscripter.exe`, menu **AI Generator** → *AI Funscript Generator panel*.

The first run of a video takes a few minutes (motion + picture analysis); runs after that are seconds.

## Install from source

```
git clone https://github.com/ZuperTheGod/ofs-ai-funscript-generator
```
Then run `1 - Setup (run once).bat` in the cloned folder. It also downloads the trained models from the release.

---

## Using it in OFS

**Generate**

- *Extension*: select an **empty track** (OFS: *Project → Add new*), press **Generate for this video**, then
  **Load into selected track**. *Undo load* reverts it.
- *OFS-AI*: press **Generate AI script for this video**. The result appears as a new track next to yours.

**Settings** (hover them in the panel for help)

| Setting | What it does |
|---|---|
| Method | *AI* (default) or *Motion tracking only* - plain tracking is very accurate on loops that repeat the same motion |
| Model | `v4` (default) or `v3` (previous model) |
| Scripting threshold | how sure the AI must be that a scene is stroking before it scripts it. Lower = scripts more (catches more off-screen action), higher = stops more readily (kissing, talking, posing) |
| Follow visible strokes | where the motion is at least this clear, strokes snap onto the visible ones (1.00 = off) |
| Fill gaps wider than x rhythm | keeps the rhythm going through sudden gaps, e.g. when the camera looks away (0 = off) |
| Use thrust sounds | on videos without music, impact sounds keep the rhythm and can turn a shot's direction |
| Rest points | every stretch starts and ends at 0 |
| Beat lock | music videos: where most strokes already land on beats, put the INs exactly on the beat |
| **Music mode** (per video) | for videos where the thrusts follow the music (PMV / cock-hero style): every IN on a beat, OUT halfway between. Wrong on normal videos - it switches itself off when you open another video |

**Check direction** - finds stretches that may be **upside down** (100 at OUT instead of IN). Save your script, select
its track and press *Check direction*. It compares your script with the AI, the motion tracker and how IN/OUT look
inside each camera shot:

- **likely flipped** - two clues disagree with your script
- **check** - one clue disagrees

In tests with deliberately flipped 12-second stretches it caught about 3 in 4, and marked ~10 % of correct scripts by
mistake - so look at the marks, don't flip blindly. The extension shows a list you click to jump; OFS-AI also colours
the stretches red/orange on the timeline.

**OFS-AI only**

- **Why view** - a band along the bottom of the timeline: grey = how sure the AI is that this is a stroking scene,
  white line = which way it leans, and a coloured dot on every generated stroke telling what decided it:
  green = visible motion, blue = AI model, yellow = rhythm fill (camera away), orange = thrust sound, grey = rest point.
  Hover it for the exact numbers.
- **Highlights** - right-click the timeline → *Start highlight* / *End highlight* to colour a section and tag it from a
  searchable tag gallery (new tags are kept). Highlights are saved next to the video (`<video>.labels.json`) and also
  as OFS chapters in the funscript.

## Command line

```
.venv\Scripts\python.exe -m funscript_ai.generate <video or folder> [--out x.funscript]
        [--settings "{\"act_thr\": 0.2, \"music_mode\": true}"] [--model v4|v3] [--motion-only] [--explain why.json]
.venv\Scripts\python.exe -m funscript_ai.direction_check <script.funscript> <video>
```
Settings keys: `act_thr`, `anchor_conf`, `slow_hole_factor`, `sound`, `rest`, `beat_lock`, `music_mode`.

## Troubleshooting

- **"no kernel image is available"** - your card needs an older GPU library build: run setup again, it tests the card
  and picks the right one.
- **The script stops early / misses the end** - fixed in 1.0: if the GPU video decoder stops early (e.g. while OFS plays
  the same video) the rest is decoded on the CPU, and incomplete caches are never reused.
- Anything else: open an [issue](../../issues) with the text from the panel's *Output* box.

## Credits & licence

- [OpenFunscripter](https://github.com/OpenFunscripter/OFS) (GPL-3.0) - OFS-AI is a modified build of it; its source
  changes are in [`ofs-fork/`](ofs-fork).
- [RAFT](https://pytorch.org/vision/stable/models/raft.html) optical flow (torchvision), [beat_this](https://github.com/CPJKU/beat_this)
  beat tracker, [PyTorch](https://pytorch.org), [librosa](https://librosa.org), [FFmpeg](https://ffmpeg.org).
- This project: **GPL-3.0** (see [LICENSE](LICENSE)). Only use it on videos you have the right to use; everyone shown
  must be an adult.
