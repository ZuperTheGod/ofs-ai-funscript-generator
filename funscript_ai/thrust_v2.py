"""v2 thrust script for the pipeline (`--method learned`).

The v2 model (fine picture + sound, trained on the user's own scripts) says where the thrusts are
and which way is in; a rhythm tracker places alternating IN/OUT points on that evidence at the
tempo the OG motion tracker measures (its own estimate where the motion tracker is unsure), so the
spacing stays a steady thrust rhythm through cuts, blur edits and off-screen stretches. On videos
without music, a shot whose thrust sounds land clearly on the OUT points is turned the right way up.

Held-out (50 videos, 2026-09-25), vs the v1 hybrid the user liked: same way up in 44.5% of
their scripted 4 s windows (40.9%), reversed 19.1% (21.0%), F1@100 0.435 (0.414), tempo 0.630
(0.635), off-screen points caught 55% (28%). On their own script of test video C: 66% same / 15%
reversed (hybrid 37% / 39%).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import gpu, thrust_model as tm

# v3 (2026-09-26): v2 fine-tuned on the user's labels with hand-placement scatter removed + their corrected test video B
# (first half). Unseen test video B second half: 3/8 of their reversed stretches fixed (v2 0/8), scripting where they
# deleted 67 s -> 23 s; test video A: every stretch in their notes right (v2 got 30.3-35.8 wrong); held-out tempo 0.635 ->
# 0.646, same-way-up unchanged, off-screen points 55% -> 46%.
_MODELS = Path(__file__).resolve().parents[1] / "models" / "thrust"
# Ensemble v2+v3 (2026-09-26, user round 2 on test video B: "V2 and V3 have some pros and cons, no clear winner"): the
# average of both models' evidence recovers 6/9 of the stretches where they said V2 was better while keeping 253/257 of
# the stretches they accepted in V3; held-out off-screen points 46% (v3) -> 54%, same-way-up unchanged.
# v4 (2026-10-17): v3 fine-tuned on 15 videos the user ran and then fixed in OFS (tests/add_correction.py, 117 min).
# v2+v4 at activity threshold 0.2 vs v2+v3 at 0.3 on the 50 held-out videos (tests/eval_v4.py): same-way-up
# 43.5% -> 44.7%, reversed 19.8% -> 18.8%, fast strokes same 23.7% -> 25.8% / reversed 19.5% -> 17.2%, off-screen
# points 53.9% -> 58.3%, f100 unchanged, tempo 63.3% -> 61.4%. v4 makes scripting stop more readily (it learned the
# user's deletions), hence the lower threshold -- at 0.3 off-screen fell to 46%.
ENSEMBLES = {"v4": ([_MODELS / "thrust_v2.pt", _MODELS / "thrust_v4.pt"], 0.2),
             "v3": ([_MODELS / "thrust_v2.pt", _MODELS / "thrust_v3.pt"], 0.3)}
DEFAULT_CHECKPOINT = _MODELS / "thrust_v4.pt"
if not DEFAULT_CHECKPOINT.exists():
    DEFAULT_CHECKPOINT = _MODELS / "thrust_v3.pt"
ENSEMBLE, DEFAULT_ACT_THR = ENSEMBLES["v4" if DEFAULT_CHECKPOINT.stem == "thrust_v4" else "v3"]


def resolve(choice) -> tuple[list[Path], float | None]:
    """--thrust-model value -> (model files to average, activity threshold or None for the default).
    None = the default ensemble; "v3"/"v4" = that ensemble; a path = that one model."""
    if choice is None or str(choice) == str(DEFAULT_CHECKPOINT):
        return list(ENSEMBLE), DEFAULT_ACT_THR
    if str(choice).lower() in ENSEMBLES:
        return ENSEMBLES[str(choice).lower()]
    return [Path(choice)], None


def extract_fine(video, out_npz: str | Path, width: int = 320, updates: int = 8, batch: int = 48, fps: float = 30.0):
    """Camera-corrected RAFT flow pooled to 18x32 + a 36x64 greyscale thumbnail per frame, the same
    settings and global-motion removal as motion.py. Cached."""
    out_npz = Path(out_npz)
    if out_npz.exists():
        from . import media
        d = dict(np.load(out_npz))
        if media.covers(video, len(d["gray"]), float(d.get("fps", fps))):
            return d                                  # a run cut short is never reused (user, 2026-10-23)
    import torch
    import torch.nn.functional as F
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    from . import media, motion
    dev = "cuda"
    info = media.probe(video)
    w, h = motion._size8(info, width)
    model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(dev).eval()
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, device=dev), torch.linspace(-1, 1, w, device=dev), indexing="ij")
    A = torch.stack([torch.ones_like(xs), xs, ys], -1).reshape(-1, 3)
    flows, grays, prev = [np.zeros((1,) + tm.FINE_FLOW + (2,), np.float16)], [], None
    with torch.inference_mode():
        for chunk in media.iter_frames(video, fps, w, h, pix="rgb24", batch=batch):
            x = torch.from_numpy(chunk.copy()).to(dev).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
            g = x.mean(1, keepdim=True)
            grays.append(((F.adaptive_avg_pool2d(g, tm.FINE_GRAY)[:, 0] + 1) * 127.5).clamp(0, 255).byte().cpu().numpy())
            seq = x if prev is None else torch.cat([prev, x], 0)
            if seq.shape[0] >= 2:
                flow = gpu.raft_flow(model, seq[:-1].contiguous(), seq[1:].contiguous(), updates)
                resid, _ = motion._remove_global_motion(flow, A)
                flows.append(F.adaptive_avg_pool2d(resid, tm.FINE_FLOW).permute(0, 2, 3, 1).half().cpu().numpy())
            prev = x[-1:]
    del model
    torch.cuda.empty_cache()
    out = dict(flow=np.concatenate(flows), gray=np.concatenate(grays), fps=np.float32(fps))
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **out)
    return out


def _audio(video, work_dir: Path, flow: dict, has_audio: bool):
    from . import media, rhythm_fill as rf
    from .audio_beats import detect_beats
    beats = downbeats = np.array([])
    flux = None
    audio_npz, beats_json, mel_npz = work_dir / "thrust_audio.npz", work_dir / "beats_fullmix.json", work_dir / "mel_flux.npz"
    if has_audio:
        wav = work_dir / "audio_22k_mono.wav"
        if not (audio_npz.exists() and beats_json.exists() and mel_npz.exists()):
            media.extract_audio(video, wav, sr=22050, channels=1)
        if not audio_npz.exists():
            np.savez_compressed(audio_npz, **tm.audio_tracks(wav))
        if not mel_npz.exists():
            np.savez_compressed(mel_npz, flux=rf.mel_flux(wav))
        beats, downbeats = detect_beats(wav, beats_json)
        wav.unlink(missing_ok=True)
        audio = dict(np.load(audio_npz))
        flux = np.load(mel_npz)["flux"]
    else:
        n = int(len(flow["cells"]) / flow["fps"] / 0.01) + 1
        z = np.zeros(n, np.float32)
        audio = dict(full=z, perc=z, low=z, mid=z, high=z, rms=z, harm_share=z, hop_s=np.float32(0.01),
                     no_audio=np.int8(1))
    return audio, np.asarray(beats), np.asarray(downbeats), flux


DEFAULT_POST = dict(anchor_conf=0.65, sound=True, slow_hole_factor=2.2, splice_visible=False, rest=True,
                    in_first=False, split_at_cuts=False, act_thr=0.3, real_outs=False, real_outs_conf=0.55,
                    quiet_spans=(), min_half_ms=0.0, beat_lock=False,
                    beat_lock_share=0.5, beat_lock_near_ms=90.0, music_mode=False)


def predict(video, work_dir, flow: dict, checkpoint=DEFAULT_CHECKPOINT, has_audio: bool = True, log=print) -> dict:
    """The expensive part: the v2 model's per-step evidence (in, out, active, level) at tm.RATE, plus the
    sound features the post-processing uses. Cached in work_dir per checkpoint, so the rhythm step can be
    re-run in seconds."""
    import torch
    work_dir = Path(work_dir)
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"no v2 thrust model at {checkpoint} (run '1 - Setup (run once).bat' to download the models)")
    audio, beats, downbeats, flux = _audio(video, work_dir, flow, has_audio)
    cache = work_dir / f"thrust_pred_{checkpoint.stem}_{int(checkpoint.stat().st_mtime)}.npz"
    P = np.load(cache)["P"] if cache.exists() else None
    expect = int(len(flow["cells"]) / float(flow["fps"]) * tm.RATE)
    if P is not None and len(P) < expect - 2 * tm.RATE:
        P = None                                  # made from motion that was cut short (user, 2026-10-23): redo
    if P is None:
        F1 = tm.build_features(flow, audio, beats, downbeats)
        log("  fine picture pass for the learned model ...")
        fl, gr = tm.fine_frames(extract_fine(video, work_dir / "fine_v1.npz"))
        seq = np.concatenate([F1, tm.mel_features(flux, len(F1))], 1).astype(np.float32)
        n = min(len(seq), 2 * len(fl))
        n -= n % 2
        ck = torch.load(checkpoint, map_location="cuda", weights_only=False)
        model = tm.make_model_v2(ck["d_seq"]).cuda()
        model.load_state_dict(ck["state"])
        model.eval()
        P = tm.predict_v2(model, seq[:n], fl[:n // 2], gr[:n // 2])
        del model
        torch.cuda.empty_cache()
        np.savez_compressed(cache, P=P)
    return dict(P=P, beats=np.asarray(beats), flux=flux)


def postprocess(pred: dict, flow: dict, og_at, og_pos, og_info: dict, cut_ms, cfg: dict | None = None):
    """Evidence -> alternating IN/OUT points. Every step is switchable through cfg (see DEFAULT_POST)."""
    from . import rhythm_fill as rf
    c = dict(DEFAULT_POST, **(cfg or {}))
    P, beats, flux = pred["P"], pred["beats"], pred["flux"]
    if c["quiet_spans"]:
        # the vision check saw a clearly non-sexual act here (kissing, posing, talking, clapping...): no scripting
        P = P.copy()
        for lo, hi in c["quiet_spans"]:
            P[int(lo * tm.RATE):int(hi * tm.RATE) + 1, 2] = 0.0
    fps = float(flow["fps"])
    og_at = np.asarray(og_at if og_at is not None else [], float)
    hp = rf.half_period_from_f0(og_info["f0"], og_info["conf"], fps, len(P), conf_thr=0.45,
                                fallback=rf.local_half_period(P))
    if pred.get("hp") is not None:
        # speed chooser: the user's likely stroke speed, from all tempo sources
        hp = np.asarray(pred["hp"], float)[:len(P)]
    if c["min_half_ms"]:
        # faster than the user ever strokes -> the tracker is counting jiggle: fold to half speed
        floor = c["min_half_ms"] / 1000 * tm.RATE
        hp = np.asarray(hp, float).copy()
        while np.any(hp < floor):
            hp = np.where(hp < floor, hp * 2, hp)
    P_track = P
    anchors_ms = np.array([])
    if len(og_at) and c["anchor_conf"] < 1.0:
        # user, 2026-09-25: the tracker slowly drifts out of sync on visible thrusts. Where the motion is very
        # clear (rhythm confidence >= anchor_conf) its reversal times pull the tracker into step (the evidence
        # still decides in vs out). test video A 44-71 s: 50-100 ms early -> 0-33 ms; held-out same-way-up 44.5 -> 43.4%.
        from scipy.ndimage import median_filter
        kind0 = rf.orient_by_evidence(og_at, og_pos, P, cut_ms)
        bad0 = rf.unreliable(og_info["conf"], fps, og_at, conf_thr=0.45, cv_thr=0.4)
        fi0 = np.clip(np.round(og_at / 1000 * fps).astype(int), 0, len(bad0) - 1)
        cm = median_filter(np.asarray(og_info["conf"], float), size=max(int(fps), 1), mode="nearest")
        clear = np.asarray(og_info["active"], bool)[fi0] & ~bad0[fi0] & (cm[fi0] >= c["anchor_conf"])
        P_track = rf.anchor_evidence(P, og_at, kind0, clear, weight=1.0, both=True)
        anchors_ms = og_at[clear]
    music = rf.has_music(beats, len(flow["cells"]) / fps)
    use_sound = c["sound"] and flux is not None and not music
    if use_sound:
        # only when the motion tempo is absurdly slow next to regular thrust sounds (test video A 40-41.5 s: 26 spm vs
        # impacts every ~0.45 s); a looser rule cost held-out tempo (0.628 -> 0.609), this one ~0.005
        hp = rf.combine_half_period(hp, rf.sound_half_period(rf.impact_track(flux), len(hp)), lo=0.0, hi=3.0)
        # clear, repeated thrust sounds keep the rhythm going off screen and pull IN points onto them
        # (user, test video A 38-41 s). Held-out: neutral (same-way-up 43.4 -> 43.0%).
        P_track = rf.sound_evidence(P_track, rf.impact_track(flux), z_on=1.5, in_weight=0.2)
    cuts_steps = (np.asarray(cut_ms, float) / 1000 * tm.RATE).astype(int) if c["split_at_cuts"] else None
    at, pos = rf.track(P_track, alpha=4.0, hp=hp, act_thr=c["act_thr"], in_first=c["in_first"],
                       split_steps=cuts_steps)
    if c["slow_hole_factor"]:
        # stretches where the spacing suddenly runs far wider than the rhythm on both sides (test video A 39.6-41.8 s) are
        # carried through with the rhythm from before or after. Held-out: off-screen points 55% -> 61%.
        holes = rf.slow_holes(at, factor=c["slow_hole_factor"])
        at, pos = rf.fill(at, pos, holes, P)[:2]
    else:
        holes = []
    if c["splice_visible"] and len(og_at):
        kind = rf.orient_by_evidence(og_at, og_pos, P, cut_ms)
        bad = rf.unreliable(og_info["conf"], fps, og_at, conf_thr=0.45, cv_thr=0.4)
        fi = np.clip(np.round(og_at / 1000 * fps).astype(int), 0, len(bad) - 1)
        reliable = np.asarray(og_info["active"], bool)[fi] & ~bad[fi]
        at, pos = rf.splice(og_at, kind, reliable, at, pos)
    flips = []
    if use_sound:
        pos, flips = rf.sound_direction(at, pos, rf.impact_track(flux), cut_ms=cut_ms, seg_s=1e9, margin=1.5, P=P)
    if c["quiet_spans"]:
        keep = np.ones(len(at), bool)
        for lo, hi in c["quiet_spans"]:
            keep &= ~((at >= lo * 1000) & (at <= hi * 1000))
        at, pos = at[keep], pos[keep]
    at, pos = rf.alternate(at, pos)
    if c["real_outs"] and "vb" in og_info:
        # fast in / slow out: each OUT where the visible motion really turns (user, 2026-09-26)
        at, pos = rf.real_outs(at, pos, np.asarray(og_info["vb"], float), np.asarray(og_info["conf"], float), fps,
                               conf_thr=c["real_outs_conf"])
    beat_spans = []
    if c["beat_lock"] and music:
        # thrusts that follow the music: snap the INs onto the beats there (user, 2026-10-24)
        at, pos, beat_spans = rf.beat_lock(at, pos, np.asarray(beats, float) * 1000, near_ms=c["beat_lock_near_ms"],
                                           min_share=c["beat_lock_share"])
    if c["music_mode"] and len(beats):
        # the user says this video's thrusts follow the music: every IN on a beat (user, 2026-10-24)
        at, pos = rf.music_mode(at, pos, np.asarray(beats, float) * 1000)
    before_rest = set(np.round(at).astype(int))
    if c["rest"]:
        at, pos = rf.rest_points(at, pos)      # user, 2026-09-26: every stretch starts and ends at 0
    explain = dict(anchors_ms=np.asarray(anchors_ms, float), holes_ms=[(float(a), float(b)) for a, b in holes],
                   rest_ms=[float(t) for t in at if int(round(t)) not in before_rest],
                   half_cycle_ms=np.asarray(hp, float) * 1000 / tm.RATE, act_thr=c["act_thr"], sound_used=bool(use_sound),
                   beat_locked_ms=[(float(a), float(b)) for a, b in beat_spans])
    return at, pos, dict(music=music, sound_flips=flips, active=P[:, 2] >= c["act_thr"], explain=explain, P=P)


def script_from_video(video, work_dir, flow: dict, og_info: dict, cut_ms, checkpoint=DEFAULT_CHECKPOINT,
                      has_audio: bool = True, og_at=None, og_pos=None, splice_visible: bool = False,
                      anchor_conf: float = 0.65, cfg: dict | None = None, log=print):
    """-> (at_ms, pos 100/0, info). `flow` = motion.extract_flow_cells result, `og_at/og_pos/og_info` =
    the OG narrowband_strokes result (its f0/conf give the tempo), cut_ms = cut times."""
    members, act_thr = resolve(checkpoint)
    members = [c for c in members if c.exists()]
    checkpoint = members[-1]
    preds = [predict(video, work_dir, flow, c, has_audio, log=log) for c in members]
    n = min(len(q["P"]) for q in preds)
    pred = dict(preds[-1], P=np.mean([q["P"][:n] for q in preds], axis=0))
    c = dict(anchor_conf=anchor_conf, splice_visible=splice_visible)
    if act_thr:
        c["act_thr"] = act_thr                 # the chosen model pair's own threshold ...
    c.update(cfg or {})                        # ... unless the caller sets one
    at, pos, info = postprocess(pred, flow, og_at, og_pos, og_info, cut_ms, c)
    log(f"  learned strokes ({' + '.join(c.stem for c in members)} + rhythm tracker): {len(at)} points, scripting "
        f"{info['active'].mean():.0%} of the video, {len(info['sound_flips'])} shots turned by thrust sound"
        f"{' (music: sound not used for direction)' if info['music'] else ''}")
    return at, pos, dict(active=info["active"], rate=tm.RATE, checkpoint=str(checkpoint), music=info["music"],
                         explain=info["explain"], P=info["P"], members=[m.stem for m in members], cfg=c,
                         sound_flips=info["sound_flips"])
