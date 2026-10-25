"""Rhythm continuation for gaps (the user's rule, 2026-09-24):

  "the best way to figure out when to do it is when the thrusting audio matches the next
   scene and you make the patterns match the new unseen scene, or if the thrusting audio is
   playing after the scene is going but it still matches the past scene then the patterns
   should match that."

Visible stretches keep their script. In a gap (camera away, black skip, blur edit, or a
stretch where the tracker lost the rhythm) the strokes continue at a steady tempo, either
carried forward from the scene before or backward from the scene after -- whichever the
evidence inside the gap (learned in/out probabilities, which include the sound) agrees with
better -- and each point then snaps to a clear in/out peak nearby when there is one.
"""
from __future__ import annotations

import numpy as np

from . import thrust_model as tm


def _rhythm(at, kind, side: str, edge_ms: float, look_ms: float = 4000.0, min_turns: int = 5):
    """Period (ms, one full in->in cycle) and the reference IN time of the rhythm on one side of a gap."""
    if side == "before":
        m = (at <= edge_ms) & (at >= edge_ms - look_ms)
    else:
        m = (at >= edge_ms) & (at <= edge_ms + look_ms)
    t, k = at[m], kind[m]
    if len(t) < min_turns:
        return None
    ins = t[k == 1]
    if len(ins) < 3:
        return None
    per = float(np.median(np.diff(ins)))
    if not (150 <= per <= 2500):
        return None
    ref = ins[-1] if side == "before" else ins[0]
    return per, float(ref)


def _synth(per, ref, a, b):
    """Alternating IN/OUT times at a steady period covering [a, b], phase-locked to ref (an IN time)."""
    n0 = int(np.floor((a - ref) / per)) - 1
    n1 = int(np.ceil((b - ref) / per)) + 1
    ev = []
    for n in range(n0, n1 + 1):
        t_in = ref + n * per
        for t, k in ((t_in, 1), (t_in + per / 2, -1)):
            if a < t < b:
                ev.append((t, k))
    return ev


def _evidence(ev, P):
    if not ev:
        return -np.inf
    s = 0.0
    for t, k in ev:
        i = int(np.clip(round(t / 1000 * tm.RATE), 0, len(P) - 1))
        lo, hi = max(i - 2, 0), min(i + 3, len(P))
        s += P[lo:hi, 0 if k == 1 else 1].max() - P[lo:hi, 1 if k == 1 else 0].max()
    return s / len(ev)


def _snap(ev, P, per, thr=0.35, frac=0.2):
    """Move each synthetic point to the strongest same-kind peak within +-frac of the period
    if that peak is clear (the 'snap to the sound when it is clear' part)."""
    out = []
    r = max(int(frac * per / 1000 * tm.RATE), 1)
    for t, k in ev:
        i = int(np.clip(round(t / 1000 * tm.RATE), 0, len(P) - 1))
        lo, hi = max(i - r, 0), min(i + r + 1, len(P))
        col = P[lo:hi, 0 if k == 1 else 1]
        j = int(np.argmax(col))
        out.append(((lo + j) / tm.RATE * 1000 if col[j] >= thr else t, k))
    return out


def unreliable(conf: np.ndarray, fps: float, turns_ms: np.ndarray, conf_thr: float = 0.35,
               cv_thr: float = 0.45, win_s: float = 3.0) -> np.ndarray:
    """Per-frame mask where the tracker counts as active but its rhythm has fallen apart:
    low rhythm confidence (1 s median) or erratic spacing of its turns (+-win_s/2 CV)."""
    from scipy.ndimage import median_filter
    T = len(conf)
    bad = median_filter(conf, size=max(int(fps), 1), mode="nearest") < conf_thr
    t = np.arange(T) / fps * 1000
    half = win_s * 500
    lo = np.searchsorted(turns_ms, t - half)
    hi = np.searchsorted(turns_ms, t + half)
    for i in range(0, T, max(int(fps // 6), 1)):
        seg = turns_ms[lo[i]:hi[i]]
        if len(seg) >= 4:
            d = np.diff(seg)
            if np.std(d) / max(np.mean(d), 1e-6) > cv_thr:
                bad[i:i + max(int(fps // 6), 1)] = True
    return bad


def gaps_from(active: np.ndarray, fps: float, turns_ms: np.ndarray, min_gap_ms: float = 700.0,
              max_turn_gap_ms: float = 1200.0, bad: np.ndarray | None = None):
    """Gaps = stretches where the tracker is inactive (or `bad`: active but its rhythm fell
    apart), plus holes between consecutive turns longer than max_turn_gap_ms."""
    if bad is not None:
        active = np.asarray(active, bool) & ~np.asarray(bad, bool)[:len(active)]
    spans = []
    a = None
    for i, on in enumerate(np.append(active, True)):
        if not on and a is None:
            a = i
        elif on and a is not None:
            if (i - a) / fps * 1000 >= min_gap_ms:
                spans.append((a / fps * 1000, i / fps * 1000))
            a = None
    if len(turns_ms) > 1:
        d = np.diff(turns_ms)
        for j in np.nonzero(d > max_turn_gap_ms)[0]:
            spans.append((turns_ms[j], turns_ms[j + 1]))
    spans.sort()
    merged = []
    for s in spans:
        if merged and s[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], s[1]))
        else:
            merged.append(s)
    return merged


def fill(at_ms, pos, gaps, P, max_gap_ms: float = 15000.0, min_evidence: float = -0.05, snap_thr: float = 0.35):
    """Keep every point outside the gaps; inside each gap put a continued rhythm (from before or
    after, chosen by the evidence P) when the model thinks the user would be scripting there."""
    at_ms, pos = np.asarray(at_ms, float), np.asarray(pos, float)
    kind = np.where(pos > 50, 1, -1)
    keep = np.ones(len(at_ms), bool)
    new = []
    report = []
    for a, b in gaps:
        if b - a > max_gap_ms:
            continue
        i0, i1 = int(a / 1000 * tm.RATE), int(b / 1000 * tm.RATE)
        if i1 <= i0 or P[i0:i1, 2].mean() < 0.4:          # the user would not be scripting here
            continue
        cands = []
        for side, edge in (("before", a), ("after", b)):
            r = _rhythm(at_ms, kind, side, edge)
            if r:
                ev = _synth(*r, a, b)
                cands.append((_evidence(ev, P), side, r, ev))
        if not cands:
            continue
        score, side, (per, _), ev = max(cands, key=lambda c: c[0])
        if score < min_evidence:
            continue
        keep &= ~((at_ms > a) & (at_ms < b))
        new += _snap(ev, P, per, thr=snap_thr)
        report.append(dict(start_s=a / 1000, end_s=b / 1000, from_scene=side, period_ms=per, evidence=float(score)))
    ev_all = [(t, int(k), 1.0) for t, k in zip(at_ms[keep], kind[keep])] + [(t, k, 0.5) for t, k in new]
    ev_all.sort()
    out = []
    for e in ev_all:
        if out and (out[-1][1] == e[1] or e[0] - out[-1][0] < 60):
            if e[2] > out[-1][2]:
                out[-1] = e
            continue
        out.append(e)
    return (np.array([e[0] for e in out]), np.array([100.0 if e[1] == 1 else 0.0 for e in out]), report)


def impact_track(mel_flux: np.ndarray, hop_s: float = 0.01, max_hz_band: int = 24) -> np.ndarray:
    """Body-impact strength at hop_s: locally z-scored onset flux of the low mel bands (up to ~1.5 kHz)."""
    from scipy.ndimage import uniform_filter1d
    x = mel_flux[:, :max_hz_band].astype(np.float64)
    w = int(3.0 / hop_s)
    mu = uniform_filter1d(x, w, axis=0)
    sd = np.sqrt(np.maximum(uniform_filter1d(x * x, w, axis=0) - mu * mu, 1e-8))
    return ((x - mu) / (sd + 1e-3)).mean(1)


def sound_direction(at_ms, pos, impact: np.ndarray, cut_ms=(), seg_s: float = 6.0, margin: float = 1.0,
                    hop_s: float = 0.01, win_ms: float = 60.0, min_pts: int = 6, P=None, max_model_agree: float = 0.2):
    """Flip stretches whose thrust sounds land on the OUT points instead of the IN points.
    A slap happens at full insertion (IN = 100), so where the impact strength around the OUT
    points beats the IN points by more than `margin` (z units) the stretch is upside down; when
    the sound is weak or musical the difference stays small and nothing is touched.
    Checked on test video A against the user's notes (2026-09-25): the strong-sound verdicts
    matched all 5 stretches they commented on; weak ones (< ~0.7) were unreliable -> left alone."""
    at_ms, pos = np.asarray(at_ms, float), np.asarray(pos, float).copy()
    if not len(at_ms) or impact is None or not len(impact):
        return pos, []
    end = at_ms.max()
    bounds = np.unique(np.concatenate([np.asarray(cut_ms, float), np.arange(0, end + seg_s * 1000, seg_s * 1000)]))
    sid = np.searchsorted(bounds, at_ms)
    r = int(win_ms / 1000 / hop_s)
    peak = np.array([impact[max(int(t / 1000 / hop_s) - r, 0):int(t / 1000 / hop_s) + r + 1].max()
                     if int(t / 1000 / hop_s) < len(impact) else np.nan for t in at_ms])
    flips = []
    for s in np.unique(sid):
        m = (sid == s) & ~np.isnan(peak)
        ins, outs = m & (pos > 50), m & (pos <= 50)
        if ins.sum() < min_pts // 2 or outs.sum() < min_pts // 2:
            continue
        d = float(peak[outs].mean() - peak[ins].mean())
        if d > margin and P is not None:
            # the learned evidence gets a veto: if it clearly backs the current direction, the sound does not
            # turn the shot (test video A: v2 +0.26 on the wrongly flipped 1:13-1:34 shot, +0.11 on the rightly flipped one)
            sel = np.nonzero(sid == s)[0]
            i = np.clip(np.round(at_ms[sel] / 1000 * tm.RATE).astype(int), 0, len(P) - 1)
            k = np.where(pos[sel] > 50, 1, -1)
            if float(np.mean((P[i, 0] - P[i, 1]) * k)) > max_model_agree:
                continue
        if d > margin:
            pos[sid == s] = 100.0 - pos[sid == s]
            flips.append((float(at_ms[sid == s].min()) / 1000, float(at_ms[sid == s].max()) / 1000, d))
    return pos, flips


def mel_flux(wav, hop_s: float = 0.01) -> np.ndarray:
    """64-band log-mel onset flux (T, 64) at hop_s -- the same features the study used."""
    import librosa
    import soundfile as sf
    y, sr = sf.read(str(wav), dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=int(hop_s * sr), n_mels=64, fmin=30)
    L = np.log1p(100 * M)
    return np.maximum(np.diff(L, axis=1, prepend=L[:, :1]), 0).T.astype(np.float32)


def has_music(beats_s, duration_s: float) -> bool:
    """Beat tracker finds a steady beat over at least half the video (test video A: 8%, a music HMV: ~100%)."""
    b = np.asarray(beats_s, float)
    if len(b) < 8 or duration_s <= 0:
        return False
    return len(b) * float(np.median(np.diff(b))) / duration_s >= 0.5


def local_half_period(P, act_thr: float = 0.3, win_s: float = 4.0, lo_ms: float = 90.0, hi_ms: float = 1200.0):
    """Per-step expected half-cycle (in->out) in steps, from the autocorrelation of p_in - p_out
    over a sliding window (the dominant stroke rhythm the evidence shows)."""
    d = P[:, 0] - P[:, 1]
    T = len(d)
    w = int(win_s * tm.RATE)
    lo, hi = max(int(lo_ms / 1000 * tm.RATE), 2), int(hi_ms / 1000 * tm.RATE)
    hp = np.full(T, np.nan)
    step = int(tm.RATE // 2)
    for c in range(0, T, step):
        a, b = max(c - w // 2, 0), min(c + w // 2, T)
        x = d[a:b] - d[a:b].mean()
        if len(x) < 2 * hi or x.std() < 1e-4:
            continue
        ac = np.correlate(x, x, "full")[len(x) - 1:]
        ac /= ac[0] + 1e-9
        # a full cycle is the first strong positive peak; half cycle = half of it
        full_lo, full_hi = 2 * lo, min(2 * hi, len(ac) - 1)
        if full_hi <= full_lo:
            continue
        k = full_lo + int(np.argmax(ac[full_lo:full_hi]))
        if ac[k] > 0.1:
            hp[c:c + step] = k / 2
    good = ~np.isnan(hp)
    if not good.any():
        return np.full(T, 0.25 * tm.RATE)
    idx = np.arange(T)
    return np.interp(idx, idx[good], hp[good])


def half_period_from_f0(f0: np.ndarray, conf: np.ndarray, fps: float, n: int, conf_thr: float = 0.3,
                        fallback: np.ndarray | None = None) -> np.ndarray:
    """Expected half-cycle in model steps from the motion tracker's measured stroke frequency
    (its tempo is the most reliable of all sources: held-out tempo_ok 0.65), where it is
    confident; elsewhere the fallback (the evidence's own autocorrelation estimate)."""
    t = np.arange(n) / tm.RATE
    src = np.arange(len(f0)) / fps
    f = np.interp(t, src, f0)
    c = np.interp(t, src, conf)
    hp = tm.RATE / (2 * np.maximum(f, 0.2))
    if fallback is not None:
        hp = np.where(c >= conf_thr, hp, fallback[:n])
    return hp


def track(P, act_thr: float = 0.3, alpha: float = 4.0, min_run_s: float = 1.5, hp: np.ndarray | None = None,
          in_first: bool = False, split_steps=None):
    """Dynamic-programming stroke tracker (after Ellis' beat tracker): choose alternating IN/OUT
    points that sit on high p_in / p_out while each half-cycle stays close to the local expected
    half period -- the evidence decides where, the rhythm constraint keeps the spacing steady.
    Runs separately over each stretch where the model thinks the user is scripting."""
    T = len(P)
    if hp is None:
        hp = local_half_period(P, act_thr)
    active = P[:, 2] >= act_thr
    if split_steps is not None and len(split_steps):
        # every shot finds its own rhythm (user, 2026-09-26: patterns start before the cut; editors sometimes
        # break the thrust phase at a camera change)
        active = active.copy()
        active[np.clip(np.asarray(split_steps, int), 0, T - 1)] = False
    out_t, out_k = [], []
    i = 0
    while i < T:
        if not active[i]:
            i += 1
            continue
        j = i
        while j < T and active[j]:
            j += 1
        if (j - i) / tm.RATE >= min_run_s:
            seg = slice(i, j)
            ev = (_dp_in_first if in_first else _dp)(P[seg, 0], P[seg, 1], hp[seg], alpha)
            out_t += [(i + t) / tm.RATE * 1000 for t, _ in ev]
            out_k += [k for _, k in ev]
        i = j
    # where two chains meet, two points of the same kind can end up side by side: keep them alternating
    # (the stronger one of a same-kind pair wins)
    keep_t, keep_k = [], []
    for t, k in zip(out_t, out_k):
        if keep_k and keep_k[-1] == k:
            i_prev = int(np.clip(round(keep_t[-1] / 1000 * tm.RATE), 0, T - 1))
            i_new = int(np.clip(round(t / 1000 * tm.RATE), 0, T - 1))
            col = 0 if k == 1 else 1
            if P[i_new, col] > P[i_prev, col]:
                keep_t[-1] = t
            continue
        keep_t.append(t)
        keep_k.append(k)
    return np.array(keep_t), np.where(np.array(keep_k) == 1, 100.0, 0.0)


def _dp(pin, pout, hp, alpha):
    n = len(pin)
    S = {1: pin - 0.5 * pout, -1: pout - 0.5 * pin}      # reward for placing a kind at t
    C = {k: S[k].copy() for k in (1, -1)}
    back = {k: np.full(n, -1, int) for k in (1, -1)}
    for t in range(1, n):
        h = hp[t]
        lo, hi = max(int(t - 2.0 * h), 0), int(t - 0.5 * h)
        if hi <= lo:
            continue
        tau = np.arange(lo, hi)
        pen = alpha * np.log((t - tau) / h) ** 2
        for k in (1, -1):
            cand = C[-k][lo:hi] - pen
            m = int(np.argmax(cand))
            if cand[m] > 0:                               # only chain when it pays off
                C[k][t] = S[k][t] + cand[m]
                back[k][t] = lo + m
    # best end, walk back; then the best chain that ends before this one started, and so on -- a dip in the
    # evidence breaks a chain, and every piece has to be kept (before 2026-09-25 only the last one was, which
    # left holes such as test video A 36-39 s)
    ev = []
    limit = n
    while limit > 0:
        e1, e2 = C[1][:limit], C[-1][:limit]
        if max(e1.max(), e2.max()) <= 0:
            break
        k = 1 if e1.max() >= e2.max() else -1
        t = int(np.argmax(C[k][:limit]))
        chain = []
        while t >= 0:
            chain.append((t, k))
            t, k = back[k][t], -k
        chain.reverse()
        ev = chain + ev
        limit = chain[0][0] - 1
    return ev


def orient_by_evidence(og_at, og_pos, P, cut_ms=()):
    """OG turn kinds (+1 IN / -1 OUT) with each shot turned to agree with the evidence P
    (vote of p_in - p_out at the OG turns, as in thrust_model.hybrid)."""
    og_at = np.asarray(og_at, float)
    k = np.where(np.asarray(og_pos) > 50, 1, -1)
    if not len(og_at):
        return k
    step = np.clip(np.round(og_at / 1000 * tm.RATE).astype(int), 0, len(P) - 1)
    lo, hi = np.clip(step - 2, 0, len(P) - 1), np.clip(step + 2, 0, len(P) - 1)
    pin = np.array([P[a:b + 1, 0].max() for a, b in zip(lo, hi)])
    pout = np.array([P[a:b + 1, 1].max() for a, b in zip(lo, hi)])
    sid = np.searchsorted(np.sort(np.asarray(cut_ms, float)), og_at)
    out = k.copy()
    for s in np.unique(sid):
        m = sid == s
        if np.sum((pin[m] - pout[m]) * k[m]) < 0:
            out[m] = -k[m]
    return out


def anchor_evidence(P, og_at, og_kind, reliable_at, weight: float = 1.0, sigma_steps: float = 1.5,
                    both: bool = False):
    """Add the visible motion reversals to the evidence: a sharp bump on the right kind at each
    OG turn the tracker is confident about, so the DP tracker locks onto real thrusts where it can
    see them (no drift) and only free-runs on its rhythm where it cannot."""
    Q = P.copy()
    T = len(Q)
    r = int(4 * sigma_steps)
    ker = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_steps) ** 2) * weight
    for t, k, ok in zip(og_at, og_kind, reliable_at):
        if not ok:
            continue
        c = int(round(t / 1000 * tm.RATE))
        a, b = max(c - r, 0), min(c + r + 1, T)
        if b <= a:
            continue
        cols = (0, 1) if both else ((0,) if k == 1 else (1,))
        for col in cols:           # both = pin the TIME of the reversal, let the evidence pick in or out
            Q[a:b, col] += ker[a - (c - r):b - (c - r)]
    return Q


def splice(og_at, og_kind, reliable_at, tr_at, tr_pos, min_run: int = 4):
    """Visible + reliable stretches of OG turns (runs of >= min_run) replace the tracker's points
    there; the tracker's rhythm covers everything else."""
    og_at = np.asarray(og_at, float)
    runs, cur = [], []
    for i, ok in enumerate(reliable_at):
        if ok:
            cur.append(i)
        else:
            if len(cur) >= min_run:
                runs.append(cur)
            cur = []
    if len(cur) >= min_run:
        runs.append(cur)
    keep = np.ones(len(tr_at), bool)
    ev = []
    for run in runs:
        a, b = og_at[run[0]], og_at[run[-1]]
        keep &= ~((tr_at >= a - 50) & (tr_at <= b + 50))
        ev += [(og_at[i], int(og_kind[i]), 1.0) for i in run]
    ev += [(t, 1 if p > 50 else -1, 0.5) for t, p in zip(np.asarray(tr_at)[keep], np.asarray(tr_pos)[keep])]
    ev.sort()
    out = []
    for e in ev:
        if out and (out[-1][1] == e[1] or e[0] - out[-1][0] < 60):
            if e[2] > out[-1][2]:
                out[-1] = e
            continue
        out.append(e)
    return np.array([e[0] for e in out]), np.array([100.0 if e[1] == 1 else 0.0 for e in out])


def sound_evidence(P, impact: np.ndarray, hop_s: float = 0.01, z_on: float = 1.5, act_boost: float = 0.6,
                   in_weight: float = 0.3, win_s: float = 1.5, min_hits: int = 3):
    """Videos without music: clear, repeated thrust sounds count as 'the user would be scripting here'
    and add a little IN evidence at each impact, so the tracker keeps going off screen and lands on the
    sounds (user, 2026-09-25, test video A 38-41 s: 'the start thrust starting off screen'). Returns a new P."""
    from scipy.signal import find_peaks
    Q = P.copy()
    T = len(Q)
    pk, _ = find_peaks(impact, height=z_on, distance=max(int(0.12 / hop_s), 1))
    if len(pk) < min_hits:
        return Q
    t_pk = pk * hop_s
    steps = np.arange(T) / tm.RATE
    # activity where at least min_hits impacts fall inside +-win_s/2
    lo = np.searchsorted(t_pk, steps - win_s / 2)
    hi = np.searchsorted(t_pk, steps + win_s / 2)
    busy = (hi - lo) >= min_hits
    Q[:, 2] = np.where(busy, np.maximum(Q[:, 2], act_boost), Q[:, 2])
    r = 2
    ker = np.exp(-0.5 * (np.arange(-r, r + 1) / 1.0) ** 2) * in_weight
    for t in t_pk:
        c = int(round(t * tm.RATE))
        if 0 <= c < T and busy[c]:
            a, b = max(c - r, 0), min(c + r + 1, T)
            Q[a:b, 0] += ker[a - (c - r):b - (c - r)]
    return Q


def sound_half_period(impact: np.ndarray, n: int, hop_s: float = 0.01, z_on: float = 1.5, win_s: float = 4.0,
                      max_cv: float = 0.35):
    """Half-cycle (model steps) from the spacing of clear thrust sounds, where they come regularly
    (>= 3 impacts within +-win_s/2, spacing CV <= max_cv); NaN elsewhere."""
    from scipy.signal import find_peaks
    pk, _ = find_peaks(impact, height=z_on, distance=max(int(0.12 / hop_s), 1))
    t = pk * hop_s
    out = np.full(n, np.nan)
    if len(t) < 3:
        return out
    steps = np.arange(n) / tm.RATE
    lo = np.searchsorted(t, steps - win_s / 2)
    hi = np.searchsorted(t, steps + win_s / 2)
    for i in range(0, n, int(tm.RATE // 4)):
        seg = t[lo[i]:hi[i]]
        if len(seg) >= 3:
            d = np.diff(seg)
            d = d[d < 1.5]
            if len(d) >= 2 and d.std() / d.mean() <= max_cv:
                out[i:i + int(tm.RATE // 4)] = d.mean() / 2 * tm.RATE    # one impact per stroke -> half = gap/2
    return out


def combine_half_period(hp_motion: np.ndarray, hp_sound: np.ndarray, lo: float = 0.6, hi: float = 1.6):
    """Keep the motion tracker's tempo unless clear thrust sounds say it is far off (test video A 40-41.5 s: the
    tracker read a 26 spm drift while the impacts came every ~0.45 s)."""
    ok = ~np.isnan(hp_sound)
    r = np.where(ok, hp_motion / np.where(ok, hp_sound, 1.0), 1.0)
    return np.where(ok & ((r < lo) | (r > hi)), hp_sound, hp_motion)


def slow_holes(at_ms, factor: float = 2.2, min_ms: float = 700.0, max_ms: float = 6000.0, look_ms: float = 3000.0):
    """Stretches between consecutive points that are far wider than the rhythm around them (test video A 39.6-41.8 s:
    ~1 s spacing between stretches at ~0.28 s) -> gaps for rhythm continuation."""
    at = np.asarray(at_ms, float)
    out = []
    if len(at) < 6:
        return out
    d = np.diff(at)
    for i, g in enumerate(d):
        if g < min_ms or g > max_ms:
            continue
        m = (at[:-1] >= at[i] - look_ms) & (at[:-1] <= at[i + 1] + look_ms)
        m[i] = False
        near = d[m]
        near = near[near < min_ms]
        if len(near) >= 4 and g > factor * np.median(near):
            out.append((float(at[i]), float(at[i + 1])))
    return out


def alternate(at_ms, pos):
    """Final clean-up: never two INs or two OUTs in a row (a flipped shot next to an unflipped one can
    leave one pair at the boundary) -- of a same-kind pair the first is kept."""
    at, pos = np.asarray(at_ms, float), np.asarray(pos, float)
    if not len(at):
        return at, pos
    keep = np.ones(len(at), bool)
    last = pos[0] > 50
    for i in range(1, len(at)):
        cur = pos[i] > 50
        if cur == last:
            keep[i] = False
        else:
            last = cur
    return at[keep], pos[keep]


def rest_points(at_ms, pos, gap_ms: float = 1500.0, max_lead_ms: float = 600.0):
    """Every stretch of action starts and ends at 0 (user, 2026-09-26): if a stretch opens on an IN, a 0 is
    put before it spaced like the stroke that follows; if it ends on an IN, a 0 follows it spaced like the
    stroke before. A stretch = points closer than gap_ms."""
    at, pos = np.asarray(at_ms, float), np.asarray(pos, float)
    if len(at) < 2:
        return at, pos
    brk = np.nonzero(np.diff(at) > gap_ms)[0]
    starts, ends = np.r_[0, brk + 1], np.r_[brk, len(at) - 1]
    add_t, add_p = [], []
    prev_end = -np.inf
    for s, e in zip(starts, ends):
        if pos[s] > 50:
            d = min(at[s + 1] - at[s], max_lead_ms) if e > s else 250.0
            t = at[s] - d
            if t > prev_end + 30 and t >= 0:
                add_t.append(t)
                add_p.append(0.0)
        if pos[e] > 50:
            d = min(at[e] - at[e - 1], max_lead_ms) if e > s else 250.0
            nxt = at[e + 1] if e + 1 < len(at) else np.inf
            t = at[e] + d
            if t < nxt - 30:
                add_t.append(t)
                add_p.append(0.0)
        prev_end = at[e]
    if not add_t:
        return at, pos
    a = np.r_[at, add_t]
    p = np.r_[pos, add_p]
    o = np.argsort(a, kind="stable")
    return a[o], p[o]


def _dp_in_first(pin, pout, hp, alpha, lo_frac: float = 0.15, hi_frac: float = 0.85):
    """Rhythm on the IN points only (one full stroke apart, close to 2 x the half period); every OUT then goes
    where the evidence says between two INs -- a fast in and a slow out is allowed (user, 2026-09-26, 2:22-2:32:
    forcing equal halves put the pattern half a wave off)."""
    n = len(pin)
    S = pin - 0.5 * pout
    C = S.copy()
    back = np.full(n, -1, int)
    for t in range(1, n):
        per = 2 * hp[t]
        lo, hi = max(int(t - 1.6 * per), 0), int(t - 0.6 * per)
        if hi <= lo:
            continue
        tau = np.arange(lo, hi)
        cand = C[lo:hi] - alpha * np.log((t - tau) / per) ** 2
        m = int(np.argmax(cand))
        if cand[m] > 0:
            C[t] = S[t] + cand[m]
            back[t] = lo + m
    ins = []
    limit = n
    while limit > 0:
        if C[:limit].max() <= 0:
            break
        t = int(np.argmax(C[:limit]))
        chain = []
        while t >= 0:
            chain.append(t)
            t = back[t]
        chain.reverse()
        ins = chain + ins
        limit = chain[0] - 1
    ev = []
    for a, b in zip(ins, ins[1:] + [None]):
        ev.append((a, 1))
        if b is None or b - a > 3.2 * hp[a]:
            continue                       # chain break: no OUT invented across it (rest points close it)
        lo, hi = a + max(int(lo_frac * (b - a)), 1), a + max(int(hi_frac * (b - a)), 2)
        ev.append((lo + int(np.argmax(pout[lo:hi])), -1))
    return ev


def real_outs(at_ms, pos, vb: np.ndarray, conf: np.ndarray, fps: float, conf_thr: float = 0.55,
              lo_frac: float = 0.12, hi_frac: float = 0.88, min_turns: int = 6):
    """Fast in, slow out (user, 2026-09-26: "match the thrust speed of the thrust in and thrust out based off the
    time from getting to point a and to point b"): between two IN points, the OUT goes where the visible motion
    really turns around -- the extreme of the unfiltered motion position, not the middle of an even wave.
    Only where the motion is clearly visible; the IN points are not moved."""
    from scipy.signal import butter, sosfiltfilt
    at, pos = np.asarray(at_ms, float), np.asarray(pos, float).copy()
    if len(at) < min_turns or len(vb) < 60:
        return at, pos
    x = sosfiltfilt(butter(2, 0.3, btype="high", fs=fps, output="sos"), np.cumsum(vb))
    T = len(x)
    fr = lambda ms: int(np.clip(round(ms / 1000 * fps), 0, T - 1))  # noqa: E731
    is_in = pos > 50
    # which extreme of x is "in" -- decided per stretch of consecutive points from the points themselves
    out = at.copy()
    idx_in = np.nonzero(is_in)[0]
    for a_i, b_i in zip(idx_in[:-1], idx_in[1:]):
        if b_i - a_i != 2:                       # expect IN, OUT, IN
            continue
        o_i = a_i + 1
        fa, fb = fr(at[a_i]), fr(at[b_i])
        if fb - fa < 4 or np.median(conf[fa:fb + 1]) < conf_thr:
            continue
        # the IN points sit on one extreme: pick the other one in between
        sgn = 1.0 if (x[fa] + x[fb]) / 2 > x[fa:fb + 1].mean() else -1.0
        lo, hi = fa + max(int(lo_frac * (fb - fa)), 1), fa + max(int(hi_frac * (fb - fa)), 2)
        j = lo + int(np.argmin(sgn * x[lo:hi + 1]))
        dj = 0.0
        if 0 < j < T - 1:
            y0, y1, y2 = -sgn * x[j - 1], -sgn * x[j], -sgn * x[j + 1]
            den = y0 - 2 * y1 + y2
            if den < 0:
                dj = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
        t_new = (j + dj) / fps * 1000
        if at[a_i] + 30 < t_new < at[b_i] - 30:
            out[o_i] = t_new
    return out, pos


def beat_lock(at_ms, pos, beats_ms, window_ms: float = 8000.0, near_ms: float = 90.0, min_share: float = 0.5,
              min_ins: int = 5):
    """HMVs where the thrusts follow the music (user, 2026-10-24): in each stretch where most IN points already sit
    near a beat (and the strokes run at the beat's pace or a simple multiple of it), snap every IN exactly onto its
    beat and re-centre the OUTs between them. Elsewhere nothing changes.
    Measured on the user's 135 clean music videos: ~11% of 8 s stretches are beat-locked like this."""
    at, pos = np.asarray(at_ms, float).copy(), np.asarray(pos, float)
    beats = np.asarray(beats_ms, float)
    if len(beats) < 8 or len(at) < 6:
        return at, pos, []
    ins = np.nonzero(pos > 50)[0]
    j = np.clip(np.searchsorted(beats, at[ins]), 1, len(beats) - 1)
    nb = np.where(np.abs(at[ins] - beats[j - 1]) <= np.abs(beats[j] - at[ins]), beats[j - 1], beats[j])
    near = np.abs(at[ins] - nb) <= near_ms
    locked = np.zeros(len(ins), bool)
    spans = []
    for a in np.arange(at[0], at[-1], window_ms / 2):
        m = (at[ins] >= a) & (at[ins] < a + window_ms)
        if m.sum() < min_ins or near[m].mean() < min_share:
            continue
        # strokes at the beat's pace (or 2x / 0.5x): the IN spacing is close to a whole number of beats
        sp = np.diff(at[ins][m])
        bi = np.median(np.diff(beats[(beats >= a) & (beats < a + window_ms)])) if ((beats >= a) & (beats < a + window_ms)).sum() > 2 else np.nan
        if not np.isfinite(bi) or len(sp) < 3:
            continue
        ratio = np.median(sp) / bi
        if min(abs(ratio - r) for r in (0.5, 1.0, 2.0)) > 0.2:
            continue
        locked |= m
        spans.append((float(a), float(a + window_ms)))
    if not locked.any():
        return at, pos, []
    at[ins[locked]] = nb[locked]
    # OUTs halfway between their neighbouring INs where those moved, keeping the order strictly increasing
    for i in range(1, len(at) - 1):
        if pos[i] <= 50 and pos[i - 1] > 50 and pos[i + 1] > 50:
            lo, hi = at[i - 1], at[i + 1]
            if not (lo < at[i] < hi) or (i - 1 in set(ins[locked]) and i + 1 in set(ins[locked])):
                at[i] = (lo + hi) / 2
    order = np.argsort(at, kind="stable")
    at, pos = at[order], pos[order]
    keep = np.r_[True, np.diff(at) > 20]
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return at[keep], pos[keep], merged


def music_mode(at_ms, pos, beats_ms, gap_ms: float = 1500.0):
    """User-chosen per video (OFS panel, 2026-10-24): the thrusts follow the music. Wherever the program is scripting,
    every IN goes onto a beat and every OUT halfway between. The pace per stretch (every beat, every 2nd / 4th beat,
    or twice per beat) is the one closest to the strokes the program found there, so slow and fast parts stay
    slow and fast. Measured on the user's 3 most beat-locked videos: 69-76% of their INs sit within 60 ms of a beat."""
    at, pos = np.asarray(at_ms, float), np.asarray(pos, float)
    beats = np.asarray(beats_ms, float)
    if len(beats) < 8 or len(at) < 4:
        return at, pos
    # stretches of continuous scripting
    br = np.nonzero(np.diff(at) > gap_ms)[0]
    starts, ends = np.r_[0, br + 1], np.r_[br, len(at) - 1]
    out_t, out_p = [], []
    for a, b in zip(starts, ends):
        t0, t1 = at[a], at[b]
        bb = beats[(beats >= t0 - 1) & (beats <= t1 + 1)]
        ins = at[a:b + 1][pos[a:b + 1] > 50]
        if len(bb) < 4 or len(ins) < 3:
            out_t += list(at[a:b + 1])
            out_p += list(pos[a:b + 1])
            continue
        iv = float(np.median(np.diff(ins)))
        bi = float(np.median(np.diff(bb)))
        k = min((0.5, 1, 2, 4), key=lambda s: abs(np.log(iv / (bi * s))))
        if k == 0.5:
            grid = np.sort(np.r_[bb, (bb[:-1] + bb[1:]) / 2])
        else:
            grid = bb[::int(k)]
        if len(grid) < 2:
            out_t += list(at[a:b + 1])
            out_p += list(pos[a:b + 1])
            continue
        mids = (grid[:-1] + grid[1:]) / 2
        tt = np.empty(len(grid) + len(mids))
        pp = np.empty_like(tt)
        tt[0::2], pp[0::2] = grid, 100.0
        tt[1::2], pp[1::2] = mids, 0.0
        if pos[a] <= 50:                     # the stretch started from an OUT: keep it starting there
            tt, pp = np.r_[t0, tt], np.r_[0.0, pp]
        out_t += list(tt)
        out_p += list(pp)
    order = np.argsort(out_t, kind="stable")
    t, p = np.asarray(out_t)[order], np.asarray(out_p)[order]
    keep = np.r_[True, np.diff(t) > 20]
    return t[keep], p[keep]
