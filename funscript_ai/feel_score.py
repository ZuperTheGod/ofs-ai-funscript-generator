"""The "feel" score: in 4 s windows where the user is scripting, is a generated
script the same way up as theirs, reversed, or lost? (Correlation of position
curves, best lag within +-50 ms; >0.3 same, <-0.3 reversed, else lost.)
Checked against the user's own per-timestamp notes on test video C (2026-09-24):
it flags the same windows they called reversed / lost.
"""
import numpy as np

from . import funscript_io as fio

WIN_S = 4.0


def _curve(at, pos, t):
    return np.interp(t, at, pos) if len(at) else np.zeros_like(t)


def feel(ref_at, ref_pos, gen_at, gen_pos):
    rt, _, _ = fio.extremes(ref_at, ref_pos, 3.0)
    if len(rt) < 4:
        return None
    n_same = n_rev = n_lost = 0
    for a in np.arange(rt[0], rt[-1] - WIN_S * 1000, WIN_S * 1000):
        b = a + WIN_S * 1000
        inside = rt[(rt >= a) & (rt < b)]
        if len(inside) < 6 or np.max(np.diff(inside)) > 1500:
            continue                                   # user not steadily scripting here
        t = np.arange(a, b, 1000 / 60)
        r = _curve(ref_at, ref_pos, t)
        g = _curve(gen_at, gen_pos, t)
        best = 0.0
        for lag in range(-3, 4):
            gs = np.roll(g, lag)
            if r.std() > 1e-6 and gs.std() > 1e-6:
                c = float(np.corrcoef(r, gs)[0, 1])
                if abs(c) > abs(best):
                    best = c
        if best > 0.3:
            n_same += 1
        elif best < -0.3:
            n_rev += 1
        else:
            n_lost += 1
    n = n_same + n_rev + n_lost
    if n == 0:
        return None
    return dict(same=n_same / n, reversed=n_rev / n, lost=n_lost / n, windows=n)


def feel_windows(ref_at, ref_pos, gen_at, gen_pos):
    """Per-window verdicts: list of (window_start_ms, seconds_into_user_stretch, 'same'|'reversed'|'lost')."""
    rt, _, _ = fio.extremes(ref_at, ref_pos, 3.0)
    out = []
    if len(rt) < 4:
        return out
    starts = [rt[0]] + [b for a, b in zip(rt[:-1], rt[1:]) if b - a > 1500]
    starts = np.array(starts)
    for a in np.arange(rt[0], rt[-1] - WIN_S * 1000, WIN_S * 1000):
        b = a + WIN_S * 1000
        inside = rt[(rt >= a) & (rt < b)]
        if len(inside) < 6 or np.max(np.diff(inside)) > 1500:
            continue
        t = np.arange(a, b, 1000 / 60)
        r, g = _curve(ref_at, ref_pos, t), _curve(gen_at, gen_pos, t)
        best = 0.0
        for lag in range(-3, 4):
            gs = np.roll(g, lag)
            if r.std() > 1e-6 and gs.std() > 1e-6:
                c = float(np.corrcoef(r, gs)[0, 1])
                if abs(c) > abs(best):
                    best = c
        into = (a - starts[starts <= a].max()) / 1000
        out.append((a, into, "same" if best > 0.3 else "reversed" if best < -0.3 else "lost"))
    return out
