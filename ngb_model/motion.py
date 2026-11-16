"""
motion.py -- T4/T5 motion front-end in pure NumPy.

Each stage implements the corresponding block of the model, using the bundled
connectivity arrays:

  1. **PR + MVP.** Compressive luminance nonlinearity `T1(L)/T2(L)`
     (`T1 = a1 L + a2 L^2`, `T2 = 1 + c1 L + c2 L^2`) divisively normalized by
     the MVP wide-field surround.  The surround is a linear operator on the
     luminance: `surround_j = g_mvp * L_j * (E @ (s0 + s1 * L))_j`, where ``E``
     (bundled in circuit_data.npz) is the per-column MVP surround operator.
  2. **Band-pass -> ON/OFF.** A stateful biphasic temporal filter
     (`gamma(2,2) - gamma(6,4)`, applied as a 20-sample sliding convolution);
     ON = [.]_+, OFF = [-.]_+.
  3. **Borst correlator.** The ratio-form detector `y*(1+pref*a)/(1+null*a)`
     over the six hex neighbors, with the slot groupings and the per-slot
     presence mask (a slot is valid iff it is an actual graph neighbor).

Luminance ``L`` is passed in [0, 1]; it is scaled to the model's working range
by ``LUM_SCALE`` internally.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import convolve1d

from .geometry import Circuit

LUM_SCALE = 1.0e5


@dataclass
class MotionConfig:
    n_delay: int = 2
    alpha: float = 100.0
    a1: float = 1e-3
    a2: float = 1e-7
    c1: float = 1e-3
    c2: float = 1e-7
    g_mvp: float = 0.33
    s0: float = 0.001760445128947395 - 0.001
    s1: float = 1.0165216804198919e-07


def _bandpass_kernel():
    tH = np.arange(0, 20, 1.0)
    def g(t, n, tau):
        from math import factorial
        return (n * t) ** n * np.exp(-n * t / tau) / (factorial(n - 1) * tau ** (n + 1)) * (t[1] - t[0])
    return g(tH, 2, 2) - g(tH, 6, 4)


@dataclass
class MotionField:
    x_a: np.ndarray
    x_b: np.ndarray
    x_c: np.ndarray
    x_d: np.ndarray


def _bandpass(X, f):
    """Band-pass: 20-sample sliding convolution, per column."""
    fl = len(f)
    T, M = X.shape
    out = np.zeros_like(X)
    buf = np.zeros((fl, M))
    for t in range(T):
        buf[:-1] = buf[1:]; buf[-1] = X[t]
        out[t] = convolve1d(buf, f, axis=0, mode="nearest", origin=-(fl // 2))[-1]
    return out


def _borst(ch, nb, mask, n_delay, alpha):
    NC = ch.shape[1]

    def g(src, i):
        p = np.clip(nb[:, i], 0, NC - 1)
        return src[:, p] * mask[:, i][None, :]

    chd = np.empty_like(ch); chd[n_delay:] = ch[:-n_delay]; chd[:n_delay] = 0.0
    y = [g(ch, i) for i in range(7)]
    yd = [g(chd, i) for i in range(7)]
    m = [mask[:, i] for i in range(7)]

    def sd(x):
        x = x.copy(); x[x <= 0] = 1.0; return x

    m0 = m[0]
    y_a = (y[0] * m0 + y[1] * m[1] + y[4] * m[4]) / sd(m0 + m[1] + m[4])
    y_c = (y[0] * m0 + y[5] * m[5] + y[6] * m[6] + y[2] * m[2] + y[3] * m[3]) / sd(m0 + m[5] + m[6] + m[2] + m[3])
    y_d = (y[0] * m0 + y[2] * m[2] + y[3] * m[3] + y[5] * m[5] + y[6] * m[6]) / sd(m0 + m[2] + m[3] + m[5] + m[6])
    dd23 = (yd[2] * m[2] + yd[3] * m[3]) / sd(m[2] + m[3])
    dd56 = (yd[5] * m[5] + yd[6] * m[6]) / sd(m[5] + m[6])
    dd543 = (yd[5] * m[5] + yd[4] * m[4] + yd[3] * m[3]) / sd(m[5] + m[4] + m[3])
    dd126 = (yd[1] * m[1] + yd[2] * m[2] + yd[6] * m[6]) / sd(m[1] + m[2] + m[6])
    a = alpha
    va = y_a * (1 + dd23 * a) / (1 + dd56 * a)
    vb = y_a * (1 + dd56 * a) / (1 + dd23 * a)
    vc = y_c * (1 + dd543 * a) / (1 + dd126 * a)
    vd = y_d * (1 + dd126 * a) / (1 + dd543 * a)
    return va, vb, vc, vd


def compute_motion(L: np.ndarray, circ: Circuit, dt: float,
                   cfg: MotionConfig | None = None) -> MotionField:
    """Cardinal motion signals x_a..x_d from luminance L (T, NC), L in [0, 1]."""
    cfg = cfg or MotionConfig()
    Ls = L * LUM_SCALE
    # PR + MVP: divisive normalization by the measured surround operator E.
    T1 = cfg.a1 * Ls + cfg.a2 * Ls ** 2
    T2 = 1.0 + cfg.c1 * Ls + cfg.c2 * Ls ** 2
    surround = cfg.g_mvp * Ls * ((cfg.s0 + cfg.s1 * Ls) @ circ.E.T)
    am = T1 / (T2 + surround)
    # Band-pass -> ON/OFF.
    f = _bandpass_kernel()
    g = _bandpass(am, f)
    on = np.maximum(g, 0.0)
    off = np.maximum(-g, 0.0)
    # Borst on ON (T4) and OFF (T5).
    T4a, T4b, T4c, T4d = _borst(on, circ.nb, circ.mask, cfg.n_delay, cfg.alpha)
    T5a, T5b, T5c, T5d = _borst(off, circ.nb, circ.mask, cfg.n_delay, cfg.alpha)
    return MotionField(x_a=T4a + T5a, x_b=T4b + T5b, x_c=T4c + T5c, x_d=T4d + T5d)
