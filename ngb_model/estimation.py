"""
estimation.py -- Solid angle and center elevation from the LPLC2 annulus.

This is the manuscript's estimator (Appendix "activation disk").  The activated
LPLC2s form an annulus around the expanding disk boundary; two processing steps
recover the full disk from that annulus, after which the covered solid angle and
the center elevation are read off the population:

  1. **Normalize** each LPLC2 by the per-frame population max (removes the
     near-miss activation asymmetry).
  2. **Nonlinear temporal filter**: an onset high-pass, a compressive power law
     (exponent ``COMPRESS_P = 0.3``), and a hold (leaky peak-hold, time constant
     ``TAU_DECAY = 0.5 s``) that fills the disk interior behind the passing
     boundary (the onset-decay state update).

The processed activity then gives:

  * ``Omega_hat = (sum of processed activity) * omega_neuron``, ``omega_neuron =
    s^2 * sqrt(3)/2`` (hex tile for LPLC2 spacing ``s = 11.9 deg``), then a fixed
    affine **direct-hit calibration** so the estimate tracks the identity line
    (matching Fig. 12b);
  * ``psi_hat`` = activity-weighted mean elevation of the active annulus.

``psi_hat`` is accurate to ~35 deg then saturates (finite retinal coverage);
``Omega_hat`` tracks the identity through the escape-relevant range.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .geometry import Circuit

S_LPLC2 = 11.9
OMEGA_NEURON = np.radians(S_LPLC2) ** 2 * np.sqrt(3) / 2.0
COMPRESS_P = 0.3
TAU_DECAY = 0.5
RESET_DROP = 0.1
ACTIVE_FRAC = 0.05

# Direct-hit calibration of the raw summed-activity estimate (slope, offset),
# fit once on a reference direct-hit so Omega_hat tracks true Omega (Fig. 12b).
# Omega_hat = (raw - OMEGA_CAL_OFFSET) / OMEGA_CAL_SLOPE.
OMEGA_CAL_SLOPE = 1.3543
OMEGA_CAL_OFFSET = 0.4206


@dataclass
class Estimates:
    Omega_hat: np.ndarray
    psi_hat: np.ndarray
    Omega_raw: np.ndarray    # before the direct-hit calibration


def _onset_decay(y: np.ndarray, dt: float):
    """The manuscript onset-decay processing. Returns processed state (T, N)."""
    T, N = y.shape
    alpha = np.exp(-dt / TAU_DECAY)
    state = np.zeros(N)
    prev_w = np.zeros(N)
    started = False
    out = np.zeros((T, N))
    for t in range(T):
        mx = y[t].max()
        if mx < 1e-6:
            out[t] = state
            continue
        w = y[t] / mx
        wc = np.power(np.maximum(w, 0.0), COMPRESS_P)
        if not started:
            state = wc.copy()
            started = True
        else:
            ns = np.maximum(alpha * state, wc)
            ns[wc > state] = wc[wc > state]
            drop = w < prev_w - RESET_DROP
            ns[drop] = wc[drop]
            state = ns
        prev_w = w
        out[t] = state
    return out


def estimate(y: np.ndarray, circ: Circuit, dt: float = 1.0 / 500.0) -> Estimates:
    """Estimate Omega(t) and psi(t) from LPLC2 activity y (T, N)."""
    T = y.shape[0]
    state = _onset_decay(y, dt)
    Omega_raw = np.full(T, np.nan)
    psi_hat = np.full(T, np.nan)
    cy = circ.cy_deg
    for t in range(T):
        s = state[t].sum()
        if s > 0.1:
            Omega_raw[t] = s * OMEGA_NEURON
            # Elevation from the processed (disk-filled, symmetric) activity:
            # weighting by the raw annulus over-reads, because the near-miss
            # boundary is brighter on its leading (upper) edge; the hold fills
            # the disk symmetrically, so its centroid is the true center.
            psi_hat[t] = (state[t] * cy).sum() / s
    Omega_hat = (Omega_raw - OMEGA_CAL_OFFSET) / OMEGA_CAL_SLOPE
    return Estimates(Omega_hat=Omega_hat, psi_hat=psi_hat, Omega_raw=Omega_raw)


def fit_omega_calibration(dh_true, dh_raw):
    """Least-squares (slope, offset) mapping raw -> true Omega over direct-hits.

    ``dh_true``/``dh_raw`` are 1-D arrays of true and raw-estimated solid angle
    pooled over one or more direct-hit trajectories.  Returns ``(slope, offset)``
    such that ``(raw - offset) / slope`` tracks the identity.
    """
    dh_true = np.asarray(dh_true, float)
    dh_raw = np.asarray(dh_raw, float)
    m = np.isfinite(dh_true) & np.isfinite(dh_raw)
    A = np.vstack([dh_true[m], np.ones(m.sum())]).T
    slope, offset = np.linalg.lstsq(A, dh_raw[m], rcond=None)[0]
    return float(slope), float(offset)
