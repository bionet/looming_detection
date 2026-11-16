"""
giant_fiber.py -- Giant Fiber readout, DNP, and integrate-and-fire.

The GF reads the LPLC2 population y(t) through its two lateral-dendrite
sub-branches with nonnegative weight vectors C1 (dorsal) and C2 (ventral):

    u1(t) = C1^T y(t),   u2(t) = C2^T y(t),   u(t) = u1 - u2 = (C1 - C2)^T y(t).

The differential weight vector is the quadratic ``[C1 - C2]_i = 1 - (psi^i/Delta)^2``
built in geometry.py; here C1 and C2 are its positive and negative parts. The
SIZ then applies the divisive-normalization peak detector (Eq. DNP)

    z(t) = u(t) / (1 + [tau du/dt]^2),   tau = 20 ms,

and an integrate-and-fire neuron emits the escape spike the first time its
membrane potential V(t) reaches threshold delta. The SIZ is a peak detector,
so the IAF is a leaky integrator (membrane time constant ``tau_m``) whose
potential tracks the amplitude of z(t); thresholding the peak amplitude keeps
the decision robust across approach speeds (a fast direct-hit has a brief but
tall z, which a pure time-integral of z would under-weight).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter1d


TAU_DNP = 0.020   # DNP time constant tau, s
TAU_M = 0.010     # IAF membrane time constant tau_m, s
# Zero-phase smoothing of u(t) before the DNP.  The raw LPLC2 population signal
# ripples at the ommatidia-ring-crossing frequency (a deterministic lattice
# effect, not noise); the figures read u(t) and z(t) through a light smoothing so each
# trace is a single clean peak.  The DNP derivative term otherwise amplifies the
# ripple into a spiky z(t).  ~12 ms tames it while leaving the u(t) peak location
# (and thus the escape-size tuning) unchanged.
SMOOTH_U_SIGMA = 0.012   # s


@dataclass
class GFResponse:
    u1: np.ndarray
    u2: np.ndarray
    u: np.ndarray
    z: np.ndarray
    V: np.ndarray            # cumulative IAF potential
    spike_index: int | None  # first index where V >= delta, else None


def split_weights(C_diff: np.ndarray):
    """Positive/negative parts of (C1 - C2) as the nonnegative C1, C2."""
    C1 = np.maximum(C_diff, 0.0)
    C2 = np.maximum(-C_diff, 0.0)
    return C1, C2


def _lowpass(x: np.ndarray, dt: float, tau: float) -> np.ndarray:
    if tau <= 0:
        return x.copy()
    alpha = dt / (tau + dt)
    y = np.empty_like(x)
    y[0] = x[0]
    for t in range(1, x.shape[0]):
        y[t] = y[t - 1] + alpha * (x[t] - y[t - 1])
    return y


def leaky_iaf(z: np.ndarray, dt: float, tau_m: float = TAU_M) -> np.ndarray:
    """Leaky-integrator membrane potential driven by the positive part of z.

    tau_m dV/dt = -V + [z]_+ ; the potential tracks (a smoothed version of)
    the z amplitude, so its maximum is the peak-detector output the IAF fires on.
    """
    drive = np.maximum(z, 0.0)
    alpha = dt / (tau_m + dt)
    V = np.empty_like(drive)
    V[0] = alpha * drive[0]
    for t in range(1, drive.shape[0]):
        V[t] = V[t - 1] + alpha * (drive[t] - V[t - 1])
    return V


def compute_gf(y: np.ndarray, C_diff: np.ndarray, dt: float,
               delta: float | None = None, tau: float = TAU_DNP,
               tau_m: float = TAU_M, smooth_sigma: float = SMOOTH_U_SIGMA) -> GFResponse:
    """GF readout, DNP output z(t), and IAF integration for population y (T, N).

    If ``delta`` is None the spike detection is skipped (spike_index = None);
    pass a calibrated threshold (see ``calibrate_delta``) to detect firing.
    ``u`` is smoothed (zero-phase Gaussian, ``smooth_sigma``) before the DNP so
    that du/dt is well-defined and each trace reads as a single peak.
    """
    C1, C2 = split_weights(C_diff)
    u1 = y @ C1
    u2 = y @ C2
    u = u1 - u2
    if smooth_sigma > 0:
        u = gaussian_filter1d(u, smooth_sigma / dt, mode="nearest")

    dudt = np.gradient(u, dt)
    z = u / (1.0 + (tau * dudt) ** 2)

    V = leaky_iaf(z, dt, tau_m)
    spike_index = None
    if delta is not None:
        hits = np.where(V >= delta)[0]
        spike_index = int(hits[0]) if hits.size else None

    return GFResponse(u1=u1, u2=u2, u=u, z=z, V=V, spike_index=spike_index)


def peak_potential(z: np.ndarray, dt: float, tau_m: float = TAU_M) -> float:
    """Maximum of the leaky IAF potential for a DNP output z(t)."""
    return float(leaky_iaf(z, dt, tau_m).max())


def calibrate_delta(dh_zs, nm_zs, dt: float, margin: float = 0.0) -> float:
    """Choose the IAF threshold delta that fires every direct-hit in ``dh_zs``.

    Following the manuscript's procedure (fire all direct-hits up to the escape
    velocity v_max), delta is set to the smallest direct-hit peak potential, so
    every listed DH crosses it. ``margin`` (in sr-like potential units) lowers
    it slightly to leave headroom. ``nm_zs`` is accepted for reference/checking.
    """
    dh_peaks = [peak_potential(z, dt) for z in dh_zs]
    delta = min(dh_peaks) - margin
    return float(delta)
