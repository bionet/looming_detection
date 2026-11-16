"""
stimulus.py -- Looming-object trajectory and retinal luminance.

A looming disk is drawn on a 512x512 image under an equidistant projection
(object radius ``r = 1 cm``, approach to closest approach at ``t_ca = 0.75 s``;
see ``_render.py``), then sampled by the fixed-point hexagonal retina
(box-average, window=16, spacing=12, radius=20, center (256,256)) to give the
per-ommatidium luminance L[t, j] in [0, 1].

  * direct hit (``b <= 1 mm``): a centered collision, ``z(t) = v (t_ca - t)``;
  * near miss (``b > 1 mm``): a straight-line fly-by that sweeps upward and exits
    the frontal hemisphere once ``z < 0``.

Ground truth per frame: angular diameter ``theta = 2 arctan(r / range)``, solid
angle ``Omega = 2 pi (1 - cos(theta/2))``, and center elevation ``psi`` (the
manuscript's convention ``degrees((256 - cy_px) / f_px)``), all masked to the
visible phase ``z > 0``.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from . import _render as RE

R_DEFAULT = 0.01          # object radius / half-size L (m)
T_CA = 0.75               # time of closest approach / collision (s)
F_PX = RE.F_PX_DEFAULT    # 217.2
B_CRIT = 1.0e-3           # direct-hit threshold on pass-by distance (m)


@dataclass
class Trajectory:
    t: np.ndarray
    dt: float
    theta: np.ndarray       # deg (angular diameter), NaN while not visible
    Omega: np.ndarray       # sr
    psi: np.ndarray         # deg (center elevation), NaN while not visible
    z: np.ndarray           # range along the optical axis (m)
    v: float
    b: float
    r: float = R_DEFAULT

    @property
    def is_direct_hit(self) -> bool:
        return self.b <= B_CRIT


def _dh_params(v: float) -> "RE.DirectHitParams":
    base = RE.DirectHitParams(
        width=512, height=512, duration_s=1.0, fps=500,
        background=0, foreground=255, f_px=F_PX, L=R_DEFAULT, v_z=v,
        z_min_pass=0.10, t0=0.6, v_pass=0.8, r_max_px=0.48 * 512, antialias=False)
    return RE._dh_scenario_direct_hit_collision(base, ttc_s=T_CA)


def _nm_params(v: float, b: float) -> "RE.NearMissParams":
    return RE.NearMissParams(
        width=512, height=512, duration_s=1.0, fps=500,
        f_px=F_PX, L=R_DEFAULT, r_max_px=0.9 * 512,
        pass_speed=v, pass_v_perp=b / T_CA, pass_t_ca=T_CA, pass_b=b)


def _hex_sample(frames: np.ndarray) -> np.ndarray:
    s = RE.HexFixedPointVideo(frames, center_rc=RE._CENTER_RC,
                              radius=RE._HEX_RADIUS, window=RE._HEX_WINDOW,
                              spacing=RE._HEX_SPACING)
    return np.nan_to_num(np.asarray(s.sample(), dtype=float))   # [0, 1], OOB -> 0


def make_stimulus(v: float, b: float, *, fps: float = 500.0):
    """Render one trajectory. Returns ``(L, traj)`` with luminance L[t, j] in
    [0, 1] (shape (T, NC)) and the ground-truth :class:`Trajectory`."""
    dt = 1.0 / fps
    if b <= B_CRIT:
        frames, pf = RE._render_direct_hit_frames(_dh_params(v))
        z = np.asarray(pf["z"], dtype=float)
        rng = z.copy()
        cy_px = np.asarray(pf["cy_px"], dtype=float)
    else:
        frames, pf = RE._render_near_miss_frames(_nm_params(v, b))
        z = np.asarray(pf["z"], dtype=float)                 # optical-axis depth
        rng = np.asarray(pf["range"], dtype=float)           # true range
        cy_px = np.asarray(pf["cy_px"], dtype=float)
    t = np.asarray(pf["t_s"], dtype=float)
    visible = z > 1e-6
    rng_safe = np.where(rng > 1e-6, rng, 1e-6)
    theta = 2.0 * np.degrees(np.arctan(R_DEFAULT / rng_safe))
    psi = np.degrees((256.0 - cy_px) / F_PX)
    theta = np.where(visible, theta, np.nan)
    psi = np.where(visible, psi, np.nan)
    Omega = 2.0 * np.pi * (1.0 - np.cos(np.radians(theta) / 2.0))
    L = _hex_sample(frames)
    traj = Trajectory(t=t, dt=dt, theta=theta, Omega=Omega, psi=psi,
                      z=z, v=v, b=b)
    return L, traj


def make_trajectory(v: float, b: float, *, fps: float = 500.0) -> Trajectory:
    """Ground-truth trajectory only (renders and discards the luminance)."""
    return make_stimulus(v, b, fps=fps)[1]


def render_luminance(v: float, b: float, *, fps: float = 500.0) -> np.ndarray:
    """Retinal luminance L[t, j] in [0, 1] for the given trajectory."""
    return make_stimulus(v, b, fps=fps)[0]
