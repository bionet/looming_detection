"""
model.py -- End-to-end looming circuit.

    trajectory + luminance          (stimulus.py)
      -> T4/T5 motion (PR+MVP->BP->Borst)   (motion.py)
      -> LPLC2 population y^i(t)             (lplc2.py)
      -> GF u(t), DNP z(t), IAF spike        (giant_fiber.py)
      -> Omega_hat, psi_hat                  (estimation.py)
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .geometry import Circuit, load_circuit
from .stimulus import Trajectory, make_stimulus
from .motion import MotionConfig, compute_motion
from .lplc2 import compute_lplc2
from .giant_fiber import GFResponse, compute_gf
from .estimation import Estimates, estimate

ONSET_CUT = 0.08


@dataclass
class Result:
    traj: Trajectory
    L: np.ndarray
    y: np.ndarray
    gf: GFResponse
    est: Estimates
    circ: Circuit


def simulate(v: float, b: float, *, circ: Circuit | None = None,
             delta: float | None = None, fps: float = 500.0,
             motion_cfg: MotionConfig | None = None,
             onset_cut: float = ONSET_CUT) -> Result:
    """Run one looming trajectory (speed ``v``, pass-by ``b``) through the
    connectome-grounded circuit end to end."""
    circ = circ or load_circuit()
    L, traj = make_stimulus(v, b, fps=fps)
    mf = compute_motion(L, circ, traj.dt, motion_cfg)
    y = compute_lplc2(mf, circ)
    n_cut = int(round(onset_cut / traj.dt))
    if n_cut > 0:
        y[:n_cut] = 0.0
    gf = compute_gf(y, circ.C_diff, traj.dt, delta=delta)
    est = estimate(y, circ, traj.dt)
    return Result(traj=traj, L=L, y=y, gf=gf, est=est, circ=circ)
