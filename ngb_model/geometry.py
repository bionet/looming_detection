"""
geometry.py -- Retina and LPLC2 population geometry.

The precomputed connectivity of the circuit is bundled as plain NumPy arrays in
``circuit_data.npz`` and loaded here:

  * ``pos`` (NC, 2): visual-field position (azimuth, elevation in deg) of each
    of the NC hexagonal ommatidia columns (NaN for the padding columns);
  * ``nb`` (NC, 7), ``mask`` (NC, 7): for each column, its center + six hex
    neighbors (slot order) and a 0/1 mask of which exist (the Borst inputs);
  * ``mvp_centers_cols``: the MVP ring-2 surround tiling (padded column lists);
  * ``arm_up/down/left/right`` (N, A): the ommatidia columns in each LPLC2's
    four dendritic subunits;
  * ``cy_deg`` (N,): LPLC2 center elevations ψ^i; ``C_diff`` (N,): the GF
    differential weights ``1 - (ψ^i/Δ)²``.

The connectivity is stored as arrays so the pipeline runs in pure NumPy.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix

_DATA = os.path.join(os.path.dirname(__file__), "circuit_data.npz")


@dataclass
class Circuit:
    pos: np.ndarray            # (NC, 2) ommatidia az/el, deg
    nb: np.ndarray             # (NC, 7) Borst neighbor columns (center + 6 slots)
    mask: np.ndarray           # (NC, 7) neighbor-exists mask
    E: csr_matrix              # (NC, NC) MVP surround operator (see motion.py)
    arms: dict                 # 'up'/'down'/'left'/'right' -> (N, A) column ids
    cy_deg: np.ndarray         # (N,) LPLC2 elevations psi^i
    cx_deg: np.ndarray         # (N,) LPLC2 azimuths
    C_diff: np.ndarray         # (N,) GF differential weights
    NC: int
    N: int

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.pos[:, 0])


_CIRCUIT = None


def load_circuit() -> Circuit:
    """Load the bundled circuit connectivity (cached)."""
    global _CIRCUIT
    if _CIRCUIT is not None:
        return _CIRCUIT
    d = np.load(_DATA)
    NC = int(d["NC"])
    E = csr_matrix((d["E_data"].astype(float),
                    (d["E_row"].astype(int), d["E_col"].astype(int))),
                   shape=(NC, NC))
    _CIRCUIT = Circuit(
        pos=d["pos"].astype(float),
        nb=d["nb"].astype(int),
        mask=d["mask"].astype(float),
        E=E,
        arms={"up": d["arm_up"].astype(int), "down": d["arm_down"].astype(int),
              "left": d["arm_left"].astype(int), "right": d["arm_right"].astype(int)},
        cy_deg=d["cy_deg"].astype(float),
        cx_deg=d["cx_deg"].astype(float),
        C_diff=d["C_diff"].astype(float),
        NC=NC, N=int(d["Nl"]),
    )
    return _CIRCUIT
