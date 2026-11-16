"""
lplc2.py -- LPLC2 population axonal drive.

Each of the four dendritic subunits computes the rectified opponent sum
(preferred minus opposite direction = the LPi inhibition) over the ommatidia in
its mask, and the four are summed into the LPLC2 axonal output ``y^i(t)``:

    y^i = [sum_up (mU - mD)]_+ + [sum_down (mD - mU)]_+
        + [sum_left (mL - mR)]_+ + [sum_right (mR - mL)]_+.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from .geometry import Circuit
from .motion import MotionField

_H_CACHE = {}


def _arm_matrices(circ: Circuit):
    key = id(circ)
    if key in _H_CACHE:
        return _H_CACHE[key]
    H = {}
    for arm in ("up", "down", "left", "right"):
        A = circ.arms[arm]
        rows, cols = [], []
        for i in range(A.shape[0]):
            cc = A[i][(A[i] >= 0) & (A[i] < circ.NC)]
            rows.extend([i] * len(cc)); cols.extend(cc.tolist())
        H[arm] = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(circ.N, circ.NC))
    _H_CACHE[key] = H
    return H


def compute_lplc2(mf: MotionField, circ: Circuit) -> np.ndarray:
    """LPLC2 population activity y[t, i], shape (T, N)."""
    H = _arm_matrices(circ)
    mU, mD, mL, mR = mf.x_c, mf.x_d, mf.x_a, mf.x_b   # up, down, left, right
    dU = np.maximum((mU - mD) @ H["up"].T, 0.0)
    dD = np.maximum((mD - mU) @ H["down"].T, 0.0)
    dL = np.maximum((mL - mR) @ H["left"].T, 0.0)
    dR = np.maximum((mR - mL) @ H["right"].T, 0.0)
    return dU + dD + dL + dR
