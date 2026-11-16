"""
ngb_model -- Connectome-grounded Drosophila looming-detection circuit.

A pure-NumPy implementation of the NeuroGraphBench (NGB) model: stimulus
generation (direct-hit / near-miss trajectories with ground-truth solid angle
and center elevation), the end-to-end circuit (T4/T5 PR+MVP+Borst motion ->
LPLC2 activation annulus -> Giant Fiber differential readout -> DNP ->
integrate-and-fire), and estimation of solid angle and center elevation.

The circuit's connectivity -- hex neighbor slots and mask, the MVP surround
operator ``E``, LPLC2 dendrite memberships and GF weights -- is bundled as arrays
in ``circuit_data.npz``.  See README.md for the symbol map and the figures that
reproduce the manuscript's results.
"""
from .geometry import Circuit, load_circuit
from .stimulus import (Trajectory, make_stimulus, make_trajectory,
                       render_luminance, R_DEFAULT, T_CA, F_PX, B_CRIT)
from .motion import MotionConfig, MotionField, compute_motion, LUM_SCALE
from .lplc2 import compute_lplc2
from .giant_fiber import (GFResponse, compute_gf, split_weights,
                          peak_potential, calibrate_delta, TAU_DNP)
from .estimation import (Estimates, estimate, fit_omega_calibration,
                         S_LPLC2, OMEGA_NEURON)
from .model import Result, simulate

__all__ = [
    "Circuit", "load_circuit",
    "Trajectory", "make_stimulus", "make_trajectory", "render_luminance",
    "R_DEFAULT", "T_CA", "F_PX", "B_CRIT",
    "MotionConfig", "MotionField", "compute_motion", "LUM_SCALE",
    "compute_lplc2",
    "GFResponse", "compute_gf", "split_weights", "peak_potential",
    "calibrate_delta", "TAU_DNP",
    "Estimates", "estimate", "fit_omega_calibration", "S_LPLC2", "OMEGA_NEURON",
    "Result", "simulate",
]

__version__ = "3.0.0"
