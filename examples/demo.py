"""
demo.py -- Minimal end-to-end run of the looming circuit.

    python examples/demo.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ngb_model import load_circuit, simulate, peak_potential, calibrate_delta


def main():
    circ = load_circuit()
    print(f"Circuit: {int(circ.valid.sum())} ommatidia; LPLC2 population: N = {circ.N} neurons")
    print(f"GF weight band: {(circ.C_diff > 0).sum()} positive / "
          f"{(circ.C_diff < 0).sum()} negative neurons\n")

    v = 0.20
    dh = simulate(v, 0.0, circ=circ)
    nm = simulate(v, 5e-3, circ=circ)
    delta = calibrate_delta([dh.gf.z], [nm.gf.z], dh.traj.dt)

    for name, r in [("direct-hit", dh), ("near-miss (b=5mm)", nm)]:
        vis = np.isfinite(r.traj.theta)
        ip = int(np.argmax(np.where(vis, r.gf.u, -np.inf)))
        Vmax = peak_potential(r.gf.z, r.traj.dt)
        print(f"{name:20s}: u peaks at theta = {r.traj.theta[ip]:.0f} deg; "
              f"peak IAF potential = {Vmax:.3f}; fires = {Vmax >= delta}")

    print(f"\nIAF threshold delta = {delta:.3f}")
    print("=> the Giant Fiber fires for the direct-hit and withholds for the near-miss.")


if __name__ == "__main__":
    main()
