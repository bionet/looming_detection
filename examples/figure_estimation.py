"""
figure_estimation.py -- Solid angle and center elevation estimates (cf.
manuscript Figure 12b).

Estimates Omega(t) and psi(t) from the processed LPLC2 activation annulus across
a bank of trajectories (several approach speeds and pass-by distances),
restricted to the approach phase before the disk boundary reaches the edge of
the LPLC2 population, and plots estimated vs true against the identity line.

    python examples/figure_estimation.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ngb_model import load_circuit, simulate

SPEEDS = np.linspace(0.10, 0.55, 8)          # m/s
BS_MM = [1, 2, 3, 5, 8, 12, 18, 26, 36, 50]  # pass-by distances, mm
OUT = os.path.join(os.path.dirname(__file__), "figure_estimation.png")


def main():
    circ = load_circuit()
    # Approach phase before the activation annulus reaches the edge of the LPLC2
    # population: the annulus sits ~one receptive-field reach (RF) inside the
    # expanding boundary, so require boundary + RF to stay within the population.
    RF = 27.0
    pop_az = float(np.abs(circ.cx_deg).max())
    pop_el = float(circ.cy_deg.max())
    O_true, O_hat, P_true, P_hat = [], [], [], []
    for v in SPEEDS:
        for b in BS_MM:
            r = simulate(v, b * 1e-3, circ=circ)
            th, psi = r.traj.theta, r.traj.psi
            rising = np.arange(len(th)) <= np.nanargmax(np.where(np.isfinite(th), th, -1))
            keep = (th >= 12.0) & rising & (th / 2 + RF <= pop_az) & \
                   (np.abs(psi) + th / 2 + RF <= pop_el + 10.0)
            O_true.append(r.traj.Omega[keep]); O_hat.append(r.est.Omega_hat[keep])
            P_true.append(r.traj.psi[keep]);  P_hat.append(r.est.psi_hat[keep])
    O_true = np.concatenate(O_true); O_hat = np.concatenate(O_hat)
    P_true = np.concatenate(P_true); P_hat = np.concatenate(P_hat)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    ax.scatter(O_true, O_hat, s=5, alpha=0.2, color="tab:red", edgecolors="none")
    lim = np.nanpercentile(O_true, 99)
    ax.plot([0, lim], [0, lim], "k:", lw=1)
    # binned mean +/- std.
    bins = np.linspace(0, lim, 12)
    idx = np.digitize(O_true, bins)
    bx = [O_true[idx == k].mean() for k in range(1, len(bins)) if np.any(idx == k)]
    by = [O_hat[idx == k].mean() for k in range(1, len(bins)) if np.any(idx == k)]
    be = [O_hat[idx == k].std() for k in range(1, len(bins)) if np.any(idx == k)]
    ax.errorbar(bx, by, yerr=be, fmt="o-", color="black", ms=4, lw=1.3, capsize=2)
    ax.set_title("Solid angle estimate")
    ax.set_xlabel("true Omega (sr)"); ax.set_ylabel("estimated Omega (sr)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)

    ax = axes[1]
    ax.scatter(P_true, P_hat, s=5, alpha=0.2, color="tab:blue", edgecolors="none")
    pm = 45
    ax.plot([0, pm], [0, pm], "k:", lw=1)
    bins = np.linspace(0, pm, 12)
    idx = np.digitize(P_true, bins)
    bx = [P_true[idx == k].mean() for k in range(1, len(bins)) if np.any(idx == k)]
    by = [P_hat[idx == k].mean() for k in range(1, len(bins)) if np.any(idx == k)]
    be = [P_hat[idx == k].std() for k in range(1, len(bins)) if np.any(idx == k)]
    ax.errorbar(bx, by, yerr=be, fmt="o-", color="black", ms=4, lw=1.3, capsize=2)
    ax.axvline(35, color="0.6", ls="--", lw=1)
    ax.set_title("Center elevation estimate")
    ax.set_xlabel("true psi (deg)"); ax.set_ylabel("estimated psi (deg)")
    ax.set_xlim(0, pm); ax.set_ylim(0, pm)

    fig.suptitle("Estimation from the LPLC2 activation annulus", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}  ({O_true.size} samples)")


if __name__ == "__main__":
    main()
