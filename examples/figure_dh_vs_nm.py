"""
figure_dh_vs_nm.py -- Direct-hit vs near-miss (cf. manuscript Figures 5, 6).

Panels:
  (a) the LPLC2 activation annulus for a direct-hit, at theta ~ 45 deg, with
      neuron centers colored by activity and the GF positive band outlined;
  (b) Fig 6a -- the DNP output z(t) for a direct-hit (b = 0) and near-misses
      (b = 2, 5, 10, 20, 40 mm) at v = 0.30 m/s, drawn along each trajectory in
      the (solid angle Omega, center elevation psi) plane. Diamonds mark the
      peak of z(t); the peak decreases with pass-by distance b, and the direct
      hit stays at psi = 0 while near-misses fan out in elevation.

    python examples/figure_dh_vs_nm.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ngb_model import load_circuit, simulate

V = 0.30
BS_MM = [0, 2, 5, 10, 20, 40]
# trace colors matching the manuscript Fig 6a (DH, b=2, 5, 10, 20, 40 mm).
COLORS = ["#1565C0", "#E53935", "#FF9800", "#4CAF50", "#9C27B0", "#795548"]
OUT = os.path.join(os.path.dirname(__file__), "figure_dh_vs_nm.png")


def main():
    circ = load_circuit()
    runs = {b: simulate(V, b * 1e-3, circ=circ) for b in BS_MM}
    colors = COLORS

    fig = plt.figure(figsize=(13, 5.5))

    # (a) annulus snapshot for the direct-hit near theta = 45 deg.
    ax = fig.add_subplot(1, 2, 1)
    dh = runs[0]
    ti = int(np.nanargmin(np.abs(dh.traj.theta - 45.0)))
    sc = ax.scatter(circ.cx_deg, circ.cy_deg, c=dh.y[ti],
                    cmap="inferno", s=45, edgecolors="none")
    ax.axhspan(-21.0, 21.0, color="tab:green", alpha=0.08)
    ax.axhline(21.0, color="tab:green", lw=0.8, ls="--")
    ax.axhline(-21.0, color="tab:green", lw=0.8, ls="--")
    disk = plt.Circle((0, dh.traj.psi[ti]), dh.traj.theta[ti] / 2, fill=False,
                      color="white", lw=1.5, ls=":")
    ax.add_patch(disk)
    ax.set_title(f"(a) LPLC2 annulus, direct-hit at theta={dh.traj.theta[ti]:.0f} deg")
    ax.set_xlabel("azimuth (deg)"); ax.set_ylabel("elevation psi (deg)")
    ax.set_aspect("equal"); ax.set_facecolor("0.15")
    fig.colorbar(sc, ax=ax, label="LPLC2 activity y", shrink=0.8)

    # (b) Fig 6a -- z(t) along the (Omega, psi) trajectory for each pass-by,
    # over the approach phase.  The DNP output z(t) is rectified to be
    # non-negative (as in the manuscript), so each trace rises to its peak
    # (diamond) and returns to the floor.
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    OMEGA_CAP = 1.2                               # cap the direct-hit's late expansion
    zmax = 0.0
    for b, c in zip(BS_MM, colors):
        r = runs[b]
        om, ps = r.traj.Omega, r.traj.psi
        z = np.maximum(r.gf.z, 0.0)               # rectified activation
        seg = np.isfinite(om) & np.isfinite(ps) & (r.traj.z > 0) & (om <= OMEGA_CAP)
        idx = np.where(seg)[0]
        pk = idx[np.argmax(z[idx])]
        lbl = "DH (b=0)" if b == 0 else f"b={b} mm"
        ax.plot(om[idx], ps[idx], z[idx], color=c, lw=1.8, label=lbl)
        ax.scatter([om[pk]], [ps[pk]], [z[pk]], marker="D", color=c,
                   s=35, edgecolors="k", linewidths=0.4, depthshade=False)
        zmax = max(zmax, z[pk])
    ax.set_xlabel("solid angle $\\Omega$ (sr)")
    ax.set_ylabel("center elevation $\\psi$ (deg)")
    ax.set_zlabel("z(t)")
    ax.set_zlim(0, zmax * 1.1)
    ax.set_title("(b) DNP output z(t) along the trajectory (Fig 6a)")
    ax.view_init(elev=20, azim=-62)
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"Direct-hit vs near-miss (v = {V:.2f} m/s)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
