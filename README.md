# NGB Looming Model

A pure-NumPy implementation of the NeuroGraphBench (NGB) connectome-grounded
*Drosophila* looming-detection circuit. Each stage implements the corresponding
block of the model, with the circuit's connectivity stored as plain NumPy arrays
(`ngb_model/circuit_data.npz`). The only runtime dependencies are NumPy, SciPy
and Matplotlib.

It reproduces the manuscript's results: the direct-hit `u`-peak near 40°, the
direct-hit vs. near-miss discrimination (Figs 5–6), and the solid-angle /
center-elevation estimates (Fig 12).

It covers the three pieces of the modeling side:

1. **Stimulus generation** — direct-hit (DH) and near-miss (NM) trajectories with
   ground-truth solid angle Ω(t) and center elevation ψ(t).
2. **The end-to-end circuit** — T4/T5 motion (PR + MVP + Borst) → the LPLC2
   activation annulus → the Giant Fiber differential readout → the
   divisive-normalization peak detector → an integrate-and-fire escape spike.
3. **Estimation** — recovery of Ω̂ and ψ̂ from the LPLC2 annulus.

## Install & quickstart

```bash
pip install -r requirements.txt       # numpy, scipy, matplotlib
```
```python
from ngb_model import load_circuit, simulate, calibrate_delta, peak_potential

circ = load_circuit()
dh = simulate(0.20, 0.0,  circ=circ)        # direct hit:  v = 0.20 m/s, b = 0
nm = simulate(0.20, 5e-3, circ=circ)        # near miss:   v = 0.20 m/s, b = 5 mm

delta = calibrate_delta([dh.gf.z], [nm.gf.z], dh.traj.dt)
peak_potential(dh.gf.z, dh.traj.dt) >= delta   # True  -> DH fires
peak_potential(nm.gf.z, nm.traj.dt) >= delta   # False -> NM withheld

dh.traj.theta, dh.traj.Omega, dh.traj.psi      # ground truth
dh.est.Omega_hat, dh.est.psi_hat               # estimated from the annulus
```
```bash
python examples/demo.py              # minimal end-to-end run
python examples/figure_dh_vs_nm.py   # DH vs NM discrimination   (cf. Figs 5, 6)
python examples/figure_estimation.py # Omega / psi estimation     (cf. Fig 12)
```

## The circuit

`circuit_data.npz` holds the circuit's connectivity as arrays: the hex retina
(890 ommatidia), each column's Borst neighbor slots + presence mask, the MVP
surround operator `E`, the 103 LPLC2 neurons' four dendritic-subunit column
memberships, their elevations, and the GF differential weights (42 positive / 61
negative, Δ = 21°). The pipeline runs on these in pure NumPy:

| Stage | Module | What it computes (manuscript symbols) |
|-------|--------|---------------------------------------|
| **Stimulus** | `stimulus.py` | `d(t)`, `psi(t)`; `theta`, `Omega`; the 512² equidistant disk, hex-sampled to retinal luminance |
| **PR + MVP** | `motion.py` | `am = T1(L)/(T2(L) + surround)`; surround = `g·L·(E @ (s0 + s1·L))` (MVP divisive normalization via the operator `E`) |
| **Band-pass / ON-OFF** | `motion.py` | biphasic temporal filter `gamma(2,2)-gamma(6,4)` (20-sample sliding convolution); ON = T4, OFF = T5 |
| **Borst** | `motion.py` | ratio detector `y·(1+pref·α)/(1+null·α)`, α=100, over the six hex slots; `x^j_a..d = T4 + T5` |
| **LPLC2** | `lplc2.py` | `y^i = Σ_arms [sum(pref − opp over mask)]_+` (the activation annulus) |
| **Giant Fiber** | `giant_fiber.py` | `u = (C1−C2)ᵀy`; DNP `z = u/(1+[τ u']²)`, τ=20 ms; IAF, threshold δ |
| **Estimation** | `estimation.py` | two-step disk recovery → `Omega_hat`; processed-activity centroid → `psi_hat` |

## Module reference
```
ngb_model/
  circuit_data.npz  bundled connectivity (retina, Borst slots+mask, MVP
                    operator E, LPLC2 dendrites, GF weights)
  geometry.py       loads circuit_data.npz -> Circuit
  _render.py        stimulus render (512² disk + hex sampler)
  stimulus.py       trajectory kinematics, ground truth, retinal luminance
  motion.py         PR+MVP (operator E) -> band-pass -> Borst  (T4/T5)
  lplc2.py          LPLC2 axonal drive y^i(t)
  giant_fiber.py    GF readout u(t), DNP z(t), integrate-and-fire
  estimation.py     Omega_hat / psi_hat from the annulus (disk recovery)
  model.py          simulate(): end-to-end driver
examples/           demo.py, figure_dh_vs_nm.py, figure_estimation.py
```
