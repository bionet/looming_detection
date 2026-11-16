"""
_render.py -- Stimulus rendering routines.

A 512x512 equidistant-projection looming-disk renderer plus the fixed-point
hexagonal retinal sampler (box-average window=16, spacing=12, radius=20, center
(256,256), luminance scaled x1e5) used to produce the per-ommatidium luminance
for the model.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np

F_PX_DEFAULT = 217.2
_RENDER_SIZE = 512; _HEX_RADIUS = 20; _HEX_WINDOW = 16; _HEX_SPACING = 12
_CENTER_RC = (256, 256); _LUM_SCALE = 1e5


@dataclass
class DirectHitParams:
    # Canvas
    width: int = 512
    height: int = 512
    background: int = 0
    foreground: int = 255

    # Timing
    duration_s: float = 1.0
    fps: int = 240

    # Center + translation (px, px/s)
    x0: float = 256.0
    y0: float = 256.0
    vx: float = 0.0
    vy: float = 0.0

    # Camera scaling
    f_px: float = F_PX_DEFAULT

    # Object size (world units)
    L: float = 0.01

    # Depth model
    depth_mode: _DH_DepthMode = "passby"

    # collision mode: z(t) = z0 - v_z*t - 0.5*accel*t^2, clamped at z_min
    # accel > 0 means the object accelerates toward the observer (z decreases faster)
    # accel = 0 is constant speed (default)
    # accel < 0 means deceleration
    v_z: float = 0.6
    z0: float = 0.6
    z_min: float = 1e-3
    accel: float = 0.0

    # passby mode: z(t) = sqrt(z_min_pass^2 + (v_pass*(t - t0))^2)
    z_min_pass: float = 0.08
    t0: float = 0.6
    v_pass: float = 0.6

    # Optional pixel-space radius cap
    r_max_px: Optional[float] = None

    antialias: bool = False
    edge_sigma: float = 1.0

    codec: str = "libx264"
    bitrate: str = "8M"
    pixelformat: str = "yuv420p"


@dataclass
class NearMissParams:
    width: int = 512
    height: int = 512
    duration_s: float = 1.0
    fps: int = 500
    background: int = 0
    foreground: int = 255
    f_px: float = F_PX_DEFAULT
    L: float = 0.01
    z_min: float = 1e-3
    r_max_px: Optional[float] = None

    # Straight-line 3D flyby parameters
    # Object approaches along z-axis and passes upward (positive y).
    # P(t) = (0, pass_b + pass_v_perp*(t - pass_t_ca), pass_speed*(pass_t_ca - t))
    # At t=pass_t_ca: closest approach at (0, pass_b, 0), range = pass_b.
    # Before: z > 0 (in front), y < pass_b (below final position).
    # After: z < 0 (behind) → exits frontal hemisphere → invisible.
    pass_speed: float = 0.8       # approach speed along z (world units/s)
    pass_v_perp: float = 0.05     # lateral speed upward (world units/s)
    pass_t_ca: float = 0.75       # time of closest approach (s)
    pass_b: float = 0.05          # miss distance (perpendicular to z-axis)


def equidistant_project_radius(L: float, rng: float, f_px: float = F_PX_DEFAULT) -> float:
    """Project object angular half-size to pixel radius using equidistant projection.

    angular_radius = arctan(L / range)
    pixel_radius = f_px × angular_radius

    At small angles: arctan(L/r) ≈ L/r, so pixel_radius ≈ f_px × L/r (= pinhole).
    At large angles: saturates at f_px × π/2 instead of diverging.
    """
    angular_radius = np.arctan2(L, max(rng, 1e-6))
    return float(f_px * angular_radius)


def equidistant_project_center(offset: float, rng: float, f_px: float = F_PX_DEFAULT) -> float:
    """Project lateral offset to pixel position using equidistant projection.

    angle = arctan(offset / range)  [for planar geometry]
    Or: angle = arctan2(offset, z)   [for 3D position]
    pixel_offset = f_px × angle
    """
    angle = np.arctan2(offset, max(rng, 1e-6))
    return float(f_px * angle)


def _dh_make_grid(width: int, height: int):
    xs = np.arange(width, dtype=np.float32)[None, :]
    ys = np.arange(height, dtype=np.float32)[:, None]
    return xs, ys


def _dh_z_of_t(t: float, p: DirectHitParams) -> float:
    if p.depth_mode == "collision":
        z = p.z0 - p.v_z * t - 0.5 * p.accel * t * t
        return float(max(p.z_min, z))
    elif p.depth_mode == "passby":
        z = np.sqrt((p.z_min_pass ** 2) + (p.v_pass * (t - p.t0)) ** 2)
        return float(z)
    else:
        raise ValueError(f"Unknown depth_mode={p.depth_mode}")


def _dh_dz_dt(t: float, p: DirectHitParams) -> float:
    if p.depth_mode == "collision":
        z_unclamped = p.z0 - p.v_z * t - 0.5 * p.accel * t * t
        if z_unclamped <= p.z_min:
            return 0.0
        return float(-p.v_z - p.accel * t)
    elif p.depth_mode == "passby":
        z = _dh_z_of_t(t, p)
        if z <= 1e-12:
            return 0.0
        return float((p.v_pass ** 2) * (t - p.t0) / z)
    else:
        raise ValueError(f"Unknown depth_mode={p.depth_mode}")


def _dh_radius_px_at_time(t: float, p: DirectHitParams) -> float:
    z = _dh_z_of_t(t, p)
    r_px = equidistant_project_radius(p.L, z, p.f_px)
    if p.r_max_px is not None:
        r_px = min(r_px, float(p.r_max_px))
    return float(max(0.0, r_px))


def _dh_ttc_true(t: float, p: DirectHitParams) -> float:
    """True time-to-collision. For constant speed: z/v_z.
    For accelerating: solve z0 - v_z*(t+τ) - 0.5*a*(t+τ)^2 = 0 for τ."""
    if p.depth_mode != "collision":
        return float(np.inf)
    z = _dh_z_of_t(t, p)
    if z <= p.z_min:
        return 0.0
    v_now = p.v_z + p.accel * t  # instantaneous speed
    if p.accel == 0.0:
        return float(z / v_now) if v_now > 0 else float(np.inf)
    # Quadratic: 0.5*a*τ^2 + v_now*τ - z = 0
    disc = v_now * v_now + 2.0 * p.accel * z
    if disc < 0:
        return float(np.inf)  # object decelerating and will stop before collision
    tau = (-v_now + np.sqrt(disc)) / p.accel
    return float(max(0.0, tau))


def _dh_render_frame(t: float, p: DirectHitParams, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    cx = p.x0 + p.vx * t
    cy = p.y0 + p.vy * t
    r = _dh_radius_px_at_time(t, p)

    d2 = (xs - cx) ** 2 + (ys - cy) ** 2

    if not p.antialias:
        mask = d2 <= (r * r)
        frame = np.full((p.height, p.width), p.background, dtype=np.uint8)
        frame[mask] = p.foreground
        return frame

    d = np.sqrt(d2 + 1e-12)
    alpha = 1.0 / (1.0 + np.exp((d - r) / max(1e-6, p.edge_sigma)))
    img = p.background + (p.foreground - p.background) * alpha
    return np.clip(img, 0, 255).astype(np.uint8)


def _dh_scenario_direct_hit_collision(base: DirectHitParams, ttc_s: float = 0.7) -> DirectHitParams:
    p = DirectHitParams(**vars(base))
    p.depth_mode = "collision"
    p.z0 = p.v_z * ttc_s
    p.x0 = p.width / 2
    p.y0 = p.height / 2
    p.vx = p.vy = 0.0
    return p


def _nm_center_xyz_vertical_passby(t: float, p: NearMissParams) -> np.ndarray:
    """Straight-line 3D flyby trajectory.

    Object moves on a straight line with constant velocity.
    The z-component (along optical axis) decreases linearly to zero at t_ca.
    The y-component (elevation) increases linearly — object sweeps upward.
    After z < 0, the object is behind the fly and exits the frontal FOV.
    """
    x = 0.0
    y = p.pass_b + p.pass_v_perp * (t - p.pass_t_ca)
    z = p.pass_speed * (p.pass_t_ca - t)
    return np.array([x, y, z], dtype=float)


def _nm_elevation_deg(C) -> float:
    x, y, z = C
    return float(np.degrees(np.arctan2(y, np.sqrt(x * x + z * z))))


def _nm_project(C, p: NearMissParams) -> Tuple[float, float, float]:
    """Project 3D center onto the image plane using equidistant projection.

    Each axis is projected independently using arctan to get the angular
    position, then scaled by f_px. When z ≤ 0, the object has passed
    behind the fly and is outside the frontal hemisphere — return zero
    radius to make it invisible.
    """
    x, y, z = C
    rng = max(np.sqrt(x * x + y * y + z * z), p.z_min)

    # Object behind the fly — outside frontal FOV
    if z <= 0:
        return p.width / 2, p.height / 2, rng

    cx = p.width / 2 + equidistant_project_center(x, z, p.f_px)
    cy = p.height / 2 - equidistant_project_center(y, z, p.f_px)

    return cx, cy, rng


def _nm_radius_px(C, p: NearMissParams) -> float:
    """Compute projected radius using equidistant projection.

    Uses range (not z) for the angular subtense, then maps through
    equidistant projection. Returns 0 when z ≤ 0 (behind the fly).
    """
    if C[2] <= 0:
        return 0.0  # object behind fly — invisible
    rng = max(np.sqrt(C[0]**2 + C[1]**2 + C[2]**2), p.z_min)
    r = equidistant_project_radius(p.L, rng, p.f_px)
    if p.r_max_px is not None:
        r = min(r, p.r_max_px)
    return float(r)


def _nm_render_frame(cx: float, cy: float, r: float, p: NearMissParams,
                     xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    frame = np.full((p.height, p.width), p.background, dtype=np.uint8)
    frame[d2 <= r * r] = p.foreground
    return frame


def _hex_ring_indices(radius: int):
    """Generate axial (q,r) coords ring-by-ring starting at 'north' and going clockwise.
    Flat-topped axial coordinates."""
    coords = [(0, 0)]
    ring_dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    for k in range(1, radius + 1):
        q, r = 0, -k
        for d_q, d_r in ring_dirs:
            for _ in range(k):
                coords.append((q, r))
                q += d_q
                r += d_r
    return coords


def _axial_to_pixel_flat(q: int, r: int, spacing: float):
    """Axial hex to pixel (x,y) using flat-topped orientation.
    spacing = vertical center-to-center distance of neighbors."""
    size = spacing / np.sqrt(3)  # scale factor so vertical step (0,-1) = spacing
    x = 1.5 * size * q
    y = np.sqrt(3) * size * (r + 0.5 * q)
    return x, y


class HexFixedPointVideo:
    def __init__(
        self,
        video_T: np.ndarray,
        center_rc: Tuple[float, float],
        *,
        radius: int = 13,
        window: int = 8,
        spacing: float = 8.0,
        nan_on_out_of_bounds: bool = True,
    ):
        assert video_T.ndim in (3, 4), "video_T must be (T,H,W) or (T,H,W,C)"
        self.video_T = video_T
        self.vmin = video_T.min()
        self.vmax = video_T.max()

        self.center_rc = (float(center_rc[0]), float(center_rc[1]))
        self.radius = int(radius)
        self.window = int(window)
        self.spacing = float(spacing)
        self.nan_on_oob = bool(nan_on_out_of_bounds)

        self.T = video_T.shape[0]
        self.H, self.W = video_T.shape[1:3]
        self.is_color = (video_T.ndim == 4)
        self.C = video_T.shape[3] if self.is_color else None

        # Build hex axial coords and fixed pixel centers once
        self.axial = _hex_ring_indices(self.radius)
        self.N = 1 + 3 * self.radius * (self.radius + 1)
        if len(self.axial) != self.N:
            raise ValueError(f"Expected {self.N} axial coords, got {len(self.axial)}")

        cy, cx = self.center_rc
        centers_rc = np.empty((self.N, 2), dtype=int)
        for i, (q, r) in enumerate(self.axial):
            dx, dy = _axial_to_pixel_flat(q, r, self.spacing)
            col = int(np.round(cx + dx))
            row = int(np.round(cy + dy))
            centers_rc[i] = (row, col)

        # (T, N, 2) replicated per-frame
        self.centers_T = np.repeat(centers_rc[None, ...], self.T, axis=0)

        # --- NEW: precompute a flat-topped hexagon OUTLINE (no fill) in pixel coords ---
        R = self.radius
        # the 6 "corner" axial coordinates at distance R, ordered around the hex
        corners_axial = [(R, 0), (0, R), (-R, R), (-R, 0), (0, -R), (R, -R)]
        hex_xy = []
        for (q, r) in corners_axial:
            dx, dy = _axial_to_pixel_flat(q, r, self.spacing)
            # store in (x=col, y=row) for matplotlib
            hex_xy.append((cx + dx, cy + dy))
        self.hex_outline_xy = np.asarray(hex_xy, dtype=float)

        # Filled by sample()
        self.values_T: Optional[np.ndarray] = None

    def sample(self) -> np.ndarray:
        """Compute per-frame mean over each hex patch. Returns values_T.

        Vectorized over hexes with a per-frame summed-area table: each patch mean
        is the box sum over the (clipped) window divided by its area. The hex
        centers are the same for every frame, so the box corners are computed
        once. (The grayscale path; color falls back to the per-patch loop.)
        """
        half = self.window // 2
        fill = np.nan if self.nan_on_oob else 0.0

        if self.is_color:
            values_T = np.empty((self.T, self.N, self.C), dtype=float)
            for t in range(self.T):
                frame = self.video_T[t]
                for i in range(self.N):
                    row, col = self.centers_T[t, i]
                    rr0, rr1 = max(0, row - half), min(self.H, row + half)
                    cc0, cc1 = max(0, col - half), min(self.W, col + half)
                    patch = frame[rr0:rr1, cc0:cc1]
                    values_T[t, i] = (patch.mean(axis=(0, 1))
                                      if patch.size else fill)
            self.values_T = values_T
            return values_T

        # Clipped box corners per hex (centers are frame-independent).
        rows = self.centers_T[0, :, 0]
        cols = self.centers_T[0, :, 1]
        r0 = np.clip(rows - half, 0, self.H); r1 = np.clip(rows + half, 0, self.H)
        c0 = np.clip(cols - half, 0, self.W); c1 = np.clip(cols + half, 0, self.W)
        area = (r1 - r0) * (c1 - c0)
        ok = area > 0
        area_safe = np.where(ok, area, 1)

        values_T = np.empty((self.T, self.N), dtype=float)
        integ = np.zeros((self.H + 1, self.W + 1), dtype=np.float64)
        for t in range(self.T):
            # Summed-area table: integ[r, c] = sum of frame[:r, :c].
            frame = self.video_T[t].astype(np.float64, copy=False)
            integ[1:, 1:] = frame.cumsum(axis=0).cumsum(axis=1)
            box = (integ[r1, c1] - integ[r0, c1] - integ[r1, c0] + integ[r0, c0])
            values_T[t] = np.where(ok, box / area_safe, fill)

        self.values_T = values_T
        return values_T


def _render_direct_hit_frames(p: DirectHitParams) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Render direct-hit stimulus in memory."""
    n_frames = int(np.round(p.duration_s * p.fps))
    xs, ys = _dh_make_grid(p.width, p.height)
    frames = np.empty((n_frames, p.height, p.width), dtype=np.float32)
    pf = {k: np.empty(n_frames, dtype=np.float32) for k in
          ("t_s", "cx_px", "cy_px", "r_px", "z", "dz_dt", "TTC_true_s")}

    for k in range(n_frames):
        t = k / p.fps
        frame = _dh_render_frame(t, p, xs, ys)
        frames[k] = frame.astype(np.float32) / 255.0
        pf["t_s"][k] = t
        pf["cx_px"][k] = p.x0 + p.vx * t
        pf["cy_px"][k] = p.y0 + p.vy * t
        pf["r_px"][k] = _dh_radius_px_at_time(t, p)
        pf["z"][k] = _dh_z_of_t(t, p)
        pf["dz_dt"][k] = _dh_dz_dt(t, p)
        pf["TTC_true_s"][k] = _dh_ttc_true(t, p)
    return frames, pf


def _render_near_miss_frames(p: NearMissParams) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Render near-miss stimulus in memory. Returns (frames, per_frame_dict)."""
    n_frames = int(p.duration_s * p.fps)
    xs = np.arange(p.width)[None, :]
    ys = np.arange(p.height)[:, None]
    frames = np.empty((n_frames, p.height, p.width), dtype=np.float32)
    pf = {k: np.empty(n_frames, dtype=np.float32) for k in
          ("t_s", "cx_px", "cy_px", "radius_px", "x", "y", "z",
           "range", "elevation_rad", "azimuth_rad", "time_to_closest_approach_s")}

    for k in range(n_frames):
        t = k / p.fps
        C = _nm_center_xyz_vertical_passby(t, p)
        cx, cy, z_eff = _nm_project(C, p)
        r = _nm_radius_px(C, p)
        frame = _nm_render_frame(cx, cy, r, p, xs, ys)
        frames[k] = frame.astype(np.float32) / 255.0

        pf["t_s"][k] = t
        pf["cx_px"][k] = cx
        pf["cy_px"][k] = cy
        pf["radius_px"][k] = r
        pf["x"][k] = C[0]
        pf["y"][k] = C[1]
        pf["z"][k] = C[2]
        pf["range"][k] = float(np.linalg.norm(C))
        pf["elevation_rad"][k] = float(np.arctan2(C[1], np.sqrt(C[0]**2 + C[2]**2)))
        pf["azimuth_rad"][k] = 0.0
        pf["time_to_closest_approach_s"][k] = max(0.0, p.pass_t_ca - t)
    return frames, pf
