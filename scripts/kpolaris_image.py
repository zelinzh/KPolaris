"""Read native Stokes images without guessing missing physical scales."""
from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np
from kpolaris_evpa import evpa_zero, qu_basis_sign


def rotate_stokes_basis(stokes, angle_deg):
    """Passive rotation: new e1=cos(a)*e1+sin(a)*e2, new e2=-sin(a)*e1+cos(a)*e2.

    This changes the polarization basis, not image coordinates or photon physics.
    Angles must be supplied from the basis geometry, never fitted to an image.
    """
    if not np.isfinite(angle_deg):
        raise ValueError("basis angle must be finite")
    out = np.array(stokes, dtype=float, copy=True)
    if out.shape[0] != 4:
        raise ValueError("expected Stokes axis of length four")
    c, s = np.cos(np.radians(2 * angle_deg)), np.sin(np.radians(2 * angle_deg))
    q, u = out[1].copy(), out[2].copy()
    out[1], out[2] = c * q + s * u, -s * q + c * u
    return out


def ipole_to_camera_stokes(stokes, evpa_0):
    """Undo ipole's documented N/W file-basis choice; keep its image coordinates.

    This alone does not align differing camera azimuths, rolls, or fields of view.
    At N, ipole save_pixel negates camera Q/U; at W it preserves them.
    """
    if isinstance(evpa_0, bytes):
        evpa_0 = evpa_0.decode()
    if evpa_0 not in ("N", "W"):
        raise ValueError("ipole camera conversion requires an explicit evpa_0=N or W")
    out = np.array(stokes, dtype=float, copy=True)
    if out.shape[0] != 4:
        raise ValueError("expected Stokes axis of length four")
    if evpa_0 == "N":
        out[1:3] *= -1  # Exact passive 90-degree rotation, without trig roundoff.
    return out


def evpa_tick_vectors(image, angle, units='M'):
    """Project electric-vector lines onto the actual camera image chart.

    Pinhole rays use n=(sx,sy,1)/|.| in the observer tetrad. Apply the
    gnomonic-chart Jacobian to the same projected/orthonormalized e1,e2
    used by initialize_pinhole_camera_ray. A common minus sign from looking
    back toward the source is immaterial for unoriented EVPA segments.
    """
    attrs = image['attrs']
    camera = attrs.get('camera', '')
    if isinstance(camera, bytes):
        camera = camera.decode()
    # Convert the stored EVPA to the internal screen before projecting its vector.
    if evpa_zero(attrs) == 'N':
        angle = np.asarray(angle) + np.pi / 2
    vx, vy = np.cos(angle), np.sin(angle)
    if camera in ('parallel', 'parallel_plane'):
        if units == 'pixel':
            extent, _ = image_extent(image, 'M')
            ny, nx = image['valid'].shape
            vx, vy = vx * nx / (extent[1]-extent[0]), vy * ny / (extent[3]-extent[2])
            length = np.hypot(vx,vy)
            return vx/length, vy/length
        return vx, vy
    if camera != 'pinhole':
        raise ValueError("EVPA segments require the camera projection in the input metadata")
    ny, nx = image['valid'].shape
    width = scalar(attrs.get('fov'))
    height = scalar(attrs.get('fovy', width))
    if not np.isfinite(width) or not np.isfinite(height) or min(width, height) <= 0:
        raise ValueError("pinhole EVPA segments require positive fov/fovy in native camera units")
    sx, sy = np.meshgrid(((np.arange(nx) + .5) / nx - .5) * width + scalar(attrs.get('x_offset', 0)),
                         ((np.arange(ny) + .5) / ny - .5) * height + scalar(attrs.get('y_offset', 0)))
    n = np.stack((sx, sy, np.ones_like(sx)), axis=0)
    n /= np.linalg.norm(n, axis=0)
    e1 = np.stack((np.ones_like(sx), np.zeros_like(sx), np.zeros_like(sx))) - n * n[0]
    e1 /= np.linalg.norm(e1, axis=0)
    e2 = np.cross(n, e1, axisa=0, axisb=0, axisc=0)
    electric = e1 * vx + e2 * vy
    vx, vy = electric[0] - sx * electric[2], electric[1] - sy * electric[2]
    if units == 'pixel':
        vx, vy = vx * nx / width, vy * ny / height
    length = np.hypot(vx, vy)
    return vx / length, vy / length


def scalar(value, default=float("nan")):
    try:
        return float(np.asarray(value).reshape(()))
    except (ValueError, TypeError):
        return default


def load_image(path: Path, frame: int = 0, frequency: int = 0) -> dict:
    if frame < 0 or frequency < 0:
        raise ValueError("frame and frequency indices must be nonnegative")
    with h5py.File(path, "r") as h:
        key = f"frame_{frame}/freq_{frequency}"
        if key not in h:
            available = [f"{f}/{n}" for f in h if f.startswith("frame_")
                         for n in h[f] if n.startswith("freq_")]
            raise ValueError(f"missing {key}; available images: {', '.join(available) or 'none'}")
        group = h[key]
        if any(s not in group for s in "IQUV"):
            raise ValueError(f"{key} must contain all four Stokes arrays I, Q, U, V")
        stokes = [np.asarray(group[s], dtype=float) for s in "IQUV"]
        if stokes[0].ndim != 2 or not stokes[0].size or any(a.shape != stokes[0].shape for a in stokes):
            raise ValueError("Stokes arrays must be nonempty 2D arrays of the same shape")
        # Selected-frequency values take precedence over frequency-zero aliases.
        attributes = dict(h.attrs)
        for loc in ("parameters/radiation", "parameters/camera", f"frame_{frame}", key):
            if loc in h:
                attributes.update(dict(h[loc].attrs))
        for name, expected in (('stokes_convention', 'camera_frame'),
                               ('polarization_basis', 'camera_screen')):
            value = attributes.get(name)
            if isinstance(value, bytes): value = value.decode()
            if value is not None and value != expected:
                raise ValueError(f"native plotting requires {name}={expected}; input declares {value!r}")
        attributes["evpa_0"] = evpa_zero(h.attrs, group.attrs)
        diag = h.get(f"frame_{frame}/diagnostics")
        reason = None if diag is None or "reason" not in diag else np.asarray(diag["reason"])
        if reason is not None and reason.shape != stokes[0].shape:
            raise ValueError("termination map shape does not match the image")
        finite = np.logical_and.reduce([np.isfinite(a) for a in stokes])
        valid = finite if reason is None else finite & (reason == 1)
        return {"path": str(Path(path).resolve()), "frame": frame, "frequency_index": frequency,
                "stokes": np.stack(stokes), "attrs": attributes, "reason": reason,
                "finite": finite, "valid": valid}


def image_summary(image: dict) -> dict:
    stokes, valid, attrs = image["stokes"], image["valid"], image["attrs"]
    scale = scalar(attrs.get("intensity_to_flux_jy_per_pixel"))
    scale_valid = scalar(attrs.get("flux_scale_valid", 1)) == 1 and np.isfinite(scale) and scale > 0
    sums = np.sum(stokes[:, valid], axis=1)
    values = {s: float(v) for s, v in zip("IQUV", sums)}
    positive = sums[0] > 0
    reasons = image["reason"]
    codes, counts = ([], []) if reasons is None else np.unique(reasons, return_counts=True)
    frequency = scalar(attrs.get("frequency_hz", attrs.get("frequency")))
    zero = evpa_zero(attrs)
    sign = qu_basis_sign(zero, "W")
    return {
        "schema": "kpolaris.image_summary.v1", "input": image["path"],
        "frame": image["frame"], "frequency_index": image["frequency_index"],
        "frequency_hz": float(frequency) if np.isfinite(frequency) else None,
        "shape": list(valid.shape), "pixels": int(valid.size),
        "valid_pixels": int(np.count_nonzero(valid)),
        "nonfinite_pixels": int(np.count_nonzero(~image["finite"])),
        "termination_available": reasons is not None,
        "termination_counts": {str(int(k)): int(v) for k, v in zip(codes, counts)},
        "all_rays_returned": bool(reasons is not None and np.all(reasons == 1)),
        "ok": bool(reasons is not None and np.all(valid)),
        "peak_I": float(np.max(stokes[0, valid])) if valid.any() else None,
        "sum_stokes_valid_pixels": values,
        "flux_valid_pixels_jy": {s: float(v * scale) for s, v in zip("IQUV", sums)} if scale_valid else None,
        "net_linear_fraction": float(np.hypot(sums[1], sums[2]) / sums[0]) if positive else None,
        "net_circular_fraction": float(sums[3] / sums[0]) if positive else None,
        "evpa_0": zero,
        "evpa_deg": float(np.degrees(.5 * np.arctan2(sums[2], sums[1]))) if positive and np.hypot(sums[1], sums[2]) > 0 else None,
        "camera_evpa_deg": float(np.degrees(.5 * np.arctan2(sign * sums[2], sign * sums[1]))) if positive and np.hypot(sums[1], sums[2]) > 0 else None,
    }


def image_extent(image: dict, units: str = "M", distance_pc=None, mass_solar=None):
    attrs = image["attrs"]
    ny, nx = image["valid"].shape
    if units == "pixel":
        return [-.5, nx - .5, -.5, ny - .5], "pixel"
    width = scalar(attrs.get("image_width_x_M", 2 * scalar(attrs.get("xspan"))))
    height = scalar(attrs.get("image_width_y_M", 2 * scalar(attrs.get("yspan", attrs.get("xspan")))))
    if not np.isfinite(width) or not np.isfinite(height) or min(width, height) <= 0:
        raise ValueError("physical field of view is unavailable; use --fov-units=pixel")
    # Image-plane offsets are specified in M for parallel cameras and in the
    # native angular screen coordinate for pinhole cameras. Convert the latter
    # using the same image-width/native-width ratio as the image writer.
    ox, oy = scalar(attrs.get("x_offset", 0)), scalar(attrs.get("y_offset", 0))
    camera = attrs.get("camera", "")
    if isinstance(camera, bytes):
        camera = camera.decode()
    if camera == "pinhole":
        radius = scalar(attrs.get("camera_radius", attrs.get("radius")))
        ox, oy = ox * radius, oy * radius
    if units == "muas":
        wx, wy = scalar(attrs.get("fovx_dsource")), scalar(attrs.get("fovy_dsource"))
        distance = distance_pc if distance_pc is not None else scalar(attrs.get("dsource"))
        mass = mass_solar if mass_solar is not None else scalar(attrs.get("mbh_solar"))
        if mass is None or not np.isfinite(mass) or mass <= 0:
            mass = None
            for model in ("riaf", "torus", "iharm", "kharma", "athenak", "bhac", "hamr"):
                value = scalar(attrs.get(f"{model}_mbh_solar"))
                if np.isfinite(value) and value > 0:
                    mass = value
                    break
        if distance_pc is not None or mass_solar is not None or not (wx > 0 and wy > 0):
            if mass is None or not np.isfinite(distance) or min(mass, distance) <= 0:
                raise ValueError("angular scale is unavailable; use --fov-units=M or provide --distance-pc and --mass-solar")
            factor = (6.67430e-8 * mass * 1.98847e33 / 2.99792458e10**2) / (distance * 3.085678e18) * 2.06265e11
            wx, wy = width * factor, height * factor
        ox, oy = ox * wx / width, oy * wy / height
        width, height = wx, wy
    return [ox - width / 2, ox + width / 2, oy - height / 2, oy + height / 2], (r"$\mu$as" if units == "muas" else r"$GM/c^2$")
