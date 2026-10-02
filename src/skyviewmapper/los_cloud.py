"""Probability of a cloud-free line of sight to the Sun from ERA5 cloud samples.

For each observer, every ERA5 cloud layer is read where the sight line toward
the Sun passes through it, in every (year, day) sample; layers are combined
per sample and the result averaged over samples.

Assumptions and approximations
------------------------------
- Each ERA5 layer is a slab above the observer's ground (heights in
  ``constants.CLOUD_LAYER_HEIGHTS_M``). The sight line is followed through
  the slab by sampling it at three heights; the layer's cloud fraction ``n``
  is the mean of the bilinearly interpolated ERA5 values there. Slab heights
  are taken above the observer's ground, not the ground under each crossing.
- Crossing points use :func:`geometry.los_layer_crossing` with the Sun's
  apparent (refracted) altitude and the effective Earth radius
  ``R / (1 - k)``, k = 0.13: the same refraction treatment as the terrain.
- ERA5 fields at the two full hours either side are interpolated linearly
  in time to each observer's maximum eclipse (clamped to that hour range).
- Slant-path correction: clouds in a layer are modelled as randomly placed
  cylinders with height/width ratio ``beta`` (a Boolean model). A sight line
  at elevation ``theta`` also meets cloud sides, so its chance of passing a
  layer of cover ``n`` is ``(1 - n) ** (1 + (4 beta / pi) cot(theta))``,
  which is ``1 - n`` overhead. Observations of cloud fraction against view
  angle support the direction and size of the effect up to ~70 deg from the
  zenith (Zhao & Di Girolamo 2004); at the eclipse's 2-12 deg elevations it
  is an extrapolation, and ``beta`` is uncertain. Results are therefore
  given with and without it. Elevations are floored at 0.5 deg.
- Layers overlap at random within each sample: the chance of a clear sight
  line is the product over layers, evaluated per sample and then averaged
  (layers are correlated in time).
"""

from dataclasses import dataclass

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, NDArray

from .constants import CLOUD_ASPECT_RATIO, CLOUD_LAYER_HEIGHTS_M
from .geometry import los_layer_crossing
from .terrain import EYE_HEIGHT_M, REFRACTION_K, effective_radius

MIN_ELEVATION_DEG = 0.5


@dataclass(frozen=True)
class LosCloud:
    """Per-observer cloud results. ``layer_cloud`` / ``crossing_km`` keyed by ERA5 variable."""

    p_clear_sky: NDArray[np.float64]
    p_clear_sky_no_slant: NDArray[np.float64]
    layer_cloud: dict[str, NDArray[np.float64]]
    crossing_km: dict[str, NDArray[np.float64]]


# --- pure helpers ------------------------------------------------------------


def slant_clear_fraction(cover: ArrayLike, elevation_deg: ArrayLike, aspect_ratio: float) -> NDArray[np.float64]:
    """Chance a sight line at ``elevation_deg`` passes a layer of cloud ``cover`` (Boolean cylinder model)."""
    n = np.clip(np.asarray(cover, dtype=float), 0.0, 1.0)
    theta = np.radians(np.maximum(np.asarray(elevation_deg, dtype=float), MIN_ELEVATION_DEG))
    exponent = 1.0 + (4.0 * aspect_ratio / np.pi) / np.tan(theta)
    return (1.0 - n) ** exponent


def bilinear_weights(
    lat: ArrayLike, lon: ArrayLike, grid_lat: NDArray[np.float64], grid_lon: NDArray[np.float64]
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Flat indices (..., 4) into a (lat, lon) field and their bilinear weights (..., 4).

    ``grid_lat``/``grid_lon`` must be ascending and regular. Raises if any
    point lies outside the grid.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    nlat, nlon = grid_lat.size, grid_lon.size
    y = (lat - grid_lat[0]) / (grid_lat[1] - grid_lat[0])
    x = (lon - grid_lon[0]) / (grid_lon[1] - grid_lon[0])
    if np.any(~np.isfinite(y)) or np.any(~np.isfinite(x)):
        raise ValueError("non-finite crossing point")
    if y.min() < 0 or y.max() > nlat - 1 or x.min() < 0 or x.max() > nlon - 1:
        raise ValueError("crossing point outside the ERA5 box; enlarge Region.era5_box")
    i = np.minimum(np.floor(y).astype(np.intp), nlat - 2)
    j = np.minimum(np.floor(x).astype(np.intp), nlon - 2)
    fy, fx = y - i, x - j
    idx = np.stack([i * nlon + j, i * nlon + j + 1, (i + 1) * nlon + j, (i + 1) * nlon + j + 1], axis=-1)
    w = np.stack([(1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx], axis=-1)
    return idx, w


def time_weight(t_utc: NDArray[np.datetime64], hour0: np.datetime64, hour1: np.datetime64) -> NDArray[np.float64]:
    """Linear weight of the later field: 0 at ``hour0``, 1 at ``hour1``, clamped."""
    span = (hour1 - hour0) / np.timedelta64(1, "ms")
    return np.clip((t_utc - hour0) / np.timedelta64(1, "ms") / span, 0.0, 1.0)


# --- main --------------------------------------------------------------------


def los_cloud(
    samples: xr.Dataset,
    lat_deg: ArrayLike,
    lon_deg: ArrayLike,
    ground_h: ArrayLike,
    t_max_utc: np.datetime64 | NDArray[np.datetime64],
    sun_alt_apparent_deg: ArrayLike,
    sun_az_deg: ArrayLike,
    layer_heights_m: dict[str, tuple[float, ...]] = CLOUD_LAYER_HEIGHTS_M,
    aspect_ratio: dict[str, float] = CLOUD_ASPECT_RATIO,
    eye_height_m: float = EYE_HEIGHT_M,
    k: float = REFRACTION_K,
    chunk: int = 2_000,
) -> LosCloud:
    """Cloud-free line-of-sight probability for observers (inputs broadcast to one shape).

    ``samples`` is the ERA5 cube (year, day, hour, latitude, longitude) with
    exactly two consecutive hours.
    """
    floats = [np.asarray(x, dtype=float) for x in (lat_deg, lon_deg, ground_h, sun_alt_apparent_deg, sun_az_deg)]
    t_arr = np.asarray(t_max_utc)
    shape = np.broadcast_shapes(t_arr.shape, *(a.shape for a in floats))
    t = np.broadcast_to(t_arr, shape).ravel()
    lat, lon, h, alt, az = (np.broadcast_to(a, shape).ravel() for a in floats)

    hours = [int(x) for x in samples["hour"].values]
    if len(hours) != 2 or hours[1] != hours[0] + 1:
        raise ValueError(f"need two consecutive ERA5 hours, got {hours}")
    day0 = np.datetime64(t[0], "D") if t.size else np.datetime64("2026-08-12", "D")
    w_t = time_weight(t, day0 + np.timedelta64(hours[0], "h"), day0 + np.timedelta64(hours[1], "h"))

    grid_lat = samples["latitude"].values.astype(float)
    grid_lon = samples["longitude"].values.astype(float)
    # (sample, hour, flat lat/lon) per layer, float32 to keep chunks small.
    fields = {
        name: samples[name].transpose("year", "day", "hour", "latitude", "longitude").values.reshape(
            -1, 2, grid_lat.size * grid_lon.size
        )
        for name in layer_heights_m
    }
    R_eff = effective_radius(k)

    n_obs = lat.size
    p_slant = np.empty(n_obs)
    p_plain = np.empty(n_obs)
    layer_cloud = {name: np.empty(n_obs) for name in layer_heights_m}
    crossing_km = {name: np.empty(n_obs) for name in layer_heights_m}

    for start in range(0, n_obs, chunk):
        sl = slice(start, min(start + chunk, n_obs))
        prod_slant = 1.0
        prod_plain = 1.0
        for name, heights in layer_heights_m.items():
            hts = np.asarray(heights)[None, :]  # (1, H)
            c = los_layer_crossing(
                lat[sl, None], lon[sl, None], alt[sl, None], az[sl, None],
                h[sl, None] + hts, h[sl, None] + eye_height_m, R=R_eff,
            )
            crossing_km[name][sl] = c.ground_distance_m[:, len(heights) // 2] / 1_000.0
            idx, w = bilinear_weights(c.lat, c.lon, grid_lat, grid_lon)  # (P, H, 4)
            f = fields[name][:, :, idx]  # (S, 2, P, H, 4)
            wt = w_t[sl][None, :, None, None]
            f_t = (1.0 - wt) * f[:, 0] + wt * f[:, 1]  # (S, P, H, 4)
            n = (f_t * w[None]).sum(axis=-1).mean(axis=-1)  # (S, P)
            layer_cloud[name][sl] = n.mean(axis=0)
            prod_plain = prod_plain * (1.0 - n)
            prod_slant = prod_slant * slant_clear_fraction(n, alt[sl][None, :], aspect_ratio[name])
        p_plain[sl] = np.mean(prod_plain, axis=0)
        p_slant[sl] = np.mean(prod_slant, axis=0)

    return LosCloud(
        p_clear_sky=p_slant.reshape(shape),
        p_clear_sky_no_slant=p_plain.reshape(shape),
        layer_cloud={k_: v.reshape(shape) for k_, v in layer_cloud.items()},
        crossing_km={k_: v.reshape(shape) for k_, v in crossing_km.items()},
    )
