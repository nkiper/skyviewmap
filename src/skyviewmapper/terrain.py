"""Terrain horizon toward the Sun, and per-cell terrain visibility.

For each observer spot, cast a ray along the Sun's azimuth over the DEM and
find the highest terrain elevation angle (the local horizon in that
direction). The Sun is visible when its apparent altitude exceeds it.

Assumptions and approximations
------------------------------
- Terrestrial refraction is modelled with an effective Earth radius
  ``R / (1 - k)``, k = 0.13 (standard atmosphere near the ground). The
  terrain's elevation angle is computed on that larger sphere and compared
  with the Sun's *apparent* (refracted) altitude. This is the usual approach
  of horizon/panorama tools.
- Ray positions are found on the real sphere (radius ``R_EARTH_M``) with
  :func:`geometry.destination_point`; the effective radius only enters the
  elevation-angle formula (central angle ``d / R_eff``).
- Samples along the ray are spaced geometrically (each ``step_ratio`` times
  farther than the last). Each sample reads the finest max-pooled copy of
  the DEM whose pixel (its shorter side, at the mosaic's poleward edge) is
  at least the sample spacing, so no pixel along the
  ray is skipped and narrow far peaks are kept. A pooled peak is placed at
  its pixel centre (up to ~2 sample spacings, i.e. ~6% of the distance, from
  its true position), and pooling spreads it sideways; both err slightly
  toward "blocked".
- Rays stop at ``max_reach_m``: the distance beyond which even the highest
  terrain in the DEM cannot rise above the Sun once curvature is included,
  with a 10% margin, capped at ``max_distance_m``.
- DEM heights are above the geoid and treated as heights above the sphere.
  Outside the mosaic the terrain is taken as 0 m (sea).
- Spots on DEM pixels at exactly 0 m are treated as sea and not used as
  observers.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import R_EARTH_M
from .geometry import destination_point
from .grid import Grid

REFRACTION_K = 0.13
EYE_HEIGHT_M = 2.0


@dataclass(frozen=True)
class Dem:
    """Heights on a regular lat/lon raster. Row 0 is the northernmost row.

    ``lat0``/``lon0`` are the centre of pixel (0, 0); ``dlat``/``dlon`` the
    (positive) pixel size in degrees.
    """

    heights: NDArray[np.int16]
    lat0: float
    lon0: float
    dlat: float
    dlon: float

    def sample(self, lat: NDArray[np.float64], lon: NDArray[np.float64]) -> NDArray[np.float64]:
        """Nearest-pixel height at (lat, lon); 0 outside the raster."""
        rows = np.rint((self.lat0 - lat) / self.dlat).astype(np.intp)
        cols = np.rint((lon - self.lon0) / self.dlon).astype(np.intp)
        n_rows, n_cols = self.heights.shape
        inside = (rows >= 0) & (rows < n_rows) & (cols >= 0) & (cols < n_cols)
        out = np.zeros(lat.shape)
        out[inside] = self.heights[rows[inside], cols[inside]]
        return out


@dataclass(frozen=True)
class TerrainVisibility:
    """Per-cell terrain visibility of the Sun. NaN for sea cells.

    ``clear_fraction``: share of the 3x3 typical spots (land only) that see
    the Sun. ``best_*``: the spot (of the 9 typical + the highest pixel) with
    the largest margin = Sun apparent altitude - horizon angle (degrees;
    positive means clear). ``highest_clear``: whether the highest pixel sees
    the Sun (1.0/0.0).
    """

    clear_fraction: NDArray[np.float64]
    best_margin_deg: NDArray[np.float64]
    best_lat: NDArray[np.float64]
    best_lon: NDArray[np.float64]
    best_elev_m: NDArray[np.float64]
    highest_clear: NDArray[np.float64]

    @property
    def best_clear(self) -> NDArray[np.bool_]:
        return self.best_margin_deg > 0


# --- pure geometry -----------------------------------------------------------


def effective_radius(k: float = REFRACTION_K, R: float = R_EARTH_M) -> float:
    """Effective Earth radius ``R / (1 - k)`` for terrestrial refraction coefficient ``k``."""
    return R / (1.0 - k)


def elevation_angle(
    distance_m: ArrayLike, target_h: ArrayLike, observer_h: ArrayLike, R_eff: float
) -> NDArray[np.float64]:
    """Elevation angle (degrees) of a point at ground distance ``d`` and height ``H``.

    On a sphere of radius ``R_eff`` with central angle ``phi = d / R_eff``:
    ``atan2((R_eff + H) cos(phi) - (R_eff + h0), (R_eff + H) sin(phi))``.
    """
    d = np.asarray(distance_m, dtype=float)
    rt = R_eff + np.asarray(target_h, dtype=float)
    r0 = R_eff + np.asarray(observer_h, dtype=float)
    phi = d / R_eff
    return np.degrees(np.arctan2(rt * np.cos(phi) - r0, rt * np.sin(phi)))


def max_reach_m(
    h_max: float, observer_h: ArrayLike, sun_alt_deg: ArrayLike, R_eff: float, cap_m: float = 200_000.0
) -> NDArray[np.float64]:
    """Distance beyond which terrain no higher than ``h_max`` cannot reach the Sun's altitude.

    Small-angle form: terrain at height difference dH and distance d appears at
    about ``dH / d - d / (2 R_eff)``; setting this equal to ``tan(alt)`` gives
    ``d = R_eff (-t + sqrt(t^2 + 2 dH / R_eff))``. A 10% margin is added and
    the result clipped to [0, cap_m]. Returns 0 where nothing can block.
    """
    t = np.tan(np.radians(np.asarray(sun_alt_deg, dtype=float)))
    dh = h_max - np.asarray(observer_h, dtype=float)
    disc = t * t + 2.0 * dh / R_eff
    with np.errstate(invalid="ignore"):
        d = R_eff * (-t + np.sqrt(disc))
    d = np.where((disc > 0) & (d > 0), 1.1 * d, 0.0)
    return np.minimum(d, cap_m)


def ray_distances(d_min_m: float = 45.0, d_max_m: float = 200_000.0, step_ratio: float = 1.03) -> NDArray[np.float64]:
    """Geometric sample distances ``d_min * step_ratio**k`` up to ``d_max``."""
    n = int(np.floor(np.log(d_max_m / d_min_m) / np.log(step_ratio))) + 1
    return d_min_m * step_ratio ** np.arange(n, dtype=np.float64)


# --- DEM pyramid -------------------------------------------------------------


def build_pyramid(dem: Dem, n_levels: int = 8) -> list[Dem]:
    """Level 0 is ``dem``; level i+1 max-pools 2x2 blocks of level i (odd edge rows/cols dropped)."""
    levels = [dem]
    for _ in range(n_levels - 1):
        prev = levels[-1]
        r, c = (s // 2 * 2 for s in prev.heights.shape)
        if r < 2 or c < 2:
            break
        h = np.asarray(prev.heights[:r, :c]).reshape(r // 2, 2, c // 2, 2).max(axis=(1, 3))
        levels.append(
            Dem(
                heights=h,
                lat0=prev.lat0 - prev.dlat / 2,
                lon0=prev.lon0 + prev.dlon / 2,
                dlat=prev.dlat * 2,
                dlon=prev.dlon * 2,
            )
        )
    return levels


def _level_for_spacing(spacing_m: NDArray[np.float64], base_px_m: float, n_levels: int) -> NDArray[np.intp]:
    """Finest pyramid level whose pixel is at least the sample spacing, so no pixel is skipped."""
    with np.errstate(divide="ignore"):
        lvl = np.ceil(np.log2(np.maximum(spacing_m, base_px_m) / base_px_m))
    return np.clip(lvl, 0, n_levels - 1).astype(np.intp)


# --- horizon -----------------------------------------------------------------


def horizon_angle(
    pyramid: list[Dem],
    lat_deg: ArrayLike,
    lon_deg: ArrayLike,
    observer_h: ArrayLike,
    az_deg: ArrayLike,
    reach_m: ArrayLike,
    R_eff: float,
    distances: NDArray[np.float64] | None = None,
    step_ratio: float = 1.03,
) -> NDArray[np.float64]:
    """Highest terrain elevation angle (degrees) along azimuth ``az`` out to ``reach_m``.

    Inputs are 1-D arrays of equal length (one per ray). Returns -inf where
    the ray has no samples (``reach_m`` below the first sample distance).
    """
    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    h0 = np.asarray(observer_h, dtype=float)
    az = np.asarray(az_deg, dtype=float)
    reach = np.asarray(reach_m, dtype=float)
    if distances is None:
        distances = ray_distances(step_ratio=step_ratio)
    n_used = int(np.searchsorted(distances, reach.max(initial=0.0), side="right"))
    d = distances[:n_used]
    if n_used == 0:
        return np.full(lat.shape, -np.inf)

    # Smaller pixel side in metres, at the mosaic's poleward edge where
    # longitude pixels are narrowest, so a level's pixels are at least the
    # sample spacing in every direction.
    base = pyramid[0]
    poleward = max(abs(base.lat0), abs(base.lat0 - base.dlat * (base.heights.shape[0] - 1)))
    base_px_m = np.radians(min(base.dlat, base.dlon * np.cos(np.radians(poleward)))) * R_EARTH_M
    level = _level_for_spacing(d * (step_ratio - 1.0), base_px_m, len(pyramid))

    lat2, lon2 = destination_point(lat[:, None], lon[:, None], az[:, None], d[None, :] / R_EARTH_M)
    heights = np.zeros(lat2.shape)
    for lvl in np.unique(level):
        cols = level == lvl
        heights[:, cols] = pyramid[lvl].sample(lat2[:, cols], lon2[:, cols])

    angle = elevation_angle(d[None, :], heights, h0[:, None], R_eff)
    angle = np.where(d[None, :] <= reach[:, None], angle, -np.inf)
    return angle.max(axis=1)


# --- per-cell spots ----------------------------------------------------------


def cell_spots(dem: Dem, grid: Grid) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """DEM (row, col) indices of the 10 observer spots in every grid cell.

    Returns two arrays of shape ``grid.shape + (10,)``: spots 0-8 form a 3x3
    pattern of pixels evenly spread over the cell, spot 9 is the cell's
    highest pixel. A pixel belongs to a cell when its centre lies in
    (south, north] x [west, east); the cell size must be a whole number of
    pixels so every cell gets the same block.
    """
    n_lat = grid.dlat / dem.dlat
    n_lon = grid.dlon / dem.dlon
    if not (np.isclose(n_lat, round(n_lat)) and np.isclose(n_lon, round(n_lon))):
        raise ValueError("grid resolution must be a whole number of DEM pixels")
    n_lat, n_lon = int(round(n_lat)), int(round(n_lon))

    # Top-left DEM pixel of each cell. Cell rows run south->north, DEM rows north->south.
    north_edge = grid.lat_min + (np.arange(grid.shape[0]) + 1) * grid.dlat
    west_edge = grid.lon_min + np.arange(grid.shape[1]) * grid.dlon
    # First pixel with centre <= north edge, and first with centre >= west edge.
    r0 = np.ceil((dem.lat0 - north_edge) / dem.dlat - 1e-6).astype(np.intp)
    c0 = np.ceil((west_edge - dem.lon0) / dem.dlon - 1e-6).astype(np.intp)
    if r0.min() < 0 or c0.min() < 0 or r0.max() + n_lat > dem.heights.shape[0] or c0.max() + n_lon > dem.heights.shape[1]:
        raise ValueError("grid extends beyond the DEM")

    off_r = np.rint((np.arange(3) + 0.5) * n_lat / 3 - 0.5).astype(np.intp)
    off_c = np.rint((np.arange(3) + 0.5) * n_lon / 3 - 0.5).astype(np.intp)
    R0, C0 = np.meshgrid(r0, c0, indexing="ij")
    typ_r = R0[..., None] + np.repeat(off_r, 3)[None, None, :]
    typ_c = C0[..., None] + np.tile(off_c, 3)[None, None, :]

    # Highest pixel per cell: view the cell blocks as (rows, n_lat, cols, n_lon).
    block = np.asarray(
        dem.heights[r0[-1] : r0[0] + n_lat, c0[0] : c0[-1] + n_lon]
    )  # rows north->south: last grid row is northernmost
    nr, nc = grid.shape
    block = block.reshape(nr, n_lat, nc, n_lon)[::-1]  # flip to south->north cell order
    flat = block.transpose(0, 2, 1, 3).reshape(nr, nc, n_lat * n_lon).argmax(axis=-1)
    high_r = R0 + flat // n_lon
    high_c = C0 + flat % n_lon

    rows = np.concatenate([typ_r, high_r[..., None]], axis=-1)
    cols = np.concatenate([typ_c, high_c[..., None]], axis=-1)
    return rows, cols


def terrain_visibility(
    pyramid: list[Dem],
    grid: Grid,
    sun_alt_apparent_deg: ArrayLike,
    sun_az_deg: ArrayLike,
    eye_height_m: float = EYE_HEIGHT_M,
    k: float = REFRACTION_K,
    chunk: int = 20_000,
) -> TerrainVisibility:
    """Terrain visibility of the Sun for every grid cell (Sun alt/az given per cell)."""
    dem = pyramid[0]
    R_eff = effective_radius(k)
    alt = np.broadcast_to(np.asarray(sun_alt_apparent_deg, dtype=float), grid.shape)
    az = np.broadcast_to(np.asarray(sun_az_deg, dtype=float), grid.shape)

    rows, cols = cell_spots(dem, grid)
    ground = np.asarray(dem.heights[rows, cols], dtype=float)  # (nr, nc, 10)
    spot_lat = dem.lat0 - rows.astype(np.float64) * dem.dlat
    spot_lon = dem.lon0 + cols.astype(np.float64) * dem.dlon
    land = ground != 0

    # Flatten to one ray per land spot.
    alt_s = np.broadcast_to(alt[..., None], ground.shape)
    az_s = np.broadcast_to(az[..., None], ground.shape)
    h0 = ground + eye_height_m
    h_max = float(np.max(pyramid[-1].heights))  # coarsest level is max-pooled: global max
    reach = max_reach_m(h_max, h0, alt_s, R_eff)

    margin = np.full(ground.shape, np.nan)
    idx = np.flatnonzero(land)
    # Rays with similar reach share sample counts: sort so chunks stay compact.
    idx = idx[np.argsort(reach.ravel()[idx])]
    for i in range(0, idx.size, chunk):
        sel = idx[i : i + chunk]
        hz = horizon_angle(
            pyramid,
            spot_lat.ravel()[sel],
            spot_lon.ravel()[sel],
            h0.ravel()[sel],
            az_s.ravel()[sel],
            reach.ravel()[sel],
            R_eff,
        )
        margin.ravel()[sel] = alt_s.ravel()[sel] - hz

    is_clear = land & (margin > 0)
    n_typical = land[..., :9].sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        clear_fraction = np.where(n_typical > 0, is_clear[..., :9].sum(axis=-1) / n_typical, np.nan)
    best = np.argmax(np.where(land, np.nan_to_num(margin, nan=-np.inf), -np.inf), axis=-1)
    sea = ~land.any(axis=-1)

    def at_best(a: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.where(sea, np.nan, np.take_along_axis(a, best[..., None], axis=-1)[..., 0])

    return TerrainVisibility(
        clear_fraction=np.where(sea, np.nan, clear_fraction),
        best_margin_deg=at_best(margin),
        best_lat=at_best(spot_lat),
        best_lon=at_best(spot_lon),
        best_elev_m=at_best(ground),
        highest_clear=np.where(land[..., 9], is_clear[..., 9].astype(float), np.nan),
    )
