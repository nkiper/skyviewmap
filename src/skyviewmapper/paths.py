"""Derive map regions from the path of a central solar eclipse over land.

1. Find the moment of greatest geocentric conjunction on the event day.
2. Run the ephemeris on a coarse global grid (0.5 deg, 60 s steps) over
   +-4 h: cells with a central phase and the Sun above the horizon form the
   path.
3. Keep 1 deg land tiles (Copernicus DEM tile list) within one coarse cell
   (~55 km) of the path, group neighbouring tiles, and split the groups at
   Copernicus DEM latitude bands and into pieces at most ``max_lon_span``
   degrees wide.
4. Give every group a map grid, DEM box, ERA5 box, ephemeris window and ERA5
   hours from its own geometry: the DEM box is widened by the farthest
   distance terrain could rise above the lowest Sun in the group, the ERA5 box
   by the farthest crossing of the top of the high cloud layer.

Assumptions: the coarse run only locates the path; every number on the maps
comes from the full-resolution run of each region. Terrain beyond a DEM band
edge is treated as sea (the DEM box is clipped there).
"""

import math
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import numpy as np
from numpy.typing import NDArray
from skyfield.jpllib import SpiceKernel
from skyfield.timelib import Timescale

from .constants import CLOUD_LAYER_HEIGHTS_M, R_EARTH_M
from .ephemeris import LocalCircumstances, angular_separation, body_track, local_circumstances
from .event import Event
from .geometry import los_layer_crossing
from .grid import Grid
from .io.dem import tile_cols
from .regions import Region
from .terrain import REFRACTION_K, effective_radius, max_reach_m

COARSE_RES_DEG = 0.5
MAX_TERRAIN_M = 6_000.0  # above any land relief near eclipse paths we map; only sets the DEM margin
KM_PER_DEG = math.radians(1.0) * R_EARTH_M / 1_000.0
# Longitude cell sizes (deg) that divide 1 deg and are whole numbers of DEM
# pixels in every band up to 80 deg (3", 4.5", 6", 9").
LON_RES_CHOICES = (0.01, 0.0125, 0.02, 0.025, 0.04, 0.05)


@dataclass(frozen=True)
class CoarsePath:
    """Coarse global circumstances: 2-D fields on ``lats`` x ``lons`` (cell centres)."""

    lats: NDArray[np.float64]
    lons: NDArray[np.float64]
    circ: LocalCircumstances

    @property
    def path(self) -> NDArray[np.bool_]:
        """Central phase with the Sun above the horizon."""
        return (np.nan_to_num(self.circ.central_s) > 0) & (np.nan_to_num(self.circ.sun_alt_apparent_deg, nan=-90) > 0)

    @property
    def eclipsed(self) -> NDArray[np.bool_]:
        """Any eclipse with the Sun above the horizon."""
        return (np.nan_to_num(self.circ.magnitude) > 0) & (np.nan_to_num(self.circ.sun_alt_apparent_deg, nan=-90) > 0)


# --- ephemeris ---------------------------------------------------------------


def geocentric_maximum(eph: SpiceKernel, ts: Timescale, day: date, step_s: float = 60.0) -> datetime:
    """Time of minimum geocentric Sun-Moon separation on ``day`` (UTC)."""
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    track = body_track(eph, ts, start, start + timedelta(days=1), step_s)
    sep = angular_separation(track.sun_m, track.moon_m)
    i = int(np.argmin(sep))
    return start + timedelta(seconds=i * step_s)


def coarse_path(
    eph: SpiceKernel, ts: Timescale, day: date, res_deg: float = COARSE_RES_DEG, half_span_h: float = 4.0
) -> CoarsePath:
    """Circumstances on a global ``res_deg`` grid around the geocentric maximum."""
    t0 = geocentric_maximum(eph, ts, day)
    track = body_track(eph, ts, t0 - timedelta(hours=half_span_h), t0 + timedelta(hours=half_span_h), 60.0)
    lats = np.arange(-90.0 + res_deg / 2, 90.0, res_deg)
    lons = np.arange(-180.0 + res_deg / 2, 180.0, res_deg)
    lat, lon = np.meshgrid(lats, lons, indexing="ij")
    circ = local_circumstances(lat, lon, 0.0, track, coarse_s=600.0)
    return CoarsePath(lats=lats, lons=lons, circ=circ)


# --- pure helpers ------------------------------------------------------------


def dilate(mask: NDArray[np.bool_], cells: int = 1) -> NDArray[np.bool_]:
    """Grow a 2-D mask by ``cells`` in every direction (8-neighbourhood), wrapping in longitude."""
    out = mask.copy()
    for _ in range(cells):
        grown = out.copy()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        grown |= np.roll(out, 1, axis=1) | np.roll(out, -1, axis=1)
        grown[1:] |= np.roll(out[:-1], 1, axis=1) | np.roll(out[:-1], -1, axis=1)
        grown[:-1] |= np.roll(out[1:], 1, axis=1) | np.roll(out[1:], -1, axis=1)
        out = grown
    return out


def components(tiles: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    """Groups of 8-connected 1 deg tiles, largest first."""
    left = set(tiles)
    groups = []
    while left:
        seed = left.pop()
        group = {seed}
        queue = deque([seed])
        while queue:
            la, lo = queue.popleft()
            for dla in (-1, 0, 1):
                for dlo in (-1, 0, 1):
                    nb = (la + dla, lo + dlo)
                    if nb in left:
                        left.remove(nb)
                        group.add(nb)
                        queue.append(nb)
        groups.append(group)
    return sorted(groups, key=len, reverse=True)


def split_group(group: set[tuple[int, int]], max_lon_span: int = 15) -> list[set[tuple[int, int]]]:
    """Split a tile group at DEM band edges and into pieces at most ``max_lon_span`` deg wide."""
    out = []
    by_band: dict[int, set[tuple[int, int]]] = {}
    for t in group:
        by_band.setdefault(tile_cols(t[0]), set()).add(t)
    for band in by_band.values():
        lon0 = min(lo for _, lo in band)
        pieces: dict[int, set[tuple[int, int]]] = {}
        for t in band:
            pieces.setdefault((t[1] - lon0) // max_lon_span, set()).add(t)
        out.extend(pieces.values())
    return out


def lon_resolution(lat_deg: float, lat_res: float = 0.01) -> float:
    """Longitude cell size giving roughly square cells at ``lat_deg``, from ``LON_RES_CHOICES``."""
    target = lat_res / max(math.cos(math.radians(lat_deg)), 0.05)
    return min(LON_RES_CHOICES, key=lambda r: abs(math.log(r / target)))


def band_limits(lat_south: int) -> tuple[float, float]:
    """Latitude range of the Copernicus DEM band containing the tile row starting at ``lat_south``.

    Bands go by the tile's equatorward latitude: |lat| < 50 (one band across
    the equator), 50-60, 60-70, 70-80, 80-85 and 85-90 in each hemisphere.
    """
    bottoms, tops = (0, 50, 60, 70, 80, 85), (50, 60, 70, 80, 85, 90)
    e = min(abs(lat_south), abs(lat_south + 1))
    k = next(i for i, (b, t) in enumerate(zip(bottoms, tops)) if b <= e < t)
    if k == 0:
        return -50.0, 50.0
    return (float(bottoms[k]), float(tops[k])) if lat_south >= 0 else (-float(tops[k]), -float(bottoms[k]))


def _expand(box: tuple[float, float, float, float], margin_km: float, step: float) -> tuple[float, float, float, float]:
    """Widen a box by ``margin_km`` and round outward to multiples of ``step`` degrees."""
    lat_min, lat_max, lon_min, lon_max = box
    dlat = margin_km / KM_PER_DEG
    poleward = max(abs(lat_min), abs(lat_max))
    dlon = margin_km / (KM_PER_DEG * max(math.cos(math.radians(min(poleward + dlat, 89.0))), 0.05))

    def down(x: float) -> float:
        return math.floor(round(x / step, 6)) * step

    def up(x: float) -> float:
        return math.ceil(round(x / step, 6)) * step

    return (
        max(down(lat_min - dlat), -90.0),
        min(up(lat_max + dlat), 90.0),
        down(lon_min - dlon),
        up(lon_max + dlon),
    )


def _name(lat: float, lon: float, i: int) -> str:
    return f"r{i:02d}_{abs(lat):.0f}{'N' if lat >= 0 else 'S'}_{abs(lon):03.0f}{'E' if lon >= 0 else 'W'}"


# --- regions -----------------------------------------------------------------


def _tiles_of(mask: NDArray[np.bool_], cp: CoarsePath) -> set[tuple[int, int]]:
    ii, jj = np.nonzero(mask)
    return {(int(math.floor(cp.lats[i])), int(math.floor(cp.lons[j]))) for i, j in zip(ii, jj)}


def path_tiles(
    cp: CoarsePath,
    land: set[tuple[int, int]],
    include: tuple[tuple[float, float, float, float], ...] = (),
    buffer_cells: int = 1,
) -> set[tuple[int, int]]:
    """Land tiles within ``buffer_cells`` coarse cells of the path (and inside ``include`` boxes, if any)."""
    tiles = _tiles_of(dilate(cp.path, buffer_cells) if buffer_cells else cp.path, cp) & land
    if include:
        tiles = {
            (la, lo)
            for la, lo in tiles
            if any(la < b[1] and la + 1 > b[0] and lo < b[3] and lo + 1 > b[2] for b in include)
        }
    return tiles


def region_for_tiles(tiles: set[tuple[int, int]], cp: CoarsePath, day: date, index: int, display_names: dict[str, str] | None = None) -> Region:
    """Map grid, DEM/ERA5 boxes, time window and ERA5 hours for one tile group."""
    lat_min = float(min(la for la, _ in tiles))
    lat_max = float(max(la for la, _ in tiles) + 1)
    lon_min = float(min(lo for _, lo in tiles))
    lon_max = float(max(lo for _, lo in tiles) + 1)

    # Coarse cells inside the box (plus half a cell) with the Sun up and some eclipse.
    in_lat = (cp.lats >= lat_min - COARSE_RES_DEG) & (cp.lats <= lat_max + COARSE_RES_DEG)
    in_lon = (cp.lons >= lon_min - COARSE_RES_DEG) & (cp.lons <= lon_max + COARSE_RES_DEG)
    sel = cp.eclipsed & in_lat[:, None] & in_lon[None, :]
    if not sel.any():
        raise ValueError(f"no eclipsed coarse cells in tile group at {lat_min}..{lat_max}, {lon_min}..{lon_max}")
    alt_min = float(np.nanmin(cp.circ.sun_alt_apparent_deg[sel]))
    t_max = cp.circ.t_max_utc[sel]
    t_lo, t_hi = t_max.min(), t_max.max()

    R_eff = effective_radius(REFRACTION_K)
    reach_km = float(max_reach_m(MAX_TERRAIN_M, 0.0, max(alt_min, 0.5), R_eff)) / 1_000.0
    top_m = max(max(h) for h in CLOUD_LAYER_HEIGHTS_M.values()) + 1_500.0  # high-layer top plus ground height
    cross_km = float(los_layer_crossing(0.0, 0.0, max(alt_min, 0.5), 0.0, top_m, 0.0, R=R_eff).ground_distance_m) / 1_000.0

    band_lo, band_hi = band_limits(int(lat_min))
    d_lat_min, d_lat_max, d_lon_min, d_lon_max = _expand((lat_min, lat_max, lon_min, lon_max), reach_km, 0.01)
    dem_box = (max(d_lat_min, band_lo), min(d_lat_max, band_hi), d_lon_min, d_lon_max)
    era5_box = _expand((lat_min, lat_max, lon_min, lon_max), cross_km + 30.0, 0.25)

    def minute(t: np.datetime64) -> datetime:
        return datetime.fromisoformat(str(np.datetime64(t, "m"))).replace(tzinfo=timezone.utc)

    start = minute(t_lo) - timedelta(minutes=10)
    end = minute(t_hi) + timedelta(minutes=11)
    h0 = (t_lo - np.datetime64(day, "h")) // np.timedelta64(1, "h")
    h1 = (t_hi - np.datetime64(day, "h")) // np.timedelta64(1, "h") + 1
    if h0 < 0 or h1 > 23:
        raise ValueError("eclipse window crosses midnight UTC; not supported")
    hours = tuple(range(int(h0), int(h1) + 1))

    lat_mid = (lat_min + lat_max) / 2
    grid = Grid(lat_min, lat_max, lon_min, lon_max, 0.01, lon_resolution(max(abs(lat_min), abs(lat_max))))
    name = _name(lat_mid, (lon_min + lon_max) / 2, index)
    return Region(
        name=name,
        grid=grid,
        dem_box=dem_box,
        window_utc=(start, end),
        era5_box=era5_box,
        era5_hours=hours,
        display_name=(display_names or {}).get(name, ""),
    )


def derive_regions(
    event: Event, cp: CoarsePath, land: set[tuple[int, int]], max_lon_span: int = 15
) -> list[Region]:
    """Regions covering the land under the central path, ordered by time of maximum."""
    core = path_tiles(cp, land, event.include, buffer_cells=0)  # land tiles under the path itself
    groups = [
        g
        for comp in components(path_tiles(cp, land, event.include))
        for g in split_group(comp, max_lon_span)
        if g & core  # drop groups reached only through the buffer
    ]
    provisional = [region_for_tiles(g, cp, event.date, 0) for g in groups]
    order = sorted(range(len(groups)), key=lambda i: provisional[i].window_utc[0])
    regions = [region_for_tiles(groups[i], cp, event.date, k + 1, event.display_names) for k, i in enumerate(order)]
    unknown = set(event.display_names) - {r.name for r in regions}
    if unknown:
        raise ValueError(
            f"display_names for unknown regions {sorted(unknown)}; derived: {', '.join(r.name for r in regions)}"
        )
    return regions


def event_regions(event: Event, download: bool = True) -> list[Region]:
    """The event's explicit regions, or regions derived from its path over land."""
    if event.explicit_regions:
        return list(event.explicit_regions)
    from .io.dem import land_tiles
    from .io.skyfield_data import load_ephemeris

    eph, ts = load_ephemeris(download=download)
    return derive_regions(event, coarse_path(eph, ts, event.date), land_tiles(download=download))
