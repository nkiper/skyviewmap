"""Region derivation from the eclipse path: synthetic paths and the real 2027 path."""

import math
from datetime import date

import numpy as np
import pytest

from skyviewmapper.ephemeris import LocalCircumstances
from skyviewmapper.event import Event, load_event
from skyviewmapper.io.dem import land_tiles, tile_cols
from skyviewmapper.paths import (
    CoarsePath,
    band_limits,
    components,
    coarse_path,
    derive_regions,
    dilate,
    lon_resolution,
    region_for_tiles,
    split_group,
)

# --- pure helpers ------------------------------------------------------------


def test_dilate_wraps_longitude() -> None:
    m = np.zeros((3, 4), bool)
    m[1, 0] = True
    d = dilate(m)
    assert d[:, [3, 0, 1]].all() and not d[:, 2].any()


def test_components_and_split() -> None:
    tiles = {(35, lo) for lo in range(-10, 12)} | {(10, 50)}
    groups = components(tiles)
    assert [len(g) for g in groups] == [22, 1]
    pieces = split_group(groups[0], max_lon_span=15)
    assert sorted(len(p) for p in pieces) == [7, 15]
    # Band edge at 50 N splits a group even when it is narrow.
    assert len(split_group({(49, 0), (50, 0)})) == 2


@pytest.mark.parametrize("lat", [0.0, 36.0, 55.0, 65.0, 75.0])
def test_lon_resolution_square_and_whole_pixels(lat: float) -> None:
    res = lon_resolution(lat)
    km_ratio = res * math.cos(math.radians(lat)) / 0.01
    assert 0.65 < km_ratio < 1.5  # roughly square cells
    px_lon = 1.0 / tile_cols(int(lat))
    assert res / px_lon == pytest.approx(round(res / px_lon))  # whole DEM pixels
    assert 1.0 / res == pytest.approx(round(1.0 / res))  # divides 1 deg


def test_band_limits() -> None:
    assert band_limits(36) == (-50.0, 50.0) and band_limits(-20) == (-50.0, 50.0)
    assert band_limits(64) == (60.0, 70.0) and band_limits(-55) == (-60.0, -50.0)


# --- synthetic coarse path ---------------------------------------------------

DAY = date(2027, 8, 2)


def synthetic_cp(t0_minute: int = 8 * 60, minutes_per_deg: float = 2.0) -> CoarsePath:
    """A path along 35-36 N from 20 W to 20 E; maxima advance eastwards; Sun at 40 deg."""
    lats = np.arange(30.25, 40.0, 0.5)
    lons = np.arange(-24.75, 25.0, 0.5)
    lat, lon = np.meshgrid(lats, lons, indexing="ij")
    on = (lat > 35.0) & (lat < 36.0) & (np.abs(lon) < 20.0)
    minutes = t0_minute + (lon + 25.0) * minutes_per_deg
    t = np.datetime64("2027-08-02T00:00", "ms") + (minutes * 60_000).astype("int64").astype("timedelta64[ms]")
    full = np.full(lat.shape, 1.0)
    circ = LocalCircumstances(
        t_max_utc=t, separation_deg=0 * full, sun_alt_deg=40 * full, sun_alt_apparent_deg=40 * full,
        sun_az_deg=95 * full, sun_radius_deg=0.26 * full, moon_radius_deg=0.28 * full, diameter_ratio=1.07 * full,
        magnitude=np.where(on, 1.03, 0.9), obscuration=np.where(on, 1.0, 0.9), central_s=np.where(on, 280.0, 0.0),
        eclipse_type=np.where(on, 3.0, 1.0), totality_s=np.where(on, 280.0, 0.0),
    )
    return CoarsePath(lats=lats, lons=lons, circ=circ)


def test_region_for_tiles_settings() -> None:
    cp = synthetic_cp()
    tiles = {(35, lo) for lo in range(-5, 5)} | {(34, 0), (36, 0)}
    r = region_for_tiles(tiles, cp, DAY, 3, {"r03_36N_000E": "Test land"})
    g = r.grid
    assert (g.lat_min, g.lat_max, g.lon_min, g.lon_max) == (34.0, 37.0, -5.0, 5.0)
    assert g.dlat == 0.01 and g.dlon == lon_resolution(37.0)
    # DEM and ERA5 boxes contain the grid with a margin; the Sun at 40 deg keeps them small.
    for box in (r.dem_box, r.era5_box):
        assert box[0] < g.lat_min and box[1] > g.lat_max and box[2] < g.lon_min and box[3] > g.lon_max
        assert box[1] - box[0] < 6.0
    # Maxima run 08:39-09:01 UT over the box (+-0.5 deg): hours 8, 9, 10 bracket them.
    assert r.era5_hours == (8, 9, 10)
    assert r.window_utc[0].strftime("%H:%M") < "08:40" < "09:02" < r.window_utc[1].strftime("%H:%M")
    assert r.name == "r03_36N_000E" and r.label == "Test land"


def test_derive_regions_drops_buffer_only_land_and_orders_by_time() -> None:
    cp = synthetic_cp()
    ev = Event(id="t", name="t", date=DAY, type="total", years=(1997, 2026))
    land = {(35, lo) for lo in range(5, 9)} | {(35, lo) for lo in range(-15, -11)} | {(37, 0)}  # (37, 0): 1 deg off the path
    regions = derive_regions(ev, cp, land)
    assert [r.grid.lon_min for r in regions] == [-15.0, 5.0]  # west first: earlier maxima
    assert [r.name[:3] for r in regions] == ["r01", "r02"]


def test_include_boxes_limit_regions() -> None:
    cp = synthetic_cp()
    ev = Event(id="t", name="t", date=DAY, type="total", years=(1997, 2026), include=((30.0, 40.0, 0.0, 10.0),))
    land = {(35, lo) for lo in range(-15, 9)}
    regions = derive_regions(ev, cp, land)
    assert len(regions) == 1 and regions[0].grid.lon_min == 0.0


# --- real 2027 path ------------------------------------------------------------


@pytest.fixture(scope="module")
def path_2027() -> CoarsePath:
    from skyviewmapper.io.skyfield_data import load_ephemeris

    try:
        eph, ts = load_ephemeris(delta_t=71.7, download=False)
    except FileNotFoundError:
        pytest.skip("DE440s not downloaded")
    return coarse_path(eph, ts, DAY)


def _cell(cp: CoarsePath, lat: float, lon: float) -> tuple[int, int]:
    return int(np.argmin(np.abs(cp.lats - lat))), int(np.argmin(np.abs(cp.lons - lon)))


@pytest.mark.ephemeris
def test_2027_path_matches_nasa(path_2027: CoarsePath) -> None:
    # NASA central line: 35 39.6 N 6 43.3 W (08:46 UT), 26 53.3 N 31 00.8 E (10:00 UT).
    for lat, lon in ((35.66, -6.72), (26.89, 31.01)):
        assert path_2027.path[_cell(path_2027, lat, lon)]
    # Northern limit 36 46.8 N at 5 19.2 W (08:48 UT): the northernmost path cell near 5.25 W
    # should be within one coarse cell (0.5 deg) of it.
    j = int(np.argmin(np.abs(path_2027.lons - (-5.32))))
    northmost = path_2027.lats[np.flatnonzero(path_2027.path[:, j]).max()]
    assert abs(northmost - 36.78) <= 0.5


@pytest.mark.ephemeris
def test_2027_regions_cover_southern_spain(path_2027: CoarsePath) -> None:
    try:
        land = land_tiles(download=False)
    except FileNotFoundError:
        pytest.skip("DEM tile list not downloaded")
    regions = derive_regions(load_event("2027-08-02"), path_2027, land)
    for lat, lon in ((36.72, -4.42), (36.01, -5.60), (25.69, 32.64)):  # Malaga, Tarifa, Luxor
        assert any(r.grid.lat_min <= lat < r.grid.lat_max and r.grid.lon_min <= lon < r.grid.lon_max for r in regions)
    for r in regions:
        assert len({tile_cols(la) for la in range(int(r.grid.lat_min), int(r.grid.lat_max))}) == 1
        assert r.window_utc[0] < r.window_utc[1] and len(r.era5_hours) >= 2
