"""Terrain tests.

Pure tests use hand-calculated values (R_eff = 6 371 000 / 0.87 =
7 322 988.5 m) and synthetic DEMs. Tests marked ``dem`` check the real
Copernicus GLO-90 tiles in data/raw/dem/glo90 and skip if absent.
"""

import math

import numpy as np
import pytest

from skyviewmapper.constants import R_EARTH_M
from skyviewmapper.grid import Grid
from skyviewmapper.io.dem import TILE_DIR, build_mosaic, tile_name
from skyviewmapper.terrain import (
    Dem,
    build_pyramid,
    cell_spots,
    effective_radius,
    elevation_angle,
    horizon_angle,
    max_reach_m,
    terrain_visibility,
)

R_EFF = effective_radius(0.13)
PX = 1.0 / 1200  # 3 arcsec


def flat_dem(lat_min: float, lat_max: float, lon_min: float, lon_max: float, h: int = 0) -> Dem:
    nr, nc = round((lat_max - lat_min) / PX), round((lon_max - lon_min) / PX)
    return Dem(np.full((nr, nc), h, dtype=np.int16), lat0=lat_max - PX / 2, lon0=lon_min + PX / 2, dlat=PX, dlon=PX)


def raise_box(dem: Dem, lat_a: float, lat_b: float, lon_a: float, lon_b: float, h: int) -> None:
    """Set pixels whose centres fall in the box to height ``h``."""
    lats = dem.lat0 - np.arange(dem.heights.shape[0]) * dem.dlat
    lons = dem.lon0 + np.arange(dem.heights.shape[1]) * dem.dlon
    rows = (lats >= lat_a) & (lats <= lat_b)
    cols = (lons >= lon_a) & (lons <= lon_b)
    dem.heights[np.ix_(rows, cols)] = h


def west_lon(d_m: float) -> float:
    """Longitude reached going d_m due west along the equator."""
    return -math.degrees(d_m / R_EARTH_M)


# --- pure geometry -----------------------------------------------------------


def test_effective_radius() -> None:
    assert R_EFF == pytest.approx(7_322_988.505747, abs=1e-3)


@pytest.mark.parametrize(
    "d, H, h0, ref",
    [
        # 100 m higher at 1 km: atan(0.1) = 5.71059 deg, less ~0.0039 deg of curvature.
        (1_000.0, 100.0, 0.0, 5.7066423467788),
        # Same height at 100 km: atan2(cos(phi) - 1, sin(phi)) = -phi/2 = -d / (2 R_eff).
        (100_000.0, 0.0, 0.0, -math.degrees(100_000.0 / (2 * 7_322_988.505747))),
        (0.0, 0.0, 0.0, 0.0),
    ],
)
def test_elevation_angle(d: float, H: float, h0: float, ref: float) -> None:
    assert elevation_angle(d, H, h0, R_EFF) == pytest.approx(ref, abs=1e-9)


def test_max_reach_hand_value() -> None:
    # Sun at 7 deg, highest terrain 3 479 m, observer at 0 m:
    # t = tan 7 = 0.122785; d = R_eff (-t + sqrt(t^2 + 2*3479/R_eff)) = 27 901.3 m; +10% = 30 691.4 m.
    assert max_reach_m(3_479.0, 0.0, 7.0, R_EFF) == pytest.approx(30_691.4, abs=0.1)


def test_max_reach_nothing_can_block() -> None:
    # Observer above every peak and Sun above the horizon.
    assert max_reach_m(1_000.0, 2_000.0, 5.0, R_EFF) == 0.0


def test_max_reach_is_capped() -> None:
    assert max_reach_m(3_479.0, 0.0, 0.0, R_EFF, cap_m=200_000.0) == 200_000.0


# --- horizon on synthetic DEMs ----------------------------------------------

OBS_LAT = PX / 2  # centre of the pixel row just north of the equator


def test_flat_ground_horizon_is_the_dip() -> None:
    # Level ground, 2 m eye: horizon dip = -acos(R_eff / (R_eff + 2)) = -0.042346 deg.
    pyramid = build_pyramid(flat_dem(-0.1, 0.1, -0.5, 0.1))
    hz = horizon_angle(pyramid, [OBS_LAT], [0.0], [2.0], [270.0], [50_000.0], R_EFF)
    assert hz[0] == pytest.approx(-0.0423456, abs=2e-4)


def test_ridge_blocks_only_in_its_direction() -> None:
    # 500 m wall from 10.0 to 10.5 km west. Horizon = elevation of its near
    # edge: 2.8118 deg at 10 km (2.6743 at 10.5 km; pooling may pull it ~200 m
    # closer: 2.8706 at 9.8 km).
    dem = flat_dem(-0.1, 0.1, -0.5, 0.1)
    raise_box(dem, -0.1, 0.1, west_lon(10_500), west_lon(10_000), 500)
    pyramid = build_pyramid(dem)
    west, east = horizon_angle(
        pyramid, [OBS_LAT, OBS_LAT], [0.0, 0.0], [2.0, 2.0], [270.0, 90.0], [50_000.0, 50_000.0], R_EFF
    )
    assert 2.674 <= west <= 2.871
    assert east == pytest.approx(-0.0423456, abs=2e-4)


def test_narrow_far_peak_is_not_skipped() -> None:
    # One 3" pixel, 2 000 m high, 80 km west. Elevation 1.1175 deg (1.0748 at
    # 82 km, 1.1620 at 78 km). Samples there are ~2.4 km apart, so only the
    # max-pooled levels guarantee it is seen.
    dem = flat_dem(-0.1, 0.1, -1.0, 0.1)
    lon_peak = west_lon(80_000)
    raise_box(dem, OBS_LAT - PX / 2, OBS_LAT + PX / 2, lon_peak - PX / 2, lon_peak + PX / 2, 2_000)
    assert (dem.heights == 2_000).sum() == 1
    hz = horizon_angle(build_pyramid(dem), [OBS_LAT], [0.0], [2.0], [270.0], [100_000.0], R_EFF)
    assert 1.0 < hz[0] < 1.3


def test_pyramid_max_pools_and_keeps_georeference() -> None:
    dem = flat_dem(0.0, 0.01, 0.0, 0.01)  # 12 x 12
    dem.heights[5, 7] = 42
    levels = build_pyramid(dem, n_levels=3)
    assert [lvl.heights.shape for lvl in levels] == [(12, 12), (6, 6), (3, 3)]
    assert levels[1].heights[2, 3] == 42 and levels[2].heights[1, 1] == 42
    # Centre of level-1 pixel (0, 0) is the corner shared by base pixels (0..1, 0..1).
    assert levels[1].lat0 == pytest.approx(0.01 - PX)
    assert levels[1].lon0 == pytest.approx(PX)


# --- cells -------------------------------------------------------------------


def test_cell_spots() -> None:
    dem = flat_dem(0.0, 0.02, 0.0, 0.03, h=100)  # 24 x 36 px; cells are 12 x 12 px
    grid = Grid(0.0, 0.02, 0.0, 0.03, 0.01)  # shape (2, 3), row 0 = south
    dem.heights[20, 15] = 900  # in the southern cell row (DEM rows 12-23), cell column 1 (cols 12-23)
    rows, cols = cell_spots(dem, grid)
    assert rows.shape == (2, 3, 10)
    assert (rows[0, 1, 9], cols[0, 1, 9]) == (20, 15)
    assert sorted(set(rows[0, 1, :9].tolist())) == [14, 18, 22]
    assert sorted(set(cols[0, 1, :9].tolist())) == [14, 18, 22]
    assert sorted(set(rows[1, 2, :9].tolist())) == [2, 6, 10]  # northern row, eastern column
    assert sorted(set(cols[1, 2, :9].tolist())) == [26, 30, 34]



@pytest.mark.parametrize("res_lon, n_lon", [(0.01, 6), (0.02, 12)])
def test_cell_spots_non_square_pixels(res_lon: float, n_lon: int) -> None:
    # Iceland-style tiles: 3" in latitude, 6" in longitude (1200 x 600 px).
    dem = Dem(np.full((24, 4 * n_lon), 100, np.int16), lat0=0.02 - PX / 2, lon0=PX, dlat=PX, dlon=2 * PX)
    grid = Grid(0.0, 0.02, 0.0, 4 * res_lon, 0.01, res_lon=res_lon)
    assert grid.shape == (2, 4)
    dem.heights[13, 2 * n_lon + 1] = 500  # southern cell row, cell column 2
    rows, cols = cell_spots(dem, grid)
    assert (rows[0, 2, 9], cols[0, 2, 9]) == (13, 2 * n_lon + 1)
    # 3x3 pattern spreads over the cell's n_lon columns: offsets round((k + 0.5) n / 3 - 0.5).
    expected = [2 * n_lon + round((k + 0.5) * n_lon / 3 - 0.5) for k in range(3)]
    assert sorted(set(cols[0, 2, :9].tolist())) == expected
    assert sorted(set(rows[0, 2, :9].tolist())) == [14, 18, 22]


def visibility_scene(hill: bool = True) -> tuple[list[Dem], Grid]:
    """Flat 100 m land, a 1 000 m wall ~21 km west, a 600 m hill pixel in one cell, one sea cell."""
    dem = flat_dem(-0.05, 0.05, -0.3, 0.05, h=100)
    raise_box(dem, -0.05, 0.05, -0.2, -0.19, 1_000)
    grid = Grid(-0.01, 0.01, -0.02, 0.02, 0.01)  # 2 x 4 cells
    # Hill: one pixel in cell (row 0, col 1) = lat -0.01..0, lon -0.01..0.
    if hill:
        raise_box(dem, -0.005 - PX / 2, -0.005 + PX / 2, -0.005 - PX / 2, -0.005 + PX / 2, 600)
    # Sea: cell (row 1, col 3) = lat 0..0.01, lon 0.01..0.02.
    raise_box(dem, 0.0, 0.01, 0.01, 0.02, 0)
    return build_pyramid(dem), grid


def test_terrain_visibility_blocked_and_clear() -> None:
    pyramid, grid = visibility_scene(hill=False)  # the hill would shade spots just east of it
    # Wall from 102 m at ~21 km: atan(898/21000) - 21000/(2 R_eff) = ~2.37 deg.
    low = terrain_visibility(pyramid, grid, sun_alt_apparent_deg=1.0, sun_az_deg=270.0)
    high = terrain_visibility(pyramid, grid, sun_alt_apparent_deg=4.0, sun_az_deg=270.0)
    land = np.ones(grid.shape, bool)
    land[1, 3] = False
    assert np.all(low.clear_fraction[land] == 0.0) and not low.best_clear[land].any()
    assert np.all(high.clear_fraction[land] == 1.0) and high.best_clear[land].all()
    assert np.isnan(low.clear_fraction[1, 3]) and np.isnan(high.best_margin_deg[1, 3])


def test_highest_point_can_see_over_the_wall() -> None:
    # Sun at 2.0 deg: blocked from 102 m (wall ~2.37 deg) but clear from the
    # 602 m hill: atan(398/21000) - 0.082 = ~1.0 deg.
    pyramid, grid = visibility_scene()
    v = terrain_visibility(pyramid, grid, sun_alt_apparent_deg=2.0, sun_az_deg=270.0)
    assert v.clear_fraction[0, 1] == 0.0
    assert v.highest_clear[0, 1] == 1.0 and v.best_clear[0, 1]
    assert v.best_elev_m[0, 1] == 600
    assert v.best_lat[0, 1] == pytest.approx(-0.005, abs=PX)
    assert v.best_lon[0, 1] == pytest.approx(-0.005, abs=PX)
    assert not v.best_clear[0, 0]  # neighbouring cell has no hill


# --- real DEM ----------------------------------------------------------------


@pytest.mark.dem
@pytest.mark.parametrize(
    "name, lat, lon, height, max_offset_m",
    [
        # Sharp rock summits: the highest pixel should be the summit pixel.
        ("Mulhacen", 37.0533, -3.3113, 3_479, 150.0),
        ("Aneto", 42.6308, 0.6578, 3_404, 150.0),
        # Ice-capped caldera rim (64 00 57 N, 16 40 29 W) on 6" x 3" tiles: the
        # highest ice pixel in the 2011-15 DSM sits ~250 m from the surveyed
        # summit. A georeferencing error on these narrower tiles would show up
        # as a whole-pixel-width multiple or km-scale offset, which this catches.
        ("Hvannadalshnukur", 64.01583, -16.67472, 2_110, 300.0),
    ],
)
def test_summit_heights(name: str, lat: float, lon: float, height: int, max_offset_m: float) -> None:
    lat_s, lon_w = math.floor(lat), math.floor(lon)
    if not (TILE_DIR / f"{tile_name(lat_s, lon_w)}.tif").exists():
        pytest.skip("GLO-90 tile not downloaded")
    dem = build_mosaic(lat_s, lat_s + 1, lon_w, lon_w + 1)
    # The highest pixel within ~0.5 km must sit close to the published summit.
    r = round((dem.lat0 - lat) / dem.dlat)
    c = round((lon - dem.lon0) / dem.dlon)
    w = 6
    win = np.asarray(dem.heights[r - w : r + w + 1, c - w : c + w + 1])
    i, j = np.unravel_index(win.argmax(), win.shape)
    m_per_deg = math.radians(1.0) * R_EARTH_M
    dy = (i - w) * dem.dlat * m_per_deg
    dx = (j - w) * dem.dlon * m_per_deg * math.cos(math.radians(lat))
    assert math.hypot(dx, dy) <= max_offset_m, name
    # GLO-90 heights are averages of the 30 m DEM, which flattens summits:
    # Mulhacen comes out 19 m low, Aneto 48 m, Hvannadalshnukur 29 m.
    assert height - 60 <= int(win.max()) <= height + 10, name
