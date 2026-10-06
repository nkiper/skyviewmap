"""Grid, region and DEM-band tests (no data needed)."""

import numpy as np
import pytest

from skyviewmapper.grid import Grid
from skyviewmapper.io.dem import build_mosaic, lon_width_factor, tile_cols

from conftest import event_2026


def test_grid_square_cells() -> None:
    g = Grid(36.0, 44.0, -10.0, 4.5, 0.01)
    assert g.shape == (800, 1450)
    assert g.lats[0] == pytest.approx(36.005) and g.lons[-1] == pytest.approx(4.495)


def test_grid_separate_longitude_resolution() -> None:
    g = Grid(63.0, 67.0, -25.0, -13.0, 0.01, res_lon=0.02)
    assert g.shape == (400, 600)
    assert g.lons[0] == pytest.approx(-24.99)
    lat, lon = g.mesh()
    assert lat.shape == lon.shape == (400, 600)
    np.testing.assert_allclose(np.diff(lon[0]), 0.02)


@pytest.mark.parametrize(
    "lat_south, factor, cols",
    [
        (40, 1.0, 1200),  # Iberia: 3" x 3"
        (-50, 1.0, 1200),  # southern tile S50 spans -50..-49: equatorward edge 49
        (55, 1.5, 800),
        (64, 2.0, 600),  # Iceland: 6" in longitude
        (75, 3.0, 400),
        (82, 5.0, 240),
        (87, 10.0, 120),
    ],
)
def test_copernicus_latitude_bands(lat_south: int, factor: float, cols: int) -> None:
    assert lon_width_factor(lat_south) == factor
    assert tile_cols(lat_south) == cols


def test_mosaic_refuses_mixed_bands() -> None:
    with pytest.raises(ValueError, match="latitude bands"):
        build_mosaic(49.0, 51.0, 0.0, 1.0)


@pytest.mark.parametrize("name", ["iberia", "iceland"])
def test_regions_are_consistent(name: str) -> None:
    r = next(x for x in event_2026().explicit_regions if x.name == name)
    lat_min, lat_max, lon_min, lon_max = r.dem_box
    g = r.grid
    # DEM covers the grid with a margin, within a single Copernicus band.
    assert lat_min < g.lat_min and lat_max > g.lat_max and lon_min < g.lon_min and lon_max >= g.lon_max
    assert len({tile_cols(lat) for lat in range(int(np.floor(lat_min)), int(np.ceil(lat_max)))}) == 1
    # Cells are a whole number of DEM pixels (3" in latitude; band width in longitude).
    px_lon = 1.0 / tile_cols(int(np.floor(lat_min)))
    assert (g.dlat * 1200) == pytest.approx(round(g.dlat * 1200))
    assert (g.dlon / px_lon) == pytest.approx(round(g.dlon / px_lon))
    assert r.window_utc[0] < r.window_utc[1]
