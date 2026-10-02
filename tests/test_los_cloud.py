"""Line-of-sight cloud tests: hand values and synthetic ERA5-like cubes.

Tests marked ``era5`` use the real cloud samples and skip if absent.
"""

import math

import numpy as np
import pytest
import xarray as xr

from skyviewmapper.grid import Grid, resample_bilinear
from skyviewmapper.io.era5 import load_cloud_samples
from skyviewmapper.los_cloud import bilinear_weights, los_cloud, slant_clear_fraction, time_weight
from skyviewmapper.regions import REGIONS
from skyviewmapper.terrain import Dem, cell_ground_height

T_1830 = np.datetime64("2026-08-12T18:30", "ms")

# --- pure helpers ------------------------------------------------------------


def test_slant_clear_fraction_hand_values() -> None:
    assert slant_clear_fraction(0.3, 30.0, 0.0) == pytest.approx(0.7)  # no correction
    assert slant_clear_fraction(0.3, 90.0, 0.5) == pytest.approx(0.7)  # overhead
    # n = 0.1, 3 deg, beta = 0.5: exponent 1 + (2/pi) cot 3 = 13.1474; 0.9^13.1474 = 0.25027
    assert slant_clear_fraction(0.1, 3.0, 0.5) == pytest.approx(0.25026876, abs=1e-7)
    assert slant_clear_fraction(0.0, 1.0, 0.5) == 1.0 and slant_clear_fraction(1.0, 60.0, 0.5) == 0.0


def test_slant_clear_fraction_floors_elevation() -> None:
    assert slant_clear_fraction(0.1, -2.0, 0.5) == slant_clear_fraction(0.1, 0.5, 0.5)


def test_bilinear_weights() -> None:
    glat = np.array([40.0, 40.25, 40.5])
    glon = np.array([-1.0, -0.75])
    idx, w = bilinear_weights(40.25, -1.0, glat, glon)  # on a node
    assert np.dot(w, np.arange(6)[idx]) == pytest.approx(2.0)  # flat index of (1, 0)
    idx, w = bilinear_weights(40.125, -0.875, glat, glon)  # centre of the first cell
    np.testing.assert_allclose(w, 0.25)
    assert sorted(idx.tolist()) == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="outside"):
        bilinear_weights(40.6, -1.0, glat, glon)


def test_time_weight() -> None:
    h18, h19 = np.datetime64("2026-08-12T18:00", "ms"), np.datetime64("2026-08-12T19:00", "ms")
    t = np.array(["2026-08-12T18:30", "2026-08-12T18:00", "2026-08-12T19:15"], dtype="datetime64[ms]")
    np.testing.assert_allclose(time_weight(t, h18, h19), [0.5, 0.0, 1.0])


def test_resample_bilinear_reproduces_linear_field() -> None:
    src = Grid(0.0, 1.0, 0.0, 2.0, 0.25)
    dst = Grid(0.0, 1.0, 0.0, 2.0, 0.05)
    lat, lon = src.mesh()
    out = resample_bilinear(2 * lat + 3 * lon, src, dst)
    dlat, dlon = dst.mesh()
    inside = (dlat >= src.lats[0]) & (dlat <= src.lats[-1]) & (dlon >= src.lons[0]) & (dlon <= src.lons[-1])
    np.testing.assert_allclose(out[inside], (2 * dlat + 3 * dlon)[inside], atol=1e-12)
    # Beyond the outermost source centres: edge value (clamped position).
    assert out[0, 0] == pytest.approx(2 * src.lats[0] + 3 * src.lons[0])


def test_cell_ground_height() -> None:
    px = 1.0 / 1200
    h = np.zeros((12, 12), dtype=np.int16)
    h[:6] = 300  # northern half of the cell
    dem = Dem(h, lat0=0.01 - px / 2, lon0=px / 2, dlat=px, dlon=px)
    # Typical spots sit in rows 2, 6, 10: one row at 300 m, two at 0 m -> median 0.
    assert cell_ground_height(dem, Grid(0.0, 0.01, 0.0, 0.01, 0.01))[0, 0] == 0.0
    h[:8] = 300
    assert cell_ground_height(dem, Grid(0.0, 0.01, 0.0, 0.01, 0.01))[0, 0] == 300.0


def test_region_cloud_grid() -> None:
    assert REGIONS["iberia"].cloud_grid.shape == (160, 290)
    assert REGIONS["iceland"].cloud_grid.shape == (80, 120)


# --- synthetic cubes ---------------------------------------------------------

LAT = np.arange(30.0, 50.01, 0.25)
LON = np.arange(-20.0, 10.01, 0.25)


def cube(lcc: np.ndarray, mcc: np.ndarray, hcc: np.ndarray) -> xr.Dataset:
    """Arrays shaped (year, day, hour, lat, lon) -> ERA5-like sample cube with hours 18, 19."""
    dims = ("year", "day", "hour", "latitude", "longitude")
    coords = {
        "year": np.arange(lcc.shape[0]),
        "day": np.arange(lcc.shape[1]),
        "hour": [18, 19],
        "latitude": LAT,
        "longitude": LON,
    }
    return xr.Dataset({"lcc": (dims, lcc), "mcc": (dims, mcc), "hcc": (dims, hcc)}, coords=coords)


def uniform(value: float, n_samples: int = 1) -> np.ndarray:
    return np.full((n_samples, 1, 2, LAT.size, LON.size), value, dtype=np.float32)


def test_uniform_layers_hand_value() -> None:
    # Sun at 5 deg, low 0.2 / mid 0.1 / high 0.3:
    # plain 0.8 * 0.9 * 0.7 = 0.504; slant (beta .5/.3/.1) = 0.037329 (see hand calc in plan).
    s = cube(uniform(0.2), uniform(0.1), uniform(0.3))
    r = los_cloud(s, 40.0, -3.0, 600.0, T_1830, 5.0, 280.0)
    assert r.p_clear_sky_no_slant == pytest.approx(0.504, abs=1e-6)
    assert r.p_clear_sky == pytest.approx(0.0373291638, abs=1e-6)
    assert r.layer_cloud["hcc"] == pytest.approx(0.3, abs=1e-6)


def test_cloud_band_west_blocks_only_low_sun() -> None:
    # High cloud only between 2.75 W and 0.75 W. With the Sun at 3 deg due west
    # from 40 N 0 E, the high layer (7.5-10.5 km) is crossed ~1.4-2.0 deg west;
    # with the Sun at 80 deg it is crossed within ~2 km of the observer.
    hcc = uniform(0.0)
    band = (LON >= -2.75) & (LON <= -0.75)
    hcc[..., band] = 1.0
    s = cube(uniform(0.0), uniform(0.0), hcc)
    r = los_cloud(s, [40.0, 40.0], [0.0, 0.0], 0.0, T_1830, [3.0, 80.0], 270.0)
    lon_cross = -r.crossing_km["hcc"][0] / (111.2 * math.cos(math.radians(40)))
    assert -2.5 < lon_cross < -1.0  # all bilinear nodes inside the band
    assert r.p_clear_sky_no_slant[0] == pytest.approx(0.0) and r.p_clear_sky[0] == pytest.approx(0.0)
    assert r.p_clear_sky_no_slant[1] == pytest.approx(1.0) and r.p_clear_sky[1] == pytest.approx(1.0)


def test_correlated_samples_are_averaged_per_sample() -> None:
    # Sample 0 fully overcast in every layer, sample 1 fully clear: P = 0.5
    # (product of layer means would be 0.5^3 = 0.125).
    lcc = np.concatenate([uniform(1.0), uniform(0.0)])
    s = cube(lcc, lcc.copy(), lcc.copy())
    r = los_cloud(s, 40.0, -3.0, 0.0, T_1830, 5.0, 280.0)
    assert r.p_clear_sky_no_slant == pytest.approx(0.5) and r.p_clear_sky == pytest.approx(0.5)


def test_time_interpolation_between_hours() -> None:
    lcc = uniform(0.0)
    lcc[:, :, 1] = 1.0  # clear at 18 UT, overcast at 19 UT
    s = cube(lcc, uniform(0.0), uniform(0.0))
    t = np.array(["2026-08-12T18:30", "2026-08-12T18:00", "2026-08-12T19:15"], dtype="datetime64[ms]")
    r = los_cloud(s, 40.0, -3.0, 0.0, t, 5.0, 280.0)
    np.testing.assert_allclose(r.p_clear_sky_no_slant, [0.5, 1.0, 0.0], atol=1e-6)


def test_crossing_outside_box_raises() -> None:
    s = cube(uniform(0.0), uniform(0.0), uniform(0.0))
    with pytest.raises(ValueError, match="outside the ERA5 box"):
        los_cloud(s, 40.0, -19.5, 0.0, T_1830, 2.0, 270.0)  # sight line runs off the west edge


# --- real data ---------------------------------------------------------------


@pytest.mark.era5
def test_real_iberia_cities_bounds() -> None:
    region = REGIONS["iberia"]
    try:
        samples = load_cloud_samples(region, download=False)
    except FileNotFoundError:
        pytest.skip("ERA5 samples not downloaded")
    lat = np.array([40.42, 39.47, 39.57, 43.46])  # Madrid, Valencia, Palma, Santander
    lon = np.array([-3.70, -0.38, 2.65, -3.81])
    t = np.full(4, T_1830)
    alt = np.array([7.3, 4.6, 2.6, 9.0])
    r = los_cloud(samples, lat, lon, 0.0, t, alt, 285.0)
    assert np.all((0 <= r.p_clear_sky) & (r.p_clear_sky <= r.p_clear_sky_no_slant + 1e-12))
    assert np.all(r.p_clear_sky_no_slant <= 1.0)
