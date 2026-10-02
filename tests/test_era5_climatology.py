"""ERA5 request/reshape and climatology tests.

Synthetic datasets need no downloads; tests marked ``era5`` use the real
cached cloud samples and skip if they are absent.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from skyviewmapper.climatology import layer_means, overhead_clear_probability
from skyviewmapper.io.era5 import era5_request, load_cloud_samples, to_sample_cube, window_days
from skyviewmapper.regions import REGIONS

# --- request -----------------------------------------------------------------


def test_window_days() -> None:
    assert window_days(7) == list(range(5, 20))
    assert window_days(0) == [12]


@pytest.mark.parametrize(
    "name, area, times",
    [
        ("iberia", [47.0, -15.0, 33.0, 7.0], ["18:00", "19:00"]),
        ("iceland", [68.0, -27.0, 62.0, -11.0], ["17:00", "18:00"]),
    ],
)
def test_era5_request(name: str, area: list[float], times: list[str]) -> None:
    req = era5_request(REGIONS[name], [1996, 2025], [11, 12])
    assert req["area"] == area  # N, W, S, E
    assert req["time"] == times
    assert req["year"] == ["1996", "2025"]
    assert req["month"] == ["08"] and req["day"] == ["11", "12"]
    assert req["variable"] == ["low_cloud_cover", "medium_cloud_cover", "high_cloud_cover"]
    assert req["data_format"] == "netcdf"


# --- reshaping ---------------------------------------------------------------

YEARS, DAYS, HOURS = [2000, 2001], [11, 12], [18, 19]


def synthetic_hourly(drop: int | None = None) -> xr.Dataset:
    """Hourly dataset in the CDS layout; lcc encodes its own (year, day, hour)."""
    times = pd.DatetimeIndex(
        [pd.Timestamp(y, 8, d, h) for y in YEARS for d in DAYS for h in HOURS]
    )
    if drop is not None:
        times = times.delete(drop)
    code = (times.year - 2000) * 0.1 + (times.day - 11) * 0.01 + (times.hour - 18) * 0.001
    lat = np.array([41.0, 40.75])  # descending, as delivered by the CDS
    lon = np.array([-4.0, -3.75, -3.5])
    shape = (len(times), lat.size, lon.size)
    lcc = np.broadcast_to(np.asarray(code)[:, None, None], shape).copy()
    lcc[:, 0, :] += 0.5  # northern row differs, to check latitude sorting
    return xr.Dataset(
        {
            "lcc": (("valid_time", "latitude", "longitude"), lcc),
            "mcc": (("valid_time", "latitude", "longitude"), np.full(shape, 0.2)),
            "hcc": (("valid_time", "latitude", "longitude"), np.zeros(shape)),
        },
        coords={"valid_time": times, "latitude": lat, "longitude": lon, "number": 0, "expver": ("valid_time", ["0001"] * len(times))},
    )


def test_to_sample_cube_places_every_sample() -> None:
    cube = to_sample_cube(synthetic_hourly(), YEARS, DAYS, HOURS)
    assert dict(cube.sizes) == {"year": 2, "day": 2, "hour": 2, "latitude": 2, "longitude": 3}
    assert cube["latitude"].values.tolist() == [40.75, 41.0]  # ascending
    assert cube["lcc"].dtype == np.float32
    # Southern row holds the bare code: 0.1 * (year - 2000) + 0.01 * (day - 11) + 0.001 * (hour - 18).
    v = cube["lcc"].sel(year=2001, day=12, hour=19, latitude=40.75, longitude=-3.5)
    assert float(v) == pytest.approx(0.111, abs=1e-6)
    assert float(cube["lcc"].sel(year=2000, day=11, hour=18, latitude=41.0, longitude=-4.0)) == pytest.approx(0.5)
    assert "expver" not in cube.coords and "number" not in cube.coords


def test_to_sample_cube_missing_sample_raises() -> None:
    with pytest.raises(ValueError, match="missing"):
        to_sample_cube(synthetic_hourly(drop=3), YEARS, DAYS, HOURS)


def test_to_sample_cube_nan_raises() -> None:
    ds = synthetic_hourly()
    ds["hcc"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        to_sample_cube(ds, YEARS, DAYS, HOURS)


# --- climatology -------------------------------------------------------------


def cube(lcc: list[float], mcc: list[float], hcc: list[float]) -> xr.Dataset:
    """Single-point cube whose samples run along ``year``."""
    def var(v: list[float]) -> tuple[tuple[str, ...], np.ndarray]:
        a = np.asarray(v, dtype=float).reshape(-1, 1, 1, 1, 1)
        return ("year", "day", "hour", "latitude", "longitude"), a

    return xr.Dataset(
        {"lcc": var(lcc), "mcc": var(mcc), "hcc": var(hcc)},
        coords={"year": np.arange(len(lcc)), "latitude": [40.0], "longitude": [-3.0]},
    )


def test_constant_layers_hand_value() -> None:
    # (1 - 0.5)(1 - 0.2)(1 - 0) = 0.4
    c = cube([0.5, 0.5], [0.2, 0.2], [0.0, 0.0])
    assert float(overhead_clear_probability(c).squeeze()) == pytest.approx(0.4)
    means = layer_means(c)
    assert float(means["lcc"].squeeze()) == pytest.approx(0.5)


def test_correlated_layers_average_per_sample() -> None:
    # One fully overcast sample and one fully clear one: P(clear) = 0.5.
    # Product of the layer means would give (1 - 0.5)(1 - 0.5) = 0.25.
    c = cube([1.0, 0.0], [1.0, 0.0], [0.0, 0.0])
    assert float(overhead_clear_probability(c).squeeze()) == pytest.approx(0.5)
    m = layer_means(c)
    product_of_means = float(((1 - m["lcc"]) * (1 - m["mcc"]) * (1 - m["hcc"])).squeeze())
    assert product_of_means == pytest.approx(0.25)


# --- real data ---------------------------------------------------------------


@pytest.mark.era5
@pytest.mark.parametrize("name", list(REGIONS))
def test_real_samples(name: str) -> None:
    region = REGIONS[name]
    try:
        samples = load_cloud_samples(region, download=False)
    except FileNotFoundError:
        pytest.skip("ERA5 samples not downloaded")
    assert dict(samples.sizes)["year"] == 30 and dict(samples.sizes)["day"] == 15
    assert dict(samples.sizes)["hour"] == len(region.era5_hours)
    lat_min, lat_max, lon_min, lon_max = region.era5_box
    assert float(samples["latitude"].min()) == lat_min and float(samples["latitude"].max()) == lat_max
    assert float(samples["longitude"].min()) == lon_min and float(samples["longitude"].max()) == lon_max
    for v in ("lcc", "mcc", "hcc"):
        assert float(samples[v].min()) >= 0.0 and float(samples[v].max()) <= 1.0
