"""ERA5 low/medium/high cloud cover: download from the CDS and arrange as samples.

Dataset: "ERA5 hourly data on single levels from 1940 to present"
(``reanalysis-era5-single-levels``), 0.25 deg, hourly instantaneous cloud
fractions (0-1) at the full hour UT. Needs a CDS API key in ~/.cdsapirc and
the dataset licence accepted on the CDS website.

The samples are kept as a cube (year, day, hour, latitude, longitude) rather
than averaged, because the probability of a clear line of sight must be
computed per sample before averaging (layers are correlated in time).
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from ..regions import Region

DATASET = "reanalysis-era5-single-levels"
VARIABLES = {"low_cloud_cover": "lcc", "medium_cloud_cover": "mcc", "high_cloud_cover": "hcc"}
DEFAULT_YEARS = tuple(range(1996, 2026))  # latest 30 complete years
EVENT_MONTH, EVENT_DAY = 8, 12
DEFAULT_HALF_WINDOW_DAYS = 7

_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = _ROOT / "data" / "raw" / "era5"
CACHE_DIR = _ROOT / "data" / "processed"


def window_days(half_window: int = DEFAULT_HALF_WINDOW_DAYS) -> list[int]:
    """Days of August within +-``half_window`` of the 12th."""
    return list(range(EVENT_DAY - half_window, EVENT_DAY + half_window + 1))


def era5_request(region: Region, years: Sequence[int], days: Sequence[int]) -> dict[str, Any]:
    """CDS request body for the region's cloud-cover box, years, days and hours."""
    lat_min, lat_max, lon_min, lon_max = region.era5_box
    return {
        "product_type": ["reanalysis"],
        "variable": list(VARIABLES),
        "year": [str(y) for y in years],
        "month": [f"{EVENT_MONTH:02d}"],
        "day": [f"{d:02d}" for d in days],
        "time": [f"{h:02d}:00" for h in region.era5_hours],
        "area": [lat_max, lon_min, lat_min, lon_max],  # N, W, S, E
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def _raw_path(region: Region, years: Sequence[int], half_window: int) -> Path:
    return RAW_DIR / f"{region.name}_{years[0]}-{years[-1]}_aug{EVENT_DAY}pm{half_window}.nc"


def download_era5(
    region: Region, years: Sequence[int] = DEFAULT_YEARS, half_window: int = DEFAULT_HALF_WINDOW_DAYS
) -> list[Path]:
    """Download (if not already present) and return the raw NetCDF file(s) for a region.

    Tries one request for all years; if the CDS refuses it as too large,
    falls back to one request per decade.
    """
    import cdsapi  # imported here so the rest of the package works without CDS credentials

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    days = window_days(half_window)
    whole = _raw_path(region, years, half_window)
    if whole.exists():
        return [whole]
    client = cdsapi.Client()
    try:
        _retrieve(client, era5_request(region, years, days), whole)
        return [whole]
    except Exception as err:  # the CDS reports cost-limit refusals as generic HTTP errors
        if "too large" not in str(err).lower() and "cost" not in str(err).lower():
            raise
    paths = []
    for start in range(0, len(years), 10):
        chunk = list(years[start : start + 10])
        path = _raw_path(region, chunk, half_window)
        if not path.exists():
            _retrieve(client, era5_request(region, chunk, days), path)
        paths.append(path)
    return paths


def _retrieve(client: Any, request: dict[str, Any], target: Path) -> None:
    tmp = target.with_suffix(".part")
    client.retrieve(DATASET, request).download(str(tmp))
    tmp.rename(target)


def to_sample_cube(ds: xr.Dataset, years: Sequence[int], days: Sequence[int], hours: Sequence[int]) -> xr.Dataset:
    """Arrange an hourly ERA5 dataset as (year, day, hour, latitude, longitude).

    Keeps lcc/mcc/hcc as float32, sorts latitude ascending, and raises if any
    expected (year, day, hour) is missing or any value is NaN.
    """
    time_dim = "valid_time" if "valid_time" in ds.dims else "time"
    t = pd.DatetimeIndex(ds[time_dim].values)
    index = pd.MultiIndex.from_arrays([t.year, t.day, t.hour], names=["year", "day", "hour"])
    ds = ds[list(VARIABLES.values())].drop_vars(time_dim).assign_coords(
        xr.Coordinates.from_pandas_multiindex(index, time_dim)
    )
    expected = pd.MultiIndex.from_product([list(years), list(days), list(hours)], names=["year", "day", "hour"])
    missing = expected[~expected.isin(ds.indexes[time_dim])]
    if len(missing):
        raise ValueError(f"{len(missing)} expected samples missing, e.g. {list(missing[:3])}")
    ds = ds.sel({time_dim: expected}).unstack(time_dim)
    ds = ds.transpose("year", "day", "hour", "latitude", "longitude").sortby("latitude")
    ds = ds.drop_vars([v for v in ("number", "expver") if v in ds.coords])
    ds = ds.astype(np.float32)
    for name, da in ds.data_vars.items():
        if bool(da.isnull().any()):
            raise ValueError(f"NaN values in {name}")
    return ds


def load_cloud_samples(
    region: Region,
    years: Sequence[int] = DEFAULT_YEARS,
    half_window: int = DEFAULT_HALF_WINDOW_DAYS,
    download: bool = True,
) -> xr.Dataset:
    """Cloud-cover sample cube for a region, cached as NetCDF in data/processed/."""
    cache = CACHE_DIR / f"era5_{region.name}_{years[0]}-{years[-1]}_aug{EVENT_DAY}pm{half_window}.nc"
    if cache.exists():
        return xr.load_dataset(cache)
    if download:
        paths = download_era5(region, years, half_window)
    else:
        paths = [_raw_path(region, years, half_window)]
        if not paths[0].exists():
            raise FileNotFoundError(paths[0])
    raw = xr.concat([xr.load_dataset(p) for p in paths], dim="valid_time")
    cube = to_sample_cube(raw, years, window_days(half_window), region.era5_hours)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cube.to_netcdf(cache)
    return cube


if __name__ == "__main__":
    from ..regions import REGIONS

    for r in REGIONS.values():
        cube = load_cloud_samples(r)
        print(f"{r.name}: {dict(cube.sizes)}")
