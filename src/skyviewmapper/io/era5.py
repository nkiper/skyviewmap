"""ERA5 low/medium/high cloud cover: download from the CDS and arrange as samples.

Dataset: "ERA5 hourly data on single levels from 1940 to present"
(``reanalysis-era5-single-levels``), 0.25 deg, hourly instantaneous cloud
fractions (0-1) at the full hour UT. Needs a CDS API key in ~/.cdsapirc and
the dataset licence accepted on the CDS website.

The samples are kept as a cube (year, day, hour, latitude, longitude) rather
than averaged, because the probability of a clear line of sight must be
computed per sample before averaging (layers are correlated in time). ``day``
is the offset in days from the event date (-N..N), so windows may cross a
month boundary; they are requested one month at a time, because a CDS request
takes the full product of its month and day lists.
"""

from collections.abc import Sequence
from datetime import date
from itertools import groupby
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from ..event import Event
from ..regions import Region

DATASET = "reanalysis-era5-single-levels"
VARIABLES = {"low_cloud_cover": "lcc", "medium_cloud_cover": "mcc", "high_cloud_cover": "hcc"}

_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = _ROOT / "data" / "raw" / "era5"
CACHE_DIR = _ROOT / "data" / "processed"


def month_segments(event: Event, year: int) -> list[tuple[int, list[int]]]:
    """The climatology window in ``year`` as (month, [days]) runs, in date order."""
    dates = event.window_dates(year)
    return [(m, [d.day for d in grp]) for m, grp in groupby(dates, key=lambda d: d.month)]


def era5_request(region: Region, years: Sequence[int], month: int, days: Sequence[int]) -> dict[str, Any]:
    """CDS request body for the region's cloud-cover box, one month's days, all years, the region's hours."""
    lat_min, lat_max, lon_min, lon_max = region.era5_box
    return {
        "product_type": ["reanalysis"],
        "variable": list(VARIABLES),
        "year": [str(y) for y in years],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in days],
        "time": [f"{h:02d}:00" for h in region.era5_hours],
        "area": [lat_max, lon_min, lat_min, lon_max],  # N, W, S, E
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def _raw_path(event: Event, region: Region, years: Sequence[int], month: int) -> Path:
    return RAW_DIR / event.id / f"{region.name}_{years[0]}-{years[-1]}_m{month:02d}.nc"


def download_era5(event: Event, region: Region) -> list[Path]:
    """Download (if not already present) and return the raw NetCDF files for a region.

    One request per month segment of the window. If the CDS refuses a request
    as too large, that segment falls back to one request per decade.
    """
    import cdsapi  # imported here so the rest of the package works without CDS credentials

    years = list(event.year_list)
    # The month split does not depend on the year, except around 29 Feb.
    segments = month_segments(event, years[-1])
    client = cdsapi.Client()
    paths = []
    for month, days in segments:
        whole = _raw_path(event, region, years, month)
        if whole.exists():
            paths.append(whole)
            continue
        whole.parent.mkdir(parents=True, exist_ok=True)
        try:
            _retrieve(client, era5_request(region, years, month, days), whole)
            paths.append(whole)
            continue
        except Exception as err:  # the CDS reports cost-limit refusals as generic HTTP errors
            if "too large" not in str(err).lower() and "cost" not in str(err).lower():
                raise
        for start in range(0, len(years), 10):
            chunk = years[start : start + 10]
            path = _raw_path(event, region, chunk, month)
            if not path.exists():
                _retrieve(client, era5_request(region, chunk, month, days), path)
            paths.append(path)
    return paths


def _retrieve(client: Any, request: dict[str, Any], target: Path) -> None:
    tmp = target.with_suffix(".part")
    client.retrieve(DATASET, request).download(str(tmp))
    tmp.rename(target)


def to_sample_cube(ds: xr.Dataset, event: Event, hours: Sequence[int]) -> xr.Dataset:
    """Arrange an hourly ERA5 dataset as (year, day, hour, latitude, longitude), day = offset from the event date.

    Keeps lcc/mcc/hcc as float32, sorts latitude ascending, and raises if any
    expected (year, day, hour) is missing or any value is NaN.
    """
    time_dim = "valid_time" if "valid_time" in ds.dims else "time"
    t = pd.DatetimeIndex(ds[time_dim].values)
    offset = np.array(
        [(date(y, m, d) - event.date.replace(year=y)).days for y, m, d in zip(t.year, t.month, t.day)], dtype=int
    )
    index = pd.MultiIndex.from_arrays([t.year, offset, t.hour], names=["year", "day", "hour"])
    ds = ds[list(VARIABLES.values())].drop_vars(time_dim).assign_coords(
        xr.Coordinates.from_pandas_multiindex(index, time_dim)
    )
    days = list(range(-event.half_window_days, event.half_window_days + 1))
    expected = pd.MultiIndex.from_product([list(event.year_list), days, list(hours)], names=["year", "day", "hour"])
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


def load_cloud_samples(event: Event, region: Region, download: bool = True) -> xr.Dataset:
    """Cloud-cover sample cube for one region of an event, cached as NetCDF in data/processed/<event>/."""
    y0, y1 = event.years
    cache = CACHE_DIR / event.id / f"era5_{region.name}_{y0}-{y1}_pm{event.half_window_days}.nc"
    if cache.exists():
        return xr.load_dataset(cache)
    if download:
        paths = download_era5(event, region)
    else:
        years = list(event.year_list)
        paths = [_raw_path(event, region, years, m) for m, _ in month_segments(event, years[-1])]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(p)
    raw = xr.concat([xr.load_dataset(p) for p in paths], dim="valid_time")
    cube = to_sample_cube(raw, event, region.era5_hours)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cube.to_netcdf(cache)
    return cube
