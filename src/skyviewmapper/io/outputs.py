"""Write a region's results: NetCDF, GeoTIFFs and a table of top spots."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.transform import from_origin

from ..constants import R_EARTH_M

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "outputs"
GEOTIFF_VARIABLES = ("p_clear_view", "p_clear_view_no_slant", "totality_s")
TOP_SPOT_COLUMNS = (
    "best_lat",
    "best_lon",
    "best_elev_m",
    "p_clear_view",
    "p_clear_view_no_slant",
    "p_clear_sky",
    "clear_fraction",
    "land_fraction",
    "t_max_utc",
    "sun_alt_apparent_deg",
    "sun_az_deg",
    "totality_s",
)


def write_netcdf(ds: xr.Dataset, path: Path) -> Path:
    """NetCDF has no boolean type: masks are stored as int8 (1 = true)."""
    out = ds.copy()
    for name, da in ds.data_vars.items():
        if da.dtype == bool:
            out[name] = da.astype(np.int8)
            out[name].attrs = da.attrs
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(path)
    return path


def write_geotiff(da: xr.DataArray, path: Path) -> Path:
    """Single-band float32 GeoTIFF in EPSG:4326 (rows north to south, NaN = nodata)."""
    lat = da["lat"].values
    lon = da["lon"].values
    dlat = float(lat[1] - lat[0])
    dlon = float(lon[1] - lon[0])
    north = float(lat[-1]) + dlat / 2
    west = float(lon[0]) - dlon / 2
    data = np.flipud(np.asarray(da.values, dtype=np.float32))
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(west, north, dlon, dlat),
        nodata=float("nan"),
        compress="deflate",
    ) as dst:
        dst.write(data, 1)
        dst.update_tags(1, long_name=str(da.attrs.get("long_name", "")), units=str(da.attrs.get("units", "")))
    return path


def _distance_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance (haversine) on the model sphere."""
    p1, p2 = math.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * R_EARTH_M / 1_000.0 * np.arcsin(np.sqrt(a))


def top_spots(
    ds: xr.Dataset,
    n: int = 25,
    min_separation_km: float = 20.0,
    min_totality_s: float = 60.0,
    min_land_fraction: float = 1.0 / 3.0,
) -> pd.DataFrame:
    """Best cells by ``p_clear_view`` with at least ``min_totality_s`` of totality, kept ``min_separation_km`` apart.

    The totality threshold keeps the list away from the edges of the path,
    where totality lasts only seconds and the exact limit is uncertain (lunar
    limb profile, Delta T). ``min_land_fraction`` (share of the cell's typical
    spots on land) drops offshore rocks and islets, such as Eldey off
    Reykjanes, that qualify only through a single land pixel.
    """
    p = ds["p_clear_view"].values
    ok = (
        np.isfinite(p)
        & ds["in_totality"].values.astype(bool)
        & (ds["totality_s"].values >= min_totality_s)
        & (ds["land_fraction"].values >= min_land_fraction - 1e-9)
    )
    flat = np.flatnonzero(ok)
    order = flat[np.argsort(-p.ravel()[flat], kind="stable")]
    lat = ds["best_lat"].values.ravel()
    lon = ds["best_lon"].values.ravel()
    chosen: list[int] = []
    for i in order:
        if len(chosen) == n:
            break
        if chosen and _distance_km(float(lat[i]), float(lon[i]), lat[chosen], lon[chosen]).min() < min_separation_km:
            continue
        chosen.append(int(i))
    table = pd.DataFrame({c: ds[c].values.ravel()[chosen] for c in TOP_SPOT_COLUMNS})
    table.insert(0, "rank", np.arange(1, len(chosen) + 1))
    table.attrs["min_totality_s"] = min_totality_s
    return table


def write_outputs(ds: xr.Dataset, out_dir: Path | None = None, min_totality_s: float = 60.0) -> list[Path]:
    """Write NetCDF, GeoTIFFs, PNG maps and the top-spots CSV for one region."""
    from ..plots import write_maps  # matplotlib only needed here

    region = str(ds.attrs["region"])
    out_dir = (out_dir or OUTPUT_DIR) / region
    paths = [write_netcdf(ds, out_dir / f"skyview_{region}.nc")]
    paths += [write_geotiff(ds[v], out_dir / f"{v}_{region}.tif") for v in GEOTIFF_VARIABLES]
    csv = out_dir / f"top_spots_{region}.csv"
    spots = top_spots(ds, min_totality_s=min_totality_s)
    spots.to_csv(csv, index=False, float_format="%.4f")
    paths.append(csv)
    paths += write_maps(ds, out_dir, spots)
    return paths
