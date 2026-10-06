"""Combination, top spots, writers and an end-to-end smoke test."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr

from skyviewmapper.grid import Grid
from skyviewmapper.io.outputs import top_spots, write_geotiff, write_netcdf
from skyviewmapper.plots import write_maps
from skyviewmapper.visibility import combine_visibility

from conftest import event_2026, region_2026

# --- combination -------------------------------------------------------------


def test_combine_visibility_hand_cases() -> None:
    # Cells: clear land in path, blocked land, sea, land outside the path.
    p = np.array([0.6, 0.6, 0.6, 0.4])
    p0 = np.array([0.8, 0.8, 0.8, 0.5])
    margin = np.array([2.0, -1.0, np.nan, 0.5])
    frac = np.array([0.5, 0.0, np.nan, 1.0])
    tot = np.array([90.0, 90.0, 90.0, 0.0])
    v = combine_visibility(p, p0, margin, frac, tot)
    np.testing.assert_allclose(v["p_clear_view"], [0.6, np.nan, np.nan, 0.4])
    np.testing.assert_allclose(v["p_clear_view_no_slant"], [0.8, np.nan, np.nan, 0.5])
    np.testing.assert_allclose(v["p_clear_view_typical"], [0.3, 0.0, np.nan, 0.4])
    assert v["terrain_blocked"].tolist() == [False, True, False, False]
    assert v["in_totality"].tolist() == [True, True, True, False]


# --- synthetic dataset -------------------------------------------------------


def synthetic_ds() -> xr.Dataset:
    """A 0.1 deg x 0.2 deg patch at 0.01 deg resolution with every output variable."""
    g = Grid(42.0, 42.1, -4.0, -3.8, 0.01)
    lat, lon = g.mesh()
    shape = g.shape
    p = 0.2 + 0.6 * (lon - lon.min()) / np.ptp(lon)  # increases eastwards
    margin = np.full(shape, 3.0)
    margin[0, 0] = -1.0  # one blocked cell
    margin[-1, -1] = np.nan  # one sea cell
    v = combine_visibility(p, p + 0.1, margin, np.full(shape, 1.0), np.where(lat > 42.03, 100.0, 0.0))
    data = {
        **v,
        "best_margin_deg": margin,
        "best_lat": lat,
        "best_lon": lon,
        "best_elev_m": np.full(shape, 800.0),
        "p_clear_sky": p,
        "clear_fraction": np.full(shape, 1.0),
        "land_fraction": np.full(shape, 1.0),
        "t_max_utc": np.full(shape, np.datetime64("2026-08-12T18:29", "ms")),
        "sun_alt_apparent_deg": np.full(shape, 8.0),
        "sun_az_deg": np.full(shape, 282.0),
        "central_s": np.where(lat > 42.03, 100.0, 0.0),
    }
    return xr.Dataset(
        {k: (("lat", "lon"), a) for k, a in data.items()},
        coords={"lat": g.lats, "lon": g.lons},
        attrs={
            "event": "test",
            "event_name": "Test eclipse",
            "central_word": "totality",
            "region": "testpatch",
            "region_label": "Test patch",
            "cloud_data": "synthetic",
            "terrain_data": "synthetic",
        },
    )


def test_top_spots_sorted_separated_and_in_path() -> None:
    ds = synthetic_ds()
    table = top_spots(ds, n=5, min_separation_km=5.0)
    assert len(table) >= 2
    assert (np.diff(table["p_clear_view"].to_numpy()) <= 0).all()
    assert (table["central_s"] > 0).all()
    lat, lon = np.radians(table["best_lat"].to_numpy()), np.radians(table["best_lon"].to_numpy())
    for i in range(len(table)):
        for j in range(i):
            d = 6371.0 * np.arccos(
                np.clip(np.sin(lat[i]) * np.sin(lat[j]) + np.cos(lat[i]) * np.cos(lat[j]) * np.cos(lon[i] - lon[j]), -1, 1)
            )
            assert d >= 5.0
    # The best cell is at the eastern edge (p rises eastwards), inside the path.
    assert table["best_lon"].iloc[0] == pytest.approx(ds["lon"].values.max(), abs=0.011)


def test_top_spots_break_ties_by_central_duration() -> None:
    ds = synthetic_ds()
    ds["p_clear_view"][:] = np.where(np.isfinite(ds["p_clear_view"].values), 0.9984, np.nan)
    ds["p_clear_view"][-2, 0] = 0.9991  # rounds to the same 100 %
    ds["central_s"][:] = np.where(ds["lat"].values[:, None] > 42.03, 100.0, 0.0)
    ds["central_s"][-3, 5] = 150.0  # longest totality
    best = top_spots(ds, n=1)
    assert best["central_s"].iloc[0] == 150.0


def test_top_spots_skip_islets() -> None:
    ds = synthetic_ds()
    best = top_spots(ds, n=1)
    i = int(np.argmin(np.abs(ds["lat"].values - best["best_lat"].iloc[0])))
    j = int(np.argmin(np.abs(ds["lon"].values - best["best_lon"].iloc[0])))
    ds["land_fraction"][i, j] = 1.0 / 9.0  # the best cell becomes a rock with one land spot
    assert (top_spots(ds, n=1)["best_lat"].iloc[0], top_spots(ds, n=1)["best_lon"].iloc[0]) != (
        best["best_lat"].iloc[0],
        best["best_lon"].iloc[0],
    )


def test_top_spots_respect_min_central() -> None:
    ds = synthetic_ds()  # 100 s of totality north of 42.03 N, none south
    assert len(top_spots(ds, min_central_s=120.0)) == 0
    assert len(top_spots(ds, n=3, min_separation_km=1.0, min_central_s=60.0)) == 3


def test_write_netcdf_roundtrip(tmp_path: Path) -> None:
    ds = synthetic_ds()
    path = write_netcdf(ds, tmp_path / "x.nc")
    back = xr.load_dataset(path)
    assert back["terrain_blocked"].dtype == np.int8 and int(back["terrain_blocked"][0, 0]) == 1
    np.testing.assert_allclose(back["p_clear_view"].values, ds["p_clear_view"].values)
    assert back["t_max_utc"].dtype.kind == "M"


def test_write_geotiff_georeference(tmp_path: Path) -> None:
    ds = synthetic_ds()
    path = write_geotiff(ds["p_clear_view"], tmp_path / "p.tif")
    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 4326
        b = src.bounds
        assert (b.left, b.right, b.bottom, b.top) == pytest.approx((-4.0, -3.8, 42.0, 42.1))
        band = src.read(1)
    # Row 0 is north: the northern row of the dataset.
    np.testing.assert_allclose(band[0], ds["p_clear_view"].values[-1].astype(np.float32), equal_nan=True)


def test_write_outputs_uses_readable_folder_and_file_names(tmp_path: Path) -> None:
    from skyviewmapper.io.outputs import write_outputs

    ds = synthetic_ds()
    ds.attrs["region_slug"] = "test-patch"
    paths = write_outputs(ds, tmp_path, min_central_s=60.0)
    assert {p.parent for p in paths} == {tmp_path / "test" / "test-patch"}
    assert all("test-patch" in p.name for p in paths)
    assert not any("testpatch" in p.name for p in paths)  # the internal name is not used


def test_write_maps(tmp_path: Path) -> None:
    ds = synthetic_ds()
    paths = write_maps(ds, tmp_path, top_spots(ds, n=5, min_separation_km=1.0))
    assert len(paths) == 4
    for p in paths:
        assert p.exists() and p.stat().st_size > 10_000


# --- end-to-end smoke test ---------------------------------------------------


@pytest.mark.ephemeris
@pytest.mark.dem
@pytest.mark.era5
def test_run_region_smoke() -> None:
    from skyviewmapper.pipeline import run_region

    # A 0.2 deg patch near Burgos, reusing Iberia's cached DEM and cloud data.
    region = dataclasses.replace(region_2026("iberia"), grid=Grid(42.2, 42.4, -3.8, -3.6, 0.01))
    try:
        ds = run_region(event_2026(), region, download=False)
    except FileNotFoundError:
        pytest.skip("data not downloaded")
    assert dict(ds.sizes) == {"lat": 20, "lon": 20}
    for v in ("p_clear_view", "p_clear_view_no_slant", "p_clear_sky", "p_clear_sky_no_slant"):
        vals = ds[v].values
        assert np.nanmin(vals) >= 0.0 and np.nanmax(vals) <= 1.0, v
    assert np.all(ds["p_clear_sky"].values <= ds["p_clear_sky_no_slant"].values + 1e-12)
    assert ds["in_totality"].values.all()  # Burgos is inside the path
    assert 400 < float(ds["ground_h_m"].mean()) < 1_200  # Burgos plateau, ~850 m
