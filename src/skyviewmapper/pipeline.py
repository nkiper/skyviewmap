"""End-to-end run for one region: eclipse, terrain, cloud, combined visibility."""

from datetime import datetime, timezone

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from .ephemeris import body_track, local_circumstances
from .grid import resample_bilinear
from .io.dem import load_dem
from .io.era5 import DEFAULT_HALF_WINDOW_DAYS, DEFAULT_YEARS, load_cloud_samples
from .io.skyfield_data import load_ephemeris
from .los_cloud import los_cloud
from .regions import Region
from .terrain import build_pyramid, cell_ground_height, terrain_visibility
from .visibility import combine_visibility

# name -> (units, description)
VARIABLES: dict[str, tuple[str, str]] = {
    "p_clear_view": ("1", "P(clear view): cloud-free line of sight (slant-corrected) and some spot in the cell sees the Sun; NaN where terrain blocks every spot"),
    "p_clear_view_no_slant": ("1", "as p_clear_view, cloud without the slant-path correction"),
    "p_clear_view_typical": ("1", "cloud-free line of sight (slant-corrected) x share of typical spots that see the Sun"),
    "p_clear_view_typical_no_slant": ("1", "as p_clear_view_typical, cloud without the slant-path correction"),
    "p_clear_sky": ("1", "P(cloud-free line of sight to the Sun), slant-corrected"),
    "p_clear_sky_no_slant": ("1", "P(cloud-free line of sight to the Sun), random overlap only"),
    "terrain_blocked": ("1", "1 where no spot in the cell sees the Sun over the terrain"),
    "in_totality": ("1", "1 inside the path of totality"),
    "clear_fraction": ("1", "share of the cell's 3x3 typical spots that see the Sun over the terrain"),
    "land_fraction": ("1", "share of the cell's 3x3 typical spots that are on land"),
    "best_margin_deg": ("degree", "best spot: Sun apparent altitude minus terrain horizon (positive = clear)"),
    "best_lat": ("degree_north", "latitude of the cell's best spot"),
    "best_lon": ("degree_east", "longitude of the cell's best spot"),
    "best_elev_m": ("m", "DEM height of the cell's best spot"),
    "highest_clear": ("1", "1 if the cell's highest pixel sees the Sun"),
    "ground_h_m": ("m", "median DEM height of the cell's typical spots"),
    "t_max_utc": ("", "time of maximum eclipse (UTC)"),
    "sun_alt_apparent_deg": ("degree", "Sun apparent altitude at maximum (standard refraction)"),
    "sun_alt_deg": ("degree", "Sun geometric altitude at maximum"),
    "sun_az_deg": ("degree", "Sun azimuth at maximum, clockwise from north"),
    "magnitude": ("1", "eclipse magnitude (fraction of the Sun's diameter covered)"),
    "obscuration": ("1", "fraction of the Sun's area covered"),
    "totality_s": ("s", "duration of totality (0 outside the path)"),
}


def run_region(
    region: Region,
    download: bool = True,
    years: tuple[int, ...] = DEFAULT_YEARS,
    half_window: int = DEFAULT_HALF_WINDOW_DAYS,
) -> xr.Dataset:
    """Compute every map variable for ``region`` on its map grid."""
    eph, ts = load_ephemeris(download=download)
    track = body_track(eph, ts, *region.window_utc)
    dem = load_dem(*region.dem_box, download=download)
    pyramid = build_pyramid(dem)
    samples = load_cloud_samples(region, years, half_window, download=download)

    # Eclipse and terrain on the map grid.
    grid = region.grid
    lat, lon = grid.mesh()
    ground = cell_ground_height(dem, grid)
    ecl = local_circumstances(lat, lon, ground, track)
    ter = terrain_visibility(pyramid, grid, ecl.sun_alt_apparent_deg, ecl.sun_az_deg)

    # Cloud on the coarse grid, then back to the map grid.
    cgrid = region.cloud_grid
    clat, clon = cgrid.mesh()
    cground = cell_ground_height(dem, cgrid)
    cecl = local_circumstances(clat, clon, cground, track)
    cloud = los_cloud(samples, clat, clon, cground, cecl.t_max_utc, cecl.sun_alt_apparent_deg, cecl.sun_az_deg)
    p_sky = np.clip(resample_bilinear(cloud.p_clear_sky, cgrid, grid), 0.0, 1.0)
    p_sky0 = np.clip(resample_bilinear(cloud.p_clear_sky_no_slant, cgrid, grid), 0.0, 1.0)

    vis = combine_visibility(p_sky, p_sky0, ter.best_margin_deg, ter.clear_fraction, ecl.totality_s)

    data: dict[str, NDArray[np.generic]] = {
        **vis,
        "p_clear_sky": p_sky,
        "p_clear_sky_no_slant": p_sky0,
        "clear_fraction": ter.clear_fraction,
        "land_fraction": ter.land_fraction,
        "best_margin_deg": ter.best_margin_deg,
        "best_lat": ter.best_lat,
        "best_lon": ter.best_lon,
        "best_elev_m": ter.best_elev_m,
        "highest_clear": ter.highest_clear,
        "ground_h_m": ground,
        "t_max_utc": ecl.t_max_utc,
        "sun_alt_apparent_deg": ecl.sun_alt_apparent_deg,
        "sun_alt_deg": ecl.sun_alt_deg,
        "sun_az_deg": ecl.sun_az_deg,
        "magnitude": ecl.magnitude,
        "obscuration": ecl.obscuration,
        "totality_s": ecl.totality_s,
    }
    ds = xr.Dataset(
        {
            # Datetimes get their units from xarray's encoding, so none is set here.
            name: (("lat", "lon"), data[name], {"long_name": VARIABLES[name][1]} | ({"units": VARIABLES[name][0]} if VARIABLES[name][0] else {}))
            for name in VARIABLES
        },
        coords={
            "lat": ("lat", grid.lats, {"units": "degree_north"}),
            "lon": ("lon", grid.lons, {"units": "degree_east"}),
        },
        attrs={
            "title": f"Visibility of the 12 Aug 2026 total solar eclipse: {region.name}",
            "region": region.name,
            "grid_res_deg": f"{grid.dlat} x {grid.dlon}",
            "cloud_data": f"ERA5 low/medium/high cloud cover, {years[0]}-{years[-1]}, Aug {12 - half_window}-{12 + half_window}, "
            f"{'/'.join(f'{h:02d}' for h in region.era5_hours)} UT",
            "terrain_data": "Copernicus GLO-90 DEM",
            "assumptions": "see README.md: refraction k=0.13; random cloud overlap per sample; "
            "slant-path cloud correction (uncertain); cloud and terrain independent",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    return ds
