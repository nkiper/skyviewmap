"""Map regions for the 12 Aug 2026 eclipse: grid, DEM extent and ephemeris window.

The path of totality crosses populated land in Iberia (northern Spain, a
corner of NE Portugal, the Balearics) and western Iceland. Each region is
computed on its own grid; it must lie within one Copernicus DEM latitude band
so its DEM mosaic is a regular grid.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

from .grid import Grid


@dataclass(frozen=True)
class Region:
    """A map region.

    ``dem_box`` (lat_min, lat_max, lon_min, lon_max) extends the grid by the
    longest sight line toward the Sun. ``window_utc`` must contain every
    cell's maximum eclipse with ~150 s to spare either side (for totality
    durations). ``era5_box`` extends the grid by the farthest point where a
    sight line can cross a cloud layer; ``era5_hours`` (UT) bracket every
    cell's maximum eclipse so it can be interpolated in time.
    """

    name: str
    grid: Grid
    dem_box: tuple[float, float, float, float]
    window_utc: tuple[datetime, datetime]
    era5_box: tuple[float, float, float, float]
    era5_hours: tuple[int, ...]


def _utc(h: int, m: int) -> datetime:
    return datetime(2026, 8, 12, h, m, tzinfo=timezone.utc)


REGIONS: dict[str, Region] = {
    # Spain, Portugal, Andorra and the Balearics. The Sun sets toward ~WNW at
    # 2-12 deg, so sight lines of up to ~200 km reach into Portugal and the
    # Atlantic.
    "iberia": Region(
        name="iberia",
        grid=Grid(lat_min=36.0, lat_max=44.0, lon_min=-10.0, lon_max=4.5, res=0.01),
        dem_box=(35.0, 45.0, -12.5, 4.5),
        window_utc=(_utc(18, 10), _utc(18, 45)),
        # Sun as low as ~2 deg: high cloud at 9 km is crossed up to ~250 km away.
        era5_box=(33.0, 47.0, -15.0, 7.0),
        era5_hours=(18, 19),  # maxima 18:26-18:38 UT
    ),
    # Iceland: Sun at ~26 deg toward WSW, so sight lines are short (a few km);
    # 0.02 deg in longitude keeps cells ~1 km square at 65 N.
    "iceland": Region(
        name="iceland",
        grid=Grid(lat_min=63.0, lat_max=67.0, lon_min=-25.0, lon_max=-13.0, res=0.01, res_lon=0.02),
        dem_box=(62.5, 67.5, -25.5, -12.5),
        window_utc=(_utc(17, 20), _utc(18, 20)),
        # Sun at ~25 deg: crossings are within ~25 km.
        era5_box=(62.0, 68.0, -27.0, -11.0),
        era5_hours=(17, 18),  # maxima 17:44-17:52 UT
    ),
}
