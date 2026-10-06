"""Map regions: grid, DEM extent, ephemeris window and ERA5 settings.

Regions come from an event file (explicit ``[[regions]]``) or are derived from
the eclipse path (:mod:`skyviewmapper.paths`). Each region must lie within one
Copernicus DEM latitude band so its DEM mosaic is a regular grid.
"""

from dataclasses import dataclass
from datetime import datetime

from .grid import Grid


@dataclass(frozen=True)
class Region:
    """A map region.

    ``dem_box`` (lat_min, lat_max, lon_min, lon_max) extends the grid by the
    longest sight line toward the Sun. ``window_utc`` must contain every
    cell's maximum eclipse with ~150 s to spare either side (for central
    durations). ``era5_box`` extends the grid by the farthest point where a
    sight line can cross a cloud layer; ``era5_hours`` (UT, consecutive)
    bracket every cell's maximum eclipse so it can be interpolated in time.
    ``cloud_coarsen``: the cloud term varies on ERA5's ~25 km scale, so it is
    computed on a grid this many times coarser and interpolated back.
    """

    name: str
    grid: Grid
    dem_box: tuple[float, float, float, float]
    window_utc: tuple[datetime, datetime]
    era5_box: tuple[float, float, float, float]
    era5_hours: tuple[int, ...]
    cloud_coarsen: int = 5
    display_name: str = ""

    @property
    def label(self) -> str:
        return self.display_name or self.name

    @property
    def cloud_grid(self) -> Grid:
        g = self.grid
        return Grid(g.lat_min, g.lat_max, g.lon_min, g.lon_max, g.dlat * self.cloud_coarsen, g.dlon * self.cloud_coarsen)
