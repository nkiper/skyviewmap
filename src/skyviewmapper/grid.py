"""Regular lat/lon grid of map cells (observer locations)."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Grid:
    """Cells of size ``res`` degrees covering [lat_min, lat_max] x [lon_min, lon_max].

    Rows run south to north, columns west to east; ``lats``/``lons`` are cell
    centres. Edges fall on multiples of ``res`` from the box corner.
    """

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    res: float

    @property
    def shape(self) -> tuple[int, int]:
        return (
            int(round((self.lat_max - self.lat_min) / self.res)),
            int(round((self.lon_max - self.lon_min) / self.res)),
        )

    @property
    def lats(self) -> NDArray[np.float64]:
        return self.lat_min + (np.arange(self.shape[0], dtype=np.float64) + 0.5) * self.res

    @property
    def lons(self) -> NDArray[np.float64]:
        return self.lon_min + (np.arange(self.shape[1], dtype=np.float64) + 0.5) * self.res

    def mesh(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """2-D (lat, lon) arrays of cell centres, shape ``self.shape``."""
        lat, lon = np.meshgrid(self.lats, self.lons, indexing="ij")
        return lat, lon


SPAIN = Grid(lat_min=36.0, lat_max=44.0, lon_min=-10.0, lon_max=4.5, res=0.01)

# DEM extent for SPAIN: the Sun sets toward ~WNW, so sight lines of up to
# ~200 km reach into Portugal and the Atlantic; a margin is added all round.
SPAIN_DEM_BOX = (35.0, 45.0, -12.5, 4.5)  # lat_min, lat_max, lon_min, lon_max
