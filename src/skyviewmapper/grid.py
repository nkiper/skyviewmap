"""Regular lat/lon grid of map cells (observer locations)."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Grid:
    """Cells of ``res`` degrees (latitude) by ``res_lon`` degrees (longitude).

    ``res_lon`` defaults to ``res``; a larger value keeps cells roughly square
    at high latitudes. Rows run south to north, columns west to east;
    ``lats``/``lons`` are cell centres. Edges fall on multiples of the
    resolution from the box corner.
    """

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    res: float
    res_lon: float | None = None

    @property
    def dlat(self) -> float:
        return self.res

    @property
    def dlon(self) -> float:
        return self.res if self.res_lon is None else self.res_lon

    @property
    def shape(self) -> tuple[int, int]:
        return (
            int(round((self.lat_max - self.lat_min) / self.dlat)),
            int(round((self.lon_max - self.lon_min) / self.dlon)),
        )

    @property
    def lats(self) -> NDArray[np.float64]:
        return self.lat_min + (np.arange(self.shape[0], dtype=np.float64) + 0.5) * self.dlat

    @property
    def lons(self) -> NDArray[np.float64]:
        return self.lon_min + (np.arange(self.shape[1], dtype=np.float64) + 0.5) * self.dlon

    def mesh(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """2-D (lat, lon) arrays of cell centres, shape ``self.shape``."""
        lat, lon = np.meshgrid(self.lats, self.lons, indexing="ij")
        return lat, lon
