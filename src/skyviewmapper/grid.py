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


def resample_bilinear(values: NDArray[np.float64], src: Grid, dst: Grid) -> NDArray[np.float64]:
    """Bilinearly interpolate cell-centre ``values`` on ``src`` to the cell centres of ``dst``.

    Destination centres beyond the outermost source centres take the edge
    value (no extrapolation).
    """
    values = np.asarray(values, dtype=float)

    def axis(x: NDArray[np.float64], x0: float, dx: float, n: int) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        pos = np.clip((x - x0) / dx, 0.0, n - 1.0)
        i = np.minimum(np.floor(pos).astype(np.intp), max(n - 2, 0))
        return i, pos - i

    i, fy = axis(dst.lats, float(src.lats[0]), src.dlat, src.shape[0])
    j, fx = axis(dst.lons, float(src.lons[0]), src.dlon, src.shape[1])
    i1 = np.minimum(i + 1, src.shape[0] - 1)
    j1 = np.minimum(j + 1, src.shape[1] - 1)
    fy, fx = fy[:, None], fx[None, :]
    return (
        values[np.ix_(i, j)] * (1 - fy) * (1 - fx)
        + values[np.ix_(i, j1)] * (1 - fy) * fx
        + values[np.ix_(i1, j)] * fy * (1 - fx)
        + values[np.ix_(i1, j1)] * fy * fx
    )
