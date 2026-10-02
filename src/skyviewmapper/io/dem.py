"""Copernicus GLO-90 DEM: download tiles and build a lat/lon mosaic.

Tiles come from the public AWS bucket ``copernicus-dem-90m`` (no account),
1 deg x 1 deg, 1200 x 1200 pixels at 3 arcsec for latitudes below 50 deg.
Ocean-only tiles do not exist (HTTP 404); a ``.missing`` marker is written so
they are not requested again, and the mosaic is filled with 0 m there.

Heights are metres above the EGM2008 geoid, rounded to int16 in the mosaic.
Tiles are pixel-is-point: pixel centres lie on multiples of 3 arcsec, with
the first one on the tile's NW corner (GDAL reports the transform shifted by
half a pixel accordingly); this is checked for every tile.
"""

import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import rasterio

from ..terrain import Dem

BASE_URL = "https://copernicus-dem-90m.s3.amazonaws.com"
TILE_PX = 1200
PX_DEG = 1.0 / TILE_PX
_ROOT = Path(__file__).resolve().parents[3]
TILE_DIR = _ROOT / "data" / "raw" / "dem" / "glo90"
CACHE_DIR = _ROOT / "data" / "processed"


def tile_name(lat_south: int, lon_west: int) -> str:
    """GLO-90 tile covering [lat_south, lat_south+1] x [lon_west, lon_west+1]."""
    ns = "N" if lat_south >= 0 else "S"
    ew = "E" if lon_west >= 0 else "W"
    return f"Copernicus_DSM_COG_30_{ns}{abs(lat_south):02d}_00_{ew}{abs(lon_west):03d}_00_DEM"


def _tile_ranges(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> tuple[range, range]:
    return (
        range(int(np.floor(lat_min)), int(np.ceil(lat_max))),
        range(int(np.floor(lon_min)), int(np.ceil(lon_max))),
    )


def download_tiles(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float, tile_dir: Path = TILE_DIR
) -> list[Path]:
    """Download every tile intersecting the box; return paths of tiles that exist."""
    tile_dir.mkdir(parents=True, exist_ok=True)
    lats, lons = _tile_ranges(lat_min, lat_max, lon_min, lon_max)
    found = []
    for lat in lats:
        for lon in lons:
            name = tile_name(lat, lon)
            path = tile_dir / f"{name}.tif"
            missing = tile_dir / f"{name}.missing"
            if missing.exists():
                continue
            if not path.exists():
                try:
                    with urllib.request.urlopen(f"{BASE_URL}/{name}/{name}.tif") as resp:
                        tmp = path.with_suffix(".part")
                        tmp.write_bytes(resp.read())
                        tmp.rename(path)
                except urllib.error.HTTPError as err:
                    if err.code in (403, 404):  # S3 answers 403/404 for absent ocean tiles
                        missing.touch()
                        continue
                    raise
            found.append(path)
    return found


def build_mosaic(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float, tile_dir: Path = TILE_DIR
) -> Dem:
    """Mosaic downloaded tiles covering whole degrees around the box into one int16 array (row 0 = north)."""
    lats, lons = _tile_ranges(lat_min, lat_max, lon_min, lon_max)
    heights = np.zeros((len(lats) * TILE_PX, len(lons) * TILE_PX), dtype=np.int16)
    north, west = lats[-1] + 1, lons[0]
    for lat in lats:
        for lon in lons:
            path = tile_dir / f"{tile_name(lat, lon)}.tif"
            if not path.exists():
                continue
            with rasterio.open(path) as src:
                t = src.transform
                if src.shape != (TILE_PX, TILE_PX) or not np.allclose((t.a, -t.e), PX_DEG):
                    raise ValueError(f"unexpected grid in {path.name}: {src.shape}, {t}")
                # Tiles are pixel-is-point: the centre of pixel (0, 0) is the
                # tile's NW corner, so the tile spans centres lat+1 .. lat+1/1200.
                centre = (t.c + t.a / 2, t.f + t.e / 2)
                if not np.allclose(centre, (lon, lat + 1), atol=1e-9):
                    raise ValueError(f"unexpected origin in {path.name}: {t}")
                data = src.read(1)
            r0 = (north - (lat + 1)) * TILE_PX
            c0 = (lon - west) * TILE_PX
            heights[r0 : r0 + TILE_PX, c0 : c0 + TILE_PX] = np.round(np.nan_to_num(data)).astype(np.int16)
    return Dem(heights=heights, lat0=north, lon0=west, dlat=PX_DEG, dlon=PX_DEG)


def load_dem(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float, download: bool = True
) -> Dem:
    """Download (if needed) and mosaic the box, caching the mosaic in data/processed/."""
    key = f"glo90_{lat_min:g}_{lat_max:g}_{lon_min:g}_{lon_max:g}"
    npy, meta = CACHE_DIR / f"{key}.npy", CACHE_DIR / f"{key}.json"
    if npy.exists() and meta.exists():
        m = json.loads(meta.read_text())
        return Dem(heights=np.load(npy, mmap_mode="r"), **m)
    if download:
        download_tiles(lat_min, lat_max, lon_min, lon_max)
    elif not any(TILE_DIR.glob("*.tif")):
        raise FileNotFoundError(TILE_DIR)
    dem = build_mosaic(lat_min, lat_max, lon_min, lon_max)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(npy, dem.heights)
    meta.write_text(json.dumps({"lat0": dem.lat0, "lon0": dem.lon0, "dlat": dem.dlat, "dlon": dem.dlon}))
    return dem


if __name__ == "__main__":
    from ..grid import SPAIN_DEM_BOX

    tiles = download_tiles(*SPAIN_DEM_BOX)
    print(f"{len(tiles)} GLO-90 tiles in {TILE_DIR}")
