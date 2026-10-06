"""Copernicus GLO-90 DEM: download tiles and build a lat/lon mosaic.

Tiles come from the public AWS bucket ``copernicus-dem-90m`` (no account),
1 deg x 1 deg, 1200 rows at 3 arcsec. Pixels widen in longitude toward the
poles (Copernicus latitude bands): 3" below 50 deg, then 4.5", 6", 9", 15"
and 30", i.e. 1200, 800, 600, 400, 240 or 120 columns. A mosaic must lie
within one band so it stays a regular grid.
Ocean-only tiles do not exist (HTTP 404); a ``.missing`` marker is written so
they are not requested again, and the mosaic is filled with 0 m there.

Heights are metres above the EGM2008 geoid, rounded to int16 in the mosaic.
Tiles are pixel-is-point: pixel centres lie on multiples of the pixel size,
with the first one on the tile's NW corner (GDAL reports the transform shifted by
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
TILE_ROWS = 1200
PX_DEG = 1.0 / TILE_ROWS  # latitude pixel size, all bands
# (upper |lat| of band, longitude pixel size / latitude pixel size)
_LON_BANDS = ((50, 1.0), (60, 1.5), (70, 2.0), (80, 3.0), (85, 5.0), (90, 10.0))
_ROOT = Path(__file__).resolve().parents[3]
TILE_DIR = _ROOT / "data" / "raw" / "dem" / "glo90"
CACHE_DIR = _ROOT / "data" / "processed"


def tile_name(lat_south: int, lon_west: int) -> str:
    """GLO-90 tile covering [lat_south, lat_south+1] x [lon_west, lon_west+1]."""
    ns = "N" if lat_south >= 0 else "S"
    ew = "E" if lon_west >= 0 else "W"
    return f"Copernicus_DSM_COG_30_{ns}{abs(lat_south):02d}_00_{ew}{abs(lon_west):03d}_00_DEM"


def lon_width_factor(lat_south: int) -> float:
    """Longitude/latitude pixel-size ratio for the tile starting at ``lat_south``."""
    equatorward = min(abs(lat_south), abs(lat_south + 1))
    return next(f for top, f in _LON_BANDS if equatorward < top)


def tile_cols(lat_south: int) -> int:
    """Number of pixel columns in the tile starting at ``lat_south``."""
    return int(round(TILE_ROWS / lon_width_factor(lat_south)))


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
    widths = {tile_cols(lat) for lat in lats}
    if len(widths) != 1:
        raise ValueError(
            f"box {lat_min}..{lat_max} spans Copernicus latitude bands with different pixel widths; split it"
        )
    cols = widths.pop()
    px_lon = 1.0 / cols
    heights = np.zeros((len(lats) * TILE_ROWS, len(lons) * cols), dtype=np.int16)
    north, west = lats[-1] + 1, lons[0]
    for lat in lats:
        for lon in lons:
            path = tile_dir / f"{tile_name(lat, lon)}.tif"
            if not path.exists():
                continue
            with rasterio.open(path) as src:
                t = src.transform
                if src.shape != (TILE_ROWS, cols) or not np.allclose((t.a, -t.e), (px_lon, PX_DEG)):
                    raise ValueError(f"unexpected grid in {path.name}: {src.shape}, {t}")
                # Tiles are pixel-is-point: the centre of pixel (0, 0) is the
                # tile's NW corner, so the tile spans centres lat+1 .. lat+1/1200.
                centre = (t.c + t.a / 2, t.f + t.e / 2)
                if not np.allclose(centre, (lon, lat + 1), atol=1e-9):
                    raise ValueError(f"unexpected origin in {path.name}: {t}")
                data = src.read(1)
            r0 = (north - (lat + 1)) * TILE_ROWS
            c0 = (lon - west) * cols
            heights[r0 : r0 + TILE_ROWS, c0 : c0 + cols] = np.round(np.nan_to_num(data)).astype(np.int16)
    return Dem(heights=heights, lat0=north, lon0=west, dlat=PX_DEG, dlon=px_lon)


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
    else:
        # Every tile must be either downloaded or known to be ocean (.missing);
        # otherwise land would silently become 0 m.
        lats, lons = _tile_ranges(lat_min, lat_max, lon_min, lon_max)
        absent = [
            tile_name(lat, lon)
            for lat in lats
            for lon in lons
            if not (TILE_DIR / f"{tile_name(lat, lon)}.tif").exists()
            and not (TILE_DIR / f"{tile_name(lat, lon)}.missing").exists()
        ]
        if absent:
            raise FileNotFoundError(f"{len(absent)} DEM tiles not downloaded, e.g. {absent[0]}")
    dem = build_mosaic(lat_min, lat_max, lon_min, lon_max)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(npy, dem.heights)
    meta.write_text(json.dumps({"lat0": dem.lat0, "lon0": dem.lon0, "dlat": dem.dlat, "dlon": dem.dlon}))
    return dem


def parse_tile_name(name: str) -> tuple[int, int]:
    """(lat_south, lon_west) of a GLO tile name such as ``Copernicus_DSM_COG_30_N40_00_W004_00_DEM``."""
    parts = name.split("_")
    ns, ew = parts[4], parts[6]
    lat = int(ns[1:]) * (1 if ns[0] == "N" else -1)
    lon = int(ew[1:]) * (1 if ew[0] == "E" else -1)
    return lat, lon


def land_tiles(download: bool = True) -> set[tuple[int, int]]:
    """(lat_south, lon_west) of every 1 deg tile that contains land, from the bucket's tile list.

    The list (~26 000 names) is cached in data/raw/dem/; it serves as a global
    1 deg land mask.
    """
    path = TILE_DIR.parent / "tileList_glo90.txt"
    if not path.exists():
        if not download:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{BASE_URL}/tileList.txt") as resp:
            path.write_bytes(resp.read())
    return {parse_tile_name(line.strip()) for line in path.read_text().splitlines() if line.strip()}

