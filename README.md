# skyviewmap

Maps where an astronomical event is most likely to be visible, taking into account
historical cloud cover *along the line of sight* and terrain blocking the view.
First target: the total solar eclipse of 12 August 2026, mapped for the populated
land in the path of totality: **Iberia** (northern Spain, a corner of NE Portugal,
the Balearics), where it happens low in the western sky near sunset, and
**western Iceland**, with the Sun at ~25°.

## Why the line of sight matters
With the Sun only a few degrees above the horizon, the line of sight passes
through cloud layers tens to hundreds of kilometres away from the observer, not
overhead. The model therefore looks up cloud cover where the ray crosses each layer
(allowing for Earth curvature), and checks the terrain toward the Sun's azimuth.

## Status
| Milestone | Status |
|---|---|
| 1. Line-of-sight / layer-height geometry | ✅ done |
| 2. Sun position at maximum eclipse per grid point (Skyfield) | ✅ done |
| 3. Terrain horizon mask (Copernicus DEM) | ✅ done |
| 3b. Multiple regions (Iberia + Iceland) | ✅ done |
| 4. ERA5 low/mid/high cloud climatology | ✅ done |
| 5. Cloud sampled along the line of sight, layers combined | ⏳ next |
| 6. End-to-end pipeline and P(clear view) map | — |

## Setup
Requires Python 3.11+.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m skyviewmapper.io.skyfield_data   # downloads DE440s ephemeris (~32 MB)
.venv/bin/python -m skyviewmapper.io.dem             # downloads GLO-90 DEM tiles, all regions (~490 MB)
.venv/bin/python -m skyviewmapper.io.era5            # downloads ERA5 cloud cover via the CDS (queued; can take a while)
.venv/bin/python -m pytest
.venv/bin/pyright                                    # type check (0 errors expected)
```

Tests marked `ephemeris` (NASA eclipse tables), `dem` (real DEM) and `era5` (real
cloud data) are skipped until the corresponding data has been downloaded. ERA5
downloads need a Copernicus CDS API key in `~/.cdsapirc` and the licence of
"ERA5 hourly data on single levels from 1940 to present" accepted on the CDS website.

## Layout
```
src/skyviewmapper/
  constants.py   Earth radius and other model parameters
  geometry.py    where a line of sight crosses a given height (spherical Earth)
  ephemeris.py   per-observer time of maximum eclipse, Sun alt/az, magnitude,
                 obscuration, totality duration
  grid.py        regular lat/lon grid of map cells
  regions.py     regions (Iberia, Iceland): grid, DEM extent, ephemeris time window
  terrain.py     horizon toward the Sun per spot; per-cell terrain visibility
  climatology.py summary statistics of the cloud samples (layer means, overhead P(clear))
  io/            data loading (skyfield_data.py: JPL ephemeris; dem.py: Copernicus GLO-90;
                 era5.py: ERA5 cloud cover from the CDS)
tests/           pytest, checked against hand calculations and NASA tables
data/, outputs/  downloaded data and results (git-ignored)
```

## Key assumptions
- Spherical Earth with mean radius 6 371 km. Heights are measured above sea level.
- Each observer is evaluated at its own time of maximum eclipse. Observer
  positions for the ephemeris use WGS84; Sun/Moon radii follow NASA's tables.
- The line of sight is a straight ray. The ephemeris returns both geometric and
  refracted (standard conditions) Sun altitude; which one feeds the cloud step
  is decided in milestone 5.
- Terrain: Copernicus GLO-90 (90 m). The terrain's elevation angle uses an
  effective Earth radius R/(1−k), k = 0.13, and is compared with the Sun's
  apparent altitude. Each cell (0.01°; 0.01° × 0.02° in Iceland, ~1 km) tests
  10 spots (a 3×3 pattern plus the highest pixel) with a 2 m eye height, giving
  the share of typical spots with a clear view and the best spot. GLO-90
  flattens sharp summits by tens of metres.
- Each region must lie within one Copernicus DEM latitude band (longitude
  pixels widen above 50°N: 6″ in Iceland vs 3″ in Iberia).
- Cloud: ERA5 hourly low/medium/high cloud cover (0.25°), 1996–2025, 5–19 August,
  at the two full hours bracketing the eclipse (18–19 UT Iberia, 17–18 UT
  Iceland). Every sample is kept: layers are combined per sample assuming
  random overlap, P(clear) = (1 − low)(1 − mid)(1 − high), and only then
  averaged, because cloud layers are correlated in time.
