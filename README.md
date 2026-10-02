# skyviewmap

Maps where an astronomical event is most likely to be visible, taking into account
historical cloud cover *along the line of sight* and terrain blocking the view.
First target: the total solar eclipse of 12 August 2026 over Spain, which happens
low in the western sky near sunset.

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
| 4. ERA5 low/mid/high cloud climatology | ⏳ next |
| 5. Cloud sampled along the line of sight, layers combined | — |
| 6. End-to-end pipeline and P(clear view) map | — |

## Setup
Requires Python 3.11+.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m skyviewmapper.io.skyfield_data   # downloads DE440s ephemeris (~32 MB)
.venv/bin/python -m skyviewmapper.io.dem             # downloads GLO-90 DEM tiles (~440 MB)
.venv/bin/python -m pytest
.venv/bin/pyright                                    # type check (0 errors expected)
```

Tests marked `ephemeris` (NASA eclipse tables) and `dem` (real DEM) are skipped
until the corresponding data has been downloaded. ERA5 downloads (from milestone 4) need a Copernicus CDS API key in `~/.cdsapirc`.

## Layout
```
src/skyviewmapper/
  constants.py   Earth radius and other model parameters
  geometry.py    where a line of sight crosses a given height (spherical Earth)
  ephemeris.py   per-observer time of maximum eclipse, Sun alt/az, magnitude,
                 obscuration, totality duration
  grid.py        map grid (0.01° cells over Spain) and DEM extent
  terrain.py     horizon toward the Sun per spot; per-cell terrain visibility
  io/            data loading (skyfield_data.py: JPL ephemeris; dem.py: Copernicus GLO-90)
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
  apparent altitude. Each 0.01° cell tests 10 spots (a 3×3 pattern plus the
  highest pixel) with a 2 m eye height, giving the share of typical spots with
  a clear view and the best spot. GLO-90 flattens sharp summits by tens of
  metres.
- Cloud layers are combined assuming random overlap (milestone 5).
