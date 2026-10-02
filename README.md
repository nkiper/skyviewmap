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
| 2. Sun position at maximum eclipse per grid point (Skyfield) | ⏳ next |
| 3. Terrain horizon mask (Copernicus DEM) | — |
| 4. ERA5 low/mid/high cloud climatology | — |
| 5. Cloud sampled along the line of sight, layers combined | — |
| 6. End-to-end pipeline and P(clear view) map | — |

## Setup
Requires Python 3.11+.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest
```

ERA5 downloads (from milestone 4) need a Copernicus CDS API key in `~/.cdsapirc`.

## Layout
```
src/skyviewmapper/
  constants.py   Earth radius and other model parameters
  geometry.py    where a line of sight crosses a given height (spherical Earth)
tests/           pytest, checked against hand-calculated values
data/, outputs/  downloaded data and results (git-ignored)
```

## Key assumptions
- Spherical Earth with mean radius 6 371 km. Heights are measured above sea level.
- The line of sight is a straight ray. Refraction is handled where the Sun's
  altitude is computed (milestone 2) and in the terrain model (milestone 3).
- Cloud layers are combined assuming random overlap (milestone 5).
