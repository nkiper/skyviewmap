# skyviewmap

Maps where a solar eclipse is most likely to be visible, taking into account
historical cloud cover *along the line of sight* and terrain blocking the view.
Events are defined in `events/`:

- **12 Aug 2026** (total, past): Iberia, where it was low in the western sky near
  sunset, and western Iceland. This was v1.
- **2 Aug 2027** (total, next): every land area under the path, from southern
  Spain and the Maghreb to Egypt, Arabia and Somalia, derived automatically.

## Why the line of sight matters
With the Sun only a few degrees above the horizon, the line of sight passes
through cloud layers tens to hundreds of kilometres away from the observer, not
overhead. The model therefore looks up cloud cover where the ray crosses each layer
(allowing for Earth curvature), and checks the terrain toward the Sun's azimuth.

## Status
| Milestone | Status |
|---|---|
| v1: geometry, ephemeris, terrain, ERA5 cloud, line-of-sight cloud, maps (2026) | ✅ done |
| v2.1: events as config files, regions derived from the path, annular eclipses; 2 Aug 2027 | ✅ done |
| v2.2: interactive web map (tiles + MapLibre viewer) | ⏳ next |
| v2.3: OpenStreetMap viewpoints, parks, roads; reachable top spots | — |
| v2.4: public site on GitHub Pages with a methods page | — |
| v2.5: check against the actual 2026 weather; forecast mode | — |

## Setup
Requires Python 3.11+.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m skyviewmapper.io.skyfield_data   # downloads DE440s ephemeris (~32 MB)
.venv/bin/python -m pytest
.venv/bin/pyright                                    # type check (0 errors expected)
```

## Running
```sh
.venv/bin/python -m skyviewmapper events                          # list event files
.venv/bin/python -m skyviewmapper regions  --event 2027-08-02     # regions, cells, DEM tiles needed
.venv/bin/python -m skyviewmapper download --event 2027-08-02     # DEM tiles + ERA5 (CDS queue: can take hours)
.venv/bin/python -m skyviewmapper run      --event 2027-08-02     # all regions
.venv/bin/python -m skyviewmapper run      --event 2026-08-12 --region iceland --no-download
.venv/bin/python -m skyviewmapper run      --event 2027-08-02 --min-central 240  # top spots: >= 4 min of totality
```

### Adding an event
Create `events/<date>.toml` with `id`, `name`, `date`, `type` (`total`,
`annular` or `hybrid`), `years` (first and last climatology year) and
optionally `half_window_days` (default 7), `include` boxes to limit the area,
`exclude` boxes to drop land (e.g. an island the coarse path only grazes),
`[display_names]` for derived regions (keyed by their stable names such as
`36N_002W`, shown by the `regions` command), or explicit `[[regions]]` (see
`events/2026-08-12.toml`). Without explicit regions, the path is found on a
coarse global grid and every land area under it becomes a region of at most
15° of longitude, each with its own map grid, DEM and ERA5 extents, time
window and ERA5 hours.

Outputs, per region in `outputs/<event>/<region>/`, where `<region>` is the
region's readable name in file-safe form (e.g. `outputs/2027-08-02/eastern-libya-and-egypt/`);
`--region` accepts either that or the internal name (e.g. `27N_028E`):
- `skyview_<region>.nc`: every variable on the map grid (probabilities, terrain,
  eclipse circumstances incl. `central_s`, the duration of totality or
  annularity), with units and descriptions.
- `p_clear_view_<region>.tif`, `p_clear_view_no_slant_<region>.tif`,
  `central_s_<region>.tif`: GeoTIFFs (EPSG:4326) for GIS tools.
- `map_*.png`: the headline chance of a clear view, the optimistic version
  without the slant correction, duration of totality/annularity, and the
  terrain map. The top spots are marked on every map (orange triangles, ranks
  1–5 numbered).
- `top_spots_<region>.csv`: the 25 best cells inside the path with at least
  `--min-central` seconds of totality/annularity (default 60), at least 20 km
  apart and at least a third land (which drops offshore rocks), with the best
  spot's coordinates. Cells are ranked by chance in whole percent, ties going
  to the longest totality/annularity. Where the clear-sky chance rises toward
  one edge of the path while the central phase shortens, the list sits near the
  chosen minimum: the threshold is the trade-off to explore. Lists are per
  region, so they can bunch against a region's edge when the best area
  continues into the next region.

## Layout
```
src/skyviewmapper/
  constants.py   Earth radius and other model parameters
  geometry.py    where a line of sight crosses a given height (spherical Earth)
  ephemeris.py   per-observer time of maximum eclipse, Sun alt/az, magnitude,
                 obscuration, totality duration
  grid.py        regular lat/lon grid of map cells
  event.py       events from events/*.toml (date, type, climatology years, regions)
  regions.py     Region: map grid, DEM and ERA5 extents, ephemeris window, ERA5 hours
  paths.py       derives regions from the eclipse path over land
  terrain.py     horizon toward the Sun per spot; per-cell terrain visibility
  climatology.py summary statistics of the cloud samples (layer means, overhead P(clear))
  los_cloud.py   P(cloud-free line of sight to the Sun) from the cloud samples
  visibility.py  combines cloud and terrain into P(clear view)
  pipeline.py    end-to-end run for one region -> xarray Dataset
  plots.py       PNG maps
  __main__.py    command line (python -m skyviewmapper run ...)
  io/            data loading (skyfield_data.py: JPL ephemeris; dem.py: Copernicus GLO-90;
                 era5.py: ERA5 cloud cover from the CDS; outputs.py: NetCDF, GeoTIFF, CSV)
events/          event definitions (TOML)
tests/           pytest, checked against hand calculations and NASA tables
data/, outputs/  downloaded data and results (git-ignored)
```

## Key assumptions
- Spherical Earth with mean radius 6 371 km. Heights are measured above sea level.
- Each observer is evaluated at its own time of maximum eclipse. The central
  phase (totality or annularity) lasts while the Sun–Moon separation is below
  the difference of their radii; maxima come out 0.5–1.2 s later than NASA's
  tables, a systematic offset of unknown origin that is irrelevant here. Observer
  positions for the ephemeris use WGS84; Sun/Moon radii follow NASA's tables.
- Refraction: terrain and cloud steps both use the Sun's apparent (refracted,
  standard conditions) altitude with a straight ray over an effective Earth
  radius R/(1−k), k = 0.13. The ephemeris also returns the geometric altitude.
- Terrain: Copernicus GLO-90 (90 m). The terrain's elevation angle uses an
  effective Earth radius R/(1−k), k = 0.13, and is compared with the Sun's
  apparent altitude. Each cell (0.01°; 0.01° × 0.02° in Iceland, ~1 km) tests
  10 spots (a 3×3 pattern plus the highest pixel) with a 2 m eye height, giving
  the share of typical spots with a clear view and the best spot. GLO-90
  flattens sharp summits by tens of metres.
- Each region must lie within one Copernicus DEM latitude band (longitude
  pixels widen above 50°N: 6″ in Iceland vs 3″ in Iberia); derived regions are
  split at band edges and terrain beyond an edge counts as sea.
- Cloud: ERA5 hourly low/medium/high cloud cover (0.25°), 30 years (set per
  event), the event date ±7 days, at the full hours bracketing each region's
  maxima, interpolated in time to each cell's maximum eclipse. Every
  sample is kept: layers are combined per sample assuming random overlap,
  P(clear) = Π(clear chance of each layer), and only then averaged, because
  cloud layers are correlated in time.
- Line of sight: each layer is a slab above the observer's ground (low 0–2 km,
  mid 2–6 km, high 6–12 km), read at three heights where the sight line
  crosses it (apparent Sun altitude, effective Earth radius as for terrain).
  At the eclipse's low Sun the high layer is crossed 50–400 km away in Iberia.
- Slant-path correction (uncertain): clouds are modelled as randomly placed
  cylinders with height/width ratio β (0.5 low, 0.3 mid, 0.1 high), so a slanted
  sight line also meets cloud sides: a layer of cover n is passed with chance
  (1 − n)^(1 + (4β/π)·cot θ). It is supported by satellite view-angle studies
  up to ~70° from the zenith but extrapolated to the eclipse's 2–25°, and it
  lowers P(clear) a lot (Iberian land mean 0.69 → 0.51, Iceland 0.18 → 0.10).
  Results are therefore given with and without it.
- The cloud term is computed on a 5× coarser grid (0.05°) and interpolated to
  the map grid; ERA5 itself is 0.25°.
- Combination: cloud and terrain are treated as independent. Headline
  P(clear view) = P(cloud-free line of sight, slant-corrected) where at least
  one spot in the cell sees the Sun over the terrain; cells where no spot does
  are masked. Variants without the slant correction and for a typical spot
  (× share of typical spots that see the Sun) are saved alongside.
