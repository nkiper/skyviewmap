# Eclipse visibility mapper

## Goal
Map where a solar eclipse (events defined in `events/*.toml`; v1 was 12 Aug 2026,
next is 2 Aug 2027) is most likely to be visible, accounting for historical cloud
cover *along the line of sight* and terrain obstruction toward the event's
azimuth. v2 adds an interactive public web map (GitHub Pages).

## Method (v1)
- Ephemeris: Skyfield for event altitude/azimuth per grid point.
- Terrain: ray-cast along the azimuth over a DEM (Copernicus GLO-30 or SRTM),
  including Earth curvature and standard refraction.
- Cloud: ERA5 hourly low/medium/high cloud cover, climatology for the date ±N days
  at the event hour. Each layer is sampled where the line of sight crosses its
  representative height. Layers are combined assuming random overlap (documented
  as an assumption).
- Output: P(clear view) map, terrain-blocked areas masked.

## Conventions
- Python 3.11+, virtual environment in .venv, dependencies in pyproject.toml.
- Core physics in small, pure, unit-tested functions; data I/O kept separate.
- Use numpy/xarray, vectorised over the grid.
- Use type hints on all function signatures (including tests): `ArrayLike`
  for array inputs, `NDArray[...]` for array outputs. Keep `cast` for untyped
  third-party returns (e.g. Skyfield) and comment why.
- Tests with pytest; check geometry against hand calculations.
- Raw downloaded data goes in data/raw/ and is git-ignored.
- Never commit credentials (CDS API key lives in ~/.cdsapirc).

## Working style
- Propose a plan before any non-trivial change; work in small steps.
- State physical assumptions and approximations explicitly in docstrings.
- Keep README.md succinct and current: update its status table, setup, layout
  and key assumptions in the same branch/PR as any change that affects them.
- Always use the project venv (macOS): `.venv/bin/python -m pip ...` and
  `.venv/bin/python -m pytest`.
- Before finishing any change, check the VS Code Problems panel and fix every
  error. It only covers open files, so also run `.venv/bin/pyright` (same
  settings, from `[tool.pyright]` in pyproject.toml) over src/ and tests/ and
  get it to 0 errors.
