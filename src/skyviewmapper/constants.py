"""Physical constants and modelling parameters, kept in one place.

Assumptions
-----------
- Earth is modelled as a sphere of mean radius R_EARTH_M (IUGG mean radius,
  6 371 008.8 m, rounded to 6 371 000 m). The ellipsoidal error in the
  horizontal position of a line-of-sight crossing is well below the ERA5
  grid spacing (~25–30 km), so a sphere is enough for v1.
- Observer positions for the ephemeris use the WGS84 ellipsoid, because
  eclipse timing and parallax are sensitive to the true observer position.
- Sun and Moon radii follow NASA's eclipse tables (Espenak): solar radius
  696 000 km and lunar radius k = 0.272281 equatorial Earth radii.
- Standard refraction conditions: 10 °C, 1010 hPa.
- Cloud layers (ERA5 low/medium/high) are treated as slabs above the
  observer's ground, roughly following ERA5's sigma boundaries (0.8 and
  0.45 of surface pressure, ~2 km and ~6 km above ground). Each slab is
  sampled at three heights.
- Cloud aspect ratios (height / width) for the slant-path correction are
  rough typical values (cumulus/stratocumulus, altocumulus, cirrus sheets)
  and are uncertain; results are always also given without the correction.
"""

R_EARTH_M: float = 6_371_000.0

WGS84_A_M: float = 6_378_137.0
WGS84_F: float = 1.0 / 298.257223563

R_SUN_M: float = 696_000_000.0
MOON_K: float = 0.272281
R_MOON_M: float = MOON_K * WGS84_A_M

REFRACTION_TEMPERATURE_C: float = 10.0
REFRACTION_PRESSURE_MBAR: float = 1010.0

# ERA5 layer -> sample heights above the observer's ground (m).
CLOUD_LAYER_HEIGHTS_M: dict[str, tuple[float, ...]] = {
    "lcc": (500.0, 1_000.0, 1_500.0),  # low: 0-2 km
    "mcc": (3_000.0, 4_000.0, 5_000.0),  # mid: 2-6 km
    "hcc": (7_500.0, 9_000.0, 10_500.0),  # high: 6-12 km
}
CLOUD_ASPECT_RATIO: dict[str, float] = {"lcc": 0.5, "mcc": 0.3, "hcc": 0.1}
