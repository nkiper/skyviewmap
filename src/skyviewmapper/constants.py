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
"""

R_EARTH_M: float = 6_371_000.0

WGS84_A_M: float = 6_378_137.0
WGS84_F: float = 1.0 / 298.257223563

R_SUN_M: float = 696_000_000.0
MOON_K: float = 0.272281
R_MOON_M: float = MOON_K * WGS84_A_M

REFRACTION_TEMPERATURE_C: float = 10.0
REFRACTION_PRESSURE_MBAR: float = 1010.0
