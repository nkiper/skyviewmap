"""Physical constants and modelling parameters, kept in one place.

Assumptions
-----------
- Earth is modelled as a sphere of mean radius R_EARTH_M (IUGG mean radius,
  6 371 008.8 m, rounded to 6 371 000 m). The ellipsoidal error in the
  horizontal position of a line-of-sight crossing is well below the ERA5
  grid spacing (~25–30 km), so a sphere is enough for v1.
"""

R_EARTH_M: float = 6_371_000.0
