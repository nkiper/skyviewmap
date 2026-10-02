"""Line-of-sight geometry on a spherical Earth.

Given an observer, the altitude/azimuth of a sky target and the height of an
atmospheric layer, find where the line of sight crosses that height.

Assumptions and approximations
------------------------------
- Spherical Earth of radius ``R`` (default ``constants.R_EARTH_M``). All heights
  (observer, layer) are measured above that sphere, i.e. approximately above
  mean sea level -- a layer height is *not* relative to local ground.
- The line of sight is a straight line: no atmospheric refraction is applied
  here. The altitude is used exactly as given. Callers that want refraction
  must pass an apparent altitude or an effective radius (e.g. ``R / (1 - k)``).
- Horizontal positions are found with the spherical great-circle "direct"
  formulas; ground distance is the great-circle arc length at radius ``R``.
- No check is made that the ray clears the terrain or the ground before
  reaching the layer.

Conventions: angles in degrees at the API (azimuth clockwise from true north,
altitude above the horizontal), distances and heights in metres. All functions
accept numpy arrays and broadcast their arguments against each other.
"""

from typing import NamedTuple

import numpy as np

from .constants import R_EARTH_M


class LayerCrossing(NamedTuple):
    """Where a line of sight crosses a layer height. NaN where it never does."""

    lat: np.ndarray
    lon: np.ndarray
    ground_distance_m: np.ndarray
    slant_range_m: np.ndarray


def slant_range_to_height(alt_deg, layer_h, observer_h=0.0, R=R_EARTH_M):
    """Distance along the line of sight from the observer to radius ``R + layer_h``.

    With r0 = R + observer_h and r1 = R + layer_h, the law of cosines gives
    ``s**2 + 2 r0 sin(a) s + (r0**2 - r1**2) = 0``, whose roots are
    ``s = -r0 sin(a) ± sqrt(r0**2 sin(a)**2 + r1**2 - r0**2)``.

    - Layer above the observer: exactly one positive root, returned.
    - Layer at the observer's height: 0.
    - Layer below the observer: only a descending ray (alt < 0) steep enough
      to reach it crosses; the first (nearer) crossing is returned. Otherwise
      NaN.

    The roots are evaluated in a form that avoids cancellation (using the
    product of roots), so results stay accurate for large altitudes and
    shallow layers.
    """
    a = np.radians(np.asarray(alt_deg, dtype=float))
    layer_h = np.asarray(layer_h, dtype=float)
    observer_h = np.asarray(observer_h, dtype=float)

    r0 = R + observer_h
    q = -r0 * np.sin(a)
    # Product of the roots: r0**2 - r1**2, written to avoid subtracting two
    # ~4e13 m**2 numbers.
    prod = -(layer_h - observer_h) * (2.0 * R + layer_h + observer_h)
    disc = q * q - prod

    with np.errstate(invalid="ignore", divide="ignore"):
        root_d = np.sqrt(disc)
        # Layer above: larger root q + sqrt(disc). When q < 0 (ascending ray)
        # compute it as prod / (q - sqrt(disc)) to avoid cancellation.
        above = np.where(q >= 0, q + root_d, prod / (q - root_d))
        # Layer below: nearer root q - sqrt(disc) = prod / (q + sqrt(disc)),
        # valid only for a descending ray (q > 0) with real roots.
        below = np.where((q > 0) & (disc >= 0), prod / (q + root_d), np.nan)

    s = np.where(layer_h > observer_h, above, np.where(layer_h < observer_h, below, 0.0))
    return s


def central_angle(slant_range, alt_deg, observer_h=0.0, R=R_EARTH_M):
    """Earth-centre angle (radians) between the observer and a point on the ray.

    The point lies ``slant_range`` along the ray, so relative to the Earth's
    centre it sits ``s cos(a)`` along the horizontal and ``r0 + s sin(a)``
    along the observer's vertical:
    ``theta = atan2(s cos(a), r0 + s sin(a))``.
    """
    a = np.radians(np.asarray(alt_deg, dtype=float))
    s = np.asarray(slant_range, dtype=float)
    r0 = R + np.asarray(observer_h, dtype=float)
    return np.arctan2(s * np.cos(a), r0 + s * np.sin(a))


def destination_point(lat_deg, lon_deg, az_deg, theta):
    """Point reached by going ``theta`` radians of arc from (lat, lon) on initial bearing ``az``.

    Spherical great-circle direct problem. Returns (lat, lon) in degrees with
    longitude normalised to [-180, 180).
    """
    phi1 = np.radians(np.asarray(lat_deg, dtype=float))
    lam1 = np.radians(np.asarray(lon_deg, dtype=float))
    az = np.radians(np.asarray(az_deg, dtype=float))
    theta = np.asarray(theta, dtype=float)

    sin_phi2 = np.sin(phi1) * np.cos(theta) + np.cos(phi1) * np.sin(theta) * np.cos(az)
    phi2 = np.arcsin(np.clip(sin_phi2, -1.0, 1.0))
    lam2 = lam1 + np.arctan2(
        np.sin(az) * np.sin(theta) * np.cos(phi1),
        np.cos(theta) - np.sin(phi1) * sin_phi2,
    )
    lon2 = (np.degrees(lam2) + 180.0) % 360.0 - 180.0
    return np.degrees(phi2), lon2


def los_layer_crossing(lat_deg, lon_deg, alt_deg, az_deg, layer_h, observer_h=0.0, R=R_EARTH_M):
    """Where the line of sight toward (alt, az) crosses height ``layer_h``.

    Combines :func:`slant_range_to_height`, :func:`central_angle` and
    :func:`destination_point`; see the module docstring for assumptions.
    All outputs are NaN where the ray never crosses the layer.
    """
    s = slant_range_to_height(alt_deg, layer_h, observer_h, R)
    theta = central_angle(s, alt_deg, observer_h, R)
    lat, lon = destination_point(lat_deg, lon_deg, az_deg, theta)
    return LayerCrossing(lat=lat, lon=lon, ground_distance_m=R * theta, slant_range_m=s)


def flat_earth_ground_distance(alt_deg, layer_h, observer_h=0.0):
    """Flat-Earth horizontal distance ``(layer_h - observer_h) / tan(alt)``.

    For comparison and testing only; it overestimates the distance at low
    altitudes because it ignores the Earth's curvature.
    """
    a = np.radians(np.asarray(alt_deg, dtype=float))
    return (np.asarray(layer_h, dtype=float) - np.asarray(observer_h, dtype=float)) / np.tan(a)
