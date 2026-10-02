"""Geometry tests against hand-calculated values.

Reference values use R = 6 371 000 m, r0 = R + h0, r1 = R + H and
    s     = -r0 sin a + sqrt(r0^2 sin^2 a + r1^2 - r0^2)
    theta = atan2(s cos a, r0 + s sin a),   d = R * theta
evaluated independently with Python's ``math`` module and rounded to 0.1 m.
"""

import math

import numpy as np
import pytest

from skyviewmapper.constants import R_EARTH_M
from skyviewmapper.geometry import (
    central_angle,
    destination_point,
    flat_earth_ground_distance,
    los_layer_crossing,
    slant_range_to_height,
)

R = R_EARTH_M
DIST_TOL = 0.1  # m
ANG_TOL = 1e-9  # deg


# (alt_deg, layer_h, observer_h, slant_m, ground_m)
HAND_CASES = [
    # Zenith: straight up, no horizontal displacement.
    (90.0, 10_000.0, 0.0, 10_000.0, 0.0),
    # Horizon: tangent ray. s = sqrt(2RH + H^2) = sqrt(1.2752e11) = 357 099.4;
    # theta = acos(R / (R + H)) = 3.20812 deg -> d = 356 726.2.
    (0.0, 10_000.0, 0.0, 357_099.4, 356_726.2),
    # Steep, low layer: nearly flat-Earth (H / tan 30 = 1 732.1).
    (30.0, 1_000.0, 0.0, 1_999.5, 1_731.4),
    # Low Sun, high cloud: flat Earth would give 102 870.5 (~8% too far).
    (5.0, 9_000.0, 0.0, 95_179.0, 94_686.6),
    (5.0, 2_000.0, 0.0, 22_495.4, 22_402.8),
    # Elevated observer, horizontal ray.
    (0.0, 2_000.0, 1_000.0, 112_893.8, 112_864.2),
    (10.0, 6_000.0, 500.0, 31_245.7, 30_742.2),
    # Descending ray from 2 km to a 1 km layer: nearer of the two crossings.
    # Farther than flat Earth (28 636.3) because the surface curves away.
    (-2.0, 1_000.0, 2_000.0, 30_781.5, 30_758.0),
]


@pytest.mark.parametrize("alt, H, h0, s_ref, d_ref", HAND_CASES)
def test_hand_calculated_slant_and_ground_distance(alt: float, H: float, h0: float, s_ref: float, d_ref: float) -> None:
    s = slant_range_to_height(alt, H, h0)
    d = R * central_angle(s, alt, h0)
    assert s == pytest.approx(s_ref, abs=DIST_TOL)
    assert d == pytest.approx(d_ref, abs=DIST_TOL)


@pytest.mark.parametrize("alt, H, h0, s_ref, d_ref", HAND_CASES)
def test_crossing_lies_on_layer_sphere(alt: float, H: float, h0: float, s_ref: float, d_ref: float) -> None:
    s = slant_range_to_height(alt, H, h0)
    a = math.radians(alt)
    r0 = R + h0
    r = math.hypot(r0 + s * math.sin(a), s * math.cos(a))
    assert r == pytest.approx(R + H, abs=1e-3)


def test_layer_at_observer_height_is_zero_distance() -> None:
    assert slant_range_to_height(10.0, 1_500.0, 1_500.0) == 0.0


@pytest.mark.parametrize(
    "alt, H, h0",
    [
        (5.0, 1_000.0, 2_000.0),  # layer below observer, ray ascending
        (0.0, 1_000.0, 2_000.0),  # layer below observer, ray horizontal
        # Descending, but shallower than the ~1.015 deg dip to the 1 km sphere.
        (-1.0, 1_000.0, 2_000.0),
    ],
)
def test_no_crossing_is_nan(alt: float, H: float, h0: float) -> None:
    c = los_layer_crossing(40.0, -3.0, alt, 270.0, H, h0)
    assert all(np.isnan(v) for v in c)


def test_flat_earth_limit_at_high_altitude() -> None:
    d = R * central_angle(slant_range_to_height(45.0, 100.0), 45.0)
    assert d == pytest.approx(flat_earth_ground_distance(45.0, 100.0), rel=1e-3)


def test_ground_distance_increases_as_altitude_decreases() -> None:
    alts = np.linspace(90.0, 0.0, 91)
    d = los_layer_crossing(40.0, -3.0, alts, 270.0, 9_000.0).ground_distance_m
    assert np.all(np.diff(d) > 0)


def test_curvature_shortens_distance_vs_flat_earth() -> None:
    alts = np.array([1.0, 5.0, 15.0])
    curved = los_layer_crossing(40.0, -3.0, alts, 270.0, 9_000.0).ground_distance_m
    flat = flat_earth_ground_distance(alts, 9_000.0)
    assert np.all(curved < flat)


# Destination point: theta = 1 deg of arc.
ONE_DEG = math.radians(1.0)


@pytest.mark.parametrize(
    "lat, lon, az, lat_ref, lon_ref",
    [
        (40.0, -3.0, 0.0, 41.0, -3.0),  # north along a meridian
        (40.0, -3.0, 180.0, 39.0, -3.0),  # south along a meridian
        (0.0, 0.0, 90.0, 0.0, 1.0),  # east along the equator
        (0.0, 0.0, 270.0, 0.0, -1.0),  # west along the equator
        (0.0, 179.5, 90.0, 0.0, -179.5),  # across the antimeridian
        # East from 40N: lat2 = asin(sin40 cos1) = 39.99268 (great circle
        # bends toward the equator); dlon = atan2(sin1 cos40, cos1 - sin40 sin lat2).
        (40.0, 0.0, 90.0, 39.992678052768035, 1.3053139801813936),
    ],
)
def test_destination_point(lat: float, lon: float, az: float, lat_ref: float, lon_ref: float) -> None:
    lat2, lon2 = destination_point(lat, lon, az, ONE_DEG)
    assert lat2 == pytest.approx(lat_ref, abs=ANG_TOL)
    assert lon2 == pytest.approx(lon_ref, abs=ANG_TOL)


def test_los_layer_crossing_end_to_end() -> None:
    # Observer near Madrid, Sun at 5 deg due west, high cloud at 9 km.
    # theta from the (5 deg, 9 km) case above, then destination_point by hand.
    c = los_layer_crossing(40.0, -3.0, 5.0, 270.0, 9_000.0)
    assert c.slant_range_m == pytest.approx(95_179.0, abs=DIST_TOL)
    assert c.ground_distance_m == pytest.approx(94_686.6, abs=DIST_TOL)
    assert c.lat == pytest.approx(39.99469063293666, abs=ANG_TOL)
    assert c.lon == pytest.approx(-4.111544580138631, abs=ANG_TOL)


def test_vectorised_grid_matches_scalar() -> None:
    lats = np.array([[36.0, 37.0], [42.0, 43.0]])[..., None]  # (2, 2, 1)
    lons = np.array([[-6.0, -2.0], [-8.0, 2.0]])[..., None]
    alts = np.array([[3.0, 6.0], [9.0, 12.0]])[..., None]
    azs = np.array([[280.0, 285.0], [290.0, 295.0]])[..., None]
    heights = np.array([1_000.0, 4_000.0, 9_000.0])  # (3,)
    obs_h = np.array([[0.0, 600.0], [200.0, 1_200.0]])[..., None]

    c = los_layer_crossing(lats, lons, alts, azs, heights, obs_h)
    assert c.lat.shape == (2, 2, 3)

    for i in range(2):
        for j in range(2):
            for k in range(3):
                ref = los_layer_crossing(
                    lats[i, j, 0], lons[i, j, 0], alts[i, j, 0], azs[i, j, 0], heights[k], obs_h[i, j, 0]
                )
                for got, want in zip(c, ref):
                    np.testing.assert_allclose(got[i, j, k], want, rtol=0, atol=1e-9, equal_nan=True)
    # Layer below the 1 200 m observer with an ascending ray: no crossing.
    assert np.isnan(c.lat[1, 1, 0])
