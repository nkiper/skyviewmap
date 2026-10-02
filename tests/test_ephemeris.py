"""Ephemeris tests.

Pure tests use hand-calculated values and a synthetic Sun/Moon track, so they
need no data. Tests marked ``ephemeris`` compare against NASA's central-line
table for 2026 Aug 12 (Espenak, eclipse.gsfc.nasa.gov, Delta T = 71.4 s) and
need DE440s in data/raw/skyfield.
"""

import math
from datetime import datetime, timezone
from typing import cast

import numpy as np
import pytest
from skyfield.jpllib import SpiceKernel
from skyfield.timelib import Timescale

from skyviewmapper.constants import R_MOON_M, R_SUN_M
from skyviewmapper.ephemeris import (
    BodyTrack,
    alt_az,
    angular_separation,
    body_track,
    eclipse_magnitude,
    local_circumstances,
    obscuration,
    observer_itrs,
)

# --- pure geometry -----------------------------------------------------------


@pytest.mark.parametrize(
    "lat, lon, h, xyz",
    [
        (0.0, 0.0, 0.0, (6_378_137.0, 0.0, 0.0)),  # equatorial radius a
        (0.0, 90.0, 100.0, (0.0, 6_378_237.0, 0.0)),
        (90.0, 0.0, 0.0, (0.0, 0.0, 6_356_752.314245)),  # polar radius b = a(1 - f)
    ],
)
def test_observer_itrs(lat: float, lon: float, h: float, xyz: tuple[float, float, float]) -> None:
    np.testing.assert_allclose(observer_itrs(lat, lon, h), xyz, atol=1e-6)


# At (0, 0): up = +x, north = +z, east = +y.
@pytest.mark.parametrize(
    "vec, alt_ref, az_ref",
    [
        ((0.0, 0.0, 1.0), 0.0, 0.0),  # north horizon
        ((0.0, 1.0, 0.0), 0.0, 90.0),  # east horizon
        ((0.0, 0.0, -1.0), 0.0, 180.0),  # south horizon
        ((0.0, -1.0, 0.0), 0.0, 270.0),  # west horizon
        ((1.0, 0.0, 1.0), 45.0, 0.0),  # halfway up, due north
        ((1.0, -1.0, 0.0), 45.0, 270.0),  # halfway up, due west
    ],
)
def test_alt_az_at_equator(vec: tuple[float, float, float], alt_ref: float, az_ref: float) -> None:
    alt, az = alt_az(np.array(vec), 0.0, 0.0)
    assert alt == pytest.approx(alt_ref, abs=1e-12)
    assert az == pytest.approx(az_ref, abs=1e-12)


def test_alt_az_zenith_at_mid_latitude() -> None:
    lat, lon = 40.0, -3.0
    up = np.array(
        [math.cos(math.radians(lat)) * math.cos(math.radians(lon)),
         math.cos(math.radians(lat)) * math.sin(math.radians(lon)),
         math.sin(math.radians(lat))]
    )
    alt, _ = alt_az(up, lat, lon)
    assert alt == pytest.approx(90.0, abs=1e-9)


def test_angular_separation() -> None:
    assert angular_separation(np.array([1.0, 0, 0]), np.array([0, 1.0, 0])) == pytest.approx(math.pi / 2)
    # Stable for tiny angles, where arccos of a dot product would lose precision.
    assert angular_separation(np.array([1.0, 0, 0]), np.array([1.0, 1e-9, 0])) == pytest.approx(1e-9, rel=1e-6)


@pytest.mark.parametrize(
    "sep, rs, rm, mag",
    [
        (0.0, 1.0, 1.03, 1.015),  # central total: (1 + 1.03) / 2
        (1.0, 1.0, 1.0, 0.5),  # Moon's limb at the Sun's centre
        (3.0, 1.0, 1.0, 0.0),  # no contact
    ],
)
def test_eclipse_magnitude(sep: float, rs: float, rm: float, mag: float) -> None:
    assert eclipse_magnitude(sep, rs, rm) == pytest.approx(mag)


@pytest.mark.parametrize(
    "sep, rs, rm, obsc",
    [
        (3.0, 1.0, 1.0, 0.0),  # disjoint
        (2.0, 1.0, 1.0, 0.0),  # touching
        (0.0, 1.0, 1.03, 1.0),  # Moon covers the Sun
        (0.0, 1.0, 0.9, 0.81),  # annular: (0.9 / 1)^2
        # Equal discs, centres one radius apart:
        # lens = 2 acos(1/2) - sqrt(3)/2 = 2pi/3 - sqrt(3)/2; / pi = 2/3 - sqrt(3)/(2pi).
        (1.0, 1.0, 1.0, 2 / 3 - math.sqrt(3) / (2 * math.pi)),  # 0.391002
    ],
)
def test_obscuration(sep: float, rs: float, rm: float, obsc: float) -> None:
    assert obscuration(sep, rs, rm) == pytest.approx(obsc, abs=1e-12)


# --- synthetic track ---------------------------------------------------------
#
# Observer at (0, 0) on the equator. The Sun sits at the zenith; the Moon moves
# along the sky so that its direction from the observer is (1, u, b) with
# u = v (t - t0). The separation is atan(sqrt(u^2 + b^2)), minimum atan(b) at
# t = t0, and totality lasts while atan(sqrt(u^2 + b^2)) < r_m - r_s, i.e.
#     duration = 2 sqrt(tan^2(r_m - r_s) - b^2) / v.

T0_UTC = np.datetime64("2026-08-12T18:00:00", "ms")
R_S, R_M, V = 0.0045, 0.0047, 2.5e-6  # rad, rad, rad/s


def synthetic_track(t0_s: float, b: float, n: int = 401) -> BodyTrack:
    obs = observer_itrs(0.0, 0.0, 0.0)
    t = np.arange(n, dtype=float)
    sun = obs + (R_SUN_M / math.sin(R_S)) * np.array([1.0, 0.0, 0.0]) * np.ones((n, 1))
    dirs = np.stack([np.ones(n), V * (t - t0_s), np.full(n, b)], axis=1)
    moon = obs + (R_MOON_M / math.sin(R_M)) * dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    return BodyTrack(t0_utc=T0_UTC, step_s=1.0, sun_m=sun, moon_m=moon)


@pytest.mark.parametrize(
    "b, duration",
    [
        # 2 sqrt(tan^2(2e-4) - 1e-8) / 2.5e-6 = 2 * 1.7320508e-4 / 2.5e-6
        (1e-4, 138.564065),
        # Central: 2 tan(2e-4) / 2.5e-6
        (0.0, 160.000002),
    ],
)
def test_synthetic_max_time_and_totality(b: float, duration: float) -> None:
    c = local_circumstances(0.0, 0.0, 0.0, synthetic_track(200.3, b))
    assert (c.t_max_utc - T0_UTC) / np.timedelta64(1, "ms") == pytest.approx(200_300, abs=1)
    assert c.separation_deg == pytest.approx(math.degrees(math.atan(b)), abs=1e-9)
    assert c.sun_alt_deg == pytest.approx(90.0, abs=1e-6)
    assert c.sun_radius_deg == pytest.approx(math.degrees(R_S), rel=1e-9)
    assert c.moon_radius_deg == pytest.approx(math.degrees(R_M), rel=1e-9)
    # (r_s + r_m - b) / (2 r_s)
    assert c.magnitude == pytest.approx((R_S + R_M - math.atan(b)) / (2 * R_S), rel=1e-6)
    assert c.obscuration == 1.0
    assert c.totality_s == pytest.approx(duration, abs=0.01)


def test_synthetic_partial_has_zero_totality() -> None:
    b = 3e-4  # > r_m - r_s, so never total
    c = local_circumstances(0.0, 0.0, 0.0, synthetic_track(200.3, b))
    assert c.totality_s == 0.0
    assert 0.0 < c.obscuration < 1.0


def test_maximum_outside_track_is_nan() -> None:
    c = local_circumstances(0.0, 0.0, 0.0, synthetic_track(450.0, 1e-4))
    assert np.isnat(c.t_max_utc)
    assert np.isnan(c.magnitude)


# --- NASA central line -------------------------------------------------------


def dms(deg: float, minutes: float) -> float:
    return math.copysign(abs(deg) + minutes / 60.0, deg)


# (UT1 hh:mm, lat, lon, Sun alt, Sun az, Moon/Sun diameter ratio, duration s)
NASA_CENTRAL_LINE = [
    # Just west of Iceland (at sea).
    ("17:46", dms(65, 10.3), dms(-25, 12.3), 26, 248, 1.039, 138.2),
    ("17:48", dms(64, 10.1), dms(-24, 45.4), 26, 250, 1.039, 138.1),
    # Iberia.
    ("18:26", dms(44, 42.8), dms(-8, 23.9), 13, 278, 1.034, 113.0),
    ("18:28", dms(43, 22.3), dms(-6, 11.3), 10, 281, 1.034, 109.3),
    ("18:30", dms(41, 49.0), dms(-3, 11.1), 8, 283, 1.033, 104.6),
    ("18:32", dms(39, 24.5), dms(2, 57.0), 2, 288, None, 95.8),
]


@pytest.fixture(scope="module")
def nasa_track(nasa_ephemeris: tuple[SpiceKernel, Timescale]) -> tuple[Timescale, BodyTrack]:
    eph, ts = nasa_ephemeris
    start = datetime(2026, 8, 12, 17, 20, tzinfo=timezone.utc)
    end = datetime(2026, 8, 12, 18, 45, tzinfo=timezone.utc)
    return ts, body_track(eph, ts, start, end)


@pytest.mark.ephemeris
@pytest.mark.parametrize("ut, lat, lon, alt, az, ratio, duration", NASA_CENTRAL_LINE)
def test_nasa_central_line(
    nasa_track: tuple[Timescale, BodyTrack],
    ut: str,
    lat: float,
    lon: float,
    alt: float,
    az: float,
    ratio: float | None,
    duration: float,
) -> None:
    ts, track = nasa_track
    c = local_circumstances(lat, lon, 0.0, track)

    hh, mm = map(int, ut.split(":"))
    # NASA times are UT1; convert to UTC with the same Delta T. Scalar Time -> datetime.
    expected_utc = cast(datetime, ts.ut1(2026, 8, 12, hh, mm, 0).utc_datetime())
    expected = np.datetime64(expected_utc.replace(tzinfo=None), "ms")
    assert abs((c.t_max_utc - expected) / np.timedelta64(1, "ms")) < 1_000
    assert c.separation_deg * 3600 < 3.0  # on the central line
    assert c.sun_alt_deg == pytest.approx(alt, abs=0.6)  # table rounds to 1 deg
    assert c.sun_az_deg == pytest.approx(az, abs=0.6)
    if ratio is not None:
        assert c.diameter_ratio == pytest.approx(ratio, abs=0.0015)
    assert c.totality_s == pytest.approx(duration, abs=0.5)


@pytest.mark.ephemeris
def test_partial_eclipse_seville(nasa_track: tuple[Timescale, BodyTrack]) -> None:
    # Published magnitude ~0.951 (timeanddate / theskylive; exact site unknown).
    _, track = nasa_track
    c = local_circumstances(37.389, -5.984, 0.0, track)
    assert c.magnitude == pytest.approx(0.951, abs=0.005)
    assert c.totality_s == 0.0


@pytest.mark.ephemeris
def test_grid_shape_and_coverage(nasa_track: tuple[Timescale, BodyTrack]) -> None:
    _, track = nasa_track
    lat, lon = np.meshgrid(np.arange(36.0, 44.01, 0.5), np.arange(-10.0, 4.51, 0.5), indexing="ij")
    c = local_circumstances(lat, lon, 0.0, track)
    assert c.magnitude.shape == lat.shape
    assert not np.isnan(c.magnitude).any()
    assert (c.totality_s > 0).any() and (c.totality_s == 0).any()
