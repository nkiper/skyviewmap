"""Local solar-eclipse circumstances for a grid of observers.

For each observer this finds the time of maximum eclipse (minimum topocentric
Sun-Moon separation) and returns the Sun's altitude/azimuth at that moment,
the eclipse magnitude and obscuration, and the duration of totality.

Assumptions and approximations
------------------------------
- Skyfield is used only for the geocentric *apparent* Sun and Moon positions
  (light-time, aberration, deflection), rotated into the Earth-fixed ITRS
  frame. Topocentric vectors are formed by subtracting the observer's ITRS
  position. This ignores the change in light-time and aberration across one
  Earth radius (< 0.01 arcsec), far below anything that matters here.
- Observers sit on the WGS84 ellipsoid at the given height. "Up" is the
  ellipsoid normal (geodetic vertical); azimuth is clockwise from true north.
- Body positions are sampled every ``step_s`` seconds and linearly
  interpolated in between. In the rotating ITRS frame the interpolation error
  for the Sun is ~100 m at 1 s steps, i.e. ~1e-9 rad -- negligible.
- The time of minimum separation is refined by fitting a parabola to the
  *squared* separation, which is exactly quadratic in time when the relative
  sky motion is linear (unlike the separation itself, which is V-shaped at a
  central eclipse).
- Sun and Moon radii follow NASA's eclipse tables (see ``constants``).
  Apparent altitude adds standard refraction (Skyfield's Bennett-type
  formula at 10 °C, 1010 hPa); the geometric altitude is returned as well.
- Central-phase start/end are found by linear interpolation of
  ``separation - |r_moon - r_sun|`` between samples. Durations are accurate
  to ~0.1 s except within ~1 s of the path edge, where they may be off by up
  to one step.
- UTC = UT1 + (UTC - UT1) as implied by the Skyfield timescale's Delta T.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from skyfield.earthlib import refract
from skyfield.framelib import itrs
from skyfield.jpllib import SpiceKernel
from skyfield.positionlib import Apparent, Barycentric
from skyfield.timelib import Timescale

from .constants import (
    R_MOON_M,
    R_SUN_M,
    REFRACTION_PRESSURE_MBAR,
    REFRACTION_TEMPERATURE_C,
    WGS84_A_M,
    WGS84_F,
)

_CENTRAL_HALF_WINDOW_S = 420.0  # longer than half of any total (<= 7.5 min) or annular (<= 12.5 min) phase


@dataclass(frozen=True)
class BodyTrack:
    """Geocentric apparent Sun and Moon positions in ITRS (metres), sampled every ``step_s``."""

    t0_utc: np.datetime64
    step_s: float
    sun_m: NDArray[np.float64]  # (n, 3)
    moon_m: NDArray[np.float64]  # (n, 3)

    def __len__(self) -> int:
        return len(self.sun_m)


@dataclass(frozen=True)
class LocalCircumstances:
    """Circumstances at each observer's maximum eclipse. Angles in degrees.

    ``magnitude`` is the fraction of the Sun's diameter covered (NASA
    definition, > 1 for total); ``obscuration`` the fraction of its area;
    ``diameter_ratio`` is Moon/Sun apparent diameter. ``central_s`` is the
    duration of the central phase (the Moon's disc wholly inside the Sun's,
    or the Sun's wholly inside the Moon's), 0 outside the central path;
    ``eclipse_type`` is 0 none, 1 partial, 2 annular, 3 total;
    ``totality_s`` equals ``central_s`` where total and 0 elsewhere.
    """

    t_max_utc: NDArray[np.datetime64]
    separation_deg: np.ndarray
    sun_alt_deg: np.ndarray
    sun_alt_apparent_deg: np.ndarray
    sun_az_deg: np.ndarray
    sun_radius_deg: np.ndarray
    moon_radius_deg: np.ndarray
    diameter_ratio: np.ndarray
    magnitude: np.ndarray
    obscuration: np.ndarray
    central_s: np.ndarray
    eclipse_type: np.ndarray
    totality_s: np.ndarray


# --- pure geometry -----------------------------------------------------------


def observer_itrs(lat_deg: ArrayLike, lon_deg: ArrayLike, height_m: ArrayLike = 0.0) -> NDArray[np.float64]:
    """Earth-fixed (ITRS/ECEF) position in metres of a point on the WGS84 ellipsoid, shape (..., 3)."""
    phi = np.radians(np.asarray(lat_deg, dtype=float))
    lam = np.radians(np.asarray(lon_deg, dtype=float))
    h = np.asarray(height_m, dtype=float)
    e2 = WGS84_F * (2.0 - WGS84_F)
    n = WGS84_A_M / np.sqrt(1.0 - e2 * np.sin(phi) ** 2)
    phi, lam, h, n = np.broadcast_arrays(phi, lam, h, n)
    return np.stack(
        [
            (n + h) * np.cos(phi) * np.cos(lam),
            (n + h) * np.cos(phi) * np.sin(lam),
            (n * (1.0 - e2) + h) * np.sin(phi),
        ],
        axis=-1,
    )


def enu_basis(lat_deg: ArrayLike, lon_deg: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Unit east, north and up vectors (each shape (..., 3)) in ITRS at a geodetic lat/lon."""
    phi = np.radians(np.asarray(lat_deg, dtype=float))
    lam = np.radians(np.asarray(lon_deg, dtype=float))
    phi, lam = np.broadcast_arrays(phi, lam)
    sp, cp, sl, cl = np.sin(phi), np.cos(phi), np.sin(lam), np.cos(lam)
    east = np.stack([-sl, cl, np.zeros_like(phi)], axis=-1)
    north = np.stack([-sp * cl, -sp * sl, cp], axis=-1)
    up = np.stack([cp * cl, cp * sl, sp], axis=-1)
    return east, north, up


def alt_az(vec: ArrayLike, lat_deg: ArrayLike, lon_deg: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Geometric altitude and azimuth (degrees) of ITRS direction ``vec`` (..., 3) seen from lat/lon."""
    vec = np.asarray(vec, dtype=float)
    east, north, up = enu_basis(lat_deg, lon_deg)
    e = np.sum(vec * east, axis=-1)
    n = np.sum(vec * north, axis=-1)
    u = np.sum(vec * up, axis=-1)
    alt = np.degrees(np.arctan2(u, np.hypot(e, n)))
    az = np.degrees(np.arctan2(e, n)) % 360.0
    return alt, az


def angular_separation(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Angle (radians) between vectors ``a`` and ``b`` along the last axis; stable near 0 and pi."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    return np.arctan2(cross, np.sum(a * b, axis=-1))


def eclipse_magnitude(sep: ArrayLike, r_sun: ArrayLike, r_moon: ArrayLike) -> NDArray[np.float64]:
    """Fraction of the Sun's diameter covered: ``(r_sun + r_moon - sep) / (2 r_sun)``, floored at 0."""
    sep, r_sun, r_moon = (np.asarray(x, dtype=float) for x in (sep, r_sun, r_moon))
    return np.maximum((r_sun + r_moon - sep) / (2.0 * r_sun), 0.0)


def obscuration(sep: ArrayLike, r_sun: ArrayLike, r_moon: ArrayLike) -> NDArray[np.float64]:
    """Fraction of the Sun's disc area covered by the Moon (flat-sky circle overlap)."""
    d, r1, r2 = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (sep, r_sun, r_moon)))
    with np.errstate(invalid="ignore", divide="ignore"):
        c1 = np.clip((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1), -1.0, 1.0)
        c2 = np.clip((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2), -1.0, 1.0)
        kite = (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
        lens = r1 * r1 * np.arccos(c1) + r2 * r2 * np.arccos(c2) - 0.5 * np.sqrt(np.maximum(kite, 0.0))
    partial = lens / (np.pi * r1 * r1)
    contained = np.minimum(r1, r2) ** 2 / (r1 * r1)
    return np.where(d >= r1 + r2, 0.0, np.where(d <= np.abs(r1 - r2), contained, partial))


# --- Skyfield ----------------------------------------------------------------


def body_track(
    eph: SpiceKernel, ts: Timescale, start_utc: datetime, end_utc: datetime, step_s: float = 1.0
) -> BodyTrack:
    """Sample geocentric apparent Sun and Moon ITRS positions from ``start_utc`` to ``end_utc``.

    Datetimes must be timezone-aware. The window must contain every
    observer's maximum eclipse (and, for central-phase durations, up to
    ~420 s either side of it).
    """
    start_utc = start_utc.astimezone(timezone.utc)
    n = int(round((end_utc - start_utc).total_seconds() / step_s)) + 1
    t0 = ts.from_datetime(start_utc)
    t = ts.tt_jd(t0.tt + np.arange(n) * (step_s / 86400.0))
    earth = cast(Barycentric, eph["earth"].at(t))
    sun = _itrs_m(earth.observe(eph["sun"]).apparent())
    moon = _itrs_m(earth.observe(eph["moon"]).apparent())
    t0_utc = np.datetime64(start_utc.replace(tzinfo=None), "ms")
    return BodyTrack(t0_utc=t0_utc, step_s=float(step_s), sun_m=sun, moon_m=moon)


def _itrs_m(position: Apparent) -> NDArray[np.float64]:
    """(n, 3) ITRS coordinates in metres.

    ``Distance.m`` is a lazily computed attribute that type checkers read as
    a method, hence the cast.
    """
    return cast(NDArray[np.float64], position.frame_xyz(itrs).m).T


# --- local circumstances -----------------------------------------------------


def local_circumstances(
    lat_deg: ArrayLike,
    lon_deg: ArrayLike,
    height_m: ArrayLike,
    track: BodyTrack,
    coarse_s: float = 30.0,
    chunk: int = 20_000,
) -> LocalCircumstances:
    """Circumstances at maximum eclipse for observers at (lat, lon, height). Inputs broadcast."""
    lat, lon, h = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (lat_deg, lon_deg, height_m)))
    shape = lat.shape
    lat, lon, h = lat.ravel(), lon.ravel(), h.ravel()

    parts = [
        _circumstances_chunk(lat[i : i + chunk], lon[i : i + chunk], h[i : i + chunk], track, coarse_s)
        for i in range(0, lat.size, chunk)
    ]
    fields = {k: np.concatenate([p[k] for p in parts]).reshape(shape) for k in parts[0]}

    t_off = fields.pop("t_offset_s")
    valid = np.isfinite(t_off)
    ms = np.where(valid, t_off * 1000.0, 0.0).round().astype("int64")
    t_max = np.where(valid, track.t0_utc + ms.astype("timedelta64[ms]"), np.datetime64("NaT", "ms"))
    return LocalCircumstances(t_max_utc=t_max, **fields)


def _sep_at(track: BodyTrack, idx: NDArray[np.intp], obs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Topocentric Sun-Moon separation (rad) at sample indices ``idx`` (P, k) for observers ``obs`` (P, 3)."""
    o = obs[:, None, :]
    return angular_separation(track.sun_m[idx] - o, track.moon_m[idx] - o)


def _circumstances_chunk(
    lat: NDArray[np.float64], lon: NDArray[np.float64], h: NDArray[np.float64], track: BodyTrack, coarse_s: float
) -> dict[str, NDArray[Any]]:
    n = len(track)
    obs = observer_itrs(lat, lon, h)
    rows = np.arange(lat.size)

    # Coarse then fine search for the sample with minimum separation.
    stride = max(1, int(round(coarse_s / track.step_s)))
    coarse_idx = np.arange(0, n, stride)
    k0 = coarse_idx[np.argmin(_sep_at(track, np.broadcast_to(coarse_idx, (lat.size, coarse_idx.size)), obs), axis=1)]
    fine_idx = np.clip(k0[:, None] + np.arange(-stride, stride + 1), 0, n - 1)
    k = fine_idx[rows, np.argmin(_sep_at(track, fine_idx, obs), axis=1)]
    at_edge = (k == 0) | (k == n - 1)

    # Parabola through squared separation at k-1, k, k+1.
    km = np.clip(k - 1, 0, n - 1)
    kp = np.clip(k + 1, 0, n - 1)
    y = _sep_at(track, np.stack([km, k, kp], axis=1), obs) ** 2
    denom = y[:, 0] - 2.0 * y[:, 1] + y[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = np.where(denom > 0, 0.5 * (y[:, 0] - y[:, 2]) / denom, 0.0)
    pos = np.clip(k + np.clip(delta, -1.0, 1.0), 0, n - 1)
    pos[at_edge] = np.nan

    # Interpolated topocentric vectors at maximum.
    safe = np.nan_to_num(pos)
    i0 = np.minimum(np.floor(safe).astype(int), n - 2)
    f = (safe - i0)[:, None]
    sun = track.sun_m[i0] * (1 - f) + track.sun_m[i0 + 1] * f - obs
    moon = track.moon_m[i0] * (1 - f) + track.moon_m[i0 + 1] * f - obs

    sep = angular_separation(sun, moon)
    r_s = np.arcsin(R_SUN_M / np.linalg.norm(sun, axis=-1))
    r_m = np.arcsin(R_MOON_M / np.linalg.norm(moon, axis=-1))
    alt, az = alt_az(sun, lat, lon)
    # Central phase: separation below the difference of the radii (total if the
    # Moon is larger, annular if smaller).
    central = _central_duration(track, obs, k, sep - np.abs(r_m - r_s))
    magnitude = eclipse_magnitude(sep, r_s, r_m)
    etype = np.where(central > 0, np.where(r_m > r_s, 3.0, 2.0), np.where(magnitude > 0, 1.0, 0.0))

    out = {
        "t_offset_s": pos * track.step_s,
        "separation_deg": np.degrees(sep),
        "sun_alt_deg": alt,
        "sun_alt_apparent_deg": refract(alt, REFRACTION_TEMPERATURE_C, REFRACTION_PRESSURE_MBAR),
        "sun_az_deg": az,
        "sun_radius_deg": np.degrees(r_s),
        "moon_radius_deg": np.degrees(r_m),
        "diameter_ratio": r_m / r_s,
        "magnitude": magnitude,
        "obscuration": obscuration(sep, r_s, r_m),
        "central_s": central,
        "eclipse_type": etype,
        "totality_s": np.where(etype == 3.0, central, 0.0),
    }
    for key in out:
        if key != "t_offset_s":
            out[key] = np.where(at_edge, np.nan, out[key])
    return out


def _central_duration(
    track: BodyTrack, obs: NDArray[np.float64], k: NDArray[np.intp], g_min: NDArray[np.float64], chunk: int = 2_000
) -> NDArray[np.float64]:
    """Seconds during which ``sep < |r_moon - r_sun|``; 0 where that never happens, NaN if out of window."""
    duration = np.zeros(k.shape)
    central = np.flatnonzero(g_min < 0)
    for start in range(0, central.size, chunk):
        sel = central[start : start + chunk]
        duration[sel] = _central_duration_chunk(track, obs[sel], k[sel], g_min[sel])
    return duration


def _central_duration_chunk(
    track: BodyTrack, obs: NDArray[np.float64], k: NDArray[np.intp], g_min: NDArray[np.float64]
) -> NDArray[np.float64]:
    n = len(track)
    w = int(np.ceil(_CENTRAL_HALF_WINDOW_S / track.step_s))
    raw = k[:, None] + np.arange(-w, w + 1)
    idx = np.clip(raw, 0, n - 1)
    o = obs[:, None, :]
    sun = track.sun_m[idx] - o
    moon = track.moon_m[idx] - o
    g = angular_separation(sun, moon) - np.abs(
        np.arcsin(R_MOON_M / np.linalg.norm(moon, axis=-1)) - np.arcsin(R_SUN_M / np.linalg.norm(sun, axis=-1))
    )
    # Use the interpolated minimum at the centre so very short central phases are still seen as negative.
    g[:, w] = np.minimum(g[:, w], g_min)
    g[(raw < 0) | (raw > n - 1)] = np.nan  # outside the track: crossing cannot be found

    def crossing(side: NDArray[np.float64]) -> NDArray[np.float64]:
        # side: g from the centre outward, shape (P, w+1); find first sample with g >= 0.
        nonneg = side >= 0
        found = nonneg.any(axis=1)
        j = np.argmax(nonneg, axis=1)  # first index >= 0 (j >= 1 since centre < 0)
        rows = np.arange(side.shape[0])
        inside, outside = side[rows, j - 1], side[rows, j]
        frac = inside / (inside - outside)
        return np.where(found, j - 1 + frac, np.nan)

    before = crossing(g[:, w::-1])
    after = crossing(g[:, w:])
    return (before + after) * track.step_s
