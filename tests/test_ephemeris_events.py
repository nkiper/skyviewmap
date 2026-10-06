"""Central-phase checks against NASA path tables for 2027 (total) and 2028 (annular).

NASA tables (Espenak, eclipse.gsfc.nasa.gov) use Delta T = 71.7 s (2027) and
71.9 s (2028); times are UT1. Skipped when the ephemeris is not downloaded.
"""

import math
from datetime import datetime, timezone
from typing import cast

import numpy as np
import pytest

from skyviewmapper.ephemeris import body_track, local_circumstances
from skyviewmapper.io.skyfield_data import load_ephemeris


def dms(deg: float, minutes: float) -> float:
    return math.copysign(abs(deg) + minutes / 60.0, deg)


# (event day, delta T, window start/end UT, [(UT1 hh:mm, lat, lon, alt, az, ratio, duration s, type)])
CASES = [
    (
        (2027, 8, 2), 71.7, ((8, 30), (10, 25)),
        [
            ("08:46", dms(35, 39.6), dms(-6, 43.3), 37, 94, 1.072, 288.3, 3),  # off southern Spain
            ("10:00", dms(26, 53.3), dms(31, 0.8), 81, 177, 1.079, 383.2, 3),  # Egypt, longest duration
            ("10:10", dms(24, 46.5), dms(34, 15.4), 81, 215, 1.079, 381.8, 3),
        ],
    ),
    (
        (2028, 1, 26), 71.9, ((15, 15), (16, 15)),
        [
            ("15:30", dms(6, 51.2), dms(-47, 0.1), 64, 185, 0.920, 610.2, 2),  # annular, tropical Atlantic
            ("16:00", dms(13, 26.0), dms(-40, 15.3), 54, 207, 0.919, 570.9, 2),
        ],
    ),
]


@pytest.mark.ephemeris
@pytest.mark.parametrize("day, delta_t, window, points", CASES)
def test_nasa_central_lines(
    day: tuple[int, int, int],
    delta_t: float,
    window: tuple[tuple[int, int], tuple[int, int]],
    points: list[tuple[str, float, float, int, int, float, float, int]],
) -> None:
    try:
        eph, ts = load_ephemeris(delta_t=delta_t, download=False)
    except FileNotFoundError:
        pytest.skip("DE440s not downloaded")
    (h0, m0), (h1, m1) = window
    track = body_track(eph, ts, datetime(*day, h0, m0, tzinfo=timezone.utc), datetime(*day, h1, m1, tzinfo=timezone.utc))
    for ut, lat, lon, alt, az, ratio, duration, etype in points:
        c = local_circumstances(lat, lon, 0.0, track)
        hh, mm = map(int, ut.split(":"))
        expected_utc = cast(datetime, ts.ut1(*day, hh, mm, 0).utc_datetime())
        expected = np.datetime64(expected_utc.replace(tzinfo=None), "ms")
        # Our maxima are systematically 0.6-1.2 s later than NASA's here (0.5 s in
        # 2026), cause unknown (e.g. lunar centre-of-figure conventions);
        # irrelevant for a climatology map, but kept visible by this bound.
        assert 0 < (c.t_max_utc - expected) / np.timedelta64(1, "ms") < 1_500, ut
        assert c.separation_deg * 3600 < 3.0, ut
        assert c.sun_alt_deg == pytest.approx(alt, abs=0.6), ut
        assert c.sun_az_deg == pytest.approx(az, abs=0.6), ut
        assert c.diameter_ratio == pytest.approx(ratio, abs=0.0015), ut
        assert c.central_s == pytest.approx(duration, abs=0.5), ut
        assert c.eclipse_type == etype, ut
        assert c.totality_s == (c.central_s if etype == 3 else 0.0), ut
