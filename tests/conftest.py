from datetime import date

import pytest
from skyfield.jpllib import SpiceKernel
from skyfield.timelib import Timescale

from skyviewmapper.event import Event, load_event
from skyviewmapper.io.skyfield_data import load_ephemeris
from skyviewmapper.regions import Region

# NASA's 2026 Aug 12 path tables were computed with Delta T = 71.4 s.
NASA_DELTA_T_S = 71.4


@pytest.fixture(scope="session")
def nasa_ephemeris() -> tuple[SpiceKernel, Timescale]:
    """(eph, ts) matching NASA's Delta T; skips if the ephemeris is not downloaded."""
    try:
        return load_ephemeris(delta_t=NASA_DELTA_T_S, download=False)
    except FileNotFoundError:
        pytest.skip("DE440s not downloaded: run `.venv/bin/python -m skyviewmapper.io.skyfield_data`")


def event_2026() -> Event:
    """The v1 event (explicit Iberia and Iceland regions)."""
    return load_event("2026-08-12")


def region_2026(name: str) -> Region:
    return next(r for r in event_2026().explicit_regions if r.name == name)


def synthetic_event(day: date = date(2000, 8, 12), years: tuple[int, int] = (2000, 2001), half_window: int = 1) -> Event:
    """A small event for reshaping tests."""
    return Event(id="test", name="Test eclipse", date=day, type="total", years=years, half_window_days=half_window)
