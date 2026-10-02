import pytest

from skyviewmapper.io.skyfield_data import load_ephemeris

# NASA's 2026 Aug 12 path tables were computed with Delta T = 71.4 s.
NASA_DELTA_T_S = 71.4


@pytest.fixture(scope="session")
def nasa_ephemeris():
    """(eph, ts) matching NASA's Delta T; skips if the ephemeris is not downloaded."""
    try:
        return load_ephemeris(delta_t=NASA_DELTA_T_S, download=False)
    except FileNotFoundError:
        pytest.skip("DE440s not downloaded: run `.venv/bin/python -m skyviewmapper.io.skyfield_data`")
