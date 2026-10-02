"""Load the JPL ephemeris and Skyfield timescale.

The ephemeris (DE440s, ~32 MB, valid 1849-2150) is downloaded on first use
into ``data/raw/skyfield/``, which is git-ignored.
"""

from pathlib import Path
from typing import cast

from skyfield.api import Loader
from skyfield.jpllib import SpiceKernel
from skyfield.timelib import Timescale

EPHEMERIS_FILE = "de440s.bsp"
DEFAULT_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "skyfield"


def load_ephemeris(
    data_dir: str | Path = DEFAULT_DIR, delta_t: float | None = None, download: bool = True
) -> tuple[SpiceKernel, Timescale]:
    """Return ``(eph, ts)``: the planetary ephemeris and a Skyfield timescale.

    ``delta_t`` (seconds, TT - UT1) overrides Skyfield's built-in model; pass
    e.g. 71.4 to reproduce NASA's 2026 eclipse tables. ``download=False``
    raises ``FileNotFoundError`` instead of fetching a missing ephemeris.
    """
    data_dir = Path(data_dir)
    if not download and not (data_dir / EPHEMERIS_FILE).exists():
        raise FileNotFoundError(data_dir / EPHEMERIS_FILE)
    data_dir.mkdir(parents=True, exist_ok=True)
    load = Loader(str(data_dir), verbose=False)
    # Loader.__call__ returns different types per file kind; a .bsp is a SpiceKernel.
    eph = cast(SpiceKernel, load(EPHEMERIS_FILE))
    return eph, load.timescale(delta_t=delta_t)


if __name__ == "__main__":
    load_ephemeris()
    print(f"Ephemeris ready in {DEFAULT_DIR}")
