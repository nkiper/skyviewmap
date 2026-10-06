"""Astronomical events, defined by TOML files in ``events/``.

An event file gives the date, type and climatology settings. Regions are
either listed explicitly (``[[regions]]``) or derived from the path of the
central eclipse over land (see :mod:`skyviewmapper.paths`).
"""

import tomllib
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from .grid import Grid
from .regions import Region

EVENTS_DIR = Path(__file__).resolve().parents[2] / "events"
EVENT_TYPES = ("total", "annular", "hybrid")
_TOP_KEYS = {"id", "name", "date", "type", "years", "half_window_days", "include", "regions", "display_names"}
_REGION_KEYS = {
    "name", "display_name", "grid", "res", "res_lon", "dem_box", "era5_box", "window_utc", "era5_hours", "cloud_coarsen",
}


@dataclass(frozen=True)
class Event:
    """A solar eclipse to map.

    ``years`` are the first and last climatology years (inclusive);
    ``include`` optionally limits derived regions to these
    (lat_min, lat_max, lon_min, lon_max) boxes; ``display_names`` renames
    derived regions.
    """

    id: str
    name: str
    date: date
    type: str
    years: tuple[int, int]
    half_window_days: int = 7
    include: tuple[tuple[float, float, float, float], ...] = ()
    explicit_regions: tuple[Region, ...] = ()
    display_names: dict[str, str] = field(default_factory=dict)

    @property
    def year_list(self) -> tuple[int, ...]:
        return tuple(range(self.years[0], self.years[1] + 1))

    @property
    def midnight_utc(self) -> datetime:
        return datetime.combine(self.date, time(0), tzinfo=timezone.utc)

    def window_dates(self, year: int) -> list[date]:
        """The climatology days (event day +- half window) transposed to ``year``."""
        centre = self.date.replace(year=year)
        return [centre + timedelta(days=k) for k in range(-self.half_window_days, self.half_window_days + 1)]

    @property
    def central_word(self) -> str:
        """'totality' or 'annularity', for labels."""
        return "annularity" if self.type == "annular" else "totality"


def _box(v: Any, what: str) -> tuple[float, float, float, float]:
    if not (isinstance(v, list) and len(v) == 4 and all(isinstance(x, int | float) for x in v)):
        raise ValueError(f"{what} must be [lat_min, lat_max, lon_min, lon_max]")
    lat_min, lat_max, lon_min, lon_max = (float(x) for x in v)
    if not (lat_min < lat_max and lon_min < lon_max):
        raise ValueError(f"{what} must have lat_min < lat_max and lon_min < lon_max")
    return lat_min, lat_max, lon_min, lon_max


def _region(cfg: dict[str, Any], day: date) -> Region:
    unknown = set(cfg) - _REGION_KEYS
    if unknown:
        raise ValueError(f"unknown region keys: {sorted(unknown)}")
    try:
        lat_min, lat_max, lon_min, lon_max = _box(cfg["grid"], "grid")
        start, end = (datetime.combine(day, time.fromisoformat(t), tzinfo=timezone.utc) for t in cfg["window_utc"])
        return Region(
            name=str(cfg["name"]),
            grid=Grid(lat_min, lat_max, lon_min, lon_max, float(cfg["res"]), cfg.get("res_lon")),
            dem_box=_box(cfg["dem_box"], "dem_box"),
            window_utc=(start, end),
            era5_box=_box(cfg["era5_box"], "era5_box"),
            era5_hours=tuple(int(h) for h in cfg["era5_hours"]),
            cloud_coarsen=int(cfg.get("cloud_coarsen", 5)),
            display_name=str(cfg.get("display_name", cfg["name"])),
        )
    except KeyError as err:
        raise ValueError(f"region is missing {err}") from None


def parse_event(cfg: dict[str, Any]) -> Event:
    """Validate a parsed TOML event definition."""
    unknown = set(cfg) - _TOP_KEYS
    if unknown:
        raise ValueError(f"unknown event keys: {sorted(unknown)}")
    for key in ("id", "name", "date", "type", "years"):
        if key not in cfg:
            raise ValueError(f"event is missing '{key}'")
    if not isinstance(cfg["date"], date) or isinstance(cfg["date"], datetime):
        raise ValueError("date must be a TOML date, e.g. date = 2027-08-02")
    if cfg["type"] not in EVENT_TYPES:
        raise ValueError(f"type must be one of {EVENT_TYPES}")
    years = cfg["years"]
    if not (isinstance(years, list) and len(years) == 2 and years[0] <= years[1]):
        raise ValueError("years must be [first, last]")
    day: date = cfg["date"]
    return Event(
        id=str(cfg["id"]),
        name=str(cfg["name"]),
        date=day,
        type=str(cfg["type"]),
        years=(int(years[0]), int(years[1])),
        half_window_days=int(cfg.get("half_window_days", 7)),
        include=tuple(_box(b, "include box") for b in cfg.get("include", [])),
        explicit_regions=tuple(_region(r, day) for r in cfg.get("regions", [])),
        display_names={str(k): str(v) for k, v in cfg.get("display_names", {}).items()},
    )


def list_events(events_dir: Path = EVENTS_DIR) -> list[str]:
    return sorted(p.stem for p in events_dir.glob("*.toml"))


def load_event(event_id: str, events_dir: Path = EVENTS_DIR) -> Event:
    """Load ``events/<event_id>.toml``."""
    path = events_dir / f"{event_id}.toml"
    if not path.exists():
        raise FileNotFoundError(f"no event '{event_id}'; available: {', '.join(list_events(events_dir)) or 'none'}")
    with path.open("rb") as f:
        event = parse_event(tomllib.load(f))
    if event.id != event_id:
        raise ValueError(f"{path.name}: id '{event.id}' does not match the file name")
    return event
