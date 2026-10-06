"""Event files: loading, validation, climatology windows."""

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from skyviewmapper.event import list_events, load_event, parse_event


def test_list_events() -> None:
    assert {"2026-08-12", "2027-08-02"} <= set(list_events())


def test_load_2026_explicit_regions() -> None:
    ev = load_event("2026-08-12")
    assert ev.date == date(2026, 8, 12) and ev.type == "total" and ev.years == (1996, 2025)
    names = [r.name for r in ev.explicit_regions]
    assert names == ["iberia", "iceland"]
    iberia = ev.explicit_regions[0]
    assert iberia.grid.shape == (800, 1450)
    assert iberia.window_utc == (
        datetime(2026, 8, 12, 18, 10, tzinfo=timezone.utc),
        datetime(2026, 8, 12, 18, 45, tzinfo=timezone.utc),
    )
    assert ev.explicit_regions[1].grid.dlon == 0.02


def test_load_2027_derived_regions() -> None:
    ev = load_event("2027-08-02")
    assert ev.explicit_regions == () and ev.year_list[0] == 1997 and len(ev.year_list) == 30


def test_window_dates_cross_month() -> None:
    ev = load_event("2027-08-02")
    days = ev.window_dates(2000)
    assert days[0] == date(2000, 7, 26) and days[7] == date(2000, 8, 2) and days[-1] == date(2000, 8, 9)


def test_central_word() -> None:
    base = {"id": "x", "name": "x", "date": date(2028, 1, 26), "years": [1998, 2027]}
    assert parse_event(base | {"type": "annular"}).central_word == "annularity"
    assert parse_event(base | {"type": "total"}).central_word == "totality"


@pytest.mark.parametrize(
    "change, message",
    [
        ({"date": None}, "missing 'date'"),
        ({"type": "lunar"}, "type must be"),
        ({"years": [2020, 1990]}, "years must be"),
        ({"colour": "blue"}, "unknown event keys"),
        ({"regions": [{"name": "a"}]}, "region is missing"),
        ({"include": [[40, 30, 0, 1]]}, "lat_min < lat_max"),
        ({"exclude": [[1, 2, 3]]}, "exclude box must be"),
    ],
)
def test_parse_errors(change: dict[str, object], message: str) -> None:
    cfg: dict[str, object] = {"id": "x", "name": "x", "date": date(2027, 8, 2), "type": "total", "years": [1997, 2026]}
    cfg.update(change)
    cfg = {k: v for k, v in cfg.items() if v is not None}
    with pytest.raises(ValueError, match=message):
        parse_event(cfg)


def test_id_must_match_file_name(tmp_path: Path) -> None:
    (tmp_path / "2030-01-01.toml").write_text('id = "other"\nname = "x"\ndate = 2030-01-01\ntype = "total"\nyears = [2000, 2029]\n')
    with pytest.raises(ValueError, match="does not match"):
        load_event("2030-01-01", events_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="available"):
        load_event("1999-08-11", events_dir=tmp_path)
