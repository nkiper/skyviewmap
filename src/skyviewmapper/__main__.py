"""Command line for skyviewmapper.

    python -m skyviewmapper events                         list event files
    python -m skyviewmapper regions  --event 2027-08-02    show an event's regions and data needs
    python -m skyviewmapper download --event 2027-08-02    fetch DEM tiles and ERA5 cloud samples
    python -m skyviewmapper run      --event 2027-08-02 [--region NAME] [--no-download] [--min-central S]
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .event import Event, list_events, load_event
from .regions import Region


def _regions(event: Event, names: list[str] | None, download: bool) -> list[Region]:
    from .paths import event_regions

    regions = event_regions(event, download=download)
    if names:
        known = {r.name for r in regions}
        unknown = [n for n in names if n not in known]
        if unknown:
            raise SystemExit(f"unknown region(s) {unknown}; {event.id} has: {', '.join(sorted(known))}")
        regions = [r for r in regions if r.name in names]
    return regions


def cmd_events(_: argparse.Namespace) -> int:
    for eid in list_events():
        ev = load_event(eid)
        kind = "explicit" if ev.explicit_regions else "derived from the path"
        print(f"{eid}  {ev.name}  ({ev.type}; regions {kind})")
    return 0


def cmd_regions(args: argparse.Namespace) -> int:
    from .io.dem import _tile_ranges, land_tiles

    event = load_event(args.event)
    regions = _regions(event, None, download=True)
    land = land_tiles()
    total_cells = total_tiles = 0
    print(f"{event.name}: {len(regions)} region(s)")
    for r in regions:
        n = r.grid.shape[0] * r.grid.shape[1]
        lats, lons = _tile_ranges(*r.dem_box)
        tiles = sum((la, lo) in land for la in lats for lo in lons)
        total_cells += n
        total_tiles += tiles
        g = r.grid
        print(
            f"  {r.name:15s} {r.label if r.display_name else '':28s} grid {g.lat_min:g}..{g.lat_max:g}N "
            f"{g.lon_min:g}..{g.lon_max:g}E ({g.dlat:g} x {g.dlon:g} deg, {n / 1e6:.2f}M cells)  "
            f"{r.window_utc[0]:%H:%M}-{r.window_utc[1]:%H:%M} UT  ERA5 hours {list(r.era5_hours)}  ~{tiles} DEM tiles"
        )
    print(f"total: {total_cells / 1e6:.2f}M cells, ~{total_tiles} DEM land tiles (~{total_tiles * 4 / 1000:.1f} GB at most)")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    from .io.dem import download_tiles
    from .io.era5 import load_cloud_samples

    event = load_event(args.event)
    regions = _regions(event, args.region, download=True)
    for r in regions:
        print(f"{r.name}: {len(download_tiles(*r.dem_box))} DEM tiles", flush=True)
    # The CDS queues requests; a few in parallel shortens the wait.
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(load_cloud_samples, event, r, True): r for r in regions}
        for fut in as_completed(futures):
            print(f"{futures[fut].name}: ERA5 {dict(fut.result().sizes)}", flush=True)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .io.outputs import write_outputs
    from .pipeline import run_region

    event = load_event(args.event)
    for region in _regions(event, args.region, download=not args.no_download):
        t0 = time.time()
        try:
            ds = run_region(event, region, download=not args.no_download)
        except FileNotFoundError as err:
            print(f"{region.name}: missing data ({err}); run `download` or drop --no-download", file=sys.stderr)
            return 1
        paths = write_outputs(ds, args.out, min_central_s=args.min_central)
        print(f"{region.name}: done in {time.time() - t0:.0f} s -> {paths[0].parent}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m skyviewmapper", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("events", help="list event files").set_defaults(func=cmd_events)

    def with_event(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--event", required=True, help="event id, e.g. 2027-08-02 (see `events`)")
        return p

    with_event(sub.add_parser("regions", help="show an event's regions")).set_defaults(func=cmd_regions)
    dl = with_event(sub.add_parser("download", help="fetch DEM tiles and ERA5 cloud samples"))
    dl.add_argument("--region", nargs="*", help="limit to these regions")
    dl.add_argument("--parallel", type=int, default=3, help="concurrent ERA5 requests (default 3)")
    dl.set_defaults(func=cmd_download)
    run = with_event(sub.add_parser("run", help="compute maps for an event"))
    run.add_argument("--region", nargs="*", help="limit to these regions")
    run.add_argument("--no-download", action="store_true", help="fail instead of downloading missing data")
    run.add_argument("--out", type=Path, default=None, help="output directory (default: outputs/)")
    run.add_argument("--min-central", type=float, default=60.0, metavar="SECONDS",
                     help="minimum totality/annularity for the top-spots table (default: 60)")
    run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
