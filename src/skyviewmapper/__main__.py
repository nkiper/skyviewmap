"""Command line: ``python -m skyviewmapper run [--region iberia|iceland|all] [--no-download] [--min-totality S]``."""

import argparse
import sys
import time
from pathlib import Path

from .regions import REGIONS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m skyviewmapper", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="compute maps for one or all regions")
    run.add_argument("--region", choices=[*REGIONS, "all"], default="all")
    run.add_argument("--no-download", action="store_true", help="fail instead of downloading missing data")
    run.add_argument("--out", type=Path, default=None, help="output directory (default: outputs/)")
    run.add_argument("--min-totality", type=float, default=60.0, metavar="SECONDS",
                     help="minimum totality for the top-spots table (default: 60)")
    args = parser.parse_args(argv)

    from .io.outputs import write_outputs
    from .pipeline import run_region

    names = list(REGIONS) if args.region == "all" else [args.region]
    for name in names:
        t0 = time.time()
        try:
            ds = run_region(REGIONS[name], download=not args.no_download)
        except FileNotFoundError as err:
            print(f"{name}: missing data ({err}); run without --no-download to fetch it", file=sys.stderr)
            return 1
        paths = write_outputs(ds, args.out, min_totality_s=args.min_totality)
        print(f"{name}: done in {time.time() - t0:.0f} s")
        for p in paths:
            print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
