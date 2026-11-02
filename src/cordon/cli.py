"""Command-line entry point. Stages run in the order listed in the README."""

from __future__ import annotations

import argparse


def _month(text: str) -> tuple[int, int]:
    year, month = text.split("-")
    return int(year), int(month)


def main() -> None:
    from . import config

    parser = argparse.ArgumentParser(prog="cordon", description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)

    stages.add_parser("zones", help="classify taxi zones against the MTA geofence")

    build = stages.add_parser("build", help="download and aggregate TLC months (about 30 GB read)")
    build.add_argument("--first", type=_month, default=config.FIRST_MONTH)
    build.add_argument("--last", type=_month, default=config.LAST_MONTH)
    build.add_argument("--keep-raw", action="store_true")

    stages.add_parser("panel", help="collapse interim tables into committed aggregates")

    estimate = stages.add_parser("estimate", help="run every estimator from committed aggregates")
    estimate.add_argument("--draws", type=int, default=config.WILD_DRAWS)

    stages.add_parser("report", help="render the site from results/")
    stages.add_parser("feecheck", help="record charge amounts around January 2026 (network)")

    args = parser.parse_args()
    if args.stage == "zones":
        from .pipeline import zones_stage

        zones_stage()
    elif args.stage == "build":
        from .pipeline import build_stage

        build_stage(args.first, args.last, keep_raw=args.keep_raw)
    elif args.stage == "panel":
        from .panel import panel_stage

        panel_stage()
    elif args.stage == "estimate":
        from .estimate import estimate_stage

        estimate_stage(draws=args.draws)
    elif args.stage == "report":
        from .report import report_stage

        report_stage()
    elif args.stage == "feecheck":
        from .pipeline import fee_check_stage

        fee_check_stage()


if __name__ == "__main__":
    main()
