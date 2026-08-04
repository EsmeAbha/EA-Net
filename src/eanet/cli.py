"""Command-line entry point: ``eanet {eval,sweep,shifts}``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from eanet.config import load_config
from eanet.data.shifts import SHIFTS
from eanet.experiment import run_experiment, run_sweep


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config field, e.g. --set tta.method=tent (repeatable)",
    )
    parser.add_argument("--out", default=None, help="write JSON results here")


def _write(results, out: str | None) -> None:
    text = json.dumps(results, indent=2, default=str)
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")
    else:
        print(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eanet", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval", help="train on source, evaluate under one shift")
    _add_common(p_eval)

    p_sweep = sub.add_parser("sweep", help="grid over shift x severity x method")
    _add_common(p_sweep)
    p_sweep.add_argument("--shifts", nargs="+", default=["none", "noise", "channel_dropout"])
    p_sweep.add_argument("--methods", nargs="+", default=["source", "norm", "tent", "sar"])
    p_sweep.add_argument("--severities", nargs="+", type=int, default=[3])

    sub.add_parser("shifts", help="list available shift kinds")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command == "shifts":
        for name in sorted(SHIFTS):
            print(name)
        return 0

    cfg = load_config(args.config, args.overrides)

    if args.command == "eval":
        results, _ = run_experiment(cfg)
        _write(results, args.out)
    elif args.command == "sweep":
        rows = run_sweep(cfg, args.shifts, args.methods, args.severities)
        _write(rows, args.out)

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
