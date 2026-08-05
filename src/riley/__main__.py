# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
from __future__ import annotations

import argparse
import importlib
import sys


_DEMO_FUNCS = {
    "demo_sphere200": "riley.pydemos.demo_sphere200",
    "demo_rabbits": "riley.pydemos.demo_rabbits",
    "demo_dicuq": "riley.pydemos.demo_dicuq",
    "demo_dic_from_exodus": "riley.pydemos.demo_dic_from_exodus",
    "demo_stereocal": "riley.pydemos.demo_stereocal",
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m riley")
    parser.add_argument(
        "command",
        nargs="?",
        help="Demo name to run, or 'test' to run the packaged pytest suite.",
    )
    args, extra_args = parser.parse_known_args()

    if args.command is None:
        parser.error(
            "expected a command such as 'demo_dicuq' or 'test'. "
            f"Available demos: {', '.join(sorted(_DEMO_FUNCS))}.",
        )

    if args.command == "test":
        import pytest

        pytest_args = ["-s", "--pyargs", "riley.pytests", *extra_args]
        raise SystemExit(pytest.main(pytest_args))

    if args.command not in _DEMO_FUNCS:
        parser.error(
            f"unknown command '{args.command}'. Available demos: "
            f"{', '.join(sorted(_DEMO_FUNCS))}, test.",
        )

    demo_module = importlib.import_module(_DEMO_FUNCS[args.command])
    demo_module.main()


if __name__ == "__main__":
    main()
