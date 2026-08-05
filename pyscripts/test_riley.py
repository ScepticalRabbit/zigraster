# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
import argparse
import os
from time import perf_counter

import pytest


FORCE_ZIG_RENDER: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-zig-render",
        action="store_true",
        help="Re-run the Zig demo renders even if cached BMPs exist.",
    )
    return parser.parse_args()


def main() -> None:
    total_start = perf_counter()
    args = parse_args()
    os.environ["RILEY_FORCE_ZIG_RENDER"] = (
        "1" if FORCE_ZIG_RENDER or args.force_zig_render else "0"
    )
    pytest_args = ["-s", "--pyargs", "riley.pytests.test_riley"]
    exit_code = pytest.main(pytest_args)
    total_elapsed = perf_counter() - total_start
    print(f"test_riley completed in {total_elapsed:.3f}s.")
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(str(err))
        sys.exit(1)
