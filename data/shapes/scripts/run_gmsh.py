from __future__ import annotations

import argparse
from pathlib import Path
import sys

from run_helpers import run_gmsh_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Gmsh to generate a 3D mesh (.msh) from a .geo file."
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input .geo file path",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output .msh file path (optional)",
    )
    parser.add_argument(
        "-t", "--threads",
        type=int,
        default=4,
        help="Number of threads for meshing (default: 4)",
    )
    args = parser.parse_args()

    geo_path = Path(args.input)
    if not geo_path.is_absolute():
        geo_path = (Path.cwd() / geo_path).resolve()

    out_msh = Path(args.output).resolve() if args.output else None
    run_gmsh_file(geo_path, out_msh=out_msh, threads=args.threads)


if __name__ == "__main__":
    main()
