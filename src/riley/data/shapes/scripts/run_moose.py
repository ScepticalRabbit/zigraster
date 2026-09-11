from __future__ import annotations

import argparse
from pathlib import Path

from run_helpers import extract_exodus_csvs, run_moose_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a MOOSE simulation input (.i) to generate "
            "Exodus output (.e)."
        )
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input MOOSE file path (.i)",
    )
    parser.add_argument(
        "-m", "--mesh",
        type=str,
        default=None,
        help="External mesh file path (.msh) to override Mesh/file",
    )
    parser.add_argument(
        "-t", "--threads",
        type=int,
        default=1,
        help="Number of threads (default: 1)",
    )
    parser.add_argument(
        "--extract-csv",
        action="store_true",
        help="Extract coords.csv and connectivity.csv from output Exodus file",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (Path.cwd() / input_path).resolve()

    mesh_path = None
    if args.mesh:
        mesh_path = Path(args.mesh)
        if not mesh_path.is_absolute():
            mesh_path = (Path.cwd() / mesh_path).resolve()

    out_exo = run_moose_file(
        input_path,
        mesh_path=mesh_path,
        threads=args.threads,
    )

    if args.extract_csv:
        extract_exodus_csvs(out_exo)


if __name__ == "__main__":
    main()
