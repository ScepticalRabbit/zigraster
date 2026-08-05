#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import subprocess
import sys


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


def python_path() -> pathlib.Path:
    venv_python = repo_root() / ".venv" / "bin" / "python"
    if venv_python.exists():
        return venv_python
    return pathlib.Path(sys.executable)


def run_script(script_name: str) -> None:
    script_path = repo_root() / "scripts" / script_name
    print(f"Running {script_name}...")
    subprocess.run(
        [str(python_path()), str(script_path)],
        cwd=repo_root(),
        check=True,
    )


def main() -> int:
    verif_scripts = [
        "paper_verif_2_helper.py",
        "paper_verif_6_helper.py",
    ]
    paper_scripts = [
        "paper_verif_1.py",
        "paper_verif_2.py",
        "paper_verif_3.py",
        "paper_verif_4.py",
        "paper_verif_5.py",
        "paper_verif_6.py",
    ]

    print("Running verification analysis scripts...")
    for script_name in verif_scripts:
        run_script(script_name)

    print("Running paper asset scripts...")
    for script_name in paper_scripts:
        run_script(script_name)

    print("Completed all verification scripts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
