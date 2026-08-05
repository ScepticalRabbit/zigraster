#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import subprocess
import sys


RUN_FULLRASTER = True
RUN_CAM = True
RUN_GEOM = True
RUN_SPHERE2000 = True
RUN_SPHERE2000ZOOM = True
RUN_DICUQ = True
RUNS = 25


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
    command = [str(python_path()), str(script_path)]
    if RUNS > 0:
        command.extend(["--runs", str(RUNS)])
    subprocess.run(
        command,
        cwd=repo_root(),
        check=True,
    )


def main() -> int:
    script_names: list[str] = []

    if RUN_FULLRASTER:
        script_names.append("bench_fullraster.py")
    if RUN_CAM:
        script_names.append("benchcam.py")
    if RUN_GEOM:
        script_names.append("bench_geom.py")
    if RUN_SPHERE2000:
        script_names.append("bench_sphere2000.py")
    if RUN_SPHERE2000ZOOM:
        script_names.append("bench_sphere2000zoom.py")
    if RUN_DICUQ:
        script_names.append("bench_dicuq.py")

    for script_name in script_names:
        run_script(script_name)

    print("Completed selected benchmark scripts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
