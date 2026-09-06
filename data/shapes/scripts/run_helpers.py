from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import netCDF4
import numpy as np


def find_gmsh_path() -> Path:
    candidates = (
        Path.home() / "gmsh/bin/gmsh",
        Path("/usr/local/bin/gmsh"),
        Path("/usr/bin/gmsh"),
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate

    on_path = shutil.which("gmsh")
    if on_path:
        return Path(on_path)

    raise FileNotFoundError("Could not locate gmsh executable.")


def find_moose_path() -> Path:
    candidates = (
        Path.home() / "proteus/proteus-opt",
        Path.home() / "moose/test/moose_test-opt",
        Path.home() / "projects/moose/modules/combined/combined-opt",
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate

    raise FileNotFoundError(
        "Could not locate MOOSE executable (checked proteus-opt, "
        "moose_test-opt, combined-opt)."
    )


def run_gmsh_file(
    geo_path: Path,
    out_msh: Path | None = None,
    threads: int = 4,
) -> Path:
    geo_path = Path(geo_path).resolve()
    if not geo_path.is_file():
        raise FileNotFoundError(f"Gmsh .geo file not found: {geo_path}")

    if out_msh is None:
        out_msh = geo_path.with_suffix(".msh")
    else:
        out_msh = Path(out_msh).resolve()

    gmsh_bin = find_gmsh_path()
    cmd = [
        str(gmsh_bin),
        "-3",
        str(geo_path),
        "-o",
        str(out_msh),
        "-nt",
        str(threads),
    ]

    print(f"Running Gmsh: {' '.join(cmd)}")
    start_time = time.perf_counter()
    result = subprocess.run(
        cmd,
        cwd=str(geo_path.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start_time

    if result.returncode != 0 or not out_msh.is_file():
        print(result.stdout)
        raise RuntimeError(
            f"Gmsh failed on {geo_path.name} (exit code {result.returncode})"
        )

    print(
        f"Generated {out_msh.name} ({out_msh.stat().st_size} bytes) "
        f"in {elapsed:.2f}s"
    )
    return out_msh


def run_moose_file(
    input_path: Path,
    mesh_path: Path | None = None,
    threads: int = 1,
    tasks: int = 1,
) -> Path:
    input_path = Path(input_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"MOOSE input not found: {input_path}")

    moose_bin = find_moose_path()
    cmd = [str(moose_bin), "-i", input_path.name]

    if mesh_path is not None:
        mesh_path = Path(mesh_path).resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
        cmd.extend(["Mesh/file=" + str(mesh_path)])

    if threads > 1:
        cmd.append(f"--n-threads={threads}")

    out_log = input_path.with_suffix(".out")
    print(f"Running MOOSE: {' '.join(cmd)}")
    print(f"Logging stdout to: {out_log.name}")

    start_time = time.perf_counter()
    with open(out_log, "w", encoding="utf-8") as out_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(input_path.parent),
            text=True,
            bufsize=1,
        )

        try:
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    out_file.write(line)
                    out_file.flush()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise

        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

    elapsed = time.perf_counter() - start_time
    print(f"MOOSE completed in {elapsed:.2f}s")

    # Locate output exodus file
    # Default MOOSE output is <input_base>_out.e
    base_name = input_path.stem
    candidates = (
        input_path.parent / f"{base_name}_out.e",
        input_path.parent / f"{base_name}.e",
    )
    for cand in candidates:
        if cand.is_file():
            return cand

    raise FileNotFoundError(
        f"Could not find generated Exodus output for {input_path.name}"
    )


def extract_exodus_csvs(
    exodus_path: Path,
    out_dir: Path | None = None,
    prefix: str = "",
) -> dict[str, Path]:
    import riley

    exodus_path = Path(exodus_path).resolve()
    if out_dir is None:
        out_dir = exodus_path.parent
    else:
        out_dir = Path(out_dir).resolve()

    coords_csv = out_dir / f"{prefix}coords.csv"
    connect_csv = out_dir / f"{prefix}connectivity.csv"
    disp_x_csv = out_dir / f"{prefix}disp_x.csv"
    disp_y_csv = out_dir / f"{prefix}disp_y.csv"
    disp_z_csv = out_dir / f"{prefix}disp_z.csv"
    temp_csv = out_dir / f"{prefix}temperature.csv"

    sim = riley.load_exodus(
        exodus_path,
        disp_keys=("disp_x", "disp_y", "disp_z"),
        nodal_keys=("temperature",),
    )
    block = sim.blocks["connect1"]
    converted = riley.convert_mesh(
        sim.coords,
        block.connect,
        riley.ConnectConvention(
            block.elem_type,
            riley.EConnectAxis.ROW,
            1,
            riley.ENodeOrder.EXODUS,
        ),
    )
    riley.verify_mesh(converted)

    np.savetxt(coords_csv, converted.coords, delimiter=",", fmt="%.8f")
    np.savetxt(connect_csv, converted.connect, delimiter=",", fmt="%d")

    out_paths = {
        "coords": coords_csv,
        "connectivity": connect_csv,
    }

    if sim.disp is not None and len(sim.disp) == 3:
        np.savetxt(disp_x_csv, sim.disp[0], delimiter=",", fmt="%.8e")
        np.savetxt(disp_y_csv, sim.disp[1], delimiter=",", fmt="%.8e")
        np.savetxt(disp_z_csv, sim.disp[2], delimiter=",", fmt="%.8e")
        out_paths["disp_x"] = disp_x_csv
        out_paths["disp_y"] = disp_y_csv
        out_paths["disp_z"] = disp_z_csv

    if "temperature" in sim.nodal_vars:
        np.savetxt(
            temp_csv,
            sim.nodal_vars["temperature"],
            delimiter=",",
            fmt="%.8e",
        )
        out_paths["temperature"] = temp_csv

    print(
        f"Saved CSVs for {prefix.rstrip('_')}: "
        f"coords {converted.coords.shape}, "
        f"connect {converted.connect.shape}, fields "
        f"{sim.disp[0].shape if sim.disp is not None else 'None'}"
    )
    return out_paths
