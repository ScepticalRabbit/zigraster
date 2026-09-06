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
    import netCDF4
    import riley

    exodus_path = Path(exodus_path).resolve()
    if out_dir is None:
        out_dir = exodus_path.parent
    else:
        out_dir = Path(out_dir).resolve()

    with netCDF4.Dataset(exodus_path, mode="r") as ds:
        connect_keys = [
            k for k in ds.variables if k.startswith("connect")
        ]
        nodal_names: list[str] = []
        if "name_nod_var" in ds.variables:
            raw = np.ma.filled(ds.variables["name_nod_var"][:], b"")
            nodal_names = [
                str(n).strip() for n in netCDF4.chartostring(raw)
            ]

    disp_k = tuple(
        k for k in ("disp_x", "disp_y", "disp_z") if k in nodal_names
    )
    disp_keys = disp_k if disp_k else None

    nodal_k = tuple(k for k in ("temperature",) if k in nodal_names)
    nodal_keys = nodal_k if nodal_k else None

    sim = riley.load_exodus(
        exodus_path,
        connect_keys=connect_keys,
        disp_keys=disp_keys,
        nodal_keys=nodal_keys,
    )

    coords_csv = out_dir / f"{prefix}coords.csv"
    np.savetxt(coords_csv, sim.coords, delimiter=",", fmt="%.8f")
    out_paths: dict[str, Path] = {"coords": coords_csv}

    if len(sim.blocks) == 1:
        block = list(sim.blocks.values())[0]
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
        connect_csv = out_dir / f"{prefix}connectivity.csv"
        np.savetxt(connect_csv, converted.connect, delimiter=",", fmt="%d")
        out_paths["connectivity"] = connect_csv
    else:
        block_names = ("cube", "cylinder")
        for idx, (b_key, block) in enumerate(sim.blocks.items()):
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
            b_name = (
                block_names[idx]
                if idx < len(block_names)
                else b_key
            )
            connect_csv = out_dir / f"{prefix}{b_name}_connectivity.csv"
            np.savetxt(
                connect_csv, converted.connect, delimiter=",", fmt="%d"
            )
            out_paths[f"{b_name}_connectivity"] = connect_csv

    if sim.disp is not None:
        if len(sim.disp) == 3:
            for axis_idx, axis in enumerate("xyz"):
                disp_csv = out_dir / f"{prefix}disp_{axis}.csv"
                np.savetxt(
                    disp_csv, sim.disp[axis_idx], delimiter=",", fmt="%.8e"
                )
                out_paths[f"disp_{axis}"] = disp_csv
        elif len(sim.disp) == 2:
            for axis_idx, axis in enumerate("xy"):
                disp_csv = out_dir / f"{prefix}disp_{axis}.csv"
                np.savetxt(
                    disp_csv, sim.disp[axis_idx], delimiter=",", fmt="%.8e"
                )
                out_paths[f"disp_{axis}"] = disp_csv
            disp_z_csv = out_dir / f"{prefix}disp_z.csv"
            np.savetxt(
                disp_z_csv,
                np.zeros_like(sim.disp[0]),
                delimiter=",",
                fmt="%.8e",
            )
            out_paths["disp_z"] = disp_z_csv

    if "temperature" in sim.nodal_vars:
        temp_csv = out_dir / f"{prefix}temperature.csv"
        np.savetxt(
            temp_csv,
            sim.nodal_vars["temperature"],
            delimiter=",",
            fmt="%.8e",
        )
        out_paths["temperature"] = temp_csv

    print(
        f"Saved CSVs for {prefix.rstrip('_')}: "
        f"coords {sim.coords.shape}, "
        f"blocks {len(sim.blocks)}, fields "
        f"{sim.disp[0].shape if sim.disp is not None else 'None'}"
    )
    return out_paths
