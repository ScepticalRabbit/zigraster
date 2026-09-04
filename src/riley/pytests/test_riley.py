# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
from PIL import Image


PROJECT_ROOT = Path.cwd()
PYTHON_EXE = Path(sys.executable)
ZIG_CMD = [
    "zig",
    "run",
    "-lc",
    "-O",
    "ReleaseFast",
    "./src/run_all_demos.zig",
]
EXACT_8BIT_COMPARE = True
FLOAT_FALLBACK_ABS_TOL = 0.5

STANDARD_DEMO_CASES = (
    (
        "sphere200",
        [str(PYTHON_EXE), "-m", "riley", "demo_sphere200"],
        "out/demo-sphere200",
        "out-riley-py/demo-sphere200",
        None,
    ),
    (
        "psf",
        [str(PYTHON_EXE), "-m", "riley", "demo_psf"],
        "out/demo-psf",
        "out-riley-py/demo-psf",
        None,
    ),
    (
        "rabbits",
        [str(PYTHON_EXE), "-m", "riley", "demo_rabbits"],
        "out/demo-rabbits",
        "out-riley-py/demo-rabbits",
        None,
    ),
    (
        "dicuq",
        [str(PYTHON_EXE), "-m", "riley", "demo_dicuq"],
        "out/demo-dicuq",
        "out-riley-py/demo-dicuq",
        2,
    ),
    (
        "stereocal",
        [str(PYTHON_EXE), "-m", "riley", "demo_stereocal"],
        "out/demo-stereocal",
        "out-riley-py/demo-stereocal",
        8,
    ),
)

EXODUS_DEMO_CASE = (
    "dic_from_exodus",
    [str(PYTHON_EXE), "-m", "riley", "demo_dic_from_exodus"],
    "out/demo-dicuq",
    "out-riley-py/demo-dicuq-from-exodus",
    2,
)

ALL_DEMO_CASES = (*STANDARD_DEMO_CASES, EXODUS_DEMO_CASE)
DEMO_CASES = ALL_DEMO_CASES


def test_raster_config_exposes_global_subpixel_sizing() -> None:
    """Python callers can tune both global accumulation tile dimensions."""
    import riley

    config = riley.RasterConfig()
    assert config.global_subpx_tile_size_min == 64
    assert config.global_subpx_tile_size_max == 1024
    assert config.global_subpx_tile_size_override == 0
    assert config.global_subpx_stripe_size_min == 256
    assert config.global_subpx_stripe_size_max == 4096
    assert config.global_subpx_stripe_size_override == 0

    config.global_subpx_tile_size_override = 128
    config.global_subpx_stripe_size_override = 512
    assert config.global_subpx_tile_size_override == 128
    assert config.global_subpx_stripe_size_override == 512


def _repo_assets_available() -> bool:
    required_paths = (
        PROJECT_ROOT / "src/run_all_demos.zig",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "texture",
    )
    return all(path.exists() for path in required_paths)


def _has_bmp_renders(dir_path: Path) -> bool:
    return dir_path.is_dir() and any(dir_path.rglob("*.bmp"))


def _has_expected_demo_renders(
    dir_path: Path,
    frames_num: int | None,
) -> bool:
    if frames_num is None:
        return _has_bmp_renders(dir_path)
    expected = {
        Path(f"cam{camera}_frame{frame}_field0.bmp")
        for camera in range(2)
        for frame in range(frames_num)
    }
    actual = {
        path.relative_to(dir_path)
        for path in dir_path.rglob("*.bmp")
    } if dir_path.is_dir() else set()
    return actual == expected


def _run_command(label: str, cmd: list[str], env: dict[str, str]) -> float:
    print(f"Running {label}: {' '.join(cmd)}")
    start_time = perf_counter()
    subprocess.run(
        cmd,
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    elapsed_time = perf_counter() - start_time
    print(f"{label} completed in {elapsed_time:.3f}s.")
    return elapsed_time


def _compare_renders(path_a: Path, path_b: Path) -> None:
    arr_a_raw = np.asarray(Image.open(path_a))
    arr_b_raw = np.asarray(Image.open(path_b))
    if arr_a_raw.shape != arr_b_raw.shape:
        raise AssertionError(
            f"shape mismatch for {path_a.name}: "
            f"zig={arr_a_raw.shape} python={arr_b_raw.shape}",
        )

    if (
        EXACT_8BIT_COMPARE
        and arr_a_raw.dtype == np.uint8
        and arr_b_raw.dtype == np.uint8
    ):
        if not np.array_equal(arr_a_raw, arr_b_raw):
            diff = np.abs(
                arr_a_raw.astype(np.int16) - arr_b_raw.astype(np.int16)
            )
            raise AssertionError(
                f"render mismatch for {path_a.name}: "
                f"max_abs_diff={int(np.max(diff))}, "
                f"diff_px={int(np.count_nonzero(diff))}",
            )
        return

    arr_a = arr_a_raw.astype(np.float32, copy=False)
    arr_b = arr_b_raw.astype(np.float32, copy=False)
    max_diff = float(np.max(np.abs(arr_a - arr_b)))
    if max_diff > FLOAT_FALLBACK_ABS_TOL:
        raise AssertionError(
            f"render mismatch for {path_a.name}: "
            f"max_abs_diff={max_diff:.4f} > tol={FLOAT_FALLBACK_ABS_TOL:.4f}",
        )


def _compare_render_dirs(dir_a: Path, dir_b: Path) -> None:
    files_a = sorted(
        path_a.relative_to(dir_a) for path_a in dir_a.rglob("*.bmp")
    )
    files_b = sorted(
        path_b.relative_to(dir_b) for path_b in dir_b.rglob("*.bmp")
    )
    if files_a != files_b:
        raise AssertionError(
            f"output file mismatch: zig={files_a}, python={files_b}"
        )

    print(f"Comparing renders in {dir_a} against {dir_b}...")
    start_time = perf_counter()
    for rel_path in files_a:
        _compare_renders(dir_a / rel_path, dir_b / rel_path)
    elapsed_time = perf_counter() - start_time
    print(f"Compare completed in {elapsed_time:.3f}s.")


@pytest.fixture(scope="session", autouse=True)
def ensure_repo_context() -> None:
    if not _repo_assets_available():
        pytest.skip(
            "Riley repo assets are not available from the current working "
            "directory. Repo parity demos are skipped.",
            allow_module_level=True,
        )


@pytest.fixture(scope="session", autouse=True)
def ensure_zig_demo_renders() -> None:
    silent_env = dict(os.environ)
    silent_env["RILEY_DEMO_SILENT"] = "1"

    force_zig_render = os.environ.get("RILEY_FORCE_ZIG_RENDER", "0") == "1"
    if force_zig_render:
        for _, _, zig_dir, _, _ in DEMO_CASES:
            shutil.rmtree(PROJECT_ROOT / zig_dir, ignore_errors=True)

    needs_zig_render = force_zig_render or any(
        not _has_expected_demo_renders(PROJECT_ROOT / zig_dir, frames_num)
        for _, _, zig_dir, _, frames_num in DEMO_CASES
    )

    if needs_zig_render:
        _run_command("zig demo render", ZIG_CMD, silent_env)
    else:
        print("Reusing cached Zig demo renders.")


def _run_demo_case(
    case_name: str,
    python_cmd: list[str],
    zig_dir: str,
    py_dir: str,
) -> None:
    silent_env = dict(os.environ)
    silent_env["RILEY_DEMO_SILENT"] = "1"

    py_dir_path = PROJECT_ROOT / py_dir
    shutil.rmtree(py_dir_path, ignore_errors=True)

    print(f"Testing demo case: {case_name}")
    _run_command(f"python render {case_name}", python_cmd, silent_env)
    _compare_render_dirs(PROJECT_ROOT / zig_dir, py_dir_path)


@pytest.mark.parametrize(
    ("case_name", "python_cmd", "zig_dir", "py_dir", "frames_num"),
    STANDARD_DEMO_CASES,
    ids=[case[0] for case in STANDARD_DEMO_CASES],
)
def test_demo_parity(
    case_name: str,
    python_cmd: list[str],
    zig_dir: str,
    py_dir: str,
    frames_num: int | None,
) -> None:
    del frames_num
    _run_demo_case(case_name, python_cmd, zig_dir, py_dir)


@pytest.mark.skipif(
    find_spec("netCDF4") is None,
    reason="netCDF4 is required for the exodus Python demo parity test.",
)
def test_demo_dic_from_exodus_parity() -> None:
    case_name, python_cmd, zig_dir, py_dir, _ = EXODUS_DEMO_CASE
    _run_demo_case(case_name, python_cmd, zig_dir, py_dir)
