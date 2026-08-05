#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import subprocess

from perf_common import repo_root


BENCH_NAMES = [
    "benchcam",
    "bench_dicuq",
    "bench_fullraster",
    "bench_geom",
    "bench_sphere2000",
    "bench_sphere2000zoom",
]


def compile_mode(simd_mode: str, suffix: str) -> None:
    root = repo_root()
    for bench_name in BENCH_NAMES:
        print(f"Compiling {bench_name}_{suffix}...")
        subprocess.run(
            [
                "zig",
                "build",
                f"install-{bench_name.replace('_', '-')}",
                "--prefix",
                ".",
                "-Doptimize=ReleaseFast",
                "-Dprecision=f64",
                f"-Dsimd={simd_mode}",
            ],
            cwd=root,
            check=True,
        )
        src = (
            root / "bin" /
            f"{bench_name}_f64_{suffix}_inner"
        )
        dst = root / "bin" / f"{bench_name}_{suffix}"
        if src.exists():
            import shutil
            shutil.copy2(src, dst)



def main() -> int:
    root = repo_root()
    (root / "bin").mkdir(parents=True, exist_ok=True)

    compile_mode("off", "scalar")
    compile_mode("on", "simd")

    print(f"Benchmark executables written to {root / 'bin'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
