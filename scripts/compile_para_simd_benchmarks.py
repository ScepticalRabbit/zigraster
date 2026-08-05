#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import subprocess

from perf_common import repo_root


BENCH_NAMES = [
    "bench_dicuq",
    "bench_fullraster",
    "bench_tiltraster",
    "bench_geom",
    "bench_sphere2000",
    "bench_sphere2000zoom",
    "bench_thread_geom",
]


BATCH_SIZE = 4


def compile_mode_parallel(suffix: str) -> None:
    root = repo_root()
    failures: list[str] = []

    for ii in range(0, len(BENCH_NAMES), BATCH_SIZE):
        batch = BENCH_NAMES[ii : ii + BATCH_SIZE]
        processes: list[tuple[str, subprocess.Popen[str]]] = []
        for bench_name in batch:
            print(f"Compiling {bench_name}_{suffix}...")
            process = subprocess.Popen(
                [
                    "zig",
                    "build",
                    f"install-{bench_name.replace('_', '-')}",
                    "--prefix",
                    ".",
                    "-Doptimize=ReleaseFast",
                    "-Dprecision=f64",
                    "-Dsimd=on",
                ],
                cwd=root,
                text=True,
            )
            processes.append((bench_name, process))

        for bench_name, process in processes:
            if process.wait() != 0:
                failures.append(bench_name)

    if failures:
        joined = ", ".join(failures)
        raise SystemExit(
            f"One or more {suffix} benchmark compilations failed: {joined}.",
        )

    for bench_name in BENCH_NAMES:
        src = root / "bin" / f"{bench_name}_f64_{suffix}"
        dst = root / "bin" / f"{bench_name}_{suffix}"
        if src.exists():
            import shutil
            shutil.copy2(src, dst)





def main() -> int:
    root = repo_root()
    (root / "bin").mkdir(parents=True, exist_ok=True)

    compile_mode_parallel("simd")

    print(f"SIMD benchmark executables written to {root / 'bin'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
