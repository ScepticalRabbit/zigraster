#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import TextIO


ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_ROUNDS = 10
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
METRICS = (
    ("setup_ms", "Setup", r"(?:Riley\s+)?Setup Time"),
    ("raster_loop_ms", "Raster loop", r"Raster loop time"),
    ("total_render_ms", "Total render", r"Total Render Time"),
)
TIMING_PATTERNS = {
    field: re.compile(
        rf"^\s*{line_label}\s*=\s*({NUMBER})\s*ms\s*$",
        re.MULTILINE | re.IGNORECASE,
    )
    for field, _, line_label in METRICS
}
CSV_FIELDS = (
    "round",
    "order",
    "case",
    "executable",
    "arguments_json",
    "setup_ms",
    "raster_loop_ms",
    "total_render_ms",
)


@dataclass(frozen=True)
class Case:
    name: str
    executable: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class Sample:
    round_number: int
    order: int
    case: Case
    setup_ms: float
    raster_loop_ms: float
    total_render_ms: float


class BenchmarkError(RuntimeError):
    pass


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def assignment(
    parser: argparse.ArgumentParser, spec: str, option: str
) -> tuple[str, str]:
    if "=" not in spec:
        parser.error(f"{option} requires NAME=VALUE")
    name, value = spec.split("=", 1)
    name = name.strip()
    if not name or not value:
        parser.error(f"{option} requires non-empty NAME and VALUE")
    return name, value


def parse_cases(
    parser: argparse.ArgumentParser,
    case_specs: list[str],
    arg_specs: list[str],
    baseline: str,
) -> list[Case]:
    executables: dict[str, str] = {}
    for spec in case_specs:
        name, executable = assignment(parser, spec, "--case")
        if name in executables:
            parser.error(f"duplicate case name: {name}")
        executables[name] = executable

    if len(executables) < 2:
        parser.error("at least two --case options are required")
    if baseline not in executables:
        parser.error(f"baseline case not found: {baseline}")

    arguments: dict[str, list[str]] = {name: [] for name in executables}
    for spec in arg_specs:
        name, argument = assignment(parser, spec, "--arg")
        if name not in arguments:
            parser.error(f"--arg refers to unknown case: {name}")
        arguments[name].append(argument)

    return [
        Case(name, executable, tuple(arguments[name]))
        for name, executable in executables.items()
    ]


def resolve_executable(executable: str) -> str:
    expanded = pathlib.Path(executable).expanduser()
    if expanded.is_absolute() or os.sep in executable:
        path = expanded if expanded.is_absolute() else ROOT / expanded
        path = path.resolve()
        if not path.is_file():
            raise BenchmarkError(f"executable does not exist: {path}")
        if not os.access(path, os.X_OK):
            raise BenchmarkError(f"file is not executable: {path}")
        return str(path)

    resolved = shutil.which(executable)
    if resolved is None:
        raise BenchmarkError(f"executable not found on PATH: {executable}")
    return resolved


def output_excerpt(stdout: str, stderr: str, limit: int = 30) -> str:
    lines = [f"stdout: {line}" for line in stdout.splitlines()]
    lines.extend(f"stderr: {line}" for line in stderr.splitlines())
    if not lines:
        return "  (no process output)"
    omitted = len(lines) - limit
    lines = lines[-limit:]
    if omitted > 0:
        lines.insert(0, f"(... {omitted} earlier lines omitted)")
    return "\n".join(f"  {line}" for line in lines)


def run_case(case: Case, executable: str) -> dict[str, float]:
    command = [executable, *case.args]
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise BenchmarkError(
            f"case '{case.name}' failed with exit code {result.returncode}: "
            f"{shlex.join(command)}\n{output_excerpt(result.stdout, result.stderr)}"
        )

    output = result.stdout + "\n" + result.stderr
    timings = {}
    for field, label, _ in METRICS:
        matches = TIMING_PATTERNS[field].findall(output)
        if len(matches) != 1:
            raise BenchmarkError(
                f"case '{case.name}' produced {len(matches)} {label} timing lines; "
                f"expected exactly one\n{output_excerpt(result.stdout, result.stderr)}"
            )
        timings[field] = float(matches[0])
    return timings


def rotated(cases: list[Case], index: int, fixed_order: bool) -> list[Case]:
    if fixed_order:
        return cases
    offset = index % len(cases)
    return cases[offset:] + cases[:offset]


def run_warmups(
    cases: list[Case],
    executables: dict[str, str],
    warmups: int,
    fixed_order: bool,
) -> None:
    for warmup in range(warmups):
        for case in rotated(cases, warmup, fixed_order):
            print(f"Warmup {warmup + 1}/{warmups}: {case.name}")
            run_case(case, executables[case.name])


def sample_row(sample: Sample) -> dict[str, object]:
    return {
        "round": sample.round_number,
        "order": sample.order,
        "case": sample.case.name,
        "executable": sample.case.executable,
        "arguments_json": json.dumps(sample.case.args),
        "setup_ms": sample.setup_ms,
        "raster_loop_ms": sample.raster_loop_ms,
        "total_render_ms": sample.total_render_ms,
    }


def run_measurements(
    cases: list[Case],
    executables: dict[str, str],
    writer: csv.DictWriter,
    csv_file: TextIO,
    rounds: int | None,
    duration: float | None,
    fixed_order: bool,
) -> list[Sample]:
    samples = []
    start = time.monotonic()
    round_index = 0
    while rounds is None or round_index < rounds:
        for order, case in enumerate(rotated(cases, round_index, fixed_order), start=1):
            print(f"Round {round_index + 1}, case {order}/{len(cases)}: {case.name}")
            sample = Sample(
                round_number=round_index + 1,
                order=order,
                case=case,
                **run_case(case, executables[case.name]),
            )
            samples.append(sample)
            writer.writerow(sample_row(sample))
            csv_file.flush()
        round_index += 1
        if duration is not None and time.monotonic() - start >= duration:
            break
    return samples


def paired_delta_percent(
    samples: list[Sample],
    baseline_by_round: dict[int, Sample],
    metric: str,
) -> float | None:
    deltas = []
    for sample in samples:
        baseline = getattr(baseline_by_round[sample.round_number], metric)
        if baseline == 0.0:
            return None
        deltas.append(100.0 * (getattr(sample, metric) - baseline) / baseline)
    return statistics.median(deltas)


def report(samples: list[Sample], cases: list[Case], baseline: str) -> None:
    by_case = {case.name: [] for case in cases}
    for sample in samples:
        by_case[sample.case.name].append(sample)
    baseline_by_round = {
        sample.round_number: sample for sample in by_case[baseline]
    }

    print(
        "\nMedians in ms; paired delta is the median of each round's "
        "(case - baseline) / baseline."
    )
    for case in cases:
        print(f"\n{case.name}" + (" (baseline)" if case.name == baseline else ""))
        for metric, label, _ in METRICS:
            median_ms = statistics.median(
                getattr(sample, metric) for sample in by_case[case.name]
            )
            delta = paired_delta_percent(
                by_case[case.name], baseline_by_round, metric
            )
            delta_text = "n/a (zero baseline)" if delta is None else f"{delta:+.2f}%"
            print(f"  {label:<13} {median_ms:10.3f} ms   paired delta {delta_text}")


def default_output_path() -> pathlib.Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return pathlib.Path("out") / f"texture_procedural_sphere_{timestamp}.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially benchmark already-built texture/procedural sphere "
            "executables and compare each named case with a baseline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example:
  python scripts/bench_texture_procedural_sphere.py \\
    --case texture=/path/to/texture-sphere \\
    --case procedural=/path/to/procedural-sphere \\
    --arg procedural=--size --arg procedural=0.25 \\
    --baseline texture --rounds 10 --output out/sphere_raw.csv

Each --arg contributes one exact argv item. Commands are executed directly without a
shell. Case order rotates by one position after every complete warmup/measurement
round unless --fixed-order is supplied. --duration is a minimum measurement time;
the current complete round always finishes so comparisons remain paired.
""",
    )
    parser.add_argument(
        "--case", action="append", required=True, metavar="NAME=EXECUTABLE",
        help="add a named case (repeat for each executable)",
    )
    parser.add_argument(
        "--arg", action="append", default=[], metavar="NAME=ARGUMENT",
        help="append one argument to a named case; repeat to preserve argv order",
    )
    parser.add_argument(
        "--baseline", required=True, metavar="NAME",
        help="case used for paired percentage differences",
    )
    parser.add_argument(
        "--warmup", type=nonnegative_int, default=1, metavar="COUNT",
        help="complete warmup rounds (default: 1)",
    )
    stopping = parser.add_mutually_exclusive_group()
    stopping.add_argument(
        "--rounds", type=positive_int, metavar="COUNT",
        help=f"complete measurement rounds (default: {DEFAULT_ROUNDS})",
    )
    stopping.add_argument(
        "--duration", type=positive_float, metavar="SECONDS",
        help="minimum measured duration, ending after a complete paired round",
    )
    parser.add_argument(
        "--fixed-order", action="store_true",
        help="disable the default per-round case-order rotation",
    )
    parser.add_argument(
        "--output", type=pathlib.Path, metavar="CSV",
        help="raw CSV path (default: timestamped file under out/)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    cases = parse_cases(parser, args.case, args.arg, args.baseline)
    rounds = args.rounds
    if rounds is None and args.duration is None:
        rounds = DEFAULT_ROUNDS

    try:
        executables = {
            case.name: resolve_executable(case.executable) for case in cases
        }
        run_warmups(cases, executables, args.warmup, args.fixed_order)

        output_path = args.output or default_output_path()
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            samples = run_measurements(
                cases,
                executables,
                writer,
                csv_file,
                rounds,
                args.duration,
                args.fixed_order,
            )
    except (BenchmarkError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"\nRaw samples written to {output_path}")
    report(samples, cases, args.baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
