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


DEFAULT_ROUNDS = 10
TIMING_PATTERNS = {
    "setup_ms": re.compile(
        r"^\s*(?:Riley\s+)?Setup Time\s*=\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*ms\s*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    "raster_loop_ms": re.compile(
        r"^\s*Raster loop time\s*=\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*ms\s*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    "total_render_ms": re.compile(
        r"^\s*Total Render Time\s*=\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*ms\s*$",
        re.MULTILINE | re.IGNORECASE,
    ),
}
METRICS = (
    ("setup_ms", "Setup"),
    ("raster_loop_ms", "Raster loop"),
    ("total_render_ms", "Total render"),
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


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


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


def split_assignment(value: str, option: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"{option} requires NAME=VALUE")
    name, assigned_value = value.split("=", 1)
    name = name.strip()
    if not name or not assigned_value:
        raise argparse.ArgumentTypeError(f"{option} requires non-empty NAME and VALUE")
    return name, assigned_value


def parse_cases(
    parser: argparse.ArgumentParser,
    case_specs: list[str],
    arg_specs: list[str],
    baseline: str,
) -> list[Case]:
    executables: dict[str, str] = {}
    case_order: list[str] = []
    for spec in case_specs:
        try:
            name, executable = split_assignment(spec, "--case")
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
        if name in executables:
            parser.error(f"duplicate case name: {name}")
        executables[name] = executable
        case_order.append(name)

    if len(case_order) < 2:
        parser.error("at least two --case options are required")
    if baseline not in executables:
        parser.error(f"baseline case not found: {baseline}")

    arguments = {name: [] for name in case_order}
    for spec in arg_specs:
        try:
            name, argument = split_assignment(spec, "--arg")
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
        if name not in arguments:
            parser.error(f"--arg refers to unknown case: {name}")
        arguments[name].append(argument)

    return [
        Case(name, executables[name], tuple(arguments[name]))
        for name in case_order
    ]


def resolve_executable(executable: str) -> str:
    expanded = pathlib.Path(executable).expanduser()
    if expanded.is_absolute() or os.sep in executable:
        path = expanded if expanded.is_absolute() else repo_root() / expanded
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
    lines = []
    if stdout:
        lines.extend(f"stdout: {line}" for line in stdout.splitlines())
    if stderr:
        lines.extend(f"stderr: {line}" for line in stderr.splitlines())
    if not lines:
        return "  (no process output)"
    omitted = max(0, len(lines) - limit)
    excerpt = lines[-limit:]
    prefix = [f"  (... {omitted} earlier lines omitted)"] if omitted else []
    return "\n".join(f"  {line}" for line in prefix + excerpt)


def parse_timings(case: Case, stdout: str, stderr: str) -> dict[str, float]:
    combined = stdout + "\n" + stderr
    timings: dict[str, float] = {}
    for metric, pattern in TIMING_PATTERNS.items():
        matches = pattern.findall(combined)
        if len(matches) != 1:
            label = dict(METRICS)[metric]
            raise BenchmarkError(
                f"case '{case.name}' produced {len(matches)} {label} timing lines; "
                "expected exactly one\n"
                + output_excerpt(stdout, stderr)
            )
        timings[metric] = float(matches[0])
    return timings


def run_case(case: Case, executable: str) -> dict[str, float]:
    command = [executable, *case.args]
    result = subprocess.run(
        command,
        cwd=repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise BenchmarkError(
            f"case '{case.name}' failed with exit code {result.returncode}: "
            f"{shlex.join(command)}\n"
            + output_excerpt(result.stdout, result.stderr)
        )
    return parse_timings(case, result.stdout, result.stderr)


def rotated(cases: list[Case], round_index: int, fixed_order: bool) -> list[Case]:
    if fixed_order:
        return cases
    offset = round_index % len(cases)
    return cases[offset:] + cases[:offset]


def run_warmups(
    cases: list[Case],
    executables: dict[str, str],
    warmups: int,
    fixed_order: bool,
) -> None:
    for warmup_index in range(warmups):
        for case in rotated(cases, warmup_index, fixed_order):
            print(f"Warmup {warmup_index + 1}/{warmups}: {case.name}")
            run_case(case, executables[case.name])


def write_sample(writer: csv.DictWriter, sample: Sample) -> None:
    writer.writerow(
        {
            "round": sample.round_number,
            "order": sample.order,
            "case": sample.case.name,
            "executable": sample.case.executable,
            "arguments_json": json.dumps(sample.case.args),
            "setup_ms": sample.setup_ms,
            "raster_loop_ms": sample.raster_loop_ms,
            "total_render_ms": sample.total_render_ms,
        }
    )


def run_measurements(
    cases: list[Case],
    executables: dict[str, str],
    writer: csv.DictWriter,
    csv_file: object,
    rounds: int | None,
    duration: float | None,
    fixed_order: bool,
) -> list[Sample]:
    samples: list[Sample] = []
    start = time.monotonic()
    round_index = 0
    while rounds is None or round_index < rounds:
        for order, case in enumerate(
            rotated(cases, round_index, fixed_order),
            start=1,
        ):
            print(f"Round {round_index + 1}, case {order}/{len(cases)}: {case.name}")
            timings = run_case(case, executables[case.name])
            sample = Sample(
                round_number=round_index + 1,
                order=order,
                case=case,
                **timings,
            )
            samples.append(sample)
            write_sample(writer, sample)
            csv_file.flush()  # type: ignore[attr-defined]
        round_index += 1
        if duration is not None and time.monotonic() - start >= duration:
            break
    return samples


def paired_delta_percent(
    case_samples: list[Sample],
    baseline_by_round: dict[int, Sample],
    metric: str,
) -> float | None:
    deltas = []
    for sample in case_samples:
        baseline_value = getattr(baseline_by_round[sample.round_number], metric)
        if baseline_value == 0.0:
            return None
        deltas.append(100.0 * (getattr(sample, metric) - baseline_value) / baseline_value)
    return statistics.median(deltas)


def report(samples: list[Sample], cases: list[Case], baseline: str) -> None:
    baseline_by_round = {
        sample.round_number: sample
        for sample in samples
        if sample.case.name == baseline
    }
    print(
        "\nMedians in ms; paired delta is the median of each round's "
        "(case - baseline) / baseline."
    )
    for case in cases:
        case_samples = [sample for sample in samples if sample.case.name == case.name]
        print(f"\n{case.name}" + (" (baseline)" if case.name == baseline else ""))
        for metric, label in METRICS:
            median_ms = statistics.median(
                getattr(sample, metric) for sample in case_samples
            )
            delta = paired_delta_percent(case_samples, baseline_by_round, metric)
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
        "--case",
        action="append",
        required=True,
        metavar="NAME=EXECUTABLE",
        help="add a named case (repeat for each executable)",
    )
    parser.add_argument(
        "--arg",
        action="append",
        default=[],
        metavar="NAME=ARGUMENT",
        help="append one argument to a named case; repeat to preserve argv order",
    )
    parser.add_argument(
        "--baseline",
        required=True,
        metavar="NAME",
        help="case used for paired percentage differences",
    )
    parser.add_argument(
        "--warmup",
        type=nonnegative_int,
        default=1,
        metavar="COUNT",
        help="complete warmup rounds (default: 1)",
    )
    stopping = parser.add_mutually_exclusive_group()
    stopping.add_argument(
        "--rounds",
        type=positive_int,
        metavar="COUNT",
        help=f"complete measurement rounds (default: {DEFAULT_ROUNDS})",
    )
    stopping.add_argument(
        "--duration",
        type=positive_float,
        metavar="SECONDS",
        help="minimum measured duration, ending after a complete paired round",
    )
    parser.add_argument(
        "--fixed-order",
        action="store_true",
        help="disable the default per-round case-order rotation",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        metavar="CSV",
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
            output_path = repo_root() / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "round",
                    "order",
                    "case",
                    "executable",
                    "arguments_json",
                    "setup_ms",
                    "raster_loop_ms",
                    "total_render_ms",
                ],
            )
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
