#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import pathlib
import re

import numpy as np


FRAME_RE = re.compile(r"cam\d+_frame(\d+)_field\d+_stats\.csv$")
CASE_RE = re.compile(
    r"^verif_6_(?P<mesh_name>[^_]+)_(?P<geom_name>[^_]+)_(?P<camera_case>.+)$",
)
GAUSS_POINTS = np.array(
    [
        -math.sqrt(3.0 / 5.0),
        0.0,
        math.sqrt(3.0 / 5.0),
    ],
    dtype=float,
)
GAUSS_WEIGHTS = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=float)
DEFAULT_EDGE_SEGMENTS_SENSOR_MULT = 4
MIN_EDGE_SEGMENTS = 4096


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verif-root",
        type=pathlib.Path,
        default=repo_root() / "verif" / "verif_6",
    )
    parser.add_argument(
        "--out-csv",
        type=pathlib.Path,
        default=repo_root() / "verif" / "verif_6" / "verif_6_summary.csv",
    )
    parser.add_argument(
        "--edge-segments",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--edge-segments-sensor-mult",
        type=int,
        default=DEFAULT_EDGE_SEGMENTS_SENSOR_MULT,
    )
    return parser.parse_args()


def parse_stats_csv(stats_path: pathlib.Path) -> dict[str, float | str]:
    stats: dict[str, float | str] = {}
    with stats_path.open(newline="") as stats_file:
        reader = csv.reader(stats_file)
        header = next(reader)
        if header != ["key", "unit", "value"]:
            raise ValueError(
                f"{stats_path} does not have key,unit,value layout",
            )
        for row in reader:
            if len(row) != 3:
                continue
            key, unit, value = row
            if unit == "name":
                stats[key] = value
            else:
                stats[key] = float(value)
    return stats


def load_scalar_csv(image_path: pathlib.Path) -> np.ndarray:
    rows: list[list[float]] = []
    with image_path.open(newline="") as image_file:
        reader = csv.reader(image_file)
        for row in reader:
            if row and row[-1] == "":
                row = row[:-1]
            rows.append([float(value) for value in row])
    return np.asarray(rows, dtype=float)


def calc_image_centroid_and_area(
    image_vals: np.ndarray,
) -> tuple[float, float, float, float]:
    rows_num, cols_num = image_vals.shape
    fill_value = float(np.max(image_vals))
    weighted_area = float(np.sum(image_vals))
    if weighted_area <= 0.0 or fill_value <= 0.0:
        return math.nan, math.nan, 0.0, fill_value

    x_centers = np.arange(cols_num, dtype=float) + 0.5
    y_centers = np.arange(rows_num, dtype=float) + 0.5

    centroid_x = float(np.sum(image_vals * x_centers[None, :]) / weighted_area)
    centroid_y = float(np.sum(image_vals * y_centers[:, None]) / weighted_area)
    area_px2 = weighted_area / fill_value
    return centroid_x, centroid_y, area_px2, fill_value


def parse_case_info(stats_path: pathlib.Path) -> tuple[str, str, str, int]:
    case_name = stats_path.parent.name
    case_match = CASE_RE.match(case_name)
    if case_match is None:
        raise ValueError(f"could not parse verification case from {case_name}")

    frame_match = FRAME_RE.match(stats_path.name)
    if frame_match is None:
        raise ValueError(f"could not parse frame index from {stats_path.name}")

    return (
        case_match.group("mesh_name"),
        case_match.group("geom_name"),
        case_match.group("camera_case"),
        int(frame_match.group(1)),
    )


def get_node_coords(
    stats: dict[str, float | str],
) -> list[tuple[float, float]]:
    node_indices: list[int] = []
    for key in stats:
        if key.startswith("N") and key.endswith("_x"):
            node_indices.append(int(key[1:-2]))

    coords = []
    for nn in sorted(node_indices):
        coords.append(
            (
                float(stats[f"N{nn}_x"]),
                float(stats[f"N{nn}_y"]),
            )
        )
    return coords


def linear_edge_nodes(
    node_start: tuple[float, float],
    node_end: tuple[float, float],
    ss: float,
) -> tuple[float, float, float, float]:
    n0 = 0.5 * (1.0 - ss)
    n1 = 0.5 * (1.0 + ss)
    dn0 = -0.5
    dn1 = 0.5

    xx = n0 * node_start[0] + n1 * node_end[0]
    yy = n0 * node_start[1] + n1 * node_end[1]
    dx_ds = dn0 * node_start[0] + dn1 * node_end[0]
    dy_ds = dn0 * node_start[1] + dn1 * node_end[1]
    return xx, yy, dx_ds, dy_ds


def quadratic_edge_nodes(
    node_start: tuple[float, float],
    node_mid: tuple[float, float],
    node_end: tuple[float, float],
    ss: float,
) -> tuple[float, float, float, float]:
    n0 = 0.5 * ss * (ss - 1.0)
    n1 = 1.0 - ss * ss
    n2 = 0.5 * ss * (ss + 1.0)
    dn0 = ss - 0.5
    dn1 = -2.0 * ss
    dn2 = ss + 0.5

    xx = n0 * node_start[0] + n1 * node_mid[0] + n2 * node_end[0]
    yy = n0 * node_start[1] + n1 * node_mid[1] + n2 * node_end[1]
    dx_ds = dn0 * node_start[0] + dn1 * node_mid[0] + dn2 * node_end[0]
    dy_ds = dn0 * node_start[1] + dn1 * node_mid[1] + dn2 * node_end[1]
    return xx, yy, dx_ds, dy_ds


def iter_edges(
    mesh_name: str,
    node_coords: list[tuple[float, float]],
) -> list[tuple[str, tuple[float, float], ...]]:
    if mesh_name == "tri3":
        return [
            ("linear", node_coords[0], node_coords[1]),
            ("linear", node_coords[1], node_coords[2]),
            ("linear", node_coords[2], node_coords[0]),
        ]
    if mesh_name == "tri6":
        return [
            ("quadratic", node_coords[0], node_coords[3], node_coords[1]),
            ("quadratic", node_coords[1], node_coords[4], node_coords[2]),
            ("quadratic", node_coords[2], node_coords[5], node_coords[0]),
        ]
    if mesh_name == "quad4":
        return [
            ("linear", node_coords[0], node_coords[1]),
            ("linear", node_coords[1], node_coords[2]),
            ("linear", node_coords[2], node_coords[3]),
            ("linear", node_coords[3], node_coords[0]),
        ]
    if mesh_name == "quad8":
        return [
            ("quadratic", node_coords[0], node_coords[4], node_coords[1]),
            ("quadratic", node_coords[1], node_coords[5], node_coords[2]),
            ("quadratic", node_coords[2], node_coords[6], node_coords[3]),
            ("quadratic", node_coords[3], node_coords[7], node_coords[0]),
        ]
    if mesh_name == "quad9":
        return [
            ("quadratic", node_coords[0], node_coords[4], node_coords[1]),
            ("quadratic", node_coords[1], node_coords[5], node_coords[2]),
            ("quadratic", node_coords[2], node_coords[6], node_coords[3]),
            ("quadratic", node_coords[3], node_coords[7], node_coords[0]),
        ]
    raise ValueError(f"unsupported mesh name {mesh_name}")


def sample_edge_points(
    edge_kind: str,
    edge_nodes: tuple[tuple[float, float], ...],
    edge_segments: int,
) -> list[tuple[float, float]]:
    samples: list[tuple[float, float]] = []
    for ii in range(edge_segments + 1):
        ss = -1.0 + 2.0 * ii / edge_segments
        if edge_kind == "linear":
            xx, yy, _, _ = linear_edge_nodes(
                edge_nodes[0],
                edge_nodes[1],
                ss,
            )
        elif edge_kind == "quadratic":
            xx, yy, _, _ = quadratic_edge_nodes(
                edge_nodes[0],
                edge_nodes[1],
                edge_nodes[2],
                ss,
            )
        else:
            raise ValueError(f"unsupported edge kind {edge_kind}")
        samples.append((xx, yy))
    return samples


def calc_edge_segments(
    sensor_x: int,
    sensor_y: int,
    edge_segments: int | None,
    edge_segments_sensor_mult: int,
) -> int:
    if edge_segments is not None:
        return edge_segments
    sensor_edge_max = max(sensor_x, sensor_y)
    return max(MIN_EDGE_SEGMENTS, edge_segments_sensor_mult * sensor_edge_max)


def distort_pixel_point(
    xx: float,
    yy: float,
    stats: dict[str, float | str],
) -> tuple[float, float]:
    model_name = str(stats["distortion_model"])
    if model_name == "none":
        return xx, yy

    focal_px_x = float(stats["focal_px_x"])
    focal_px_y = float(stats["focal_px_y"])
    offset_x = float(stats["offset_x"])
    offset_y = float(stats["offset_y"])

    x_norm = (xx - offset_x) / focal_px_x
    y_norm = (yy - offset_y) / focal_px_y
    r2 = x_norm * x_norm + y_norm * y_norm
    r4 = r2 * r2
    r6 = r4 * r2
    k1 = float(stats["distortion_k1"])
    k2 = float(stats["distortion_k2"])
    k3 = float(stats["distortion_k3"])
    p1 = float(stats["distortion_p1"])
    p2 = float(stats["distortion_p2"])
    radial_scale = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    x_dist = (
        x_norm * radial_scale +
        2.0 * p1 * x_norm * y_norm +
        p2 * (r2 + 2.0 * x_norm * x_norm)
    )
    y_dist = (
        y_norm * radial_scale +
        p1 * (r2 + 2.0 * y_norm * y_norm) +
        2.0 * p2 * x_norm * y_norm
    )
    return x_dist * focal_px_x + offset_x, y_dist * focal_px_y + offset_y


def build_distorted_boundary_polyline(
    mesh_name: str,
    node_coords: list[tuple[float, float]],
    edge_segments: int,
    stats: dict[str, float | str],
) -> list[tuple[float, float]]:
    polyline: list[tuple[float, float]] = []
    for edge_idx, edge in enumerate(iter_edges(mesh_name, node_coords)):
        edge_points = sample_edge_points(edge[0], edge[1:], edge_segments)
        if edge_idx > 0:
            edge_points = edge_points[1:]
        for xx, yy in edge_points:
            polyline.append(distort_pixel_point(xx, yy, stats))
    return polyline


def calc_polyline_area_px2(polyline: list[tuple[float, float]]) -> float:
    if len(polyline) < 3:
        return 0.0
    area = 0.0
    for ii, (xx0, yy0) in enumerate(polyline):
        xx1, yy1 = polyline[(ii + 1) % len(polyline)]
        area += xx0 * yy1 - xx1 * yy0
    return 0.5 * abs(area)


def build_reference_mask_from_polyline(
    polyline: list[tuple[float, float]],
    sensor_x: int,
    sensor_y: int,
    eps: float = 1.0e-12,
) -> np.ndarray:
    if len(polyline) < 2:
        raise ValueError("reference polyline must have at least 2 points")

    xx_centers = np.arange(sensor_x, dtype=float) + 0.5
    mask = np.zeros((sensor_y, sensor_x), dtype=float)

    seg_x0 = np.array([point[0] for point in polyline], dtype=float)
    seg_y0 = np.array([point[1] for point in polyline], dtype=float)
    seg_x1 = np.roll(seg_x0, -1)
    seg_y1 = np.roll(seg_y0, -1)

    for rr in range(sensor_y):
        yy = float(rr) + 0.5
        seg_dy = seg_y1 - seg_y0
        non_horizontal_mask = np.abs(seg_dy) > eps
        upward_mask = (
            (seg_y0 <= yy + eps) &
            (yy < seg_y1 - eps) &
            non_horizontal_mask
        )
        downward_mask = (
            (seg_y1 <= yy + eps) &
            (yy < seg_y0 - eps) &
            non_horizontal_mask
        )
        crossing_mask = upward_mask | downward_mask

        row_mask = np.zeros(sensor_x, dtype=bool)
        if np.any(crossing_mask):
            event_x_arr = seg_x0[crossing_mask] + (
                (yy - seg_y0[crossing_mask]) *
                (seg_x1[crossing_mask] - seg_x0[crossing_mask]) /
                seg_dy[crossing_mask]
            )
            event_winding_arr = np.where(
                upward_mask[crossing_mask],
                1,
                -1,
            ).astype(int)
            order = np.argsort(event_x_arr)
            event_x_arr = event_x_arr[order]
            event_winding_arr = event_winding_arr[order]
            winding_prefix = np.cumsum(event_winding_arr, dtype=int)
            event_idx = np.searchsorted(
                event_x_arr,
                xx_centers,
                side="right",
            ) - 1

            inside_mask = event_idx >= 0
            row_winding = np.zeros(sensor_x, dtype=int)
            row_winding[inside_mask] = winding_prefix[event_idx[inside_mask]]
            row_mask = row_winding != 0

        horizontal_on_row_mask = (
            (np.abs(seg_y0 - yy) <= eps) &
            (np.abs(seg_y1 - yy) <= eps)
        )
        if np.any(horizontal_on_row_mask):
            x_min_arr = np.minimum(
                seg_x0[horizontal_on_row_mask],
                seg_x1[horizontal_on_row_mask],
            )
            x_max_arr = np.maximum(
                seg_x0[horizontal_on_row_mask],
                seg_x1[horizontal_on_row_mask],
            )
            for x_min, x_max in zip(x_min_arr, x_max_arr, strict=True):
                row_mask |= (
                    (xx_centers >= x_min - eps) &
                    (xx_centers <= x_max + eps)
                )

        touching_row_mask = (
            (yy >= np.minimum(seg_y0, seg_y1) - eps) &
            (yy <= np.maximum(seg_y0, seg_y1) + eps) &
            non_horizontal_mask
        )
        if np.any(touching_row_mask):
            x_on_seg_arr = seg_x0[touching_row_mask] + (
                (yy - seg_y0[touching_row_mask]) *
                (seg_x1[touching_row_mask] - seg_x0[touching_row_mask]) /
                seg_dy[touching_row_mask]
            )
            for x_on_seg in x_on_seg_arr:
                xx_idx = int(round(x_on_seg - 0.5))
                if 0 <= xx_idx < sensor_x:
                    xx_center = xx_centers[xx_idx]
                    if abs(xx_center - x_on_seg) <= eps:
                        row_mask[xx_idx] = True

        mask[rr, row_mask] = 1.0

    return mask


def normalize_render_mask(
    image_vals: np.ndarray,
) -> np.ndarray:
    fill_value = float(np.max(image_vals))
    if fill_value <= 0.0:
        return np.zeros_like(image_vals)
    return image_vals / fill_value


def calc_mask_diff_pct(
    mask_ref: np.ndarray,
    mask_render: np.ndarray,
) -> float:
    abs_diff = np.abs(mask_ref - mask_render)
    return 100.0 * float(np.mean(abs_diff))


def analyse_stats_file(
    stats_path: pathlib.Path,
    edge_segments: int | None,
    edge_segments_sensor_mult: int,
) -> dict[str, object]:
    stats = parse_stats_csv(stats_path)
    image_path = stats_path.with_name(stats_path.name.replace("_stats.csv", ".csv"))
    image_vals = load_scalar_csv(image_path)

    mesh_name, geom_name, camera_case, frame_idx = parse_case_info(stats_path)
    centroid_num_x, centroid_num_y, area_num_px2, fill_value = (
        calc_image_centroid_and_area(image_vals)
    )

    node_coords = get_node_coords(stats)
    sensor_x = int(round(float(stats["sensor_pixels_x"])))
    sensor_y = int(round(float(stats["sensor_pixels_y"])))
    edge_segments_num = calc_edge_segments(
        sensor_x,
        sensor_y,
        edge_segments,
        edge_segments_sensor_mult,
    )
    polyline = build_distorted_boundary_polyline(
        mesh_name,
        node_coords,
        edge_segments_num,
        stats,
    )
    _ = calc_polyline_area_px2(polyline)
    mask_ref = build_reference_mask_from_polyline(
        polyline,
        sensor_x,
        sensor_y,
    )
    centroid_ref_x, centroid_ref_y, _, _ = calc_image_centroid_and_area(mask_ref)
    mask_render = normalize_render_mask(image_vals)
    if mask_render.shape != mask_ref.shape:
        raise ValueError(
            f"mask shape mismatch for {stats_path}: "
            f"{mask_ref.shape} vs {mask_render.shape}",
        )
    mask_diff_pct = calc_mask_diff_pct(mask_ref, mask_render)

    centroid_diff_x = centroid_num_x - centroid_ref_x
    centroid_diff_y = centroid_num_y - centroid_ref_y
    centroid_dist = math.hypot(centroid_diff_x, centroid_diff_y)

    area_ref_px2 = math.nan
    area_diff_px2 = math.nan
    area_rel_diff = math.nan
    area_err_pct = math.nan

    return {
        "mesh_name": mesh_name,
        "geom_name": geom_name,
        "camera_case": camera_case,
        "frame_idx": frame_idx,
        "node_count": len(node_coords),
        "fill_value": fill_value,
        "centroid_ref_x_px": centroid_ref_x,
        "centroid_ref_y_px": centroid_ref_y,
        "centroid_num_x_px": centroid_num_x,
        "centroid_num_y_px": centroid_num_y,
        "centroid_diff_x_px": centroid_diff_x,
        "centroid_diff_y_px": centroid_diff_y,
        "centroid_dist_px": centroid_dist,
        "area_ref_px2": area_ref_px2,
        "area_num_px2": area_num_px2,
        "area_diff_px2": area_diff_px2,
        "area_rel_diff": area_rel_diff,
        "area_err_pct": area_err_pct,
        "sensor_x_px": sensor_x,
        "sensor_y_px": sensor_y,
        "edge_segments": edge_segments_num,
        "mask_diff_pct": mask_diff_pct,
        "distortion_model": stats["distortion_model"],
    }


def write_summary_csv(
    out_path: pathlib.Path,
    rows: list[dict[str, object]],
) -> None:
    field_names = [
        "mesh_name",
        "geom_name",
        "camera_case",
        "frame_idx",
        "node_count",
        "fill_value",
        "centroid_ref_x_px",
        "centroid_ref_y_px",
        "centroid_num_x_px",
        "centroid_num_y_px",
        "centroid_diff_x_px",
        "centroid_diff_y_px",
        "centroid_dist_px",
        "area_ref_px2",
        "area_num_px2",
        "area_diff_px2",
        "area_rel_diff",
        "area_err_pct",
        "sensor_x_px",
        "sensor_y_px",
        "edge_segments",
        "mask_diff_pct",
        "distortion_model",
        "mesh_name",
        "geom_name",
        "camera_case",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def run_self_test() -> None:
    side_leng = 10.0
    tri_height = 0.5 * math.sqrt(3.0) * side_leng
    tri3_nodes = [
        (0.0, 0.0),
        (side_leng, 0.0),
        (0.5 * side_leng, tri_height),
    ]
    polyline = build_distorted_boundary_polyline(
        "tri3",
        tri3_nodes,
        256,
        {
            "distortion_model": "none",
            "focal_px_x": 1.0,
            "focal_px_y": 1.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "distortion_k1": 0.0,
            "distortion_k2": 0.0,
            "distortion_k3": 0.0,
            "distortion_p1": 0.0,
            "distortion_p2": 0.0,
        },
    )
    area_exact = 0.25 * math.sqrt(3.0) * side_leng * side_leng
    area_calc = calc_polyline_area_px2(polyline)
    if not math.isclose(area_calc, area_exact, rel_tol=0.0, abs_tol=1.0e-10):
        raise RuntimeError("distorted polyline self-test failed")


def main() -> int:
    args = parse_args()
    run_self_test()
    print("Running verif_6 self-test...")
    print("Self-test passed.")

    def sort_key(p: pathlib.Path) -> tuple[str, int, int, int]:
        m = re.match(
            r"cam(\d+)_frame(\d+)_field(\d+)_stats\.csv$",
            p.name,
        )
        if m:
            return (
                p.parent.name,
                int(m.group(1)),
                int(m.group(2)),
                int(m.group(3)),
            )
        return (p.parent.name, 0, 0, 0)

    stats_files_all = sorted(
        args.verif_root.glob("verif_6_*/*_stats.csv"),
        key=sort_key,
    )
    print(f"Found {len(stats_files_all)} total verif_6 stats files.")
    if not stats_files_all:
        raise FileNotFoundError(f"no stats files found under {args.verif_root}")

    rows = []
    for ii, stats_path in enumerate(stats_files_all, start=1):
        if ii == 1 or ii % 25 == 0 or ii == len(stats_files_all):
            print(
                "Analysing file "
                f"{ii}/{len(stats_files_all)}: "
                f"{stats_path.parent.name}/{stats_path.name}",
            )
        rows.append(
            analyse_stats_file(
                stats_path,
                args.edge_segments,
                args.edge_segments_sensor_mult,
            )
        )

    print(f"Writing summary CSV to {args.out_csv}...")
    write_summary_csv(args.out_csv, rows)
    max_centroid = max(abs(float(row["centroid_dist_px"])) for row in rows)
    max_mask_diff = max(abs(float(row["mask_diff_pct"])) for row in rows)
    print(f"Processed {len(rows)} verif_6 frames")
    print(f"Wrote summary to {args.out_csv}")
    print(f"Max centroid error: {max_centroid:.6f} px")
    print("Max area rel error: nan")
    print(f"Max mask diff: {max_mask_diff:.6f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
