"""Generate independent OpenCV and NumPy distortion oracle data."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.polynomial import polynomial as nppoly


MODEL_NONE = 0
MODEL_BROWN = 1
MODEL_BROWN_EXT = 2
MODEL_POLYNOMIAL = 3
MODEL_BROWN_POLYNOMIAL = 4
MODEL_BROWN_EXT_POLYNOMIAL = 5
COEFFS_NUM = 10


@dataclass(slots=True)
class OracleCase:
    """Complete specification for one distortion oracle case."""

    name: str
    model: int
    brown: tuple[float, ...] = (0.0,) * 14
    order: int = 1
    forward_u: tuple[float, ...] | None = None
    forward_v: tuple[float, ...] | None = None
    inverse_u: tuple[float, ...] | None = None
    inverse_v: tuple[float, ...] | None = None


def _coeffs(*values: float) -> tuple[float, ...]:
    return tuple(values) + (0.0,) * (COEFFS_NUM - len(values))


def get_cases() -> list[OracleCase]:
    """Return diagnostic cases covering every existing model family."""

    poly_linear_u = _coeffs(1.0e-3, 8.0e-3, -3.0e-3)
    poly_linear_v = _coeffs(-2.0e-3, 4.0e-3, -7.0e-3)
    poly_quad_u = _coeffs(5.0e-4, 6.0e-3, -2.0e-3, 8.0e-3, 1.2e-2, -5.0e-3)
    poly_quad_v = _coeffs(-7.0e-4, 3.0e-3, -5.0e-3, -9.0e-3, 7.0e-3, 1.0e-2)
    poly_cubic_u = _coeffs(
        3.0e-4, 5.0e-3, -2.0e-3, 7.0e-3, -8.0e-3,
        4.0e-3, 1.1e-2, -6.0e-3, 9.0e-3, -5.0e-3,
    )
    poly_cubic_v = _coeffs(
        -4.0e-4, 2.0e-3, -6.0e-3, -5.0e-3, 1.0e-2,
        8.0e-3, -7.0e-3, 5.0e-3, -1.2e-2, 6.0e-3,
    )
    inverse_u = _coeffs(-4.0e-4, -4.0e-3, 2.0e-3, -5.0e-3, 7.0e-3, 3.0e-3)
    inverse_v = _coeffs(6.0e-4, -2.0e-3, 5.0e-3, 4.0e-3, -6.0e-3, -8.0e-3)

    cases = [
        OracleCase("none", MODEL_NONE),
        OracleCase("brown_zero", MODEL_BROWN),
        OracleCase("brown_k1", MODEL_BROWN, (-0.12,) + (0.0,) * 13),
        OracleCase("brown_k2", MODEL_BROWN, (0.0, 0.08) + (0.0,) * 12),
        OracleCase("brown_k3", MODEL_BROWN, (0.0, 0.0, 0.0, 0.0, -0.04) + (0.0,) * 9),
        OracleCase("brown_p1", MODEL_BROWN, (0.0, 0.0, 1.5e-3) + (0.0,) * 11),
        OracleCase("brown_p2", MODEL_BROWN, (0.0, 0.0, 0.0, -2.0e-3) + (0.0,) * 10),
        OracleCase(
            "brown_mixed",
            MODEL_BROWN,
            (-0.11, 0.035, 1.2e-3, -1.7e-3, -0.008) + (0.0,) * 9,
        ),
        OracleCase("brown_ext_zero", MODEL_BROWN_EXT),
        OracleCase(
            "brown_ext_k4",
            MODEL_BROWN_EXT,
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.07) + (0.0,) * 8,
        ),
        OracleCase(
            "brown_ext_k5",
            MODEL_BROWN_EXT,
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.04) + (0.0,) * 7,
        ),
        OracleCase(
            "brown_ext_k6",
            MODEL_BROWN_EXT,
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.025) + (0.0,) * 6,
        ),
        OracleCase(
            "brown_ext_mixed",
            MODEL_BROWN_EXT,
            (-0.10, 0.03, 1.1e-3, -1.6e-3, -0.007, 0.045, -0.018, 0.006)
            + (0.0,) * 6,
        ),
        OracleCase(
            "brown_ext_s1", MODEL_BROWN_EXT, (0.0,) * 8 + (2.0e-3,) + (0.0,) * 5
        ),
        OracleCase(
            "brown_ext_s2", MODEL_BROWN_EXT, (0.0,) * 9 + (-1.5e-3,) + (0.0,) * 4
        ),
        OracleCase(
            "brown_ext_s3", MODEL_BROWN_EXT, (0.0,) * 10 + (1.7e-3,) + (0.0,) * 3
        ),
        OracleCase(
            "brown_ext_s4", MODEL_BROWN_EXT, (0.0,) * 11 + (-1.2e-3,) + (0.0,) * 2
        ),
        OracleCase("brown_ext_tau_x", MODEL_BROWN_EXT, (0.0,) * 12 + (0.035, 0.0)),
        OracleCase("brown_ext_tau_y", MODEL_BROWN_EXT, (0.0,) * 12 + (0.0, -0.052)),
        OracleCase(
            "brown_ext_full14",
            MODEL_BROWN_EXT,
            (
                -0.10, 0.03, 1.1e-3, -1.6e-3, -0.007, 0.045, -0.018,
                0.006, 1.8e-3, -1.1e-3, 1.4e-3, -9.0e-4, 0.031, -0.047,
            ),
        ),
        OracleCase(
            "polynomial_linear",
            MODEL_POLYNOMIAL,
            order=1,
            forward_u=poly_linear_u,
            forward_v=poly_linear_v,
        ),
        OracleCase(
            "polynomial_quadratic",
            MODEL_POLYNOMIAL,
            order=2,
            forward_u=poly_quad_u,
            forward_v=poly_quad_v,
        ),
        OracleCase(
            "polynomial_cubic",
            MODEL_POLYNOMIAL,
            order=3,
            forward_u=poly_cubic_u,
            forward_v=poly_cubic_v,
        ),
        OracleCase(
            "polynomial_inverse_only",
            MODEL_POLYNOMIAL,
            order=2,
            inverse_u=inverse_u,
            inverse_v=inverse_v,
        ),
        OracleCase(
            "polynomial_both_maps",
            MODEL_POLYNOMIAL,
            order=2,
            forward_u=poly_quad_u,
            forward_v=poly_quad_v,
            inverse_u=inverse_u,
            inverse_v=inverse_v,
        ),
        OracleCase(
            "brown_polynomial",
            MODEL_BROWN_POLYNOMIAL,
            (-0.09, 0.025, 9.0e-4, -1.3e-3, -0.006) + (0.0,) * 9,
            order=3,
            forward_u=poly_cubic_u,
            forward_v=poly_cubic_v,
        ),
        OracleCase(
            "brown_ext_polynomial",
            MODEL_BROWN_EXT_POLYNOMIAL,
            (-0.08, 0.02, 8.0e-4, -1.1e-3, -0.005, 0.035, -0.012, 0.004)
            + (0.0,) * 6,
            order=2,
            forward_u=poly_quad_u,
            forward_v=poly_quad_v,
        ),
        OracleCase(
            "brown_ext_full14_polynomial",
            MODEL_BROWN_EXT_POLYNOMIAL,
            (
                -0.08, 0.02, 8.0e-4, -1.1e-3, -0.005, 0.035, -0.012,
                0.004, 1.5e-3, -8.0e-4, 1.2e-3, -7.0e-4, 0.027, -0.041,
            ),
            order=3,
            forward_u=poly_cubic_u,
            forward_v=poly_cubic_v,
        ),
        OracleCase(
            "brown_polynomial_inverse_only",
            MODEL_BROWN_POLYNOMIAL,
            (-0.07, 0.018, 7.0e-4, -9.0e-4, -0.004) + (0.0,) * 9,
            order=2,
            inverse_u=inverse_u,
            inverse_v=inverse_v,
        ),
        OracleCase(
            "brown_ext_polynomial_both_maps",
            MODEL_BROWN_EXT_POLYNOMIAL,
            (
                -0.06,
                0.016,
                6.0e-4,
                -8.0e-4,
                -0.003,
                0.025,
                -0.009,
                0.003,
                1.0e-3,
                -6.0e-4,
                9.0e-4,
                -5.0e-4,
                0.019,
                -0.029,
            ),
            order=2,
            forward_u=poly_quad_u,
            forward_v=poly_quad_v,
            inverse_u=inverse_u,
            inverse_v=inverse_v,
        ),
    ]

    powers = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2),
              (3, 0), (2, 1), (1, 2), (0, 3))
    for output_name in ("u", "v"):
        for coeff_idx, (power_x, power_y) in enumerate(powers):
            values = [0.0] * COEFFS_NUM
            values[coeff_idx] = 7.0e-3
            coeffs = tuple(values)
            zero = (0.0,) * COEFFS_NUM
            cases.append(
                OracleCase(
                    f"polynomial_basis_{output_name}_{power_x}_{power_y}",
                    MODEL_POLYNOMIAL,
                    order=1 if coeff_idx < 3 else 2 if coeff_idx < 6 else 3,
                    forward_u=coeffs if output_name == "u" else zero,
                    forward_v=coeffs if output_name == "v" else zero,
                )
            )
    return cases


def get_points() -> np.ndarray:
    """Return a compact asymmetric sample of a wide normalized field."""

    x_values = np.array([-0.80, -0.57, -0.29, 0.0, 0.23, 0.51, 0.80])
    y_values = np.array([-0.60, -0.33, -0.11, 0.17, 0.39, 0.60])
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    diagnostic = np.array(
        [
            [0.0, 0.0], [0.71, -0.23], [-0.37, 0.53], [0.12, -0.49],
            [-0.73, 0.08], [0.63, 0.44], [-0.19, -0.56], [0.79, 0.31],
            [-0.61, -0.41], [0.42, -0.52], [-0.48, 0.27], [0.06, 0.58],
        ]
    )
    return np.vstack((grid, diagnostic))


def _coefficient_matrix(coefficients: tuple[float, ...], order: int) -> np.ndarray:
    matrix = np.zeros((4, 4), dtype=np.float64)
    powers = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2),
              (3, 0), (2, 1), (1, 2), (0, 3))
    term_count = (3, 6, 10)[order - 1]
    for coeff, (power_x, power_y) in zip(coefficients[:term_count], powers):
        matrix[power_x, power_y] = coeff
    return matrix


def evaluate_polynomial(
    points: np.ndarray,
    coefficients_u: tuple[float, ...],
    coefficients_v: tuple[float, ...],
    order: int,
) -> np.ndarray:
    """Evaluate an identity-plus-displacement polynomial using NumPy."""

    matrix_u = _coefficient_matrix(coefficients_u, order)
    matrix_v = _coefficient_matrix(coefficients_v, order)
    x = points[:, 0]
    y = points[:, 1]
    return np.column_stack(
        (
            x + nppoly.polyval2d(x, y, matrix_u),
            y + nppoly.polyval2d(x, y, matrix_v),
        )
    )


def evaluate_polynomial_jacobian(case: OracleCase, points: np.ndarray) -> np.ndarray:
    """Evaluate polynomial Jacobians using NumPy's analytic derivatives."""

    matrix_u = _coefficient_matrix(case.forward_u or (0.0,) * COEFFS_NUM, case.order)
    matrix_v = _coefficient_matrix(case.forward_v or (0.0,) * COEFFS_NUM, case.order)
    du_dx = nppoly.polyder(matrix_u, axis=0)
    du_dy = nppoly.polyder(matrix_u, axis=1)
    dv_dx = nppoly.polyder(matrix_v, axis=0)
    dv_dy = nppoly.polyder(matrix_v, axis=1)
    x = points[:, 0]
    y = points[:, 1]
    jac = np.empty((points.shape[0], 2, 2), dtype=np.float64)
    jac[:, 0, 0] = 1.0 + nppoly.polyval2d(x, y, du_dx)
    jac[:, 0, 1] = nppoly.polyval2d(x, y, du_dy)
    jac[:, 1, 0] = nppoly.polyval2d(x, y, dv_dx)
    jac[:, 1, 1] = 1.0 + nppoly.polyval2d(x, y, dv_dy)
    return jac


def evaluate_brown(case: OracleCase, points: np.ndarray) -> np.ndarray:
    """Evaluate Brown-Conrady using OpenCV projectPoints."""

    object_points = np.column_stack((points, np.ones(points.shape[0])))
    projected, _ = cv2.projectPoints(
        object_points,
        np.zeros(3),
        np.zeros(3),
        np.eye(3),
        np.asarray(case.brown, dtype=np.float64),
    )
    return projected.reshape((-1, 2))


def _invert_polynomial(case: OracleCase, targets: np.ndarray) -> np.ndarray:
    """Invert an inverse-map polynomial with NumPy-derived Newton Jacobians."""

    inverse_case = OracleCase(
        case.name,
        MODEL_POLYNOMIAL,
        order=case.order,
        forward_u=case.inverse_u,
        forward_v=case.inverse_v,
    )
    values = targets.copy()
    for _ in range(30):
        mapped = evaluate_polynomial(
            values,
            inverse_case.forward_u or (0.0,) * COEFFS_NUM,
            inverse_case.forward_v or (0.0,) * COEFFS_NUM,
            inverse_case.order,
        )
        residual = mapped - targets
        if np.max(np.abs(residual)) < 1.0e-14:
            return values
        jac = evaluate_polynomial_jacobian(inverse_case, values)
        delta = np.linalg.solve(jac, -residual[..., np.newaxis]).squeeze(-1)
        values += delta
    raise RuntimeError(f"polynomial oracle inversion failed for {case.name}")


def evaluate_case(case: OracleCase, points: np.ndarray) -> np.ndarray:
    """Evaluate one complete forward distortion case independently."""

    values = points
    if case.model in (MODEL_BROWN, MODEL_BROWN_EXT,
                      MODEL_BROWN_POLYNOMIAL, MODEL_BROWN_EXT_POLYNOMIAL):
        values = evaluate_brown(case, values)
    if case.model in (MODEL_POLYNOMIAL, MODEL_BROWN_POLYNOMIAL,
                      MODEL_BROWN_EXT_POLYNOMIAL):
        if case.forward_u is not None and case.forward_v is not None:
            values = evaluate_polynomial(
                values, case.forward_u, case.forward_v, case.order
            )
        elif case.inverse_u is not None and case.inverse_v is not None:
            values = _invert_polynomial(case, values)
        else:
            raise ValueError(f"missing polynomial map for {case.name}")
    return values.copy()


def evaluate_inverse_case(case: OracleCase, observed: np.ndarray) -> np.ndarray:
    """Evaluate Riley's declared inverse-stage order using independent tools."""

    values = observed
    if case.model in (
        MODEL_POLYNOMIAL,
        MODEL_BROWN_POLYNOMIAL,
        MODEL_BROWN_EXT_POLYNOMIAL,
    ):
        if case.inverse_u is not None and case.inverse_v is not None:
            values = evaluate_polynomial(
                values,
                case.inverse_u,
                case.inverse_v,
                case.order,
            )
        elif case.forward_u is not None and case.forward_v is not None:
            values = _invert_polynomial(
                OracleCase(
                    case.name,
                    MODEL_POLYNOMIAL,
                    order=case.order,
                    inverse_u=case.forward_u,
                    inverse_v=case.forward_v,
                ),
                values,
            )
        else:
            raise ValueError(f"missing polynomial map for {case.name}")
    if case.model in (
        MODEL_BROWN,
        MODEL_BROWN_EXT,
        MODEL_BROWN_POLYNOMIAL,
        MODEL_BROWN_EXT_POLYNOMIAL,
    ):
        values = cv2.undistortPoints(
            values.reshape((-1, 1, 2)),
            np.eye(3),
            np.asarray(case.brown, dtype=np.float64),
        ).reshape((-1, 2))
    return values.copy()


def evaluate_numerical_jacobian(case: OracleCase, points: np.ndarray) -> np.ndarray:
    """Differentiate the complete independent map by central differences."""

    step = 1.0e-6
    jac = np.empty((points.shape[0], 2, 2), dtype=np.float64)
    for axis in range(2):
        offset = np.zeros_like(points)
        offset[:, axis] = step
        derivative = (evaluate_case(case, points + offset) -
                      evaluate_case(case, points - offset)) / (2.0 * step)
        jac[:, :, axis] = derivative
    return jac


def _manifest_row(case_id: int, case: OracleCase) -> list[float | int]:
    forward_u = case.forward_u or (0.0,) * COEFFS_NUM
    forward_v = case.forward_v or (0.0,) * COEFFS_NUM
    inverse_u = case.inverse_u or (0.0,) * COEFFS_NUM
    inverse_v = case.inverse_v or (0.0,) * COEFFS_NUM
    return [
        case_id,
        case.model,
        case.order,
        int(case.forward_u is not None),
        int(case.inverse_u is not None),
        *case.brown,
        *forward_u,
        *forward_v,
        *inverse_u,
        *inverse_v,
    ]


def generate_distortion_oracles(gold_root: Path) -> list[Path]:
    """Generate compact committed distortion oracle CSV files."""

    gold_root.mkdir(parents=True, exist_ok=True)
    cases_path = gold_root / "distortion_oracle_cases.csv"
    points_path = gold_root / "distortion_oracle_points.csv"
    jacobians_path = gold_root / "distortion_oracle_jacobians.csv"
    cases = get_cases()
    points = get_points()
    jacobian_points = points[::5]

    with cases_path.open("w", newline="") as out_file:
        writer = csv.writer(out_file, lineterminator="\n")
        for case_id, case in enumerate(cases):
            writer.writerow(_manifest_row(case_id, case))

    with points_path.open("w", newline="") as out_file:
        writer = csv.writer(out_file, lineterminator="\n")
        for case_id, case in enumerate(cases):
            observed = evaluate_case(case, points)
            recovered = evaluate_inverse_case(case, observed)
            for point_id, (ideal, distorted, inverse) in enumerate(
                zip(points, observed, recovered)
            ):
                writer.writerow([case_id, point_id, *ideal, *distorted, *inverse])

    with jacobians_path.open("w", newline="") as out_file:
        writer = csv.writer(out_file, lineterminator="\n")
        for case_id, case in enumerate(cases):
            if case.model == MODEL_POLYNOMIAL and case.forward_u is not None:
                jacobians = evaluate_polynomial_jacobian(case, jacobian_points)
            else:
                jacobians = evaluate_numerical_jacobian(case, jacobian_points)
            for point_id, (ideal, jac) in enumerate(zip(jacobian_points, jacobians)):
                writer.writerow([case_id, point_id, *ideal, *jac.ravel()])

    return [cases_path, points_path, jacobians_path]
