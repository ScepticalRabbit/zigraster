// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const buildconfig = @import("../riley/zig/buildconfig.zig");
const F = buildconfig.F;
const S = buildconfig.SimdWidth;
const VecSB = buildconfig.VecSB;
const VecSF = buildconfig.VecSF;
const cam = @import("../riley/zig/camera.zig");
const csvio = @import("../riley/zig/csvio.zig");
const tcfg = @import("../dev_support/testconfig.zig");

const cases_path = "gold/verif/distortion_oracle_cases.csv";
const points_path = "gold/verif/distortion_oracle_points.csv";
const jacobians_path = "gold/verif/distortion_oracle_jacobians.csv";
const case_cols_num: usize = 59;
const points_cols_num: usize = 8;
const jac_cols_num: usize = 8;
const brown_start: usize = 5;
const forward_u_start: usize = 19;
const forward_v_start: usize = 29;
const inverse_u_start: usize = 39;
const inverse_v_start: usize = 49;

fn copyCoefficients(row: []const F, start: usize) [10]F {
    var coefficients: [10]F = undefined;
    @memcpy(&coefficients, row[start .. start + coefficients.len]);
    return coefficients;
}

fn buildPolynomial(row: []const F) cam.BidirectionalPolynomial {
    const order: cam.PolynomialOrder = @enumFromInt(
        @as(u8, @intFromFloat(row[2])),
    );
    const forward_map: ?cam.PolynomialMap = if (row[3] != 0.0)
        .{
            .order = order,
            .coeffs_u = copyCoefficients(row, forward_u_start),
            .coeffs_v = copyCoefficients(row, forward_v_start),
        }
    else
        null;
    const inv_map: ?cam.PolynomialMap = if (row[4] != 0.0)
        .{
            .order = order,
            .coeffs_u = copyCoefficients(row, inverse_u_start),
            .coeffs_v = copyCoefficients(row, inverse_v_start),
        }
    else
        null;
    return .{ .forward_map = forward_map, .inv_map = inv_map };
}

fn buildBrown(row: []const F) cam.BrownConrady {
    return .{
        .k1 = row[brown_start + 0],
        .k2 = row[brown_start + 1],
        .k3 = row[brown_start + 4],
        .p1 = row[brown_start + 2],
        .p2 = row[brown_start + 3],
    };
}

fn buildBrownExt(row: []const F) cam.BrownConradyExt {
    return .{
        .k1 = row[brown_start + 0],
        .k2 = row[brown_start + 1],
        .k3 = row[brown_start + 4],
        .k4 = row[brown_start + 5],
        .k5 = row[brown_start + 6],
        .k6 = row[brown_start + 7],
        .p1 = row[brown_start + 2],
        .p2 = row[brown_start + 3],
        .s1 = row[brown_start + 8],
        .s2 = row[brown_start + 9],
        .s3 = row[brown_start + 10],
        .s4 = row[brown_start + 11],
        .tau_x = row[brown_start + 12],
        .tau_y = row[brown_start + 13],
    };
}

fn buildModel(row: []const F) !cam.DistortionModel {
    const model_tag: u8 = @intFromFloat(row[1]);
    return switch (model_tag) {
        0 => .none,
        1 => .{ .brown_conrady = buildBrown(row) },
        2 => .{ .brown_conrady_ext = buildBrownExt(row) },
        3 => .{ .polynomial = buildPolynomial(row) },
        4 => .{ .brown_conrady_polynomial = .{
            .brown_conrady = buildBrown(row),
            .polynomial = buildPolynomial(row),
        } },
        5 => .{ .brown_conrady_ext_polynomial = .{
            .brown_conrady_ext = buildBrownExt(row),
            .polynomial = buildPolynomial(row),
        } },
        else => error.InvalidOracleModel,
    };
}

fn reportMismatch(
    operation: []const u8,
    case_id: usize,
    point_id: usize,
    expected: [2]F,
    actual: [2]F,
    tolerance: F,
) void {
    std.debug.print(
        "distortion oracle {s} mismatch: case={d}, point={d}, " ++
            "expected=({e},{e}), actual=({e},{e}), " ++
            "error=({e},{e}), tolerance={e}\n",
        .{
            operation,
            case_id,
            point_id,
            expected[0],
            expected[1],
            actual[0],
            actual[1],
            @abs(actual[0] - expected[0]),
            @abs(actual[1] - expected[1]),
            tolerance,
        },
    );
}

fn expectPairApprox(
    operation: []const u8,
    case_id: usize,
    point_id: usize,
    expected: [2]F,
    actual: [2]F,
    tolerance: F,
) !void {
    const error_x = @abs(actual[0] - expected[0]);
    const error_y = @abs(actual[1] - expected[1]);
    if (error_x > tolerance or error_y > tolerance) {
        reportMismatch(operation, case_id, point_id, expected, actual, tolerance);
        return error.DistortionOracleMismatch;
    }
}

fn checkScalarPoints(cases: anytype, points: anytype) !void {
    const tolerance = tcfg.DISTORTION_ORACLE_TOL;
    for (0..points.dims[0]) |rr| {
        const point_row = points.slice[rr * points_cols_num ..][0..points_cols_num];
        const case_id: usize = @intFromFloat(point_row[0]);
        const point_id: usize = @intFromFloat(point_row[1]);
        const case_row = cases.slice[case_id * case_cols_num ..][0..case_cols_num];
        const model = try buildModel(case_row);
        const ideal = [2]F{ point_row[2], point_row[3] };
        const expected_observed = [2]F{ point_row[4], point_row[5] };
        const expected_inverse = [2]F{ point_row[6], point_row[7] };
        const actual_observed = cam.forwardDistortionModelScal(
            model,
            ideal[0],
            ideal[1],
        );
        try expectPairApprox(
            "scalar forward",
            case_id,
            point_id,
            expected_observed,
            actual_observed,
            tolerance.forward_abs_norm,
        );

        const recovered = try cam.invDistortionModelScal(
            model,
            expected_observed[0],
            expected_observed[1],
        );
        try expectPairApprox(
            "scalar inverse",
            case_id,
            point_id,
            expected_inverse,
            .{ recovered.x, recovered.y },
            tolerance.inverse_abs_norm,
        );
    }
}

fn checkSIMDPoints(cases: anytype, points: anytype) !void {
    const tolerance = tcfg.DISTORTION_ORACLE_TOL.backend_abs_norm;
    var row_start: usize = 0;
    while (row_start < points.dims[0]) {
        const first_row = points.slice[row_start * points_cols_num ..][0..points_cols_num];
        const case_id: usize = @intFromFloat(first_row[0]);
        var lane_count: usize = 0;
        while (lane_count < S and row_start + lane_count < points.dims[0]) {
            const row = points.slice[(row_start + lane_count) * points_cols_num ..][0..points_cols_num];
            if (@as(usize, @intFromFloat(row[0])) != case_id) break;
            lane_count += 1;
        }

        var ideal_x = [_]F{0.0} ** S;
        var ideal_y = [_]F{0.0} ** S;
        var observed_x = [_]F{0.0} ** S;
        var observed_y = [_]F{0.0} ** S;
        var active = [_]bool{false} ** S;
        for (0..lane_count) |lane| {
            const row = points.slice[(row_start + lane) * points_cols_num ..][0..points_cols_num];
            ideal_x[lane] = row[2];
            ideal_y[lane] = row[3];
            observed_x[lane] = row[4];
            observed_y[lane] = row[5];
            active[lane] = true;
        }

        const case_row = cases.slice[case_id * case_cols_num ..][0..case_cols_num];
        const model = try buildModel(case_row);
        const actual_forward = switch (model) {
            .brown_conrady => |brown| cam.forwardDistortionSIMD(
                cam.BrownConrady,
                brown,
                @as(VecSF, ideal_x),
                @as(VecSF, ideal_y),
            ),
            .brown_conrady_ext => |brown| cam.forwardDistortionSIMD(
                cam.BrownConradyExt,
                brown,
                @as(VecSF, ideal_x),
                @as(VecSF, ideal_y),
            ),
            else => null,
        };
        if (actual_forward) |forward| {
            const forward_x: [S]F = forward.x_d;
            const forward_y: [S]F = forward.y_d;
            for (0..lane_count) |lane| {
                const row = points.slice[(row_start + lane) * points_cols_num ..][0..points_cols_num];
                try expectPairApprox(
                    "SIMD forward",
                    case_id,
                    @intFromFloat(row[1]),
                    .{ row[4], row[5] },
                    .{ forward_x[lane], forward_y[lane] },
                    tolerance,
                );
            }
        }
        const solved = try cam.invDistortionModelSIMD(
            model,
            @as(VecSF, observed_x),
            @as(VecSF, observed_y),
            @as(VecSB, active),
        );
        const recovered_x: [S]F = solved.x;
        const recovered_y: [S]F = solved.y;
        for (0..lane_count) |lane| {
            const row = points.slice[(row_start + lane) * points_cols_num ..][0..points_cols_num];
            try expectPairApprox(
                "SIMD inverse",
                case_id,
                @intFromFloat(row[1]),
                .{ row[6], row[7] },
                .{ recovered_x[lane], recovered_y[lane] },
                tolerance,
            );
        }
        row_start += lane_count;
    }
}

fn modelJacobian(model: cam.DistortionModel, x: F, y: F) ?[2][2]F {
    return switch (model) {
        .brown_conrady => |brown| brown.forwardWithJac(x, y).jac,
        .brown_conrady_ext => |brown| brown.forwardWithJac(x, y).jac,
        .polynomial => |polynomial| if (polynomial.forward_map) |forward_map|
            forward_map.forwardWithJac(x, y).jac
        else
            null,
        else => null,
    };
}

fn checkJacobians(cases: anytype, jacobians: anytype) !void {
    const tolerance = tcfg.DISTORTION_ORACLE_TOL;
    for (0..jacobians.dims[0]) |rr| {
        const row = jacobians.slice[rr * jac_cols_num ..][0..jac_cols_num];
        const case_id: usize = @intFromFloat(row[0]);
        const case_row = cases.slice[case_id * case_cols_num ..][0..case_cols_num];
        const model = try buildModel(case_row);
        const actual = modelJacobian(model, row[2], row[3]) orelse continue;
        const expected = [2][2]F{
            .{ row[4], row[5] },
            .{ row[6], row[7] },
        };
        for (0..2) |jj| {
            for (0..2) |ii| {
                const scale = @max(1.0, @abs(expected[jj][ii]));
                const allowed = @max(
                    tolerance.jac_abs,
                    tolerance.jac_rel * scale,
                );
                if (@abs(actual[jj][ii] - expected[jj][ii]) > allowed) {
                    std.debug.print(
                        "distortion oracle Jacobian mismatch: case={d}, " ++
                            "point={d}, entry=({d},{d}), expected={e}, " ++
                            "actual={e}, tolerance={e}\n",
                        .{
                            case_id,
                            @as(usize, @intFromFloat(row[1])),
                            jj,
                            ii,
                            expected[jj][ii],
                            actual[jj][ii],
                            allowed,
                        },
                    );
                    return error.DistortionOracleJacobianMismatch;
                }
            }
        }
    }
}

pub fn run(allocator: std.mem.Allocator, io: std.Io) !void {
    var cases = try csvio.loadScalarCsv2D(allocator, io, cases_path);
    defer {
        allocator.free(cases.slice);
        cases.deinit(allocator);
    }
    var points = try csvio.loadScalarCsv2D(allocator, io, points_path);
    defer {
        allocator.free(points.slice);
        points.deinit(allocator);
    }
    var jacobians = try csvio.loadScalarCsv2D(allocator, io, jacobians_path);
    defer {
        allocator.free(jacobians.slice);
        jacobians.deinit(allocator);
    }

    try std.testing.expectEqual(case_cols_num, cases.dims[1]);
    try std.testing.expectEqual(points_cols_num, points.dims[1]);
    try std.testing.expectEqual(jac_cols_num, jacobians.dims[1]);
    try checkScalarPoints(cases, points);
    try checkSIMDPoints(cases, points);
    try checkJacobians(cases, jacobians);
}
