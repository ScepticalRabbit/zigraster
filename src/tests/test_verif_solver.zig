// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const gk = @import("../riley/zig/geometrykernels.zig");
const cam = @import("../riley/zig/camera.zig");
const orch = @import("../dev_support/orchestration.zig");
const tcfg = @import("../dev_support/testconfig.zig");
const verif = @import("../dev_support/verif.zig");
const vconst = @import("../dev_support/verifconstants.zig");
const solver_verif = @import("../verif_1_solver.zig");

const tri_points = [_]verif.SamplePoint{
    .{ .xi_true = 1.0 / 3.0, .eta_true = 1.0 / 3.0 },
    .{ .xi_true = 0.05, .eta_true = 0.05 },
    .{ .xi_true = 0.90, .eta_true = 0.05 },
    .{ .xi_true = 0.05, .eta_true = 0.90 },
    .{ .xi_true = 0.20, .eta_true = 0.35 },
    .{ .xi_true = 0.65, .eta_true = 0.20 },
};

const quad_points = [_]verif.SamplePoint{
    .{ .xi_true = 0.0, .eta_true = 0.0 },
    .{ .xi_true = -0.90, .eta_true = -0.90 },
    .{ .xi_true = 0.90, .eta_true = -0.90 },
    .{ .xi_true = 0.90, .eta_true = 0.90 },
    .{ .xi_true = -0.90, .eta_true = 0.90 },
    .{ .xi_true = -0.35, .eta_true = 0.55 },
};

const quad_ibi_points = [_]verif.SamplePoint{
    .{ .xi_true = 0.50, .eta_true = 0.50 },
    .{ .xi_true = 0.05, .eta_true = 0.05 },
    .{ .xi_true = 0.95, .eta_true = 0.05 },
    .{ .xi_true = 0.95, .eta_true = 0.95 },
    .{ .xi_true = 0.05, .eta_true = 0.95 },
    .{ .xi_true = 0.30, .eta_true = 0.70 },
};

fn checkCase(
    comptime mesh_type: gk.MeshType,
    allocator: std.mem.Allocator,
    io: std.Io,
    case_spec: vconst.DistortCase,
    points: []const verif.SamplePoint,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const local_alloc = arena.allocator();

    const camera = try cam.CameraPrepared.init(local_alloc, case_spec.camera_input);
    const sim_data = try orch.loadData(local_alloc, io, case_spec.data_dir);
    const frame_idx = if (sim_data.field) |field| @min(@as(usize, 1), field.getTimeN() - 1) else 0;
    const nodes = solver_verif.frameNodes(mesh_type, &sim_data, frame_idx);
    const solver_nodes = verif.worldNodesToSolverCoords(
        mesh_type,
        &camera,
        nodes.x[0..],
        nodes.y[0..],
        nodes.z[0..],
    );

    for (points) |point| {
        const record = try solver_verif.evalSample(
            mesh_type,
            &camera,
            nodes.x[0..],
            nodes.y[0..],
            nodes.z[0..],
            &solver_nodes,
            point,
        );
        try std.testing.expect(record.converged);
        try std.testing.expect(record.in_domain);
        try std.testing.expectApproxEqAbs(point.xi_true, record.xi_rec, tcfg.VERIF_TOL.para_abs);
        try std.testing.expectApproxEqAbs(point.eta_true, record.eta_rec, tcfg.VERIF_TOL.para_abs);
        try std.testing.expect(record.reproj_err <= tcfg.VERIF_TOL.reproj_abs_px);
    }
}

test "verification solver recovers known parent coordinates" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    for (vconst.distort_cases) |case_spec| {
        const selected = std.mem.eql(u8, case_spec.case_name, "shear") or
            std.mem.eql(u8, case_spec.case_name, "bulge");
        if (!selected) continue;

        switch (case_spec.mesh_type) {
            .tri3 => try checkCase(.tri3, allocator, io, case_spec, &tri_points),
            .tri3opt => {},
            .tri6 => try checkCase(.tri6, allocator, io, case_spec, &tri_points),
            .quad4ibi => try checkCase(.quad4ibi, allocator, io, case_spec, &quad_ibi_points),
            .quad4newton => try checkCase(.quad4newton, allocator, io, case_spec, &quad_points),
            .quad8 => try checkCase(.quad8, allocator, io, case_spec, &quad_points),
            .quad9 => try checkCase(.quad9, allocator, io, case_spec, &quad_points),
        }
    }
}
