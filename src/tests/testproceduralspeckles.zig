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
const expected_config = @import("expected_speckle_config.zig");
const meshio = @import("../riley/zig/meshio.zig");
const meshpipeline = @import("../riley/zig/meshpipeline.zig");
const shaderops = @import("../riley/zig/shaderops_common.zig");
const uvio = @import("../riley/zig/uvio.zig");

fn evaluatorName(evaluator: buildconfig.SpeckleEvaluator) []const u8 {
    return switch (evaluator) {
        .cell_hash => "cell-hash",
        .list_naive => "list-naive",
        .list_indexed => "list-indexed",
        .mask_1bit => "mask-1bit",
        .mask_u8 => "mask-u8",
    };
}

test "selected speckle evaluator prepares its production resources" {
    try std.testing.expectEqualStrings(
        expected_config.evaluator,
        evaluatorName(buildconfig.speckle_evaluator),
    );
    try std.testing.expectEqualStrings(
        expected_config.shape,
        @tagName(buildconfig.speckle_shape),
    );

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = std.testing.io;

    const data_dir = "data/min/tri3_sphere200/";
    const sim_data = try meshio.loadSimData(
        allocator,
        io,
        data_dir ++ "coords.csv",
        data_dir ++ "connect.csv",
        null,
        null,
    );
    const uvs = try uvio.loadUVMap(allocator, io, data_dir ++ "uvs.csv");
    const params = shaderops.Speckle2DParams{
        .seed = 12345,
        .cells_per_uv = .{ 24.0, 20.0 },
        .occupancy = 0.8,
        .radius_mean = 0.42,
        .radius_jitter = 0.06,
        .edge_softness = 0.03,
    };
    const mesh_input = meshpipeline.MeshInput{
        .mesh_type = .tri3,
        .coords = sim_data.coords,
        .connect = sim_data.connect,
        .disp = null,
        .shader = .{ .func = .{
            .uvs = uvs.array,
            .coord_mode = .uv,
            .builtin = .speckle,
            .params = params.toFuncShaderParams(),
            .bits = 8,
            .scaling = .auto,
            .normal_type = .none,
        } },
    };
    const mesh_static = try meshpipeline.initMeshStatic(allocator, &mesh_input);
    const func_static = switch (mesh_static.shader) {
        .func => |func| func,
        else => return error.UnexpectedShaderVariant,
    };

    const expected_evaluator = expected_config.evaluator;
    try std.testing.expectEqual(
        std.mem.startsWith(u8, expected_evaluator, "list-"),
        func_static.speckle_list != null,
    );
    try std.testing.expectEqual(
        std.mem.startsWith(u8, expected_evaluator, "mask-"),
        func_static.speckle_mask != null,
    );
}
