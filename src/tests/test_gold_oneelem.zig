// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const buildconfig = @import("../riley/zig/buildconfig.zig");
const camera = @import("../riley/zig/camera.zig");
const common = @import("../dev_support/tests.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const iio = @import("../riley/zig/imageio.zig");
const meshio = @import("../riley/zig/meshio.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const orch = @import("../dev_support/orchestration.zig");
const policy = @import("../dev_support/testpolicy.zig");
const rastcfg = @import("../riley/zig/rasterconfig.zig");
const riley = @import("../riley/zig/riley.zig");
const shaderops = @import("../riley/zig/shaderops_common.zig");
const tcfg = @import("../dev_support/testconfig.zig");

const F = buildconfig.F;
const CameraInput = camera.CameraInput;
const MeshInput = mo.MeshInput;
const Timestamp = std.Io.Clock.Timestamp;

const pixel_num_oneelem = [_]u32{ 128, 128 };

pub fn runOneElemCaseTest(
    allocator: std.mem.Allocator,
    io: std.Io,
    case_name: []const u8,
    mesh_type: gk.MeshType,
    gold_dir_root: []const u8,
    data_dir_root: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const prepared = try orch.prepareSingleMeshCase(
        aa,
        io,
        case_name,
        mesh_type,
        pixel_num_oneelem,
        1.15,
        data_dir_root,
    );

    const case_dir_name = try std.fmt.allocPrint(
        aa,
        "oneelem_{s}_{s}_func_checker",
        .{ case_name, @tagName(mesh_type) },
    );

    const gold_dir = try std.fmt.allocPrint(
        aa,
        "{s}/{s}",
        .{ gold_dir_root, case_dir_name },
    );

    const shader_params = shaderops.FuncShaderParams{
        .coord_scale = .{ 4.0, 4.0 },
        .settings = .{ .checker = .{} },
    };

    const mesh_input = MeshInput{
        .mesh_type = mesh_type,
        .coords = prepared.sim_data.coords,
        .connect = prepared.sim_data.connect,
        .disp = prepared.sim_data.field,
        .shader = .{
            .func = .{
                .uvs = null,
                .coord_mode = .para,
                .builtin = .checker,
                .params = shader_params,
                .bits = 8,
                .scaling = .auto,
                .normal_type = .none,
            },
        },
    };

    const camera_input = CameraInput{
        .pixels_num = prepared.camera.pixels_num,
        .pixels_size = prepared.camera.pixels_size,
        .pos_world = prepared.camera.pos_world,
        .rot_world = prepared.camera.rot_world,
        .roi_cent_world = prepared.camera.roi_cent_world,
        .focal_length = prepared.camera.focal_length,
        .sub_sample = 1,
        .distortion = .none,
    };

    var run_config = config;
    run_config.save_strategy = .memory;
    run_config.background_value = 127.5;

    const start_time = Timestamp.now(io, .awake);
    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), run_config.total_threads) },
    };

    const result = try riley.raster(
        aa,
        &render_groups,
        &[_]CameraInput{camera_input},
        &[_]MeshInput{mesh_input},
        run_config,
        null,
    );

    var render_result = result orelse return error.NoResult;
    defer aa.free(render_result.slice);

    const end_time = Timestamp.now(io, .awake);
    const duration_ms = @as(
        F,
        @floatFromInt(start_time.durationTo(end_time).raw.nanoseconds),
    ) / 1e6;

    const frames_num = if (render_result.dims.len == 5)
        render_result.dims[1]
    else
        render_result.dims[0];

    for (0..frames_num) |ff| {
        const gold_path = try common.findGoldPath(
            aa,
            io,
            gold_dir,
            0,
            ff,
            0,
            false,
        );

        common.compareNDArrayToGold(
            aa,
            io,
            &render_result,
            0,
            ff,
            0,
            1,
            gold_path,
            tcfg.REL_TOL,
            tcfg.ABS_TOL,
        ) catch |err| {
            if (tcfg.TEST_CASE_VERBOSE) {
                std.debug.print(
                    "FAIL {s} frame {d} ({d:.2} ms)\n",
                    .{ case_dir_name, ff, duration_ms },
                );
            }
            return err;
        };
    }

    if (tcfg.TEST_CASE_VERBOSE) {
        std.debug.print(
            "PASS {s} ({d:.2} ms, {d} frames)\n",
            .{ case_dir_name, duration_ms, frames_num },
        );
    }
}

pub fn run(allocator: std.mem.Allocator, io: std.Io) !void {
    const config = tcfg.getRasterConfig(.testing);
    const gold_dir_root = policy.goldRoot(.basic);
    const data_dir_root = "data/edge";

    const midside_mesh_types = [_]gk.MeshType{ .tri6, .quad8, .quad9 };
    const full_mesh_types = [_]gk.MeshType{
        .tri3,
        .tri6,
        .quad4,
        .quad8,
        .quad9,
    };

    const cases = [_]struct {
        name: []const u8,
        mesh_types: []const gk.MeshType,
    }{
        .{ .name = "distort_rot", .mesh_types = &full_mesh_types },
        .{ .name = "distort_shear", .mesh_types = &full_mesh_types },
        .{ .name = "distort_stretch", .mesh_types = &full_mesh_types },
        .{ .name = "distort_bulge", .mesh_types = &midside_mesh_types },
        .{ .name = "distort_tan", .mesh_types = &midside_mesh_types },
        .{ .name = "vertbulge", .mesh_types = &midside_mesh_types },
        .{ .name = "bulgein_rot", .mesh_types = &midside_mesh_types },
        .{ .name = "bulgeout_rot", .mesh_types = &midside_mesh_types },
    };

    for (cases) |case| {
        for (case.mesh_types) |mesh_type| {
            try runOneElemCaseTest(
                allocator,
                io,
                case.name,
                mesh_type,
                gold_dir_root,
                data_dir_root,
                config,
            );
        }
    }
}
