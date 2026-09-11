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
const cameraops = @import("../riley/zig/cameraops.zig");
const common = @import("../dev_support/tests.zig");
const common_full = @import("../dev_support/fullfixtures.zig");
const fullcase_hull = @import("fullcase_hull.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
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

const HullStatusCase = fullcase_hull.HullStatusCase;
const HullPsfCase = fullcase_hull.HullPsfCase;
const NewtonSeedCase = fullcase_hull.NewtonSeedCase;

fn runOneElemHullCaseTest(
    allocator: std.mem.Allocator,
    io: std.Io,
    case_name: []const u8,
    mesh_type: gk.MeshType,
    is_offscreen: bool,
    hull_case: HullStatusCase,
    psf_case: HullPsfCase,
    seed_case: NewtonSeedCase,
    gold_dir_root: []const u8,
    data_dir_root: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const base_case_name = if (is_offscreen)
        (if (std.mem.startsWith(u8, case_name, "vertbulge"))
            "vertbulge"
        else
            "distort_bulge")
    else
        case_name;

    const prepared = try orch.prepareSingleMeshCase(
        aa,
        io,
        base_case_name,
        mesh_type,
        fullcase_hull.pixel_num_hull,
        1.15,
        data_dir_root,
    );

    var cam_inp = CameraInput{
        .pixels_num = prepared.camera.pixels_num,
        .pixels_size = prepared.camera.pixels_size,
        .pos_world = prepared.camera.pos_world,
        .rot_world = prepared.camera.rot_world,
        .roi_cent_world = prepared.camera.roi_cent_world,
        .focal_length = prepared.camera.focal_length,
        .sub_sample = 2,
        .distortion = .none,
        .psf = psf_case.psf,
    };

    if (is_offscreen) {
        const metrics = cameraops.calcPlaneMetrics(cam_inp);
        cam_inp.pos_world.set(0, cam_inp.pos_world.get(0) + 0.5 * metrics.roi_plane_size[0]);
        cam_inp.roi_cent_world.set(
            0,
            cam_inp.roi_cent_world.get(0) + 0.5 * metrics.roi_plane_size[0],
        );
    }

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

    const case_dir_name = try fullcase_hull.formatCaseDirName(
        aa,
        case_name,
        mesh_type,
        hull_case,
        psf_case,
        seed_case,
    );
    const gold_dir = try std.fmt.allocPrint(
        aa,
        "{s}/{s}",
        .{ gold_dir_root, case_dir_name },
    );

    var run_config = config;
    run_config.save_strategy = .memory;
    run_config.hull_mode = hull_case.mode;
    run_config.newton_seed_mode = seed_case.seed_mode;
    run_config.newton_seed_reuse = seed_case.seed_reuse;

    const start_time = Timestamp.now(io, .awake);
    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), run_config.total_threads) },
    };

    const result = try riley.raster(
        aa,
        &render_groups,
        &[_]CameraInput{cam_inp},
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
    ) / 1.0e6;

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
            tcfg.FULL_GOLD_REL_TOL,
            tcfg.FULL_GOLD_ABS_TOL,
        ) catch |err| {
            const fail_dir_name = try std.fmt.allocPrint(
                aa,
                "full_hull/{s}",
                .{case_dir_name},
            );
            try common.saveComparisonArtifactsFromResult(
                aa,
                io,
                common.default_fails_root,
                fail_dir_name,
                &render_result,
                0,
                ff,
                0,
                gold_path,
                1,
            );
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

fn runScene2HullCaseTest(
    allocator: std.mem.Allocator,
    io: std.Io,
    mesh_type: gk.MeshType,
    prep: *const common_full.Scene2Prepared,
    textures: *const common_full.FullTextures,
    hull_case: HullStatusCase,
    psf_case: HullPsfCase,
    seed_case: NewtonSeedCase,
    gold_dir_root: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    var cam_inp = common_full.createScene2Camera(fullcase_hull.pixel_num_hull, 2);
    cam_inp.psf = psf_case.psf;

    const meshes = common_full.buildScene2Meshes(prep, textures);

    const case_dir_name = try fullcase_hull.formatCaseDirName(
        aa,
        "scene2",
        mesh_type,
        hull_case,
        psf_case,
        seed_case,
    );
    const gold_dir = try std.fmt.allocPrint(
        aa,
        "{s}/{s}",
        .{ gold_dir_root, case_dir_name },
    );

    var run_config = config;
    run_config.save_strategy = .memory;
    run_config.hull_mode = hull_case.mode;
    run_config.newton_seed_mode = seed_case.seed_mode;
    run_config.newton_seed_reuse = seed_case.seed_reuse;

    const start_time = Timestamp.now(io, .awake);
    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), run_config.total_threads) },
    };

    const result = try riley.raster(
        aa,
        &render_groups,
        &[_]CameraInput{cam_inp},
        &meshes,
        run_config,
        null,
    );

    var render_result = result orelse return error.NoResult;
    defer aa.free(render_result.slice);

    const end_time = Timestamp.now(io, .awake);
    const duration_ms = @as(
        F,
        @floatFromInt(start_time.durationTo(end_time).raw.nanoseconds),
    ) / 1.0e6;

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
            tcfg.FULL_GOLD_REL_TOL,
            tcfg.FULL_GOLD_ABS_TOL,
        ) catch |err| {
            const fail_dir_name = try std.fmt.allocPrint(
                aa,
                "full_hull/{s}",
                .{case_dir_name},
            );
            try common.saveComparisonArtifactsFromResult(
                aa,
                io,
                common.default_fails_root,
                fail_dir_name,
                &render_result,
                0,
                ff,
                0,
                gold_path,
                1,
            );
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
    var config = tcfg.getRasterConfig(.testing);
    config.background_value = 127.5;
    const gold_dir_root = policy.goldRoot(.full_hull);
    const data_dir_root = "data/edge";

    var textures = try common_full.FullTextures.init(allocator, io);
    defer textures.deinit(allocator);

    const newton_mesh_types = [_]gk.MeshType{
        .tri6,
        .quad4,
        .quad8,
        .quad9,
    };
    const midside_mesh_types = [_]gk.MeshType{
        .tri6,
        .quad8,
        .quad9,
    };

    const oneelem_cases = [_]struct {
        name: []const u8,
        mesh_types: []const gk.MeshType,
        is_offscreen: bool,
    }{
        .{ .name = "distort_rot", .mesh_types = &newton_mesh_types, .is_offscreen = false },
        .{ .name = "distort_shear", .mesh_types = &newton_mesh_types, .is_offscreen = false },
        .{
            .name = "distort_stretch",
            .mesh_types = &newton_mesh_types,
            .is_offscreen = false,
        },
        .{ .name = "distort_bulge", .mesh_types = &midside_mesh_types, .is_offscreen = false },
        .{ .name = "distort_tan", .mesh_types = &midside_mesh_types, .is_offscreen = false },
        .{ .name = "vertbulge", .mesh_types = &midside_mesh_types, .is_offscreen = false },
        .{ .name = "bulgein_rot", .mesh_types = &midside_mesh_types, .is_offscreen = false },
        .{ .name = "bulgeout_rot", .mesh_types = &midside_mesh_types, .is_offscreen = false },
        .{
            .name = "vertbulge_offscreen",
            .mesh_types = &midside_mesh_types,
            .is_offscreen = true,
        },
        .{
            .name = "distort_bulge_offscreen",
            .mesh_types = &midside_mesh_types,
            .is_offscreen = true,
        },
    };

    for (oneelem_cases) |elem_case| {
        for (elem_case.mesh_types) |mesh_type| {
            for (fullcase_hull.hull_status_cases) |hull_case| {
                for (fullcase_hull.hull_psf_cases) |psf_case| {
                    for (fullcase_hull.newton_seed_cases) |seed_case| {
                        try runOneElemHullCaseTest(
                            allocator,
                            io,
                            elem_case.name,
                            mesh_type,
                            elem_case.is_offscreen,
                            hull_case,
                            psf_case,
                            seed_case,
                            gold_dir_root,
                            data_dir_root,
                            config,
                        );
                    }
                }
            }
        }
    }

    for (newton_mesh_types) |mesh_type| {
        var prep2 = try common_full.prepareScene2(allocator, io, mesh_type);
        defer prep2.deinit(allocator);

        for (fullcase_hull.hull_status_cases) |hull_case| {
            for (fullcase_hull.hull_psf_cases) |psf_case| {
                for (fullcase_hull.newton_seed_cases) |seed_case| {
                    try runScene2HullCaseTest(
                        allocator,
                        io,
                        mesh_type,
                        &prep2,
                        &textures,
                        hull_case,
                        psf_case,
                        seed_case,
                        gold_dir_root,
                        config,
                    );
                }
            }
        }
    }

    try runNewtonSeedMatrixTests(allocator, io, &textures, config);
}

fn runNewtonSeedMatrixTests(
    allocator: std.mem.Allocator,
    io: std.Io,
    textures: *const common_full.FullTextures,
    config: rastcfg.RasterConfig,
) !void {
    const curved_mesh_types = [_]gk.MeshType{
        .tri6,
        .quad8,
        .quad9,
    };
    const seed_modes = [_]rastcfg.NewtonSeedMode{
        .centroid,
        .hull,
    };
    const seed_reuses = [_]rastcfg.NewtonSeedReuse{
        .off,
        .last_conv,
    };

    for (curved_mesh_types) |mesh_type| {
        var prep2 = try common_full.prepareScene2(allocator, io, mesh_type);
        defer prep2.deinit(allocator);

        const meshes = common_full.buildScene2Meshes(&prep2, textures);
        const cam_inp = common_full.createScene2Camera(
            fullcase_hull.pixel_num_hull,
            2,
        );

        var baseline_slice_opt: ?[]F = null;
        var baseline_dims_opt: ?[]usize = null;
        var baseline_arena = std.heap.ArenaAllocator.init(allocator);
        defer baseline_arena.deinit();

        for (seed_modes) |seed_mode| {
            for (seed_reuses) |seed_reuse| {
                var arena = std.heap.ArenaAllocator.init(allocator);
                defer arena.deinit();
                const aa = arena.allocator();

                var run_config = config;
                run_config.save_strategy = .memory;
                run_config.hull_mode = .on_no_fallback;
                run_config.newton_seed_mode = seed_mode;
                run_config.newton_seed_reuse = seed_reuse;

                const render_groups = [_]riley.RenderGroupSpec{
                    .{ .io = io, .workers = 1 },
                };

                const result = try riley.raster(
                    aa,
                    &render_groups,
                    &[_]CameraInput{cam_inp},
                    &meshes,
                    run_config,
                    null,
                );

                const current_img = result orelse return error.NoResult;

                for (current_img.slice) |pixel_val| {
                    try std.testing.expect(std.math.isFinite(pixel_val));
                }

                if (baseline_slice_opt == null) {
                    baseline_slice_opt = try baseline_arena.allocator().dupe(
                        F,
                        current_img.slice,
                    );
                    baseline_dims_opt = try baseline_arena.allocator().dupe(
                        usize,
                        current_img.dims,
                    );
                } else {
                    try std.testing.expectEqualSlices(
                        usize,
                        baseline_dims_opt.?,
                        current_img.dims,
                    );
                    var diff_sum: F = 0.0;
                    for (baseline_slice_opt.?, current_img.slice) |base_val, curr_val| {
                        diff_sum += @abs(base_val - curr_val);
                    }
                    const mean_diff = diff_sum / @as(
                        F,
                        @floatFromInt(current_img.slice.len),
                    );
                    try std.testing.expect(mean_diff <= 0.05);
                }
            }
        }
    }
}
