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
const common = @import("gen_gold_full_common.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const iio = @import("../riley/zig/imageio.zig");
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

pub const HullStatusCase = struct {
    tag: []const u8,
    mode: rastcfg.HullMode,
};

pub const HullPsfCase = struct {
    tag: []const u8,
    psf: camera.PointSpreadFunc,
};

pub const hull_status_cases = [_]HullStatusCase{
    .{ .tag = "hull_off", .mode = .off },
    .{ .tag = "hull_onnofallback", .mode = .on_no_fallback },
    .{ .tag = "hull_onfallback", .mode = .on_convex_fallback },
};

pub const hull_psf_cases = [_]HullPsfCase{
    .{ .tag = "psf_none", .psf = .{ .pixel_box = .{} } },
    .{
        .tag = "psf_gauss",
        .psf = .{
            .gaussian = .{
                .sigma_px = 1.5,
                .supp_rad_px = 4.5,
                .separable = .yes,
            },
        },
    },
};

pub const pixel_num_hull = [_]u32{ 128, 128 };

fn formatCaseDirName(
    allocator: std.mem.Allocator,
    case_name: []const u8,
    mesh_type: gk.MeshType,
    hull_case: HullStatusCase,
    psf_case: HullPsfCase,
) ![]const u8 {
    return std.fmt.allocPrint(
        allocator,
        "{s}_{s}_{s}_{s}",
        .{ case_name, @tagName(mesh_type), hull_case.tag, psf_case.tag },
    );
}

fn runOneElemHullCase(
    allocator: std.mem.Allocator,
    io: std.Io,
    case_name: []const u8,
    mesh_type: gk.MeshType,
    is_offscreen: bool,
    hull_case: HullStatusCase,
    psf_case: HullPsfCase,
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
        pixel_num_hull,
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
                .params = .{
                    .coord_scale = .{ 24.0, 24.0 },
                    .coord_offset = .{ 0.0, 0.0 },
                },
                .normal_type = .none,
            },
        },
    };

    const case_dir_name = try formatCaseDirName(
        aa,
        case_name,
        mesh_type,
        hull_case,
        psf_case,
    );
    const gold_dir = try std.fmt.allocPrint(
        aa,
        "{s}/{s}",
        .{ gold_dir_root, case_dir_name },
    );

    var out_dir = try orch.openDirEnsured(io, gold_dir);
    out_dir.close(io);

    var run_config = config;
    run_config.save_strategy = .disk;
    run_config.hull_mode = hull_case.mode;

    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), run_config.total_threads) },
    };

    _ = try riley.raster(
        aa,
        &render_groups,
        &[_]CameraInput{cam_inp},
        &[_]MeshInput{mesh_input},
        run_config,
        gold_dir,
    );
}

fn runScene2HullCase(
    allocator: std.mem.Allocator,
    io: std.Io,
    mesh_type: gk.MeshType,
    prep: *const common.Scene2Prepared,
    textures: *const common.FullTextures,
    hull_case: HullStatusCase,
    psf_case: HullPsfCase,
    gold_dir_root: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    var cam_inp = common.createScene2Camera(pixel_num_hull, 2);
    cam_inp.psf = psf_case.psf;

    const meshes = common.buildScene2Meshes(prep, textures);

    const case_dir_name = try formatCaseDirName(
        aa,
        "scene2",
        mesh_type,
        hull_case,
        psf_case,
    );
    const gold_dir = try std.fmt.allocPrint(
        aa,
        "{s}/{s}",
        .{ gold_dir_root, case_dir_name },
    );

    var out_dir = try orch.openDirEnsured(io, gold_dir);
    out_dir.close(io);

    var run_config = config;
    run_config.save_strategy = .disk;
    run_config.hull_mode = hull_case.mode;

    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), run_config.total_threads) },
    };

    _ = try riley.raster(
        aa,
        &render_groups,
        &[_]CameraInput{cam_inp},
        &meshes,
        run_config,
        gold_dir,
    );
}

pub fn generate(allocator: std.mem.Allocator, io: std.Io) !void {
    var config = tcfg.getRasterConfig(.gold_gen);
    config.save_strategy = .disk;
    config.image_save_mode = .grey;
    config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .fimg, .bits = null, .scaling = .none },
        .{ .format = .bmp, .bits = 8, .scaling = .none },
    };

    const gold_dir_root = policy.goldRoot(.full_hull);
    const data_dir_root = "data/edge";

    var textures = try common.FullTextures.init(allocator, io);
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

    // 1. Single-element cases
    for (oneelem_cases) |elem_case| {
        for (elem_case.mesh_types) |mesh_type| {
            for (hull_status_cases) |hull_case| {
                for (hull_psf_cases) |psf_case| {
                    try runOneElemHullCase(
                        allocator,
                        io,
                        elem_case.name,
                        mesh_type,
                        elem_case.is_offscreen,
                        hull_case,
                        psf_case,
                        gold_dir_root,
                        data_dir_root,
                        config,
                    );
                }
            }
        }
    }

    // 2. Scene 2 cases
    for (newton_mesh_types) |mesh_type| {
        var prep2 = try common.prepareScene2(allocator, io, mesh_type);
        defer prep2.deinit(allocator);

        for (hull_status_cases) |hull_case| {
            for (hull_psf_cases) |psf_case| {
                try runScene2HullCase(
                    allocator,
                    io,
                    mesh_type,
                    &prep2,
                    &textures,
                    hull_case,
                    psf_case,
                    gold_dir_root,
                    config,
                );
            }
        }
    }
}
