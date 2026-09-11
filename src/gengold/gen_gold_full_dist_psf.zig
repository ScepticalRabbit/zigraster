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
const fullcase_dist_psf = @import("../tests/fullcase_dist_psf.zig");
const fullfixtures = @import("../dev_support/fullfixtures.zig");
const iio = @import("../riley/zig/imageio.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const orch = @import("../dev_support/orchestration.zig");
const policy = @import("../dev_support/testpolicy.zig");
const rastcfg = @import("../riley/zig/rasterconfig.zig");
const riley = @import("../riley/zig/riley.zig");
const tcfg = @import("../dev_support/testconfig.zig");

const F = buildconfig.F;
const CameraInput = camera.CameraInput;
const MeshInput = mo.MeshInput;

pub const DistCase = fullcase_dist_psf.DistCase;
pub const PsfCase = fullcase_dist_psf.PsfCase;
pub const dist_cases = fullcase_dist_psf.dist_cases;
pub const psf_cases = fullcase_dist_psf.psf_cases;
pub const ssaa_levels = fullcase_dist_psf.ssaa_levels;

pub fn generateDistPsfCase(
    allocator: std.mem.Allocator,
    io: std.Io,
    prep: *const fullfixtures.Scene1Prepared,
    ssaa: u32,
    dist_case: DistCase,
    psf_case: PsfCase,
    gold_dir_root: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    const case_name = try fullcase_dist_psf.formatDistPsfCaseName(
        allocator,
        ssaa,
        dist_case.tag,
        psf_case.tag,
    );
    defer allocator.free(case_name);

    const out_dir_path = try std.fmt.allocPrint(
        allocator,
        "{s}/{s}",
        .{ gold_dir_root, case_name },
    );
    defer allocator.free(out_dir_path);

    var out_dir = try orch.openDirEnsured(io, out_dir_path);
    out_dir.close(io);

    const mesh = fullcase_dist_psf.buildScene1Mesh(prep);
    const meshes = [_]MeshInput{mesh};

    var camera_input = prep.camera_input;
    camera_input.sub_sample = ssaa;
    camera_input.distortion = dist_case.distortion;
    camera_input.psf = psf_case.psf;

    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), config.total_threads) },
    };

    var case_config = config;
    case_config.background_value = fullfixtures.grey_background_scene1;

    const images = try riley.raster(
        allocator,
        &render_groups,
        &[_]CameraInput{camera_input},
        &meshes,
        case_config,
        out_dir_path,
    );

    if (images) |img| {
        allocator.free(img.slice);
    }
}

pub fn generateAllFullDistPsfCases(
    allocator: std.mem.Allocator,
    io: std.Io,
    gold_dir_root: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    var prep = try fullfixtures.prepareScene1(allocator, io);
    defer prep.deinit(allocator);

    for (ssaa_levels) |ssaa| {
        for (dist_cases) |dist_case| {
            for (psf_cases) |psf_case| {
                try generateDistPsfCase(
                    allocator,
                    io,
                    &prep,
                    ssaa,
                    dist_case,
                    psf_case,
                    gold_dir_root,
                    config,
                );
            }
        }
    }
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const allocator = arena.allocator();

    var config = tcfg.getRasterConfig(.gold_gen);
    config.save_strategy = .disk;
    config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .fimg, .bits = null, .scaling = .none },
        .{ .format = .bmp, .bits = 8, .scaling = .auto },
    };
    config.background_value = fullfixtures.grey_background_scene1;

    const gold_dir_root = policy.goldRoot(.full_dist_psf);
    try generateAllFullDistPsfCases(allocator, io, gold_dir_root, config);
}
