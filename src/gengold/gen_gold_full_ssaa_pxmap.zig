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
const fullcase_ssaa_pxmap = @import("../tests/fullcase_ssaa_pxmap.zig");
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

pub const DistCase = fullcase_ssaa_pxmap.DistCase;
pub const PsfCase = fullcase_ssaa_pxmap.PsfCase;
pub const PxMapCase = fullcase_ssaa_pxmap.PxMapCase;
pub const dist_cases = fullcase_ssaa_pxmap.dist_cases;
pub const psf_cases = fullcase_ssaa_pxmap.psf_cases;
pub const pxmap_cases = fullcase_ssaa_pxmap.pxmap_cases;
pub const ssaa_levels = fullcase_ssaa_pxmap.ssaa_levels;

pub fn generateSsaaPxmapCase(
    allocator: std.mem.Allocator,
    io: std.Io,
    prep: *const fullfixtures.Scene1Prepared,
    ssaa: u32,
    dist_case: DistCase,
    psf_case: PsfCase,
    pxmap_case: PxMapCase,
    gold_dir_root: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    const case_name = try fullcase_ssaa_pxmap.formatSsaaPxmapCaseName(
        allocator,
        ssaa,
        dist_case.tag,
        psf_case.tag,
        pxmap_case.tag,
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

    const mesh = fullcase_ssaa_pxmap.buildScene1Mesh(prep);
    const meshes = [_]MeshInput{mesh};

    var camera_input = prep.camera_input;
    camera_input.sub_sample = ssaa;
    camera_input.distortion = dist_case.distortion;
    camera_input.psf = psf_case.psf;
    camera_input.subpixel_center_map = pxmap_case.map_mode;

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

pub fn generateAllFullSsaaPxmapCases(
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
                for (pxmap_cases) |pxmap_case| {
                    try generateSsaaPxmapCase(
                        allocator,
                        io,
                        &prep,
                        ssaa,
                        dist_case,
                        psf_case,
                        pxmap_case,
                        gold_dir_root,
                        config,
                    );
                }
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

    const gold_dir_root = policy.goldRoot(.full_ssaa_pxmap);
    try generateAllFullSsaaPxmapCases(allocator, io, gold_dir_root, config);
}
