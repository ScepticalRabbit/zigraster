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
const fullcase_tex = @import("../tests/fullcase_texture.zig");
const fullfixtures = @import("../dev_support/fullfixtures.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
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

pub const FullTexSamplingCase = fullcase_tex.FullTexSamplingCase;
pub const all_tex_samp_configs = fullcase_tex.all_tex_samp_configs;

pub fn generateFullTexCase(
    allocator: std.mem.Allocator,
    io: std.Io,
    mesh_type: gk.MeshType,
    prep: *const fullfixtures.Scene0Prepared,
    textures: *const fullfixtures.FullTextures,
    case: FullTexSamplingCase,
    gold_dir_root: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    const case_name = try case.formatDirName(allocator, mesh_type);
    defer allocator.free(case_name);

    const out_dir_path = try std.fmt.allocPrint(
        allocator,
        "{s}/{s}",
        .{ gold_dir_root, case_name },
    );
    defer allocator.free(out_dir_path);

    var out_dir = try orch.openDirEnsured(io, out_dir_path);
    out_dir.close(io);

    const meshes = fullcase_tex.buildTexSamplingCaseMeshes(
        mesh_type,
        prep,
        textures,
        case,
    );

    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), config.total_threads) },
    };

    const images = try riley.raster(
        allocator,
        &render_groups,
        &[_]CameraInput{prep.camera_input},
        &meshes,
        config,
        out_dir_path,
    );

    if (images) |img| {
        allocator.free(img.slice);
    }
}

pub fn generateAllFullTextureCases(
    allocator: std.mem.Allocator,
    io: std.Io,
    gold_dir_root: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    var textures = try fullfixtures.FullTextures.init(allocator, io);
    defer textures.deinit(allocator);

    const mesh_types = [_]gk.MeshType{
        .tri3,
        .tri6,
        .quad4,
        .quad8,
        .quad9,
    };
    const bool_values = [_]bool{ false, true };
    const dtypes = [_]@typeInfo(FullTexSamplingCase).@"struct".fields[1].type{
        .u8,
        .u16,
        .f64,
    };

    for (mesh_types) |mesh_type| {
        var prep = try fullfixtures.prepareScene0(allocator, io, mesh_type);
        defer prep.deinit(allocator);

        for (bool_values) |is_rgb| {
            for (dtypes) |dtype| {
                for (all_tex_samp_configs) |samp_cfg| {
                    try generateFullTexCase(
                        allocator,
                        io,
                        mesh_type,
                        &prep,
                        &textures,
                        .{
                            .is_rgb = is_rgb,
                            .dtype = dtype,
                            .samp_cfg = samp_cfg,
                        },
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

    const gold_dir_root = policy.goldRoot(.full_texture);
    std.debug.print(
        "Generating Full Suite: texture cases in {s}...\n",
        .{gold_dir_root},
    );
    try generateAllFullTextureCases(
        allocator,
        io,
        gold_dir_root,
        config,
    );
    std.debug.print("Done full texture cases.\n", .{});
}
