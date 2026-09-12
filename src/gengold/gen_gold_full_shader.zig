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
const fullcase_shader = @import("../tests/fullcase_shader.zig");
const fullfixtures = @import("../dev_support/fullfixtures.zig");
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

pub const FullNodalCase = fullcase_shader.FullNodalCase;
pub const FullTexCase = fullcase_shader.FullTexCase;
pub const FullFuncCase = fullcase_shader.FullFuncCase;
pub const FullShaderCase = fullcase_shader.FullShaderCase;

pub fn generateFullShaderCase(
    allocator: std.mem.Allocator,
    io: std.Io,
    mesh_type: gk.MeshType,
    prep: *const fullfixtures.Scene0Prepared,
    textures: *const fullfixtures.FullTextures,
    case: FullShaderCase,
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

    const meshes = fullcase_shader.buildShaderCaseMeshes(
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

pub fn generateAllFullShaderCases(
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
    const normal_types = [_]shaderops.NormalType{
        .none,
        .avg,
        .exact,
    };
    const bool_values = [_]bool{ false, true };
    const builtins = [_]shaderops.FuncShaderBuiltin{
        .constant,
        .linear,
        .quadratic,
        .sinusoidal,
        .sinusoidal_approx,
        .checker,
        .checker_smooth,
        .lambertian_normal_z,
        .eggbox,
    };
    const coord_modes = [_]shaderops.FuncCoordMode{
        .para,
        .uv,
        .world_reference,
        .world_deformed,
    };
    const dtypes = [_]@typeInfo(FullTexCase).@"struct".fields[1].type{
        .u8,
        .u16,
        .f64,
    };

    for (mesh_types) |mesh_type| {
        var prep = try fullfixtures.prepareScene0(allocator, io, mesh_type);
        defer prep.deinit(allocator);

        // 1. Nodal cases (18 per mesh)
        for (bool_values) |is_rgb| {
            for (normal_types) |norm| {
                // Scalenone
                try generateFullShaderCase(
                    allocator,
                    io,
                    mesh_type,
                    &prep,
                    &textures,
                    .{
                        .nodal = .{
                            .is_rgb = is_rgb,
                            .scaling = .none,
                            .scale_over = .over_frames,
                            .normal_type = norm,
                        },
                    },
                    gold_dir_root,
                    config,
                );
                // Scalewithin
                try generateFullShaderCase(
                    allocator,
                    io,
                    mesh_type,
                    &prep,
                    &textures,
                    .{
                        .nodal = .{
                            .is_rgb = is_rgb,
                            .scaling = .auto,
                            .scale_over = .within_frames,
                            .normal_type = norm,
                        },
                    },
                    gold_dir_root,
                    config,
                );
                // Scaleover
                try generateFullShaderCase(
                    allocator,
                    io,
                    mesh_type,
                    &prep,
                    &textures,
                    .{
                        .nodal = .{
                            .is_rgb = is_rgb,
                            .scaling = .auto,
                            .scale_over = .over_frames,
                            .normal_type = norm,
                        },
                    },
                    gold_dir_root,
                    config,
                );
            }
        }

        // 2. Texture cases (18 per mesh)
        for (bool_values) |is_rgb| {
            for (dtypes) |dtype| {
                for (normal_types) |norm| {
                    try generateFullShaderCase(
                        allocator,
                        io,
                        mesh_type,
                        &prep,
                        &textures,
                        .{
                            .tex = .{
                                .is_rgb = is_rgb,
                                .dtype = dtype,
                                .normal_type = norm,
                            },
                        },
                        gold_dir_root,
                        config,
                    );
                }
            }
        }

        // 3. Function cases (216 per mesh)
        for (bool_values) |is_rgb| {
            for (builtins) |builtin| {
                for (coord_modes) |coord| {
                    for (normal_types) |norm| {
                        try generateFullShaderCase(
                            allocator,
                            io,
                            mesh_type,
                            &prep,
                            &textures,
                            .{
                                .func = .{
                                    .is_rgb = is_rgb,
                                    .builtin = builtin,
                                    .coord = coord,
                                    .normal_type = norm,
                                },
                            },
                            gold_dir_root,
                            config,
                        );
                    }
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

    const gold_dir_root = policy.goldRoot(.full_shader);
    std.debug.print(
        "Generating Full Suite: shader cases in {s}...\n",
        .{gold_dir_root},
    );
    try generateAllFullShaderCases(
        allocator,
        io,
        gold_dir_root,
        config,
    );
    std.debug.print("Done full shader cases.\n", .{});
}
