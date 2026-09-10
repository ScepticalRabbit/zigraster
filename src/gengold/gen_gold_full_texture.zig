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
const common = @import("gen_gold_full_common.zig");
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
const texops = @import("../riley/zig/textureops.zig");

const F = buildconfig.F;
const CameraInput = camera.CameraInput;
const MeshInput = mo.MeshInput;

pub const FullTexSamplingCase = struct {
    is_rgb: bool,
    dtype: enum { u8, u16, f64 },
    samp_cfg: texops.TexSampConfig,

    pub fn formatDirName(
        self: FullTexSamplingCase,
        allocator: std.mem.Allocator,
        elem: gk.MeshType,
    ) ![]const u8 {
        const elem_str = switch (elem) {
            .tri3opt => "tri3",
            else => @tagName(elem),
        };
        const colour_str = if (self.is_rgb) "rgb" else "mono";
        const dtype_str = @tagName(self.dtype);
        const sampler_str = @tagName(self.samp_cfg.sample);
        const mode_str = @tagName(self.samp_cfg.mode);

        return std.fmt.allocPrint(
            allocator,
            "scene0_{s}_tex_{s}_{s}_{s}_{s}",
            .{ elem_str, colour_str, dtype_str, sampler_str, mode_str },
        );
    }
};

pub const all_tex_samp_configs = [_]texops.TexSampConfig{
    .{ .sample = .nearest, .mode = .direct },
    .{ .sample = .linear, .mode = .direct },
    .{ .sample = .cubic_catmull_rom, .mode = .direct },
    .{ .sample = .cubic_catmull_rom, .mode = .lut },
    .{ .sample = .cubic_catmull_rom, .mode = .lut_lerp },
    .{ .sample = .cubic_mitchell_netravali, .mode = .direct },
    .{ .sample = .cubic_mitchell_netravali, .mode = .lut },
    .{ .sample = .cubic_mitchell_netravali, .mode = .lut_lerp },
    .{ .sample = .lanczos3, .mode = .direct },
    .{ .sample = .lanczos3, .mode = .lut },
    .{ .sample = .lanczos3, .mode = .lut_lerp },
    .{ .sample = .cubic_bspline, .mode = .direct },
    .{ .sample = .cubic_bspline, .mode = .lut },
    .{ .sample = .cubic_bspline, .mode = .lut_lerp },
    .{ .sample = .quintic_bspline, .mode = .direct },
    .{ .sample = .quintic_bspline, .mode = .lut },
    .{ .sample = .quintic_bspline, .mode = .lut_lerp },
    .{ .sample = .lanczos2, .mode = .direct },
    .{ .sample = .lanczos2, .mode = .lut },
    .{ .sample = .lanczos2, .mode = .lut_lerp },
};

pub fn buildTexSamplingCaseMeshes(
    mesh_type: gk.MeshType,
    prep: *const common.Scene0Prepared,
    textures: *const common.FullTextures,
    case: FullTexSamplingCase,
) [2]MeshInput {
    const sphere_shader: shaderops.ShaderInput = if (case.is_rgb)
        switch (case.dtype) {
            .u8 => .{
                .tex_rgb_u8 = .{
                    .uvs = prep.sphere_uvs.array,
                    .tex = textures.tex_u8_rgb,
                    .samp_cfg = case.samp_cfg,
                    .normal_type = .none,
                },
            },
            .u16 => .{
                .tex_rgb_u16 = .{
                    .uvs = prep.sphere_uvs.array,
                    .tex = textures.tex_u16_rgb,
                    .samp_cfg = case.samp_cfg,
                    .normal_type = .none,
                },
            },
            .f64 => .{
                .tex_rgb_f = .{
                    .uvs = prep.sphere_uvs.array,
                    .tex = textures.tex_f64_rgb,
                    .samp_cfg = case.samp_cfg,
                    .normal_type = .none,
                },
            },
        }
    else
        switch (case.dtype) {
            .u8 => .{
                .tex_u8 = .{
                    .uvs = prep.sphere_uvs.array,
                    .tex = textures.tex_u8_mono,
                    .samp_cfg = case.samp_cfg,
                    .normal_type = .none,
                },
            },
            .u16 => .{
                .tex_u16 = .{
                    .uvs = prep.sphere_uvs.array,
                    .tex = textures.tex_u16_mono,
                    .samp_cfg = case.samp_cfg,
                    .normal_type = .none,
                },
            },
            .f64 => .{
                .tex_f = .{
                    .uvs = prep.sphere_uvs.array,
                    .tex = textures.tex_f64_mono,
                    .samp_cfg = case.samp_cfg,
                    .normal_type = .none,
                },
            },
        };

    const cyl_shader: shaderops.ShaderInput = if (case.is_rgb)
        switch (case.dtype) {
            .u8 => .{
                .tex_rgb_u8 = .{
                    .uvs = prep.cylinder_uvs.array,
                    .tex = textures.tex_u8_rgb,
                    .samp_cfg = case.samp_cfg,
                    .normal_type = .none,
                },
            },
            .u16 => .{
                .tex_rgb_u16 = .{
                    .uvs = prep.cylinder_uvs.array,
                    .tex = textures.tex_u16_rgb,
                    .samp_cfg = case.samp_cfg,
                    .normal_type = .none,
                },
            },
            .f64 => .{
                .tex_rgb_f = .{
                    .uvs = prep.cylinder_uvs.array,
                    .tex = textures.tex_f64_rgb,
                    .samp_cfg = case.samp_cfg,
                    .normal_type = .none,
                },
            },
        }
    else
        switch (case.dtype) {
            .u8 => .{
                .tex_u8 = .{
                    .uvs = prep.cylinder_uvs.array,
                    .tex = textures.tex_u8_mono,
                    .samp_cfg = case.samp_cfg,
                    .normal_type = .none,
                },
            },
            .u16 => .{
                .tex_u16 = .{
                    .uvs = prep.cylinder_uvs.array,
                    .tex = textures.tex_u16_mono,
                    .samp_cfg = case.samp_cfg,
                    .normal_type = .none,
                },
            },
            .f64 => .{
                .tex_f = .{
                    .uvs = prep.cylinder_uvs.array,
                    .tex = textures.tex_f64_mono,
                    .samp_cfg = case.samp_cfg,
                    .normal_type = .none,
                },
            },
        };

    return [_]MeshInput{
        .{
            .mesh_type = mesh_type,
            .coords = prep.sphere_coords,
            .connect = prep.sphere_connect,
            .disp = prep.sphere_disp,
            .shader = sphere_shader,
        },
        .{
            .mesh_type = mesh_type,
            .coords = prep.cylinder_coords,
            .connect = prep.cylinder_connect,
            .disp = prep.cylinder_disp,
            .shader = cyl_shader,
        },
    };
}

pub fn generateFullTexCase(
    allocator: std.mem.Allocator,
    io: std.Io,
    mesh_type: gk.MeshType,
    prep: *const common.Scene0Prepared,
    textures: *const common.FullTextures,
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

    const meshes = buildTexSamplingCaseMeshes(mesh_type, prep, textures, case);

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
    var textures = try common.FullTextures.init(allocator, io);
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
        var prep = try common.prepareScene0(allocator, io, mesh_type);
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
    std.debug.print("Generating Full Suite: texture cases in {s}...\n", .{gold_dir_root});
    try generateAllFullTextureCases(
        allocator,
        io,
        gold_dir_root,
        config,
    );
    std.debug.print("Done full texture cases.\n", .{});
}
