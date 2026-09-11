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
const common_full = @import("../dev_support/fullfixtures.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const shaderops = @import("../riley/zig/shaderops_common.zig");
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
    prep: *const common_full.Scene0Prepared,
    textures: *const common_full.FullTextures,
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
    else switch (case.dtype) {
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
    else switch (case.dtype) {
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
