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
const mo = @import("../riley/zig/meshpipeline.zig");

const F = buildconfig.F;
const CameraInput = camera.CameraInput;
const MeshInput = mo.MeshInput;
const SubPixelCenterMap = camera.SubPixelCenterMap;

pub const DistCase = struct {
    tag: []const u8,
    distortion: camera.DistortionModel,
};

pub const PsfCase = struct {
    tag: []const u8,
    psf: camera.PointSpreadFunc,
};

pub const PxMapCase = struct {
    tag: []const u8,
    map_mode: SubPixelCenterMap,
};

pub const dist_cases = [_]DistCase{
    .{
        .tag = "dist_bc",
        .distortion = .{ .brown_conrady = .{ .k1 = -1500.0, .k2 = 5.0e6 } },
    },
    .{
        .tag = "dist_bce",
        .distortion = .{
            .brown_conrady_ext = .{
                .k1 = 1500.0,
                .k2 = -5.0e6,
                .k4 = 200.0,
            },
        },
    },
    .{
        .tag = "dist_poly_bc",
        .distortion = .{
            .brown_conrady_polynomial = .{
                .brown_conrady = .{ .k1 = -1000.0 },
                .polynomial = .{
                    .forward_map = common_full.getRepresentativePolynomialMap(),
                },
            },
        },
    },
};

pub const psf_cases = [_]PsfCase{
    .{
        .tag = "psf_box",
        .psf = .{ .pixel_box = .{} },
    },
    .{
        .tag = "psf_gauss_halo",
        .psf = .{
            .gaussian = .{
                .sigma_px = 1.5,
                .supp_rad_px = 4.5,
                .separable = .yes,
            },
        },
    },
};

pub const pxmap_cases = [_]PxMapCase{
    .{ .tag = "full_in_mem", .map_mode = .full_in_mem },
    .{ .tag = "per_tile", .map_mode = .per_tile },
    .{ .tag = "affine_jac", .map_mode = .affine_jac },
};

pub const ssaa_levels = [_]u32{ 1, 2, 3, 4 };

pub fn formatSsaaPxmapCaseName(
    allocator: std.mem.Allocator,
    ssaa: u32,
    dist_tag: []const u8,
    psf_tag: []const u8,
    pxmap_tag: []const u8,
) ![]const u8 {
    return std.fmt.allocPrint(
        allocator,
        "cube_tri3_ssaa{d}_{s}_{s}_{s}",
        .{ ssaa, dist_tag, psf_tag, pxmap_tag },
    );
}

pub fn buildScene1Mesh(prep: *const common_full.Scene1Prepared) MeshInput {
    return .{
        .mesh_type = .tri3,
        .coords = prep.coords,
        .connect = prep.connect,
        .disp = null,
        .shader = .{
            .func = .{
                .uvs = prep.uvs.array,
                .builtin = .checker,
                .params = .{
                    .coord_scale = .{ 24.0, 24.0 },
                    .coord_offset = .{ 0.0, 0.0 },
                    .settings = .{ .checker = .{} },
                },
                .coord_mode = .uv,
                .bits = 8,
                .scaling = .auto,
                .normal_type = .none,
            },
        },
    };
}
