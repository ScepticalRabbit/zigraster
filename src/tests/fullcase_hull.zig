// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const camera = @import("../riley/zig/camera.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const rastcfg = @import("../riley/zig/rasterconfig.zig");

pub const HullStatusCase = struct {
    tag: []const u8,
    mode: rastcfg.HullMode,
};

pub const HullPsfCase = struct {
    tag: []const u8,
    psf: camera.PointSpreadFunc,
};

pub const NewtonSeedCase = struct {
    tag: []const u8,
    seed_mode: rastcfg.NewtonSeedMode,
    seed_reuse: rastcfg.NewtonSeedReuse,
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

pub const newton_seed_cases = [_]NewtonSeedCase{
    .{ .tag = "seed_centroid", .seed_mode = .centroid, .seed_reuse = .off },
    .{ .tag = "seed_hull", .seed_mode = .hull, .seed_reuse = .off },
    .{ .tag = "seed_centroid_reuse", .seed_mode = .centroid, .seed_reuse = .last_conv },
    .{ .tag = "seed_hull_reuse", .seed_mode = .hull, .seed_reuse = .last_conv },
};

pub const pixel_num_hull = [_]u32{ 128, 128 };

pub fn formatCaseDirName(
    allocator: std.mem.Allocator,
    case_name: []const u8,
    mesh_type: gk.MeshType,
    hull_case: HullStatusCase,
    psf_case: HullPsfCase,
    seed_case: NewtonSeedCase,
) ![]const u8 {
    return std.fmt.allocPrint(
        allocator,
        "{s}_{s}_{s}_{s}_{s}",
        .{
            case_name,
            @tagName(mesh_type),
            hull_case.tag,
            psf_case.tag,
            seed_case.tag,
        },
    );
}
