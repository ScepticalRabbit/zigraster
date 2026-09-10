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

pub const DistCase = struct {
    tag: []const u8,
    distortion: camera.DistortionModel,
};

pub const PsfCase = struct {
    tag: []const u8,
    psf: camera.PointSpreadFunc,
};

pub const dist_cases = [_]DistCase{
    .{
        .tag = "dist_none",
        .distortion = .none,
    },
    .{
        .tag = "dist_light_barrel_bc",
        .distortion = .{ .brown_conrady = .{ .k1 = -1000.0 } },
    },
    .{
        .tag = "dist_light_pincushion_bc",
        .distortion = .{ .brown_conrady = .{ .k1 = 1000.0 } },
    },
    .{
        .tag = "dist_extreme_barrel_bc",
        .distortion = .{ .brown_conrady = .{ .k1 = -2500.0, .k2 = 1.0e7 } },
    },
    .{
        .tag = "dist_extreme_pincushion_bc",
        .distortion = .{ .brown_conrady = .{ .k1 = 2500.0, .k2 = 1.0e7 } },
    },
    .{
        .tag = "dist_light_barrel_bce",
        .distortion = .{ .brown_conrady_ext = .{ .k1 = -1000.0 } },
    },
    .{
        .tag = "dist_light_pincushion_bce",
        .distortion = .{ .brown_conrady_ext = .{ .k1 = 1000.0 } },
    },
    .{
        .tag = "dist_extreme_barrel_bce",
        .distortion = .{
            .brown_conrady_ext = .{
                .k1 = -2500.0,
                .k2 = 1.0e7,
                .k4 = 500.0,
            },
        },
    },
    .{
        .tag = "dist_extreme_pincushion_bce",
        .distortion = .{
            .brown_conrady_ext = .{
                .k1 = 2500.0,
                .k2 = 1.0e7,
                .k4 = 500.0,
            },
        },
    },
    .{
        .tag = "dist_poly_bc",
        .distortion = .{
            .brown_conrady_polynomial = .{
                .brown_conrady = .{ .k1 = -1000.0 },
                .polynomial = .{
                    .forward_map = common.getRepresentativePolynomialMap(),
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
        .tag = "psf_gauss_sep",
        .psf = .{
            .gaussian = .{
                .sigma_px = 1.5,
                .supp_rad_px = 4.5,
                .separable = .yes,
            },
        },
    },
    .{
        .tag = "psf_gauss_nonsep",
        .psf = .{
            .gaussian = .{
                .sigma_px = 1.5,
                .supp_rad_px = 4.5,
                .separable = .no,
            },
        },
    },
    .{
        .tag = "psf_aniso_gauss",
        .psf = .{
            .anisotropic_gaussian = .{
                .sigma_x_px = 1.5,
                .sigma_y_px = 0.1,
                .theta_rad = 0.0,
                .supp_rad_px = 4.5,
                .separable = .no,
            },
        },
    },
};

pub const ssaa_levels = [_]u32{ 1, 4 };

pub fn formatDistPsfCaseName(
    allocator: std.mem.Allocator,
    ssaa: u32,
    dist_tag: []const u8,
    psf_tag: []const u8,
) ![]const u8 {
    return std.fmt.allocPrint(
        allocator,
        "cube_tri3_ssaa{d}_{s}_{s}",
        .{ ssaa, dist_tag, psf_tag },
    );
}

pub fn buildScene1Mesh(prep: *const common.Scene1Prepared) MeshInput {
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

pub fn generateDistPsfCase(
    allocator: std.mem.Allocator,
    io: std.Io,
    prep: *const common.Scene1Prepared,
    ssaa: u32,
    dist_case: DistCase,
    psf_case: PsfCase,
    gold_dir_root: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    const case_name = try formatDistPsfCaseName(
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

    const mesh = buildScene1Mesh(prep);
    const meshes = [_]MeshInput{mesh};

    var camera_input = prep.camera_input;
    camera_input.sub_sample = ssaa;
    camera_input.distortion = dist_case.distortion;
    camera_input.psf = psf_case.psf;

    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), config.total_threads) },
    };

    var case_config = config;
    case_config.background_value = common.grey_background_scene1;

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
    var prep = try common.prepareScene1(allocator, io);
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
    config.background_value = common.grey_background_scene1;

    const gold_dir_root = policy.goldRoot(.full_dist_psf);
    try generateAllFullDistPsfCases(allocator, io, gold_dir_root, config);
}
