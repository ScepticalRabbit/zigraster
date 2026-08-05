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
const policy = @import("testpolicy.zig");
const F = buildconfig.F;
const orch = @import("orchestration.zig");
const tcfg = @import("testconfig.zig");
const cam = @import("../riley/zig/camera.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const iio = @import("../riley/zig/imageio.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const NDArray = @import("../riley/zig/ndarray.zig").NDArray;
const rastcfg = @import("../riley/zig/rasterconfig.zig");
const riley = @import("../riley/zig/riley.zig");
const shaderops = @import("../riley/zig/shaderops.zig");

pub const gold_root = policy.goldRoot(.psf);
pub const pixel_num = [_]u32{ 512, 512 };
pub const fov_scale: F = 1.1;
pub const tile_size_small: u16 = 16;
pub const tile_size_large: u16 = 64;
pub const grey_background_value: F = 0.5;
pub const checker_squares_per_axis: F = 36.0;

pub const midside_mesh_types = [_]gk.MeshType{ .tri6, .quad8, .quad9 };
pub const full_mesh_types = [_]gk.MeshType{
    .tri3,
    .tri6,
    .quad4ibi,
    .quad4newton,
    .quad8,
    .quad9,
};

pub const DistortionCase = struct {
    name: []const u8,
    mesh_types: []const gk.MeshType,
};

pub const ShaderCase = struct {
    tag: []const u8,
    builtin: shaderops.FuncShaderBuiltin,
    params: shaderops.FuncShaderParams = .{},
    use_uvs: bool = true,
    background_value: F = grey_background_value,
};

pub const PsfCase = struct {
    tag: []const u8,
    psf: cam.PointSpreadFunc,
};

pub const RenderCase = struct {
    distortion_case_name: []const u8,
    mesh_type: gk.MeshType,
    shader_case: ShaderCase,
    psf_case: PsfCase,
};

pub const distortion_cases = [_]DistortionCase{
    .{ .name = "distort_bulge", .mesh_types = &midside_mesh_types },
    .{ .name = "distort_shear", .mesh_types = &full_mesh_types },
};

pub const shader_cases = [_]ShaderCase{
    .{
        .tag = "texfunc_checker",
        .builtin = .checker,
        .params = .{
            .coord_scale = .{
                checker_squares_per_axis,
                checker_squares_per_axis,
            },
        },
        .use_uvs = true,
        .background_value = grey_background_value,
    },
    .{
        .tag = "texfunc_constant",
        .builtin = .constant,
        .params = .{
            .output_scale = 2.0,
            .output_offset = 0.0,
        },
        .use_uvs = false,
        .background_value = 0.0,
    },
};

pub const psf_cases = [_]PsfCase{
    .{
        .tag = "pixel_box",
        .psf = .{ .pixel_box = .{} },
    },
    .{
        .tag = "gaussian_sep",
        .psf = .{ .gaussian = .{
            .sigma_px = 0.6,
            .supp_rad_px = 2.0,
            .separable = .yes,
        } },
    },
    .{
        .tag = "gaussian_nonsep",
        .psf = .{ .gaussian = .{
            .sigma_px = 0.6,
            .supp_rad_px = 2.0,
            .separable = .no,
        } },
    },
    .{
        .tag = "anisotropic_gaussian",
        .psf = .{ .anisotropic_gaussian = .{
            .sigma_x_px = 1.2,
            .sigma_y_px = 0.2,
            .theta_rad = std.math.pi / 6.0,
            .supp_rad_px = 3.0,
            .separable = .no,
        } },
    },
};

pub fn caseDirName(
    allocator: std.mem.Allocator,
    render_case: RenderCase,
) ![]const u8 {
    return std.fmt.allocPrint(
        allocator,
        "{s}_{s}_{s}_{s}",
        .{
            render_case.distortion_case_name,
            @tagName(render_case.mesh_type),
            render_case.shader_case.tag,
            render_case.psf_case.tag,
        },
    );
}

fn buildMeshInput(
    prepared: *const orch.SingleMeshPrepared,
    render_case: RenderCase,
) mo.MeshInput {
    return .{
        .mesh_type = render_case.mesh_type,
        .coords = prepared.sim_data.coords,
        .connect = prepared.sim_data.connect,
        .disp = prepared.sim_data.field,
        .shader = .{
            .func = .{
                .uvs = if (render_case.shader_case.use_uvs)
                    prepared.uvs.array
                else
                    null,
                .coord_mode = if (render_case.shader_case.use_uvs) .uv else .para,
                .builtin = render_case.shader_case.builtin,
                .params = render_case.shader_case.params,
                .normal_type = .none,
            },
        },
    };
}

fn buildCameraInput(
    prepared: *const orch.SingleMeshPrepared,
    psf: cam.PointSpreadFunc,
) cam.CameraInput {
    return .{
        .pixels_num = prepared.camera.pixels_num,
        .pixels_size = prepared.camera.pixels_size,
        .pos_world = prepared.camera.pos_world,
        .rot_world = prepared.camera.rot_world,
        .roi_cent_world = prepared.camera.roi_cent_world,
        .focal_length = prepared.camera.focal_length,
        .sub_sample = prepared.camera.sub_sample,
        .distortion = prepared.camera.distortion,
        .psf = psf,
        .coord_sys = prepared.camera.coord_sys,
    };
}

fn baseRasterConfig(
    tile_size_override: ?u16,
    save_strategy: rastcfg.SaveStrategy,
    image_save_opts: []const iio.ImageSaveOpts,
    background_value: F,
) rastcfg.RasterConfig {
    var config = tcfg.getRasterConfig(.gold);
    config.save_strategy = save_strategy;
    config.tile_size_override = tile_size_override;
    config.background_value = background_value;
    config.image_save_opts = image_save_opts;
    return config;
}

pub fn renderCase(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    render_case: RenderCase,
    tile_size_override: ?u16,
) !NDArray(F) {
    return renderCaseWithRasterHalo(
        outer_alloc,
        io,
        render_case,
        tile_size_override,
        null,
    );
}

pub fn renderCaseWithRasterHalo(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    render_case: RenderCase,
    tile_size_override: ?u16,
    raster_halo_px_override: ?u16,
) !NDArray(F) {
    var arena = std.heap.ArenaAllocator.init(outer_alloc);
    defer arena.deinit();
    const aa = arena.allocator();

    const prepared = try orch.prepareSingleMeshCase(
        aa,
        io,
        render_case.distortion_case_name,
        render_case.mesh_type,
        pixel_num,
        fov_scale,
        "data/edge",
    );
    const mesh_input = buildMeshInput(&prepared, render_case);
    const camera_input = buildCameraInput(&prepared, render_case.psf_case.psf);

    var config = baseRasterConfig(
        tile_size_override,
        .memory,
        &[_]iio.ImageSaveOpts{
            .{ .format = .csv, .bits = null, .scaling = .none },
        },
        render_case.shader_case.background_value,
    );
    config.raster_halo_px_override = raster_halo_px_override;
    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), config.total_threads) },
    };

    return (try riley.raster(
        outer_alloc,
        &render_groups,
        &[_]cam.CameraInput{camera_input},
        &[_]mo.MeshInput{mesh_input},
        config,
        null,
    )) orelse error.NoResult;
}

pub fn saveGoldCase(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    render_case: RenderCase,
) !void {
    var arena = std.heap.ArenaAllocator.init(outer_alloc);
    defer arena.deinit();
    const aa = arena.allocator();

    const prepared = try orch.prepareSingleMeshCase(
        aa,
        io,
        render_case.distortion_case_name,
        render_case.mesh_type,
        pixel_num,
        fov_scale,
        "data/edge",
    );
    const mesh_input = buildMeshInput(&prepared, render_case);
    const camera_input = buildCameraInput(&prepared, render_case.psf_case.psf);
    const case_dir_name = try caseDirName(aa, render_case);
    const gold_dir = try std.fmt.allocPrint(aa, "{s}/{s}", .{ gold_root, case_dir_name });

    const config = baseRasterConfig(
        null,
        .disk,
        &[_]iio.ImageSaveOpts{
            .{ .format = .fimg, .bits = null, .scaling = .none },
            .{ .format = .bmp, .bits = 8, .scaling = .auto },
        },
        render_case.shader_case.background_value,
    );
    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), config.total_threads) },
    };

    if (try riley.raster(
        outer_alloc,
        &render_groups,
        &[_]cam.CameraInput{camera_input},
        &[_]mo.MeshInput{mesh_input},
        config,
        gold_dir,
    )) |result| {
        outer_alloc.free(result.slice);
        var result_mut = result;
        result_mut.deinit(outer_alloc);
    }
}

pub fn saveAllGoldCases(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
) !void {
    for (distortion_cases) |distortion_case| {
        for (distortion_case.mesh_types) |mesh_type| {
            for (shader_cases) |shader_case| {
                for (psf_cases) |psf_case| {
                    try saveGoldCase(
                        outer_alloc,
                        io,
                        .{
                            .distortion_case_name = distortion_case.name,
                            .mesh_type = mesh_type,
                            .shader_case = shader_case,
                            .psf_case = psf_case,
                        },
                    );
                }
            }
        }
    }
}

pub fn expectResultsApproxEq(
    lhs: *const NDArray(F),
    rhs: *const NDArray(F),
    rel_tol: F,
    abs_tol: F,
) !void {
    try std.testing.expectEqual(lhs.dims.len, rhs.dims.len);
    for (lhs.dims, rhs.dims) |ld, rd| {
        try std.testing.expectEqual(ld, rd);
    }
    for (lhs.slice, rhs.slice) |l, r| {
        try std.testing.expectApproxEqRel(r, l, rel_tol);
        try std.testing.expectApproxEqAbs(r, l, abs_tol);
    }
}
