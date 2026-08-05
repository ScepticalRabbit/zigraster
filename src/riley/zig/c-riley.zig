// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const buildconfig = @import("buildconfig.zig");
const F = buildconfig.F;

const cam = @import("camera.zig");
const cameraio = @import("cameraio.zig");
const cameraops = @import("cameraops.zig");
const gk = @import("geometrykernels.zig");
const iio = @import("imageio.zig");
const imageops = @import("imageops.zig");
const meshio = @import("meshio.zig");
const mo = @import("meshpipeline.zig");
const ndarray = @import("ndarray.zig");
const riley = @import("riley.zig");
const rotation = @import("rotation.zig");
const rastcfg = @import("rasterconfig.zig");
const sceneops = @import("sceneops.zig");
const shaderops = @import("shaderops.zig");
const texops = @import("textureops.zig");
const vec = @import("vecstack.zig");

// --------------------------------------------------------------------------
// Public C ABI Contract
//
// The public Riley C ABI is fixed to the production Riley build:
// - precision: f64
// - SIMD: on
//
// This keeps the exported ABI stable for C, Cython and Python callers.
// If you need alternate precision or SIMD experiments, use the native Zig
// entry points rather than this public C interface.
// --------------------------------------------------------------------------
comptime {
    if (buildconfig.default_precision != f64) {
        @compileError("The public Riley C ABI must be built with f64.");
    }
    if (buildconfig.default_simd != .on) {
        @compileError("The public Riley C ABI must be built with SIMD on.");
    }
}

const error_buf_len: usize = 512;
const default_image_save_opts = [_]iio.ImageSaveOpts{
    .{
        .format = .bmp,
        .bits = 8,
        .scaling = .auto,
    },
};

var last_error_buf: [error_buf_len]u8 = [_]u8{0} ** error_buf_len;

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const CVec2U32 = extern struct {
    x: u32,
    y: u32,
};

pub const CVec2F64 = extern struct {
    x: F,
    y: F,
};

pub const CVec3F64 = extern struct {
    x: F,
    y: F,
    z: F,
};

pub const CArray2DF64 = extern struct {
    elems: [*c]const F,
    rows_num: usize,
    cols_num: usize,
};

pub const CArray2DUsize = extern struct {
    elems: [*c]const usize,
    rows_num: usize,
    cols_num: usize,
};

pub const CArray3DF64 = extern struct {
    elems: [*c]const F,
    dim0: usize,
    dim1: usize,
    dim2: usize,
};

pub const CArray3DU8 = extern struct {
    elems: [*c]const u8,
    dim0: usize,
    dim1: usize,
    dim2: usize,
};

pub const CArray3DU16 = extern struct {
    elems: [*c]const u16,
    dim0: usize,
    dim1: usize,
    dim2: usize,
};

pub const CDims5Usize = extern struct {
    dim0: usize,
    dim1: usize,
    dim2: usize,
    dim3: usize,
    dim4: usize,
};

pub const CImageBuffF64 = extern struct {
    elems: [*c]F,
    dims: CDims5Usize,
};

pub const CCameraInput = extern struct {
    pixels_num: CVec2U32,
    pixels_size: CVec2F64,
    pos_world: CVec3F64,
    rot_world: CVec3F64,
    roi_cent_world: CVec3F64,
    focal_length: F,
    sub_sample: u32,
    distortion_model: u32,
    distortion_k1: F,
    distortion_k2: F,
    distortion_k3: F,
    distortion_k4: F,
    distortion_k5: F,
    distortion_k6: F,
    distortion_p1: F,
    distortion_p2: F,
    distortion_poly_order: u32,
    distortion_poly_has_forward: u8,
    distortion_poly_has_inv: u8,
    distortion_poly_forward_u: [10]F,
    distortion_poly_forward_v: [10]F,
    distortion_poly_inv_u: [10]F,
    distortion_poly_inv_v: [10]F,
    coord_sys: u32,
    subpixel_center_map: u32,
    psf_type: u32,
    psf_sigma_x: F,
    psf_sigma_y: F,
    psf_theta: F,
    psf_supp_rad: F,
    psf_separable: u32,
};

pub const CFuncShaderParams = extern struct {
    coord_scale_0: F,
    coord_scale_1: F,
    coord_offset_0: F,
    coord_offset_1: F,
    output_scale: F,
    output_offset: F,
    constant_value: F,
    constant_value_rgb_0: F,
    constant_value_rgb_1: F,
    constant_value_rgb_2: F,
    linear_coeff_0: F,
    linear_coeff_1: F,
    linear_coeff_2: F,
    linear_coeff_rgb_00: F,
    linear_coeff_rgb_01: F,
    linear_coeff_rgb_02: F,
    linear_coeff_rgb_10: F,
    linear_coeff_rgb_11: F,
    linear_coeff_rgb_12: F,
    linear_coeff_rgb_20: F,
    linear_coeff_rgb_21: F,
    linear_coeff_rgb_22: F,
    quadratic_coeff_0: F,
    quadratic_coeff_1: F,
    quadratic_coeff_2: F,
    quadratic_coeff_3: F,
    quadratic_coeff_4: F,
    quadratic_coeff_5: F,
    quadratic_coeff_rgb_00: F,
    quadratic_coeff_rgb_01: F,
    quadratic_coeff_rgb_02: F,
    quadratic_coeff_rgb_03: F,
    quadratic_coeff_rgb_04: F,
    quadratic_coeff_rgb_05: F,
    quadratic_coeff_rgb_10: F,
    quadratic_coeff_rgb_11: F,
    quadratic_coeff_rgb_12: F,
    quadratic_coeff_rgb_13: F,
    quadratic_coeff_rgb_14: F,
    quadratic_coeff_rgb_15: F,
    quadratic_coeff_rgb_20: F,
    quadratic_coeff_rgb_21: F,
    quadratic_coeff_rgb_22: F,
    quadratic_coeff_rgb_23: F,
    quadratic_coeff_rgb_24: F,
    quadratic_coeff_rgb_25: F,
    wave_num_scalar_0: F,
    wave_num_scalar_1: F,
    wave_num_rgb_0: F,
    wave_num_rgb_1: F,
    wave_num_rgb_2: F,
    sinusoidal_bias: F,
    sinusoidal_amp_0: F,
    sinusoidal_amp_1: F,
    sinusoidal_bias_rgb_0: F,
    sinusoidal_bias_rgb_1: F,
    sinusoidal_bias_rgb_2: F,
    sinusoidal_amp_rgb_0: F,
    sinusoidal_amp_rgb_1: F,
    sinusoidal_amp_rgb_2: F,
    checker_level_0: F,
    checker_level_1: F,
    checker_smooth_frequency: F,
    lambertian_coeff_0: F,
    lambertian_coeff_1: F,
    lambertian_coeff_rgb_00: F,
    lambertian_coeff_rgb_01: F,
    lambertian_coeff_rgb_10: F,
    lambertian_coeff_rgb_11: F,
    lambertian_coeff_rgb_20: F,
    lambertian_coeff_rgb_21: F,
    eggbox_mean: F,
    eggbox_contrast: F,
    eggbox_pitch_0: F,
    eggbox_pitch_1: F,
    eggbox_phase_0: F,
    eggbox_phase_1: F,
    extra_0: F,
    extra_1: F,
    extra_2: F,
    extra_3: F,
};

pub const CMeshInput = extern struct {
    mesh_type: u32,
    coords: CArray2DF64,
    connect: CArray2DUsize,
    disp: CArray3DF64,
    shader_tag: u32,
    uvs: CArray2DF64,
    tex: CArray3DF64,
    tex_u8: CArray3DU8,
    tex_u16: CArray3DU16,
    texture_storage: u32,
    sample: u32,
    sample_mode: u32,
    bits: c_int,
    scaling_tag: u32,
    scaling_min: F,
    scaling_max: F,
    nodal_field: CArray3DF64,
    scale_over: u32,
    func_shader_builtin: u32,
    func_shader_coord_mode: u32,
    func_shader_params: CFuncShaderParams,
    normal_type: u32,
};

pub const CRasterConfig = extern struct {
    render_mode: u32,
    total_threads: u16,
    frame_batch_size_per_group: u16,
    max_geom_jobs_in_flight_per_group: u16,
    max_geom_workers_per_job: u16,
    geom_scheduling_mode: u32,
    max_raster_workers_per_job: u16,
    save_strategy: u32,
    image_save_mode: u32,
    hull_mode: u32,
    newton_seed_mode: u32,
    newton_seed_reuse: u32,
    report: u32,
    tile_size_min: u16,
    tile_size_max: u16,
    background_value: F,
    disk_save_overlap: u8,
    tile_size_override: u16,
    save_frame_buff_count: usize,
    save_format: u32,
    save_bits: u32,
    save_scaling: u32,
    save_scaling_min: F,
    save_scaling_max: F,
    full_stats_save_solver_csv: u8,
    full_stats_save_iter_map: u8,
    full_stats_save_xi_map: u8,
    full_stats_save_eta_map: u8,
    full_stats_save_conv_map: u8,
    full_stats_save_jac_det_map: u8,
    full_stats_save_tile_timing_map: u8,
    full_stats_save_tile_density_map: u8,
    full_stats_save_tile_occupancy_map: u8,
    full_stats_save_depth_map: u8,
    full_stats_save_earlyout_map: u8,
    full_stats_save_pixel_occupancy_map: u8,
    full_stats_save_normals_map: u8,
};

const MeshInputBuilt = struct {
    mesh_input: mo.MeshInput,
    disp_array: ?ndarray.NDArray(F) = null,
    uvs_array: ?ndarray.NDArray(F) = null,
    tex_array_u8: ?ndarray.NDArray(u8) = null,
    tex_array_u16: ?ndarray.NDArray(u16) = null,
    tex_array_f: ?ndarray.NDArray(F) = null,
    nodal_field_array: ?ndarray.NDArray(F) = null,

    fn deinit(self: *MeshInputBuilt, allocator: std.mem.Allocator) void {
        if (self.disp_array) |*disp_array| {
            disp_array.deinit(allocator);
        }
        if (self.uvs_array) |*uvs_array| {
            uvs_array.deinit(allocator);
        }
        if (self.tex_array_u8) |*tex_array| {
            tex_array.deinit(allocator);
        }
        if (self.tex_array_u16) |*tex_array| {
            tex_array.deinit(allocator);
        }
        if (self.tex_array_f) |*tex_array| {
            tex_array.deinit(allocator);
        }
        if (self.nodal_field_array) |*nodal_field_array| {
            nodal_field_array.deinit(allocator);
        }
    }
};

fn clearLastError() void {
    @memset(last_error_buf[0..], 0);
}

fn setLastErrorSlice(msg: []const u8) void {
    clearLastError();
    const copy_len = @min(msg.len, error_buf_len - 1);
    @memcpy(last_error_buf[0..copy_len], msg[0..copy_len]);
}

fn setLastError(err: anyerror) void {
    var msg_buf: [error_buf_len]u8 = undefined;
    const msg = std.fmt.bufPrint(
        msg_buf[0..],
        "{s}",
        .{@errorName(err)},
    ) catch @errorName(err);
    setLastErrorSlice(msg);
}

pub export fn rileyGetLastError(
    out_buf: [*c]u8,
    out_buf_len: usize,
) usize {
    if (out_buf == null or out_buf_len == 0) {
        return 0;
    }

    const msg_len = std.mem.indexOfScalar(
        u8,
        last_error_buf[0..],
        0,
    ) orelse last_error_buf.len;
    const copy_len = @min(msg_len, out_buf_len - 1);
    @memcpy(out_buf[0..copy_len], last_error_buf[0..copy_len]);
    out_buf[copy_len] = 0;
    return copy_len;
}

fn cVec3ToVec3(in_vec: CVec3F64) vec.Vec3f {
    return vec.initVec3(
        F,
        in_vec.x,
        in_vec.y,
        in_vec.z,
    );
}

fn vec3ToCVec3(in_vec: vec.Vec3f) CVec3F64 {
    return .{
        .x = in_vec.get(0),
        .y = in_vec.get(1),
        .z = in_vec.get(2),
    };
}

fn dimsToArray(in_dims: CDims5Usize) [5]usize {
    return .{
        in_dims.dim0,
        in_dims.dim1,
        in_dims.dim2,
        in_dims.dim3,
        in_dims.dim4,
    };
}

fn dimsFromArray(in_dims: [5]usize) CDims5Usize {
    return .{
        .dim0 = in_dims[0],
        .dim1 = in_dims[1],
        .dim2 = in_dims[2],
        .dim3 = in_dims[3],
        .dim4 = in_dims[4],
    };
}

fn array3ToDims(in_array: anytype) [3]usize {
    return .{
        in_array.dim0,
        in_array.dim1,
        in_array.dim2,
    };
}

fn cConstSlice(
    comptime T: type,
    ptr: [*c]const T,
    len: usize,
) ![]const T {
    if (ptr == null and len > 0) {
        return error.NullPointer;
    }
    return ptr[0..len];
}

fn cMutSlice(
    comptime T: type,
    ptr: [*c]T,
    len: usize,
) ![]T {
    if (ptr == null and len > 0) {
        return error.NullPointer;
    }
    return ptr[0..len];
}

fn meshTypeFromC(mesh_type: u32) !gk.MeshType {
    return switch (mesh_type) {
        @intFromEnum(gk.MeshType.tri3) => .tri3,
        @intFromEnum(gk.MeshType.tri3opt) => .tri3opt,
        @intFromEnum(gk.MeshType.tri6) => .tri6,
        @intFromEnum(gk.MeshType.quad4ibi) => .quad4ibi,
        @intFromEnum(gk.MeshType.quad4newton) => .quad4newton,
        @intFromEnum(gk.MeshType.quad8) => .quad8,
        @intFromEnum(gk.MeshType.quad9) => .quad9,
        else => error.InvalidMeshType,
    };
}

fn renderModeFromC(render_mode: u32) !riley.RenderMode {
    return switch (render_mode) {
        @intFromEnum(riley.RenderMode.in_order) => .in_order,
        @intFromEnum(riley.RenderMode.offline) => .offline,
        else => error.InvalidRenderMode,
    };
}

fn geometrySchedulingModeFromC(
    geom_scheduling_mode: u32,
) !rastcfg.GeometrySchedulingMode {
    return switch (geom_scheduling_mode) {
        @intFromEnum(rastcfg.GeometrySchedulingMode.spread) => .spread,
        @intFromEnum(rastcfg.GeometrySchedulingMode.pack) => .pack,
        @intFromEnum(rastcfg.GeometrySchedulingMode.auto) => .auto,
        else => error.InvalidGeometrySchedulingMode,
    };
}

fn saveStrategyFromC(save_strategy: u32) !riley.SaveStrategy {
    return switch (save_strategy) {
        @intFromEnum(riley.SaveStrategy.disk) => .disk,
        @intFromEnum(riley.SaveStrategy.memory) => .memory,
        @intFromEnum(riley.SaveStrategy.both) => .both,
        @intFromEnum(riley.SaveStrategy.none) => .none,
        else => error.InvalidSaveStrategy,
    };
}

fn imageSaveModeFromC(image_save_mode: u32) !riley.ImageSaveMode {
    return switch (image_save_mode) {
        @intFromEnum(riley.ImageSaveMode.grey) => .grey,
        @intFromEnum(riley.ImageSaveMode.rgb) => .rgb,
        @intFromEnum(riley.ImageSaveMode.multifield) => .multifield,
        else => error.InvalidImageMode,
    };
}

fn reportModeFromC(report_mode: u32) !riley.ReportMode {
    return switch (report_mode) {
        @intFromEnum(riley.ReportMode.off) => .off,
        @intFromEnum(riley.ReportMode.bench) => .bench,
        @intFromEnum(riley.ReportMode.full_stats) => .full_stats,
        else => error.InvalidReportMode,
    };
}

fn subpxCenterMapFromC(subpx_map: u32) !cam.SubPixelCenterMap {
    return switch (subpx_map) {
        @intFromEnum(cam.SubPixelCenterMap.full_in_mem) => .full_in_mem,
        @intFromEnum(cam.SubPixelCenterMap.per_tile) => .per_tile,
        @intFromEnum(cam.SubPixelCenterMap.affine_jac) => .affine_jac,
        else => error.InvalidSubPixelCenterMap,
    };
}

fn hullModeFromC(hull_mode: u32) !rastcfg.HullMode {
    return switch (hull_mode) {
        @intFromEnum(rastcfg.HullMode.off) => .off,
        @intFromEnum(rastcfg.HullMode.on_no_fallback) => .on_no_fallback,
        @intFromEnum(rastcfg.HullMode.on_convex_fallback) => .on_convex_fallback,
        else => error.InvalidHullMode,
    };
}

fn newtonSeedModeFromC(
    seed_mode: u32,
) !rastcfg.NewtonSeedMode {
    return switch (seed_mode) {
        @intFromEnum(rastcfg.NewtonSeedMode.centroid) => .centroid,
        @intFromEnum(rastcfg.NewtonSeedMode.hull) => .hull,
        else => error.InvalidNewtonSeedMode,
    };
}

fn newtonSeedReuseFromC(
    seed_reuse: u32,
) !rastcfg.NewtonSeedReuse {
    return switch (seed_reuse) {
        @intFromEnum(rastcfg.NewtonSeedReuse.off) => .off,
        @intFromEnum(rastcfg.NewtonSeedReuse.last_conv) => .last_conv,
        else => error.InvalidNewtonSeedReuse,
    };
}

fn coordSysFromC(coord_sys: u32) !cam.CameraCoordSys {
    return switch (coord_sys) {
        @intFromEnum(cam.CameraCoordSys.opengl) => .opengl,
        @intFromEnum(cam.CameraCoordSys.opencv) => .opencv,
        else => error.InvalidCoordSys,
    };
}

fn imageFormatFromC(format_tag: u32) !iio.ImageFormat {
    return switch (format_tag) {
        0 => .csv,
        1 => .fimg,
        2 => .ppm,
        3 => .bmp,
        4 => .tiff,
        else => error.InvalidImageFormat,
    };
}

fn psfSeparableFromC(sep_tag: u32) !cam.SeparablePSF {
    return switch (sep_tag) {
        0 => .no,
        1 => .yes,
        else => error.InvalidSeparablePSF,
    };
}

fn psfFromC(in_camera: *const CCameraInput) !cam.PointSpreadFunc {
    return switch (in_camera.psf_type) {
        0 => .{ .pixel_box = .{} },
        1 => .{ .gaussian = .{
            .sigma_px = in_camera.psf_sigma_x,
            .supp_rad_px = in_camera.psf_supp_rad,
            .separable = try psfSeparableFromC(in_camera.psf_separable),
        } },
        2 => .{ .anisotropic_gaussian = .{
            .sigma_x_px = in_camera.psf_sigma_x,
            .sigma_y_px = in_camera.psf_sigma_y,
            .theta_rad = in_camera.psf_theta,
            .supp_rad_px = in_camera.psf_supp_rad,
            .separable = try psfSeparableFromC(in_camera.psf_separable),
        } },
        else => error.InvalidPsfType,
    };
}

fn distortionFromC(in_camera: *const CCameraInput) !cam.DistortionModel {
    const poly_order: cam.PolynomialOrder = switch (in_camera.distortion_poly_order) {
        0, 2 => .quadratic,
        1 => .linear,
        3 => .cubic,
        else => return error.InvalidPolynomialOrder,
    };
    const poly_has_forward = in_camera.distortion_poly_has_forward != 0;
    const poly_has_inv = in_camera.distortion_poly_has_inv != 0;
    var polynomial: ?cam.BidirectionalPolynomial = null;
    if (poly_has_forward or poly_has_inv) {
        var poly: cam.BidirectionalPolynomial = .{};
        if (poly_has_forward) {
            poly.forward_map = .{
                .order = poly_order,
                .coeffs_u = in_camera.distortion_poly_forward_u,
                .coeffs_v = in_camera.distortion_poly_forward_v,
            };
        }
        if (poly_has_inv) {
            poly.inv_map = .{
                .order = poly_order,
                .coeffs_u = in_camera.distortion_poly_inv_u,
                .coeffs_v = in_camera.distortion_poly_inv_v,
            };
        }
        polynomial = poly;
    }

    return switch (in_camera.distortion_model) {
        0 => .none,
        1 => .{ .brown_conrady = .{
            .k1 = in_camera.distortion_k1,
            .k2 = in_camera.distortion_k2,
            .k3 = in_camera.distortion_k3,
            .p1 = in_camera.distortion_p1,
            .p2 = in_camera.distortion_p2,
        } },
        2 => .{ .brown_conrady_ext = .{
            .k1 = in_camera.distortion_k1,
            .k2 = in_camera.distortion_k2,
            .k3 = in_camera.distortion_k3,
            .k4 = in_camera.distortion_k4,
            .k5 = in_camera.distortion_k5,
            .k6 = in_camera.distortion_k6,
            .p1 = in_camera.distortion_p1,
            .p2 = in_camera.distortion_p2,
        } },
        3 => .{ .polynomial = polynomial orelse return error.MissingPolynomialMap },
        4 => .{ .brown_conrady_polynomial = .{
            .brown_conrady = .{
                .k1 = in_camera.distortion_k1,
                .k2 = in_camera.distortion_k2,
                .k3 = in_camera.distortion_k3,
                .p1 = in_camera.distortion_p1,
                .p2 = in_camera.distortion_p2,
            },
            .polynomial = polynomial orelse return error.MissingPolynomialMap,
        } },
        5 => .{ .brown_conrady_ext_polynomial = .{
            .brown_conrady_ext = .{
                .k1 = in_camera.distortion_k1,
                .k2 = in_camera.distortion_k2,
                .k3 = in_camera.distortion_k3,
                .k4 = in_camera.distortion_k4,
                .k5 = in_camera.distortion_k5,
                .k6 = in_camera.distortion_k6,
                .p1 = in_camera.distortion_p1,
                .p2 = in_camera.distortion_p2,
            },
            .polynomial = polynomial orelse return error.MissingPolynomialMap,
        } },
        else => error.InvalidDistortionModel,
    };
}

fn texSampleFromC(sample: u32) !texops.TexSamp {
    const mitchell = texops.TexSamp.cubic_mitchell_netravali;
    return switch (sample) {
        @intFromEnum(texops.TexSamp.nearest) => .nearest,
        @intFromEnum(texops.TexSamp.linear) => .linear,
        @intFromEnum(texops.TexSamp.cubic_catmull_rom) => .cubic_catmull_rom,
        @intFromEnum(mitchell) => .cubic_mitchell_netravali,
        @intFromEnum(texops.TexSamp.lanczos3) => .lanczos3,
        @intFromEnum(texops.TexSamp.cubic_bspline) => .cubic_bspline,
        @intFromEnum(texops.TexSamp.quintic_bspline) => .quintic_bspline,
        @intFromEnum(texops.TexSamp.lanczos2) => .lanczos2,
        else => error.InvalidTexSample,
    };
}

fn texSampleModeFromC(
    sample_mode: u32,
) !texops.TexSampMode {
    return switch (sample_mode) {
        @intFromEnum(texops.TexSampMode.direct) => .direct,
        @intFromEnum(texops.TexSampMode.lut) => .lut,
        @intFromEnum(texops.TexSampMode.lut_lerp) => .lut_lerp,
        else => error.InvalidTexSampleMode,
    };
}

fn scaleOverFromC(scale_over: u32) !shaderops.ScaleOver {
    return switch (scale_over) {
        @intFromEnum(shaderops.ScaleOver.within_frames) => .within_frames,
        @intFromEnum(shaderops.ScaleOver.over_frames) => .over_frames,
        else => error.InvalidScaleOver,
    };
}

fn funcShaderBuiltinFromC(
    func_shader_builtin: u32,
) !shaderops.FuncShaderBuiltin {
    const lambertian = shaderops.FuncShaderBuiltin.lambertian_normal_z;
    const eggbox = shaderops.FuncShaderBuiltin.eggbox;
    return switch (func_shader_builtin) {
        @intFromEnum(shaderops.FuncShaderBuiltin.constant) => .constant,
        @intFromEnum(shaderops.FuncShaderBuiltin.linear) => .linear,
        @intFromEnum(shaderops.FuncShaderBuiltin.quadratic) => .quadratic,
        @intFromEnum(shaderops.FuncShaderBuiltin.sinusoidal) => .sinusoidal,
        @intFromEnum(shaderops.FuncShaderBuiltin.sinusoidal_approx) => .sinusoidal_approx,
        @intFromEnum(shaderops.FuncShaderBuiltin.checker) => .checker,
        @intFromEnum(shaderops.FuncShaderBuiltin.checker_smooth) => .checker_smooth,
        @intFromEnum(lambertian) => .lambertian_normal_z,
        @intFromEnum(eggbox) => .eggbox,
        else => error.InvalidFuncShaderBuiltin,
    };
}

fn funcCoordModeFromC(
    coord_mode: u32,
) !shaderops.FuncCoordMode {
    return switch (coord_mode) {
        @intFromEnum(shaderops.FuncCoordMode.uv) => .uv,
        @intFromEnum(shaderops.FuncCoordMode.para) => .para,
        @intFromEnum(shaderops.FuncCoordMode.world_reference) => .world_reference,
        @intFromEnum(shaderops.FuncCoordMode.world_deformed) => .world_deformed,
        else => error.InvalidFuncCoordMode,
    };
}

fn normalTypeFromC(normal_type: u32) !shaderops.NormalType {
    return switch (normal_type) {
        @intFromEnum(shaderops.NormalType.none) => .none,
        @intFromEnum(shaderops.NormalType.exact) => .exact,
        @intFromEnum(shaderops.NormalType.avg) => .avg,
        else => error.InvalidNormalType,
    };
}

fn funcShaderParamsFromC(
    builtin: shaderops.FuncShaderBuiltin,
    in_params: CFuncShaderParams,
) shaderops.FuncShaderParams {
    return .{
        .coord_scale = .{
            in_params.coord_scale_0,
            in_params.coord_scale_1,
        },
        .coord_offset = .{
            in_params.coord_offset_0,
            in_params.coord_offset_1,
        },
        .output_scale = in_params.output_scale,
        .output_offset = in_params.output_offset,
        .settings = switch (builtin) {
            .constant => .{
                .constant = .{
                    .value = in_params.constant_value,
                    .value_rgb = .{
                        in_params.constant_value_rgb_0,
                        in_params.constant_value_rgb_1,
                        in_params.constant_value_rgb_2,
                    },
                },
            },
            .linear => .{
                .linear = .{
                    .coeffs = .{
                        in_params.linear_coeff_0,
                        in_params.linear_coeff_1,
                        in_params.linear_coeff_2,
                    },
                    .coeffs_rgb = .{
                        .{
                            in_params.linear_coeff_rgb_00,
                            in_params.linear_coeff_rgb_01,
                            in_params.linear_coeff_rgb_02,
                        },
                        .{
                            in_params.linear_coeff_rgb_10,
                            in_params.linear_coeff_rgb_11,
                            in_params.linear_coeff_rgb_12,
                        },
                        .{
                            in_params.linear_coeff_rgb_20,
                            in_params.linear_coeff_rgb_21,
                            in_params.linear_coeff_rgb_22,
                        },
                    },
                },
            },
            .quadratic => .{
                .quadratic = .{
                    .coeffs = .{
                        in_params.quadratic_coeff_0,
                        in_params.quadratic_coeff_1,
                        in_params.quadratic_coeff_2,
                        in_params.quadratic_coeff_3,
                        in_params.quadratic_coeff_4,
                        in_params.quadratic_coeff_5,
                    },
                    .coeffs_rgb = .{
                        .{
                            in_params.quadratic_coeff_rgb_00,
                            in_params.quadratic_coeff_rgb_01,
                            in_params.quadratic_coeff_rgb_02,
                            in_params.quadratic_coeff_rgb_03,
                            in_params.quadratic_coeff_rgb_04,
                            in_params.quadratic_coeff_rgb_05,
                        },
                        .{
                            in_params.quadratic_coeff_rgb_10,
                            in_params.quadratic_coeff_rgb_11,
                            in_params.quadratic_coeff_rgb_12,
                            in_params.quadratic_coeff_rgb_13,
                            in_params.quadratic_coeff_rgb_14,
                            in_params.quadratic_coeff_rgb_15,
                        },
                        .{
                            in_params.quadratic_coeff_rgb_20,
                            in_params.quadratic_coeff_rgb_21,
                            in_params.quadratic_coeff_rgb_22,
                            in_params.quadratic_coeff_rgb_23,
                            in_params.quadratic_coeff_rgb_24,
                            in_params.quadratic_coeff_rgb_25,
                        },
                    },
                },
            },
            .sinusoidal => .{
                .sinusoidal = .{
                    .wave_num_scalar = .{
                        in_params.wave_num_scalar_0,
                        in_params.wave_num_scalar_1,
                    },
                    .wave_num_rgb = .{
                        in_params.wave_num_rgb_0,
                        in_params.wave_num_rgb_1,
                        in_params.wave_num_rgb_2,
                    },
                    .bias = in_params.sinusoidal_bias,
                    .amplitudes = .{
                        in_params.sinusoidal_amp_0,
                        in_params.sinusoidal_amp_1,
                    },
                    .bias_rgb = .{
                        in_params.sinusoidal_bias_rgb_0,
                        in_params.sinusoidal_bias_rgb_1,
                        in_params.sinusoidal_bias_rgb_2,
                    },
                    .amplitudes_rgb = .{
                        in_params.sinusoidal_amp_rgb_0,
                        in_params.sinusoidal_amp_rgb_1,
                        in_params.sinusoidal_amp_rgb_2,
                    },
                },
            },
            .sinusoidal_approx => .{
                .sinusoidal_approx = .{
                    .wave_num_scalar = .{
                        in_params.wave_num_scalar_0,
                        in_params.wave_num_scalar_1,
                    },
                    .wave_num_rgb = .{
                        in_params.wave_num_rgb_0,
                        in_params.wave_num_rgb_1,
                        in_params.wave_num_rgb_2,
                    },
                    .bias = in_params.sinusoidal_bias,
                    .amplitudes = .{
                        in_params.sinusoidal_amp_0,
                        in_params.sinusoidal_amp_1,
                    },
                    .bias_rgb = .{
                        in_params.sinusoidal_bias_rgb_0,
                        in_params.sinusoidal_bias_rgb_1,
                        in_params.sinusoidal_bias_rgb_2,
                    },
                    .amplitudes_rgb = .{
                        in_params.sinusoidal_amp_rgb_0,
                        in_params.sinusoidal_amp_rgb_1,
                        in_params.sinusoidal_amp_rgb_2,
                    },
                },
            },
            .checker => .{
                .checker = .{
                    .levels = .{
                        in_params.checker_level_0,
                        in_params.checker_level_1,
                    },
                },
            },
            .checker_smooth => .{
                .checker_smooth = .{
                    .frequency = in_params.checker_smooth_frequency,
                },
            },
            .lambertian_normal_z => .{
                .lambertian_normal_z = .{
                    .coeffs = .{
                        in_params.lambertian_coeff_0,
                        in_params.lambertian_coeff_1,
                    },
                    .coeffs_rgb = .{
                        .{
                            in_params.lambertian_coeff_rgb_00,
                            in_params.lambertian_coeff_rgb_01,
                        },
                        .{
                            in_params.lambertian_coeff_rgb_10,
                            in_params.lambertian_coeff_rgb_11,
                        },
                        .{
                            in_params.lambertian_coeff_rgb_20,
                            in_params.lambertian_coeff_rgb_21,
                        },
                    },
                },
            },
            .eggbox => .{
                .eggbox = .{
                    .mean = in_params.eggbox_mean,
                    .contrast = in_params.eggbox_contrast,
                    .pitch = .{
                        in_params.eggbox_pitch_0,
                        in_params.eggbox_pitch_1,
                    },
                    .phase = .{
                        in_params.eggbox_phase_0,
                        in_params.eggbox_phase_1,
                    },
                },
            },
        },
    };
}

fn bitsFromC(bits: c_int) !?u8 {
    if (bits < 0) {
        return null;
    }
    if (bits > std.math.maxInt(u8)) {
        return error.InvalidBits;
    }
    return @intCast(bits);
}

fn scaleStrategyFromC(
    scaling_tag: u32,
    scaling_min: F,
    scaling_max: F,
) !imageops.ScaleStrategy {
    return switch (scaling_tag) {
        0 => .none,
        1 => .auto,
        2 => .{ .fixed = .{ scaling_min, scaling_max } },
        3 => .{ .frac = .{ scaling_min, scaling_max } },
        else => error.InvalidScaleStrategy,
    };
}

fn buildCoordsFromC(in_coords: *const CArray2DF64) !meshio.Coords {
    if (in_coords.cols_num != 3) {
        return error.InvalidCoordsShape;
    }
    const coords_slice = try cConstSlice(
        F,
        in_coords.elems,
        in_coords.rows_num * in_coords.cols_num,
    );
    return meshio.Coords.init(
        @constCast(coords_slice),
        in_coords.rows_num,
    );
}

fn buildConnectFromC(
    in_connect: *const CArray2DUsize,
) !meshio.Connect {
    const connect_slice = try cConstSlice(
        usize,
        in_connect.elems,
        in_connect.rows_num * in_connect.cols_num,
    );
    return meshio.Connect.init(
        @constCast(connect_slice),
        in_connect.rows_num,
        in_connect.cols_num,
    );
}

fn buildArray2DF64(
    allocator: std.mem.Allocator,
    in_array: *const CArray2DF64,
    cols_num_expected: ?usize,
) !ndarray.NDArray(F) {
    if (cols_num_expected) |expected_cols_num| {
        if (in_array.cols_num != expected_cols_num) {
            return error.InvalidArray2DShape;
        }
    }
    const slice_in = try cConstSlice(
        F,
        in_array.elems,
        in_array.rows_num * in_array.cols_num,
    );
    var dims = [_]usize{
        in_array.rows_num,
        in_array.cols_num,
    };
    return try ndarray.NDArray(F).init(
        allocator,
        @constCast(slice_in),
        dims[0..],
    );
}

fn buildArray3DF64(
    allocator: std.mem.Allocator,
    in_array: *const CArray3DF64,
) !ndarray.NDArray(F) {
    const dims = array3ToDims(in_array.*);
    const elems_num = dims[0] * dims[1] * dims[2];
    const slice_in = try cConstSlice(
        F,
        in_array.elems,
        elems_num,
    );
    var dims_mut = dims;
    return try ndarray.NDArray(F).init(
        allocator,
        @constCast(slice_in),
        dims_mut[0..],
    );
}

fn buildTexArray(
    comptime T: type,
    allocator: std.mem.Allocator,
    in_array: anytype,
    channels_num: usize,
) !ndarray.NDArray(T) {
    const dims = array3ToDims(in_array.*);
    if (dims[0] != channels_num or dims[1] == 0 or dims[2] == 0) {
        return error.InvalidTexShape;
    }
    const elems_num = dims[0] * dims[1] * dims[2];
    const slice_in = try cConstSlice(T, in_array.elems, elems_num);
    var dims_mut = dims;
    return try ndarray.NDArray(T).init(allocator, @constCast(slice_in), dims_mut[0..]);
}

fn buildOptionalFieldFromC(
    allocator: std.mem.Allocator,
    in_array: *const CArray3DF64,
) !struct {
    field: ?meshio.Field,
    array: ?ndarray.NDArray(F),
} {
    const dims = array3ToDims(in_array.*);
    if (dims[0] == 0 or dims[1] == 0 or dims[2] == 0) {
        return .{
            .field = null,
            .array = null,
        };
    }
    const array = try buildArray3DF64(allocator, in_array);
    return .{
        .field = .{
            .array = array,
            .array_mem = array.slice,
        },
        .array = array,
    };
}

fn buildCameraInput(
    in_camera: *const CCameraInput,
) !cam.CameraInput {
    return .{
        .pixels_num = .{
            in_camera.pixels_num.x,
            in_camera.pixels_num.y,
        },
        .pixels_size = .{
            in_camera.pixels_size.x,
            in_camera.pixels_size.y,
        },
        .pos_world = cVec3ToVec3(in_camera.pos_world),
        .rot_world = rotation.Rotation.init(
            in_camera.rot_world.x,
            in_camera.rot_world.y,
            in_camera.rot_world.z,
        ),
        .roi_cent_world = cVec3ToVec3(in_camera.roi_cent_world),
        .focal_length = in_camera.focal_length,
        .sub_sample = in_camera.sub_sample,
        .distortion = try distortionFromC(in_camera),
        .psf = try psfFromC(in_camera),
        .coord_sys = try coordSysFromC(in_camera.coord_sys),
        .subpixel_center_map = try subpxCenterMapFromC(
            in_camera.subpixel_center_map,
        ),
    };
}

fn buildMeshInput(
    allocator: std.mem.Allocator,
    in_mesh: *const CMeshInput,
) !MeshInputBuilt {
    const mesh_type = try meshTypeFromC(in_mesh.mesh_type);
    const nodes_per_elem = mesh_type.getNodesNum();

    if (in_mesh.coords.cols_num != 3) {
        return error.InvalidCoordsShape;
    }
    if (in_mesh.connect.cols_num != nodes_per_elem) {
        return error.InvalidConnectShape;
    }

    const coords = try buildCoordsFromC(&in_mesh.coords);
    const connect = try buildConnectFromC(&in_mesh.connect);

    var built = MeshInputBuilt{
        .mesh_input = .{
            .mesh_type = mesh_type,
            .coords = coords,
            .connect = connect,
            .disp = null,
            .shader = undefined,
        },
    };

    const disp_built = try buildOptionalFieldFromC(
        allocator,
        &in_mesh.disp,
    );
    built.mesh_input.disp = disp_built.field;
    built.disp_array = disp_built.array;

    const bits = try bitsFromC(in_mesh.bits);
    const scaling = try scaleStrategyFromC(
        in_mesh.scaling_tag,
        in_mesh.scaling_min,
        in_mesh.scaling_max,
    );
    const normal_type = try normalTypeFromC(in_mesh.normal_type);
    if (in_mesh.texture_storage > 2) return error.InvalidTextureStorage;

    switch (in_mesh.shader_tag) {
        0 => {
            if (in_mesh.uvs.cols_num != 2) {
                return error.InvalidUVShape;
            }

            var uvs_array = try buildArray2DF64(
                allocator,
                &in_mesh.uvs,
                2,
            );
            errdefer uvs_array.deinit(allocator);

            const samp_cfg = texops.TexSampConfig{
                .sample = try texSampleFromC(in_mesh.sample),
                .mode = try texSampleModeFromC(in_mesh.sample_mode),
            };
            if (!samp_cfg.isValid()) {
                return error.InvalidTexSampleConfig;
            }
            built.uvs_array = uvs_array;
            if (in_mesh.texture_storage == 1) {
                var tex_array = try buildTexArray(
                    u16,
                    allocator,
                    &in_mesh.tex_u16,
                    1,
                );
                errdefer tex_array.deinit(allocator);
                built.mesh_input.shader = .{ .tex_u16 = .{
                    .uvs = uvs_array,
                    .tex = texops.Tex(u16, 1){
                        .array = tex_array,
                        .rows_num = in_mesh.tex_u16.dim1,
                        .cols_num = in_mesh.tex_u16.dim2,
                    },
                    .samp_cfg = samp_cfg,
                    .bits = bits,
                    .scaling = scaling,
                    .normal_type = normal_type,
                } };
                built.tex_array_u16 = tex_array;
            } else if (in_mesh.texture_storage == 2) {
                var tex_array = try buildTexArray(F, allocator, &in_mesh.tex, 1);
                errdefer tex_array.deinit(allocator);
                built.mesh_input.shader = .{ .tex_f = .{
                    .uvs = uvs_array,
                    .tex = texops.Tex(F, 1){ .array = tex_array, .rows_num = in_mesh.tex.dim1, .cols_num = in_mesh.tex.dim2 },
                    .samp_cfg = samp_cfg,
                    .bits = bits,
                    .scaling = scaling,
                    .normal_type = normal_type,
                } };
                built.tex_array_f = tex_array;
            } else {
                var tex_array = try buildTexArray(
                    u8,
                    allocator,
                    &in_mesh.tex_u8,
                    1,
                );
                errdefer tex_array.deinit(allocator);
                built.mesh_input.shader = .{ .tex_u8 = .{
                    .uvs = uvs_array,
                    .tex = texops.Tex(u8, 1){
                        .array = tex_array,
                        .rows_num = in_mesh.tex_u8.dim1,
                        .cols_num = in_mesh.tex_u8.dim2,
                    },
                    .samp_cfg = samp_cfg,
                    .bits = bits,
                    .scaling = scaling,
                    .normal_type = normal_type,
                } };
                built.tex_array_u8 = tex_array;
            }
        },
        1 => {
            if (in_mesh.uvs.cols_num != 2) {
                return error.InvalidUVShape;
            }

            var uvs_array = try buildArray2DF64(
                allocator,
                &in_mesh.uvs,
                2,
            );
            errdefer uvs_array.deinit(allocator);

            const samp_cfg = texops.TexSampConfig{
                .sample = try texSampleFromC(in_mesh.sample),
                .mode = try texSampleModeFromC(in_mesh.sample_mode),
            };
            if (!samp_cfg.isValid()) {
                return error.InvalidTexSampleConfig;
            }
            built.uvs_array = uvs_array;
            if (in_mesh.texture_storage == 1) {
                var tex_array = try buildTexArray(
                    u16,
                    allocator,
                    &in_mesh.tex_u16,
                    3,
                );
                errdefer tex_array.deinit(allocator);
                built.mesh_input.shader = .{ .tex_rgb_u16 = .{
                    .uvs = uvs_array,
                    .tex = texops.Tex(u16, 3){
                        .array = tex_array,
                        .rows_num = in_mesh.tex_u16.dim1,
                        .cols_num = in_mesh.tex_u16.dim2,
                    },
                    .samp_cfg = samp_cfg,
                    .bits = bits,
                    .scaling = scaling,
                    .normal_type = normal_type,
                } };
                built.tex_array_u16 = tex_array;
            } else if (in_mesh.texture_storage == 2) {
                var tex_array = try buildTexArray(F, allocator, &in_mesh.tex, 3);
                errdefer tex_array.deinit(allocator);
                built.mesh_input.shader = .{ .tex_rgb_f = .{
                    .uvs = uvs_array,
                    .tex = texops.Tex(F, 3){ .array = tex_array, .rows_num = in_mesh.tex.dim1, .cols_num = in_mesh.tex.dim2 },
                    .samp_cfg = samp_cfg,
                    .bits = bits,
                    .scaling = scaling,
                    .normal_type = normal_type,
                } };
                built.tex_array_f = tex_array;
            } else {
                var tex_array = try buildTexArray(
                    u8,
                    allocator,
                    &in_mesh.tex_u8,
                    3,
                );
                errdefer tex_array.deinit(allocator);
                built.mesh_input.shader = .{ .tex_rgb_u8 = .{
                    .uvs = uvs_array,
                    .tex = texops.Tex(u8, 3){
                        .array = tex_array,
                        .rows_num = in_mesh.tex_u8.dim1,
                        .cols_num = in_mesh.tex_u8.dim2,
                    },
                    .samp_cfg = samp_cfg,
                    .bits = bits,
                    .scaling = scaling,
                    .normal_type = normal_type,
                } };
                built.tex_array_u8 = tex_array;
            }
        },
        2 => {
            const nodal_built = try buildOptionalFieldFromC(
                allocator,
                &in_mesh.nodal_field,
            );
            if (nodal_built.field == null or nodal_built.array == null) {
                return error.MissingNodalField;
            }
            built.mesh_input.shader = .{ .nodal = .{
                .field = nodal_built.field.?,
                .bits = bits,
                .scaling = scaling,
                .scale_over = try scaleOverFromC(in_mesh.scale_over),
                .normal_type = normal_type,
            } };
            built.nodal_field_array = nodal_built.array;
        },
        5 => {
            const nodal_built = try buildOptionalFieldFromC(
                allocator,
                &in_mesh.nodal_field,
            );
            if (nodal_built.field == null or nodal_built.array == null) {
                return error.MissingNodalField;
            }
            if (nodal_built.field.?.getFieldsN() != 3) {
                return error.InvalidNodalRgbFieldCount;
            }
            built.mesh_input.shader = .{ .nodal = .{
                .field = nodal_built.field.?,
                .bits = bits,
                .scaling = scaling,
                .scale_over = try scaleOverFromC(in_mesh.scale_over),
                .normal_type = normal_type,
            } };
            built.nodal_field_array = nodal_built.array;
        },
        3 => {
            var uvs_array_opt: ?ndarray.NDArray(F) = null;
            if (in_mesh.uvs.rows_num > 0 and in_mesh.uvs.cols_num > 0) {
                uvs_array_opt = try buildArray2DF64(
                    allocator,
                    &in_mesh.uvs,
                    2,
                );
            }
            errdefer if (uvs_array_opt) |*uvs_array| {
                uvs_array.deinit(allocator);
            };

            const builtin = try funcShaderBuiltinFromC(
                in_mesh.func_shader_builtin,
            );
            built.mesh_input.shader = .{ .func = .{
                .uvs = uvs_array_opt,
                .coord_mode = try funcCoordModeFromC(
                    in_mesh.func_shader_coord_mode,
                ),
                .builtin = builtin,
                .params = funcShaderParamsFromC(
                    builtin,
                    in_mesh.func_shader_params,
                ),
                .bits = bits,
                .scaling = scaling,
                .normal_type = normal_type,
            } };
            built.uvs_array = uvs_array_opt;
        },
        4 => {
            var uvs_array_opt: ?ndarray.NDArray(F) = null;
            if (in_mesh.uvs.rows_num > 0 and in_mesh.uvs.cols_num > 0) {
                uvs_array_opt = try buildArray2DF64(
                    allocator,
                    &in_mesh.uvs,
                    2,
                );
            }
            errdefer if (uvs_array_opt) |*uvs_array| {
                uvs_array.deinit(allocator);
            };

            const builtin = try funcShaderBuiltinFromC(
                in_mesh.func_shader_builtin,
            );
            built.mesh_input.shader = .{ .func_rgb = .{
                .uvs = uvs_array_opt,
                .coord_mode = try funcCoordModeFromC(
                    in_mesh.func_shader_coord_mode,
                ),
                .builtin = builtin,
                .params = funcShaderParamsFromC(
                    builtin,
                    in_mesh.func_shader_params,
                ),
                .bits = bits,
                .scaling = scaling,
                .normal_type = normal_type,
            } };
            built.uvs_array = uvs_array_opt;
        },
        else => return error.InvalidShaderTag,
    }

    return built;
}

fn buildCameraInputSlice(
    allocator: std.mem.Allocator,
    in_cameras: [*c]const CCameraInput,
    cameras_len: usize,
) ![]cam.CameraInput {
    const cameras_slice = try cConstSlice(
        CCameraInput,
        in_cameras,
        cameras_len,
    );
    const cameras_out = try allocator.alloc(cam.CameraInput, cameras_len);
    for (cameras_slice, 0..) |camera_in, nn| {
        cameras_out[nn] = try buildCameraInput(&camera_in);
    }
    return cameras_out;
}

fn buildMeshInputSlice(
    allocator: std.mem.Allocator,
    in_meshes: [*c]const CMeshInput,
    meshes_len: usize,
) ![]MeshInputBuilt {
    const meshes_in = try cConstSlice(
        CMeshInput,
        in_meshes,
        meshes_len,
    );
    const meshes_out = try allocator.alloc(MeshInputBuilt, meshes_len);
    errdefer allocator.free(meshes_out);

    for (meshes_in, 0..) |mesh_in, nn| {
        meshes_out[nn] = buildMeshInput(allocator, &mesh_in) catch |err| {
            for (0..nn) |mm| {
                meshes_out[mm].deinit(allocator);
            }
            return err;
        };
    }
    return meshes_out;
}

fn extractMeshInputs(
    allocator: std.mem.Allocator,
    built_meshes: []const MeshInputBuilt,
) ![]mo.MeshInput {
    const mesh_inputs = try allocator.alloc(mo.MeshInput, built_meshes.len);
    for (built_meshes, 0..) |built_mesh, nn| {
        mesh_inputs[nn] = built_mesh.mesh_input;
    }
    return mesh_inputs;
}

fn deinitMeshInputSlice(
    allocator: std.mem.Allocator,
    built_meshes: []MeshInputBuilt,
) void {
    for (built_meshes) |*built_mesh| {
        built_mesh.deinit(allocator);
    }
    allocator.free(built_meshes);
}

fn buildRasterConfig(
    allocator: std.mem.Allocator,
    in_config: *const CRasterConfig,
) !riley.RasterConfig {
    var config = riley.RasterConfig{};
    config.render_mode = try renderModeFromC(in_config.render_mode);
    config.total_threads = @max(@as(u16, 1), in_config.total_threads);
    config.frame_batch_size_per_group =
        @max(@as(u16, 1), in_config.frame_batch_size_per_group);
    config.max_geom_jobs_in_flight_per_group =
        @max(@as(u16, 1), in_config.max_geom_jobs_in_flight_per_group);
    config.max_geom_workers_per_job =
        @max(@as(u16, 1), in_config.max_geom_workers_per_job);
    config.geom_scheduling_mode = try geometrySchedulingModeFromC(
        in_config.geom_scheduling_mode,
    );
    config.max_raster_workers_per_job =
        @max(@as(u16, 1), in_config.max_raster_workers_per_job);
    config.save_strategy = try saveStrategyFromC(in_config.save_strategy);
    config.image_save_mode = try imageSaveModeFromC(in_config.image_save_mode);
    config.hull_mode = try hullModeFromC(in_config.hull_mode);
    config.newton_seed_mode = try newtonSeedModeFromC(
        in_config.newton_seed_mode,
    );
    config.newton_seed_reuse = try newtonSeedReuseFromC(
        in_config.newton_seed_reuse,
    );
    config.report = try reportModeFromC(in_config.report);
    config.tile_size_min = if (in_config.tile_size_min == 0)
        config.tile_size_min
    else
        in_config.tile_size_min;
    config.tile_size_max = if (in_config.tile_size_max == 0)
        config.tile_size_max
    else
        in_config.tile_size_max;
    config.background_value = in_config.background_value;
    config.disk_save_overlap = in_config.disk_save_overlap != 0;
    config.tile_size_override = if (in_config.tile_size_override == 0)
        null
    else
        in_config.tile_size_override;
    if (in_config.save_frame_buff_count != 0) {
        config.save_frame_buff_count = in_config.save_frame_buff_count;
    }

    const save_opts = try allocator.alloc(iio.ImageSaveOpts, 1);
    save_opts[0] = .{
        .format = try imageFormatFromC(in_config.save_format),
        .bits = if (in_config.save_bits == 0)
            null
        else
            @as(u8, @intCast(in_config.save_bits)),
        .scaling = try scaleStrategyFromC(
            in_config.save_scaling,
            in_config.save_scaling_min,
            in_config.save_scaling_max,
        ),
    };
    config.image_save_opts = save_opts;
    config.full_stats_opts = .{
        .save_solver_csv = in_config.full_stats_save_solver_csv != 0,
        .save_iter_map = in_config.full_stats_save_iter_map != 0,
        .save_xi_map = in_config.full_stats_save_xi_map != 0,
        .save_eta_map = in_config.full_stats_save_eta_map != 0,
        .save_conv_map = in_config.full_stats_save_conv_map != 0,
        .save_jac_det_map = in_config.full_stats_save_jac_det_map != 0,
        .save_tile_timing_map = in_config.full_stats_save_tile_timing_map != 0,
        .save_tile_density_map = in_config.full_stats_save_tile_density_map != 0,
        .save_tile_occupancy_map = in_config.full_stats_save_tile_occupancy_map != 0,
        .save_depth_map = in_config.full_stats_save_depth_map != 0,
        .save_earlyout_map = in_config.full_stats_save_earlyout_map != 0,
        .save_pixel_occupancy_map = in_config.full_stats_save_pixel_occupancy_map != 0,
        .save_normals_map = in_config.full_stats_save_normals_map != 0,
    };

    return config;
}

const RenderGroupRuntime = struct {
    managed_ios: []std.Io.Threaded,
    render_groups: []riley.RenderGroupSpec,

    fn deinit(
        self: *RenderGroupRuntime,
        allocator: std.mem.Allocator,
    ) void {
        for (self.managed_ios) |*managed_io| {
            managed_io.deinit();
        }
        allocator.free(self.managed_ios);
        allocator.free(self.render_groups);
    }
};

fn initRenderGroups(
    allocator: std.mem.Allocator,
    total_threads_in: u16,
    num_frames: usize,
) !RenderGroupRuntime {
    const total_threads = @max(@as(u16, 1), total_threads_in);
    const frames_available = @max(@as(usize, 1), num_frames);
    var render_group_count: u16 = 1;
    if (total_threads < frames_available) {
        render_group_count = total_threads;
    } else {
        for (1..frames_available + 1) |group_count| {
            if (@as(usize, total_threads) % group_count == 0) {
                render_group_count = @intCast(group_count);
            }
        }
    }
    const workers_per_group = total_threads / render_group_count;

    const managed_ios = try allocator.alloc(std.Io.Threaded, render_group_count);
    errdefer allocator.free(managed_ios);
    const render_groups = try allocator.alloc(
        riley.RenderGroupSpec,
        render_group_count,
    );
    errdefer allocator.free(render_groups);

    for (0..render_group_count) |gg| {
        managed_ios[gg] = initThreadedIo(
            allocator,
            workers_per_group,
        );
        render_groups[gg] = .{
            .io = managed_ios[gg].io(),
            .workers = workers_per_group,
        };
    }

    return .{
        .managed_ios = managed_ios,
        .render_groups = render_groups,
    };
}

fn buildImageBuff(
    allocator: std.mem.Allocator,
    in_buff: *const CImageBuffF64,
) !ndarray.NDArray(F) {
    const dims = dimsToArray(in_buff.dims);
    var elems_num: usize = 1;
    for (dims) |dim| {
        elems_num *= dim;
    }
    const image_slice = try cMutSlice(
        F,
        in_buff.elems,
        elems_num,
    );
    return try ndarray.NDArray(F).init(
        allocator,
        image_slice,
        dims[0..],
    );
}

fn initThreadedIo(
    gpa: std.mem.Allocator,
    total_threads: u16,
) std.Io.Threaded {
    const threads = @max(@as(u16, 1), total_threads);
    const limit: std.Io.Limit = if (threads <= 1)
        .nothing
    else
        .limited(threads - 1);

    return std.Io.Threaded.init(gpa, .{
        .argv0 = .empty,
        .environ = .empty,
        .async_limit = limit,
        .concurrent_limit = limit,
    });
}

fn cameraInputToC(in_camera: cam.CameraInput) CCameraInput {
    var out_camera = CCameraInput{
        .pixels_num = .{
            .x = in_camera.pixels_num[0],
            .y = in_camera.pixels_num[1],
        },
        .pixels_size = .{
            .x = in_camera.pixels_size[0],
            .y = in_camera.pixels_size[1],
        },
        .pos_world = vec3ToCVec3(in_camera.pos_world),
        .rot_world = .{
            .x = in_camera.rot_world.alpha_z,
            .y = in_camera.rot_world.beta_y,
            .z = in_camera.rot_world.gamma_x,
        },
        .roi_cent_world = vec3ToCVec3(in_camera.roi_cent_world),
        .focal_length = in_camera.focal_length,
        .sub_sample = in_camera.sub_sample,
        .distortion_model = 0,
        .distortion_k1 = 0.0,
        .distortion_k2 = 0.0,
        .distortion_k3 = 0.0,
        .distortion_k4 = 0.0,
        .distortion_k5 = 0.0,
        .distortion_k6 = 0.0,
        .distortion_p1 = 0.0,
        .distortion_p2 = 0.0,
        .distortion_poly_order = @intFromEnum(cam.PolynomialOrder.quadratic),
        .distortion_poly_has_forward = 0,
        .distortion_poly_has_inv = 0,
        .distortion_poly_forward_u = [_]F{0.0} ** 10,
        .distortion_poly_forward_v = [_]F{0.0} ** 10,
        .distortion_poly_inv_u = [_]F{0.0} ** 10,
        .distortion_poly_inv_v = [_]F{0.0} ** 10,
        .coord_sys = @intFromEnum(in_camera.coord_sys),
        .subpixel_center_map = @intFromEnum(in_camera.subpixel_center_map),
        .psf_type = 0,
        .psf_sigma_x = 0.0,
        .psf_sigma_y = 0.0,
        .psf_theta = 0.0,
        .psf_supp_rad = 0.0,
        .psf_separable = 1,
    };

    switch (in_camera.distortion) {
        .none => {},
        .brown_conrady => |model| {
            out_camera.distortion_model = 1;
            out_camera.distortion_k1 = model.k1;
            out_camera.distortion_k2 = model.k2;
            out_camera.distortion_k3 = model.k3;
            out_camera.distortion_p1 = model.p1;
            out_camera.distortion_p2 = model.p2;
        },
        .brown_conrady_ext => |model| {
            out_camera.distortion_model = 2;
            out_camera.distortion_k1 = model.k1;
            out_camera.distortion_k2 = model.k2;
            out_camera.distortion_k3 = model.k3;
            out_camera.distortion_k4 = model.k4;
            out_camera.distortion_k5 = model.k5;
            out_camera.distortion_k6 = model.k6;
            out_camera.distortion_p1 = model.p1;
            out_camera.distortion_p2 = model.p2;
        },
        .polynomial => |poly| {
            out_camera.distortion_model = 3;
            if (poly.forward_map) |forward_map| {
                out_camera.distortion_poly_order = @intFromEnum(forward_map.order);
                out_camera.distortion_poly_has_forward = 1;
                out_camera.distortion_poly_forward_u = forward_map.coeffs_u;
                out_camera.distortion_poly_forward_v = forward_map.coeffs_v;
            }
            if (poly.inv_map) |inv_map| {
                out_camera.distortion_poly_order = @intFromEnum(inv_map.order);
                out_camera.distortion_poly_has_inv = 1;
                out_camera.distortion_poly_inv_u = inv_map.coeffs_u;
                out_camera.distortion_poly_inv_v = inv_map.coeffs_v;
            }
        },
        .brown_conrady_polynomial => |chain| {
            out_camera.distortion_model = 4;
            out_camera.distortion_k1 = chain.brown_conrady.k1;
            out_camera.distortion_k2 = chain.brown_conrady.k2;
            out_camera.distortion_k3 = chain.brown_conrady.k3;
            out_camera.distortion_p1 = chain.brown_conrady.p1;
            out_camera.distortion_p2 = chain.brown_conrady.p2;
            if (chain.polynomial.forward_map) |forward_map| {
                out_camera.distortion_poly_order = @intFromEnum(forward_map.order);
                out_camera.distortion_poly_has_forward = 1;
                out_camera.distortion_poly_forward_u = forward_map.coeffs_u;
                out_camera.distortion_poly_forward_v = forward_map.coeffs_v;
            }
            if (chain.polynomial.inv_map) |inv_map| {
                out_camera.distortion_poly_order = @intFromEnum(inv_map.order);
                out_camera.distortion_poly_has_inv = 1;
                out_camera.distortion_poly_inv_u = inv_map.coeffs_u;
                out_camera.distortion_poly_inv_v = inv_map.coeffs_v;
            }
        },
        .brown_conrady_ext_polynomial => |chain| {
            out_camera.distortion_model = 5;
            out_camera.distortion_k1 = chain.brown_conrady_ext.k1;
            out_camera.distortion_k2 = chain.brown_conrady_ext.k2;
            out_camera.distortion_k3 = chain.brown_conrady_ext.k3;
            out_camera.distortion_k4 = chain.brown_conrady_ext.k4;
            out_camera.distortion_k5 = chain.brown_conrady_ext.k5;
            out_camera.distortion_k6 = chain.brown_conrady_ext.k6;
            out_camera.distortion_p1 = chain.brown_conrady_ext.p1;
            out_camera.distortion_p2 = chain.brown_conrady_ext.p2;
            if (chain.polynomial.forward_map) |forward_map| {
                out_camera.distortion_poly_order = @intFromEnum(forward_map.order);
                out_camera.distortion_poly_has_forward = 1;
                out_camera.distortion_poly_forward_u = forward_map.coeffs_u;
                out_camera.distortion_poly_forward_v = forward_map.coeffs_v;
            }
            if (chain.polynomial.inv_map) |inv_map| {
                out_camera.distortion_poly_order = @intFromEnum(inv_map.order);
                out_camera.distortion_poly_has_inv = 1;
                out_camera.distortion_poly_inv_u = inv_map.coeffs_u;
                out_camera.distortion_poly_inv_v = inv_map.coeffs_v;
            }
        },
    }

    switch (in_camera.psf) {
        .pixel_box => {
            out_camera.psf_type = 0;
            out_camera.psf_separable = 1;
        },
        .gaussian => |g| {
            out_camera.psf_type = 1;
            out_camera.psf_sigma_x = g.sigma_px;
            out_camera.psf_supp_rad = g.supp_rad_px;
            out_camera.psf_separable = if (g.separable == .yes)
                @as(u32, 1)
            else
                @as(u32, 0);
        },
        .anisotropic_gaussian => |ag| {
            out_camera.psf_type = 2;
            out_camera.psf_sigma_x = ag.sigma_x_px;
            out_camera.psf_sigma_y = ag.sigma_y_px;
            out_camera.psf_theta = ag.theta_rad;
            out_camera.psf_supp_rad = ag.supp_rad_px;
            out_camera.psf_separable = if (ag.separable == .yes)
                @as(u32, 1)
            else
                @as(u32, 0);
        },
    }
    return out_camera;
}

fn rasterSceneInternal(
    allocator: std.mem.Allocator,
    in_meshes: [*c]const CMeshInput,
    meshes_len: usize,
    in_cameras: [*c]const CCameraInput,
    cameras_len: usize,
    in_config: *const CRasterConfig,
    out_dir_path: ?[*:0]const u8,
    out_image: ?*CImageBuffF64,
) !void {
    const built_meshes = try buildMeshInputSlice(
        allocator,
        in_meshes,
        meshes_len,
    );
    defer deinitMeshInputSlice(allocator, built_meshes);

    const mesh_inputs = try extractMeshInputs(allocator, built_meshes);
    defer allocator.free(mesh_inputs);

    const camera_inputs = try buildCameraInputSlice(
        allocator,
        in_cameras,
        cameras_len,
    );
    defer allocator.free(camera_inputs);

    const raster_config = try buildRasterConfig(allocator, in_config);

    var image_arr_opt: ?ndarray.NDArray(F) = null;
    if (out_image) |image_buff| {
        image_arr_opt = try buildImageBuff(allocator, image_buff);
    }
    defer if (image_arr_opt) |*image_arr| {
        image_arr.deinit(allocator);
    };

    var render_group_runtime = try initRenderGroups(
        std.heap.smp_allocator,
        raster_config.total_threads,
        cameras_len,
    );
    defer render_group_runtime.deinit(std.heap.smp_allocator);

    const out_dir_path_slice = if (out_dir_path) |path|
        std.mem.span(path)
    else
        null;

    try riley.rasterInto(
        std.heap.smp_allocator,
        render_group_runtime.render_groups,
        camera_inputs,
        mesh_inputs,
        raster_config,
        out_dir_path_slice,
        if (image_arr_opt) |*image_arr| image_arr else null,
    );
}

pub export fn rileyRoiCentFromCoords(
    in_coords: *const CArray2DF64,
    out_cent: *CVec3F64,
) c_int {
    clearLastError();

    const coords = buildCoordsFromC(in_coords) catch |err| {
        setLastError(err);
        return 1;
    };
    const roi_cent = sceneops.boundsCenter(&coords);
    out_cent.* = vec3ToCVec3(roi_cent);
    return 0;
}

pub export fn rileyPosFillFrameFromRot(
    in_coords: *const CArray2DF64,
    pixels_num: CVec2U32,
    pixels_size: CVec2F64,
    focal_length: F,
    rot_world: CVec3F64,
    frame_fill: F,
    out_pos: *CVec3F64,
) c_int {
    clearLastError();

    const coords = buildCoordsFromC(in_coords) catch |err| {
        setLastError(err);
        return 1;
    };
    const rot = rotation.Rotation.init(
        rot_world.x,
        rot_world.y,
        rot_world.z,
    );
    const cam_pos = cameraops.posFillFrameFromRot(
        &coords,
        .{ pixels_num.x, pixels_num.y },
        .{ pixels_size.x, pixels_size.y },
        focal_length,
        rot,
        frame_fill,
    );
    out_pos.* = vec3ToCVec3(cam_pos);
    return 0;
}

pub export fn rileyRoiCentOverMeshes(
    in_meshes: [*c]const CMeshInput,
    meshes_len: usize,
    out_cent: *CVec3F64,
) c_int {
    clearLastError();

    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const built_meshes = buildMeshInputSlice(
        aa,
        in_meshes,
        meshes_len,
    ) catch |err| {
        setLastError(err);
        return 1;
    };
    defer deinitMeshInputSlice(aa, built_meshes);

    const mesh_inputs = extractMeshInputs(aa, built_meshes) catch |err| {
        setLastError(err);
        return 1;
    };

    const roi_cent = sceneops.boundsCenterOverMeshes(mesh_inputs);
    out_cent.* = vec3ToCVec3(roi_cent);
    return 0;
}

pub export fn rileyPosFillFrameFromRotOverMeshes(
    in_meshes: [*c]const CMeshInput,
    meshes_len: usize,
    pixels_num: CVec2U32,
    pixels_size: CVec2F64,
    focal_length: F,
    rot_world: CVec3F64,
    frame_fill: F,
    out_pos: *CVec3F64,
) c_int {
    clearLastError();

    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const built_meshes = buildMeshInputSlice(
        aa,
        in_meshes,
        meshes_len,
    ) catch |err| {
        setLastError(err);
        return 1;
    };
    defer deinitMeshInputSlice(aa, built_meshes);

    const mesh_inputs = extractMeshInputs(aa, built_meshes) catch |err| {
        setLastError(err);
        return 1;
    };

    const rot = rotation.Rotation.init(
        rot_world.x,
        rot_world.y,
        rot_world.z,
    );
    const cam_pos = cameraops.posFillFrameFromRotOverMeshes(
        mesh_inputs,
        .{ pixels_num.x, pixels_num.y },
        .{ pixels_size.x, pixels_size.y },
        focal_length,
        rot,
        frame_fill,
    );
    out_pos.* = vec3ToCVec3(cam_pos);
    return 0;
}

pub export fn rileyCalcOutputDimsScene(
    in_meshes: [*c]const CMeshInput,
    meshes_len: usize,
    in_cameras: [*c]const CCameraInput,
    cameras_len: usize,
    in_config: *const CRasterConfig,
    out_dims: *CDims5Usize,
) c_int {
    clearLastError();

    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const built_meshes = buildMeshInputSlice(
        aa,
        in_meshes,
        meshes_len,
    ) catch |err| {
        setLastError(err);
        return 1;
    };
    defer deinitMeshInputSlice(aa, built_meshes);

    const mesh_inputs = extractMeshInputs(aa, built_meshes) catch |err| {
        setLastError(err);
        return 1;
    };
    const camera_inputs = buildCameraInputSlice(
        aa,
        in_cameras,
        cameras_len,
    ) catch |err| {
        setLastError(err);
        return 1;
    };

    const raster_config = buildRasterConfig(aa, in_config) catch |err| {
        setLastError(err);
        return 1;
    };

    const dims = riley.calcAllFramesImageDims(
        camera_inputs,
        mesh_inputs,
        raster_config,
    ) catch |err| {
        setLastError(err);
        return 1;
    };
    out_dims.* = dimsFromArray(dims);
    return 0;
}

pub export fn rileySaveCamera(
    out_dir_path: [*:0]const u8,
    file_name: [*:0]const u8,
    camera_idx: usize,
    camera_in: *const CCameraInput,
) c_int {
    clearLastError();

    const camera = buildCameraInput(camera_in) catch |err| {
        setLastError(err);
        return 1;
    };

    var threaded_io = initThreadedIo(std.heap.smp_allocator, 1);
    defer threaded_io.deinit();
    const io = threaded_io.io();
    const cwd = std.Io.Dir.cwd();
    cwd.createDir(io, std.mem.span(out_dir_path), .default_dir) catch |err| {
        if (err != error.PathAlreadyExists) {
            setLastError(err);
            return 1;
        }
    };

    var out_dir = cwd.openDir(
        io,
        std.mem.span(out_dir_path),
        .{},
    ) catch |err| {
        setLastError(err);
        return 1;
    };
    defer out_dir.close(io);

    cameraio.saveCamera(
        io,
        out_dir,
        std.mem.span(file_name),
        camera_idx,
        camera,
    ) catch |err| {
        setLastError(err);
        return 1;
    };
    return 0;
}

pub export fn rileyLoadCamera(
    dir_path: [*:0]const u8,
    file_name: [*:0]const u8,
    camera_out: *CCameraInput,
) c_int {
    clearLastError();

    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();

    var threaded_io = initThreadedIo(std.heap.smp_allocator, 1);
    defer threaded_io.deinit();
    const io = threaded_io.io();
    const cwd = std.Io.Dir.cwd();

    var dir = cwd.openDir(
        io,
        std.mem.span(dir_path),
        .{},
    ) catch |err| {
        setLastError(err);
        return 1;
    };
    defer dir.close(io);

    const camera = cameraio.loadCamera(
        arena.allocator(),
        io,
        dir,
        std.mem.span(file_name),
    ) catch |err| {
        setLastError(err);
        return 1;
    };
    camera_out.* = cameraInputToC(camera);
    return 0;
}

pub export fn rileySaveStereoPair(
    out_dir_path: [*:0]const u8,
    stereo_file_name: [*:0]const u8,
    cam0_in: *const CCameraInput,
    cam1_in: *const CCameraInput,
) c_int {
    clearLastError();

    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const cam0 = buildCameraInput(cam0_in) catch |err| {
        setLastError(err);
        return 1;
    };
    const cam1 = buildCameraInput(cam1_in) catch |err| {
        setLastError(err);
        return 1;
    };

    var threaded_io = initThreadedIo(std.heap.smp_allocator, 1);
    defer threaded_io.deinit();
    const io = threaded_io.io();
    const cwd = std.Io.Dir.cwd();
    cwd.createDir(io, std.mem.span(out_dir_path), .default_dir) catch |err| {
        if (err != error.PathAlreadyExists) {
            setLastError(err);
            return 1;
        }
    };

    var out_dir = cwd.openDir(
        io,
        std.mem.span(out_dir_path),
        .{},
    ) catch |err| {
        setLastError(err);
        return 1;
    };
    defer out_dir.close(io);

    cameraio.saveStereoPair(
        io,
        out_dir,
        std.mem.span(stereo_file_name),
        .{ .cameras = .{ cam0, cam1 } },
    ) catch |err| {
        _ = aa;
        setLastError(err);
        return 1;
    };
    return 0;
}

pub export fn rileyLoadStereoPair(
    dir_path: [*:0]const u8,
    stereo_file_name: [*:0]const u8,
    cam0_out: *CCameraInput,
    cam1_out: *CCameraInput,
) c_int {
    clearLastError();

    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    var threaded_io = initThreadedIo(std.heap.smp_allocator, 1);
    defer threaded_io.deinit();
    const io = threaded_io.io();
    const cwd = std.Io.Dir.cwd();

    var dir = cwd.openDir(
        io,
        std.mem.span(dir_path),
        .{},
    ) catch |err| {
        setLastError(err);
        return 1;
    };
    defer dir.close(io);

    const stereo_pair = cameraio.loadStereoPair(
        aa,
        io,
        dir,
        std.mem.span(stereo_file_name),
    ) catch |err| {
        setLastError(err);
        return 1;
    };
    cam0_out.* = cameraInputToC(stereo_pair.cameras[0]);
    cam1_out.* = cameraInputToC(stereo_pair.cameras[1]);
    return 0;
}

pub export fn rileyRaster(
    in_meshes: [*c]const CMeshInput,
    meshes_len: usize,
    in_cameras: [*c]const CCameraInput,
    cameras_len: usize,
    in_config: *const CRasterConfig,
    out_dir_path: ?[*:0]const u8,
    out_image: ?*CImageBuffF64,
) c_int {
    clearLastError();

    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();

    rasterSceneInternal(
        arena.allocator(),
        in_meshes,
        meshes_len,
        in_cameras,
        cameras_len,
        in_config,
        out_dir_path,
        out_image,
    ) catch |err| {
        setLastError(err);
        return 1;
    };

    return 0;
}
