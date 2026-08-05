// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const ndarray = @import("ndarray.zig");
const buildconfig = @import("buildconfig.zig");
const cam = @import("camera.zig");
const mo = @import("meshpipeline.zig");
const rastcfg = @import("rasterconfig.zig");
const report = @import("report.zig");

const F = buildconfig.F;

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const ValidSummary = struct {
    num_time: usize,
    raw_num_fields: u8,
    out_num_fields: u8,
    img_dims: [5]usize,
};

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn checkRenderInpsErr(
    render_groups: anytype,
    cam_inps: []const cam.CameraInput,
    meshes: []const mo.MeshInput,
    config: rastcfg.RasterConfig,
    imgs_arr: ?*ndarray.NDArray(F),
    require_out_buff: bool,
    bench_capt: ?[]report.FrameBenchCapture,
) !ValidSummary {
    if (render_groups.len == 0) {
        return error.NoRenderGroups;
    }
    if (cam_inps.len == 0) {
        return error.NoCameras;
    }
    if (meshes.len == 0) {
        return error.NoMeshes;
    }
    for (meshes) |mesh| {
        if (mesh.mesh_type == .tri3opt) {
            for (cam_inps) |cam_inp| {
                if (!cam.isNoDistortion(cam_inp.distortion)) {
                    return error.DistortionNotSuppedWithTri3Opt;
                }
            }
        }
    }
    for (render_groups) |render_group| {
        if (render_group.workers == 0) {
            return error.InvalidRenderGroupWorkers;
        }
    }

    if (config.total_threads == 0) return error.InvalidTotalThreads;
    if (config.frame_batch_size_per_group == 0) return error.InvalidFrameBatchSize;
    if (config.max_geom_jobs_in_flight_per_group == 0) return error.InvalidGeomJobsInFlight;
    if (config.max_geom_workers_per_job == 0) return error.InvalidGeomWorkersPerJob;
    if (config.max_raster_workers_per_job == 0) return error.InvalidRasterWorkersPerJob;
    if (config.tile_size_min == 0) return error.InvalidTileSizeMin;
    if (config.tile_size_max == 0) return error.InvalidTileSizeMax;
    if (config.tile_size_min > config.tile_size_max) return error.InvalidTileSizeRange;
    if (config.tile_size_override) |tile_size_override| {
        if (tile_size_override < config.tile_size_min or
            tile_size_override > config.tile_size_max)
        {
            return error.InvalidTileSizeOverride;
        }
    }
    if (!std.math.isFinite(config.background_value)) {
        return error.InvalidBackgroundValue;
    }
    if ((config.save_strategy == .disk or config.save_strategy == .both) and
        config.image_save_opts.len == 0)
    {
        return error.InvalidImageSaveOpts;
    }
    if (config.report == .full_stats and config.full_stats_opts.formats.len == 0) {
        return error.InvalidFullStatsFormats;
    }

    for (cam_inps) |cam_inp| {
        try checkCamInpErr(cam_inp);
    }

    const num_time = mo.countFrames(meshes);
    if (num_time == 0) {
        return error.NoMeshFrames;
    }

    const raw_num_fields = mo.countOutputFields(meshes);
    if (raw_num_fields == 0) {
        return error.NoOutputFields;
    }

    if (bench_capt) |capt| {
        if (capt.len != cam_inps.len * num_time) {
            return error.InvalidBenchCaptureBuff;
        }
    }

    const out_num_fields = try calcOutFieldsForImgSaveMode(
        config.image_save_mode,
        raw_num_fields,
    );
    const img_dims = calcAllFramesImgDims(
        cam_inps,
        num_time,
        out_num_fields,
    );
    try validOutBuffErr(
        config,
        imgs_arr,
        require_out_buff,
        img_dims,
    );

    return .{
        .num_time = num_time,
        .raw_num_fields = raw_num_fields,
        .out_num_fields = out_num_fields,
        .img_dims = img_dims,
    };
}

pub fn checkRenderInpsAssert(
    render_groups: anytype,
    cam_inps: []const cam.CameraInput,
    meshes: []const mo.MeshInput,
    config: rastcfg.RasterConfig,
    imgs_arr: ?*ndarray.NDArray(F),
    require_out_buff: bool,
    bench_capt: ?[]report.FrameBenchCapture,
) ValidSummary {
    std.debug.assert(render_groups.len > 0);
    std.debug.assert(cam_inps.len > 0);
    std.debug.assert(meshes.len > 0);
    for (render_groups) |render_group| {
        std.debug.assert(render_group.workers > 0);
    }

    std.debug.assert(config.total_threads > 0);
    std.debug.assert(config.frame_batch_size_per_group > 0);
    std.debug.assert(config.max_geom_jobs_in_flight_per_group > 0);
    std.debug.assert(config.max_geom_workers_per_job > 0);
    std.debug.assert(config.max_raster_workers_per_job > 0);
    std.debug.assert(config.tile_size_min > 0);
    std.debug.assert(config.tile_size_max > 0);
    std.debug.assert(config.tile_size_min <= config.tile_size_max);
    if (config.tile_size_override) |tile_size_override| {
        std.debug.assert(tile_size_override >= config.tile_size_min);
        std.debug.assert(tile_size_override <= config.tile_size_max);
    }
    std.debug.assert(std.math.isFinite(config.background_value));
    if (config.save_strategy == .disk or config.save_strategy == .both) {
        std.debug.assert(config.image_save_opts.len > 0);
    }
    if (config.report == .full_stats) {
        std.debug.assert(config.full_stats_opts.formats.len > 0);
    }

    for (cam_inps) |cam_inp| {
        checkCamInpAssert(cam_inp);
    }

    const num_time = mo.countFrames(meshes);
    const raw_num_fields = mo.countOutputFields(meshes);
    const out_num_fields = calcOutFieldsForImgSaveMode(
        config.image_save_mode,
        raw_num_fields,
    ) catch unreachable;
    const img_dims = calcAllFramesImgDims(
        cam_inps,
        num_time,
        out_num_fields,
    );
    std.debug.assert(num_time > 0);
    std.debug.assert(raw_num_fields > 0);

    if (bench_capt) |capt| {
        std.debug.assert(capt.len == cam_inps.len * num_time);
    }
    validOutBuffAssert(config, imgs_arr, require_out_buff, img_dims);

    return .{
        .num_time = num_time,
        .raw_num_fields = raw_num_fields,
        .out_num_fields = out_num_fields,
        .img_dims = img_dims,
    };
}

// --------------------------------------------------------------------------------------
// Generic Low-Level Helpers
// --------------------------------------------------------------------------------------

pub fn calcOutFieldsForImgSaveMode(
    img_save_mode: rastcfg.ImageSaveMode,
    raw_num_fields: u8,
) !u8 {
    return switch (img_save_mode) {
        .multifield => raw_num_fields,
        .grey => switch (raw_num_fields) {
            1, 3 => 1,
            else => error.UnsuppedImageModeFieldCount,
        },
        .rgb => switch (raw_num_fields) {
            1, 3 => 3,
            else => error.UnsuppedImageModeFieldCount,
        },
    };
}

fn calcAllFramesImgDims(
    cam_inps: []const cam.CameraInput,
    num_time: usize,
    out_num_fields: u8,
) [5]usize {
    std.debug.assert(cam_inps.len > 0);

    var max_pix_num = cam_inps[0].pixels_num;
    for (cam_inps[1..]) |cam_inp| {
        max_pix_num[0] = @max(max_pix_num[0], cam_inp.pixels_num[0]);
        max_pix_num[1] = @max(max_pix_num[1], cam_inp.pixels_num[1]);
    }

    return .{
        cam_inps.len,
        num_time,
        @as(usize, out_num_fields),
        max_pix_num[1],
        max_pix_num[0],
    };
}

fn validOutBuffErr(
    config: rastcfg.RasterConfig,
    imgs_arr: ?*ndarray.NDArray(F),
    require_out_buff: bool,
    exp_dims: [5]usize,
) !void {
    if (config.save_strategy == .memory or config.save_strategy == .both) {
        if (imgs_arr) |imgs_arr_req| {
            try validAllFramesBuff(imgs_arr_req, exp_dims);
        } else if (require_out_buff) {
            return error.InvalidOutputBuff;
        }
    } else if (imgs_arr != null) {
        return error.InvalidOutputBuff;
    }
}

fn validOutBuffAssert(
    config: rastcfg.RasterConfig,
    imgs_arr: ?*ndarray.NDArray(F),
    require_out_buff: bool,
    exp_dims: [5]usize,
) void {
    if (config.save_strategy == .memory or config.save_strategy == .both) {
        if (imgs_arr) |imgs_arr_req| {
            validAllFramesBuff(imgs_arr_req, exp_dims) catch unreachable;
        } else {
            std.debug.assert(!require_out_buff);
        }
    } else {
        std.debug.assert(imgs_arr == null);
    }
}

fn validAllFramesBuff(
    imgs_arr: *const ndarray.NDArray(F),
    exp_dims: [5]usize,
) !void {
    if (imgs_arr.dims.len != exp_dims.len) {
        return error.InvalidOutputBuff;
    }
    for (exp_dims, 0..) |exp_dim, dd| {
        if (imgs_arr.dims[dd] != exp_dim) {
            return error.InvalidOutputBuff;
        }
    }
}

fn isFiniteSlice(vals: []const F) bool {
    for (vals) |value| {
        if (!std.math.isFinite(value)) {
            return false;
        }
    }
    return true;
}

fn isFiniteVec3(vec: anytype) bool {
    return isFiniteSlice(vec.slice[0..]);
}

fn isValidPolynomialMap(map: cam.PolynomialMap) bool {
    const term_count = map.order.termCount();
    return isFiniteSlice(map.coeffs_u[0..term_count]) and
        isFiniteSlice(map.coeffs_v[0..term_count]);
}

fn isValidBidirectionalPolynomial(poly: cam.BidirectionalPolynomial) bool {
    if (poly.forward_map == null and poly.inv_map == null) {
        return false;
    }
    if (poly.forward_map) |forward_map| {
        if (!isValidPolynomialMap(forward_map)) return false;
    }
    if (poly.inv_map) |inv_map| {
        if (!isValidPolynomialMap(inv_map)) return false;
    }
    return true;
}

fn isValidDistortion(distortion: cam.DistortionModel) bool {
    return switch (distortion) {
        .none => true,
        .brown_conrady => |bc| isFiniteSlice(&[_]F{
            bc.k1,
            bc.k2,
            bc.k3,
            bc.p1,
            bc.p2,
        }),
        .brown_conrady_ext => |bc| isFiniteSlice(&[_]F{
            bc.k1,
            bc.k2,
            bc.k3,
            bc.k4,
            bc.k5,
            bc.k6,
            bc.p1,
            bc.p2,
        }),
        .polynomial => |poly| isValidBidirectionalPolynomial(poly),
        .brown_conrady_polynomial => |chain| isFiniteSlice(&[_]F{
            chain.brown_conrady.k1,
            chain.brown_conrady.k2,
            chain.brown_conrady.k3,
            chain.brown_conrady.p1,
            chain.brown_conrady.p2,
        }) and isValidBidirectionalPolynomial(chain.polynomial),
        .brown_conrady_ext_polynomial => |chain| isFiniteSlice(&[_]F{
            chain.brown_conrady_ext.k1,
            chain.brown_conrady_ext.k2,
            chain.brown_conrady_ext.k3,
            chain.brown_conrady_ext.k4,
            chain.brown_conrady_ext.k5,
            chain.brown_conrady_ext.k6,
            chain.brown_conrady_ext.p1,
            chain.brown_conrady_ext.p2,
        }) and isValidBidirectionalPolynomial(chain.polynomial),
    };
}

fn isValidPsf(psf: cam.PointSpreadFunc) bool {
    return switch (psf) {
        .pixel_box => |box| std.math.isFinite(box.supp_rad_px) and
            box.supp_rad_px >= 0.0,
        .gaussian => |gauss| isFiniteSlice(&[_]F{
            gauss.sigma_px,
            gauss.supp_rad_px,
        }) and
            gauss.sigma_px > 0.0 and
            gauss.supp_rad_px >= 0.0,
        .anisotropic_gaussian => |gauss| isFiniteSlice(&[_]F{
            gauss.sigma_x_px,
            gauss.sigma_y_px,
            gauss.theta_rad,
            gauss.supp_rad_px,
        }) and
            gauss.sigma_x_px > 0.0 and
            gauss.sigma_y_px > 0.0 and
            gauss.supp_rad_px >= 0.0,
    };
}

fn checkCamInpErr(cam_inp: cam.CameraInput) !void {
    if (cam_inp.pixels_num[0] == 0 or cam_inp.pixels_num[1] == 0) {
        return error.InvalidCameraPixels;
    }
    if (!isFiniteSlice(&cam_inp.pixels_size) or
        cam_inp.pixels_size[0] <= 0.0 or
        cam_inp.pixels_size[1] <= 0.0)
    {
        return error.InvalidCameraPixelSize;
    }
    if (!std.math.isFinite(cam_inp.focal_length) or
        cam_inp.focal_length <= 0.0)
    {
        return error.InvalidCameraFocalLength;
    }
    if (cam_inp.sub_sample == 0) {
        return error.InvalidCameraSubSample;
    }
    if (!isFiniteVec3(cam_inp.pos_world) or
        !isFiniteVec3(cam_inp.roi_cent_world) or
        !isFiniteSlice(&[_]F{
            cam_inp.rot_world.alpha_z,
            cam_inp.rot_world.beta_y,
            cam_inp.rot_world.gamma_x,
        }) or
        !isFiniteSlice(cam_inp.rot_world.matrix.slice[0..]))
    {
        return error.NonFiniteCameraInput;
    }
    if (!isValidDistortion(cam_inp.distortion)) {
        return error.InvalidCameraDistortion;
    }
    if (!isValidPsf(cam_inp.psf)) {
        return error.InvalidCameraPsf;
    }
}

fn checkCamInpAssert(cam_inp: cam.CameraInput) void {
    std.debug.assert(cam_inp.pixels_num[0] > 0);
    std.debug.assert(cam_inp.pixels_num[1] > 0);
    std.debug.assert(isFiniteSlice(&cam_inp.pixels_size));
    std.debug.assert(cam_inp.pixels_size[0] > 0.0);
    std.debug.assert(cam_inp.pixels_size[1] > 0.0);
    std.debug.assert(std.math.isFinite(cam_inp.focal_length));
    std.debug.assert(cam_inp.focal_length > 0.0);
    std.debug.assert(cam_inp.sub_sample > 0);
    std.debug.assert(isFiniteVec3(cam_inp.pos_world));
    std.debug.assert(isFiniteVec3(cam_inp.roi_cent_world));
    std.debug.assert(isFiniteSlice(&[_]F{
        cam_inp.rot_world.alpha_z,
        cam_inp.rot_world.beta_y,
        cam_inp.rot_world.gamma_x,
    }));
    std.debug.assert(isFiniteSlice(cam_inp.rot_world.matrix.slice[0..]));
    std.debug.assert(isValidDistortion(cam_inp.distortion));
    std.debug.assert(isValidPsf(cam_inp.psf));
}
