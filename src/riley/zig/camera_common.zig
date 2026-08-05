// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");

const vec = @import("vecstack.zig");
const matrix = @import("matstack.zig");
const rotation = @import("rotation.zig");
const ndarray = @import("ndarray.zig");
const buildconfig = @import("buildconfig.zig");
const F = buildconfig.F;
const cm = @import("cameramodels.zig");
const camera_scalar = @import("camera_scalar.zig");
const camera_simd = @import("camera_simd.zig");

const cfg = buildconfig.config;
const camera_impl = if (cfg.simd == .on) camera_simd else camera_scalar;

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const CameraInput = struct {
    pixels_num: [2]u32,
    pixels_size: [2]F,
    pos_world: vec.Vec3f,
    rot_world: rotation.Rotation,
    roi_cent_world: vec.Vec3f,
    focal_length: F,
    sub_sample: u32,
    distortion: cm.DistortionModel = .none,
    psf: cm.PointSpreadFunc = .{ .pixel_box = .{} },
    coord_sys: CameraCoordSys = .opengl,
    subpixel_center_map: SubPixelCenterMap = .per_tile,
};

pub const StereoPairInput = struct {
    cameras: [2]CameraInput,
};

pub const CameraCoordSys = enum {
    opengl,
    opencv,
};

pub const SubPixelCenterMap = enum {
    full_in_mem,
    per_tile,
    affine_jac,
};

pub const FOVScaling = struct {
    plane_dist: F,
    plane_size: [2]F,
    leng_per_pixel: [2]F,
    pixel_per_leng: [2]F,
};

const CameraPrepared = CameraPreparedType(camera_impl);

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub inline fn isNoDistortion(distortion: anytype) bool {
    return switch (distortion) {
        .none => true,
        .polynomial => |poly| poly.forward_map == null and poly.inv_map == null,
        .brown_conrady_polynomial => |chain| chain.polynomial.forward_map == null and
            chain.polynomial.inv_map == null and
            std.meta.eql(chain.brown_conrady, cm.BrownConrady{}),
        .brown_conrady_ext_polynomial => |chain| chain.polynomial.forward_map == null and
            chain.polynomial.inv_map == null and
            std.meta.eql(chain.brown_conrady_ext, cm.BrownConradyExt{}),
        else => false,
    };
}

pub inline fn calcPixelCenterCoord(pixel: usize) F {
    return @as(F, @floatFromInt(pixel)) + 0.5;
}

pub inline fn storeIdealPairScratch(
    ideal_pixel_centers: []F,
    scratch_idx: usize,
    ideal_x: F,
    ideal_y: F,
) void {
    const plane_stride = ideal_pixel_centers.len / 2;
    ideal_pixel_centers[scratch_idx] = ideal_x;
    ideal_pixel_centers[plane_stride + scratch_idx] = ideal_y;
}

pub inline fn getIdealXPlaneScratch(ideal_pixel_centers: []F) []F {
    const plane_stride = ideal_pixel_centers.len / 2;
    return ideal_pixel_centers[0..plane_stride];
}

pub inline fn getIdealYPlaneScratch(ideal_pixel_centers: []F) []F {
    const plane_stride = ideal_pixel_centers.len / 2;
    return ideal_pixel_centers[plane_stride .. plane_stride * 2];
}

pub fn allCamerasSharePixels(cameras: []const CameraPrepared) bool {
    std.debug.assert(cameras.len != 0);

    const pixels_num = cameras[0].pixels_num;
    for (cameras[1..]) |camera| {
        if (!std.meta.eql(camera.pixels_num, pixels_num)) {
            return false;
        }
    }
    return true;
}

// --------------------------------------------------------------------------------------
// Major Internal Types
// --------------------------------------------------------------------------------------

pub fn CameraPreparedType(comptime CameraBackend: type) type {
    return struct {
        const Self = @This();

        pixels_num: [2]u32,
        pixels_size: [2]F,
        pos_world: vec.Vec3f,
        rot_world: rotation.Rotation,
        roi_cent_world: vec.Vec3f,
        focal_length: F,
        sub_sample: u32,
        sensor_size: [2]F,
        image_dims: [2]F,
        image_dist: F,
        cam_to_world_mat: matrix.Mat44f,
        world_to_cam_mat: matrix.Mat44f,
        distortion: cm.DistortionModel,
        psf: cm.PointSpreadFunc,
        prep_psf: cm.PreparedPSF,
        coord_sys: CameraCoordSys,
        ideal_pixel_centers: ndarray.NDArray(F),
        pixel_center_jac: ndarray.NDArray(F),
        subpixel_center_map: SubPixelCenterMap,

        pub fn init(
            allocator: std.mem.Allocator,
            input: CameraInput,
        ) !Self {
            const subpixel_center_map = input.subpixel_center_map;
            const actual_sub_sample = if (input.sub_sample == 0)
                @as(u32, 2)
            else
                input.sub_sample;
            const sensor_size = calcSensorSize(
                input.pixels_num,
                input.pixels_size,
            );

            const pos_w = input.pos_world;
            const rot_matrix = input.rot_world.matrix;
            const image_dist: F = pos_w.sub(input.roi_cent_world).vecLen();

            const image_dims = [2]F{
                (image_dist / input.focal_length) * sensor_size[0],
                (image_dist / input.focal_length) * sensor_size[1],
            };

            var cam_to_world_mat = matrix.Mat44f.initIdentity();
            cam_to_world_mat.insertColVec(3, 0, 3, pos_w);
            cam_to_world_mat.insertSubMat(0, 0, 3, 3, rot_matrix);

            const world_to_cam_mat = matrix.Mat44Ops.inv(F, cam_to_world_mat);

            const ideal_pixel_centers = switch (subpixel_center_map) {
                .full_in_mem => blk: {
                    const sub_samp_u: usize = @intCast(actual_sub_sample);
                    const dims = [_]usize{
                        input.pixels_num[1] * sub_samp_u,
                        input.pixels_num[0] * sub_samp_u,
                        2,
                    };
                    break :blk try ndarray.NDArray(F).initFlat(
                        allocator,
                        dims[0..],
                    );
                },
                else => blk: {
                    const dims = [_]usize{ 0, 0, 2 };
                    break :blk try ndarray.NDArray(F).initFlat(
                        allocator,
                        dims[0..],
                    );
                },
            };
            const pixel_center_jac = switch (subpixel_center_map) {
                .affine_jac => blk: {
                    const dims = [_]usize{
                        input.pixels_num[1],
                        input.pixels_num[0],
                        6,
                    };
                    break :blk try ndarray.NDArray(F).initFlat(
                        allocator,
                        dims[0..],
                    );
                },
                else => blk: {
                    const dims = [_]usize{ 0, 0, 6 };
                    break :blk try ndarray.NDArray(F).initFlat(
                        allocator,
                        dims[0..],
                    );
                },
            };

            var self = Self{
                .pixels_num = input.pixels_num,
                .pixels_size = input.pixels_size,
                .pos_world = input.pos_world,
                .rot_world = input.rot_world,
                .roi_cent_world = input.roi_cent_world,
                .focal_length = input.focal_length,
                .sub_sample = actual_sub_sample,
                .sensor_size = sensor_size,
                .image_dims = image_dims,
                .image_dist = image_dist,
                .cam_to_world_mat = cam_to_world_mat,
                .world_to_cam_mat = world_to_cam_mat,
                .distortion = input.distortion,
                .psf = input.psf,
                .prep_psf = try cm.preparePSF(
                    allocator,
                    input.psf,
                    actual_sub_sample,
                ),
                .coord_sys = input.coord_sys,
                .ideal_pixel_centers = ideal_pixel_centers,
                .pixel_center_jac = pixel_center_jac,
                .subpixel_center_map = subpixel_center_map,
            };

            switch (subpixel_center_map) {
                .full_in_mem => try self.initFullIdealPixelCenters(),
                .affine_jac => try CameraBackend.initPixelCenterJac(&self),
                .per_tile => {},
            }

            return self;
        }

        pub fn deinit(
            self: *const Self,
            allocator: std.mem.Allocator,
        ) void {
            var prep_psf = self.prep_psf;
            prep_psf.deinit(allocator);
            allocator.free(self.ideal_pixel_centers.slice);
            self.ideal_pixel_centers.deinit(allocator);
            allocator.free(self.pixel_center_jac.slice);
            self.pixel_center_jac.deinit(allocator);
        }

        pub inline fn calcPinholeRasterPoint(
            self: *const Self,
            observed_x_px: F,
            observed_y_px: F,
        ) ![2]F {
            return CameraBackend.calcPinholeRasterPoint(
                self,
                observed_x_px,
                observed_y_px,
            );
        }

        pub inline fn calcPinholeRasterPointSIMD(
            self: *const Self,
            v_observed_x_px: buildconfig.VecSF,
            v_observed_y_px: buildconfig.VecSF,
            v_lane_active: buildconfig.VecSB,
        ) !struct { x: buildconfig.VecSF, y: buildconfig.VecSF } {
            return CameraBackend.calcPinholeRasterPointSIMD(
                self,
                v_observed_x_px,
                v_observed_y_px,
                v_lane_active,
            );
        }

        pub inline fn fillTileIdealCentersPerTile(
            self: *const Self,
            scratch_x_px_min: i32,
            scratch_x_px_max: i32,
            scratch_y_px_min: i32,
            scratch_y_px_max: i32,
            subpx_tile_size: usize,
            ideal_pixel_centers: []F,
        ) !void {
            return CameraBackend.fillTileIdealCentersPerTile(
                self,
                scratch_x_px_min,
                scratch_x_px_max,
                scratch_y_px_min,
                scratch_y_px_max,
                subpx_tile_size,
                ideal_pixel_centers,
            );
        }

        pub inline fn fillTileIdealCentersAffineJac(
            self: *const Self,
            scratch_x_px_min: i32,
            scratch_x_px_max: i32,
            scratch_y_px_min: i32,
            scratch_y_px_max: i32,
            subpx_tile_size: usize,
            ideal_pixel_centers: []F,
        ) void {
            CameraBackend.fillTileIdealCentersAffineJac(
                self,
                scratch_x_px_min,
                scratch_x_px_max,
                scratch_y_px_min,
                scratch_y_px_max,
                subpx_tile_size,
                ideal_pixel_centers,
            );
        }

        fn initFullIdealPixelCenters(self: *Self) !void {
            const sub_samp_u: usize = @intCast(self.sub_sample);
            const sub_samp_f = @as(F, @floatFromInt(self.sub_sample));
            const subpx_step = 1.0 / sub_samp_f;
            const subpx_off = 0.5 / sub_samp_f;

            const slice = self.ideal_pixel_centers.slice;
            const stride_y = self.ideal_pixel_centers.strides[0];
            const stride_x = self.ideal_pixel_centers.strides[1];

            for (0..self.pixels_num[1] * sub_samp_u) |jj| {
                const subpx_y_f = @as(F, @floatFromInt(jj)) *
                    subpx_step + subpx_off;
                const row_off = jj * stride_y;

                for (0..self.pixels_num[0] * sub_samp_u) |ii| {
                    const subpx_x_f = @as(F, @floatFromInt(ii)) *
                        subpx_step + subpx_off;
                    const ideal = try self.calcPinholeRasterPoint(
                        subpx_x_f,
                        subpx_y_f,
                    );
                    const col_off = ii * stride_x;
                    slice[row_off + col_off + 0] = ideal[0];
                    slice[row_off + col_off + 1] = ideal[1];
                }
            }
        }

        pub fn calcFocalPx(self: *const Self) struct { fx: F, fy: F } {
            return .{
                .fx = self.focal_length / self.pixels_size[0],
                .fy = self.focal_length / self.pixels_size[1],
            };
        }

        pub fn calcRasterOffsets(self: *const Self) struct { x_off: F, y_off: F } {
            return .{
                .x_off = 0.5 * @as(F, @floatFromInt(self.pixels_num[0])),
                .y_off = 0.5 * @as(F, @floatFromInt(self.pixels_num[1])),
            };
        }
    };
}

// --------------------------------------------------------------------------------------
// Generic Low-Level Helpers
// --------------------------------------------------------------------------------------

fn calcSensorSize(pixels_num: [2]u32, pixels_size: [2]F) [2]F {
    return .{
        @as(F, @floatFromInt(pixels_num[0])) * pixels_size[0],
        @as(F, @floatFromInt(pixels_num[1])) * pixels_size[1],
    };
}

const test_tol: F = 1e-4;
const unit_abs_tol: F = if (F == f32) 1e-4 else 1e-10;
const unit_rel_tol: F = if (F == f32) 1e-3 else 1e-6;
const sum_abs_tol: F = if (F == f32) 1e-5 else 1e-12;
const pix_num = [_]u32{ 500, 500 };
const pix_size = [_]F{ 5e-3, 5e-3 };
const foc_leng: F = 50.0;
const rotat_world = rotation.Rotation.init(
    0,
    0,
    std.math.degreesToRadians(-45),
);
const bb: F = 20.0;
const coord_n: usize = 8;
const coord_x = [_]F{ -bb, bb, bb, -bb, -bb, bb, bb, -bb };
const coord_y = [_]F{ bb, bb, -bb, -bb, bb, bb, -bb, -bb };
const coord_z = [_]F{ bb, bb, bb, bb, -bb, -bb, -bb, -bb };
const roi_world_arr = [_]F{ 0, 0, 0 };
const roi_world = vec.Vec3f.initSlice(&roi_world_arr);
const sub_samp: u32 = 2;

fn expectApproxEqRelAbs(
    expected: F,
    actual: F,
    rel_tol: F,
    abs_tol: F,
) !void {
    const diff = @abs(expected - actual);
    if (diff <= abs_tol) {
        return;
    }

    const scale = @max(@abs(expected), @abs(actual));
    if (scale == 0.0) {
        return error.TestExpectedApproxEqRelAbs;
    }
    if (diff / scale > rel_tol) {
        return error.TestExpectedApproxEqRelAbs;
    }
}

fn checkDistortionGridInv(
    distortion: anytype,
    rel_tol: F,
    abs_tol: F,
) !void {
    const grid_num = 25;
    const min_coord = -0.45;
    const max_coord = 0.45;
    const coord_step = (max_coord - min_coord) / @as(F, grid_num - 1);

    for (0..grid_num) |jj| {
        const y = min_coord + @as(F, @floatFromInt(jj)) * coord_step;
        for (0..grid_num) |ii| {
            const x = min_coord + @as(F, @floatFromInt(ii)) * coord_step;
            const distorted = distortion.forward(x, y);
            const recovered = try distortion.inv(distorted[0], distorted[1]);
            try expectApproxEqRelAbs(x, recovered.x, rel_tol, abs_tol);
            try expectApproxEqRelAbs(y, recovered.y, rel_tol, abs_tol);
        }
    }
}

// --------------------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------------------

test "CameraPrepared.init" {
    const input = CameraInput{
        .pixels_num = pix_num,
        .pixels_size = pix_size,
        .pos_world = vec.initVec3(F, 0.0, 800.0, 800.0),
        .rot_world = rotat_world,
        .roi_cent_world = roi_world,
        .focal_length = foc_leng,
        .sub_sample = sub_samp,
        .subpixel_center_map = .full_in_mem,
    };

    const camera = try CameraPrepared.init(std.testing.allocator, input);
    defer camera.deinit(std.testing.allocator);

    try std.testing.expectEqual(pix_num, camera.pixels_num);
    try std.testing.expectEqual(pix_size, camera.pixels_size);
    try std.testing.expectEqual(foc_leng, camera.focal_length);
    try std.testing.expectEqual(sub_samp, camera.sub_sample);
}

test "BrownConrady.forwardInv" {
    const bc = cm.BrownConrady{
        .k1 = -0.2,
        .k2 = 0.03,
        .k3 = -0.005,
        .p1 = 0.001,
        .p2 = -0.0015,
    };

    const x_ideal = 0.1;
    const y_ideal = -0.15;

    const distorted = bc.forward(x_ideal, y_ideal);
    const recovered = try bc.inv(distorted[0], distorted[1]);

    try std.testing.expectApproxEqAbs(x_ideal, recovered.x, unit_abs_tol);
    try std.testing.expectApproxEqAbs(y_ideal, recovered.y, unit_abs_tol);
}

test "BrownConradyExt.forwardInv" {
    const bc_ext = cm.BrownConradyExt{
        .k1 = -0.18,
        .k2 = 0.02,
        .k3 = -0.004,
        .k4 = 0.01,
        .k5 = -0.002,
        .k6 = 0.0004,
        .p1 = 0.0012,
        .p2 = -0.0018,
    };

    const x_ideal = -0.12;
    const y_ideal = 0.18;

    const distorted = bc_ext.forward(x_ideal, y_ideal);
    const recovered = try bc_ext.inv(distorted[0], distorted[1]);

    try std.testing.expectApproxEqAbs(x_ideal, recovered.x, unit_abs_tol);
    try std.testing.expectApproxEqAbs(y_ideal, recovered.y, unit_abs_tol);
}

test "Polynomial.forwardOnlyRoundTrip" {
    const model = cm.DistortionModel{
        .polynomial = .{
            .forward_map = .{
                .order = .linear,
                .coeffs_u = .{ 0.0, 0.04, -0.015 } ++ [_]F{0.0} ** 7,
                .coeffs_v = .{ 0.0, 0.01, 0.03 } ++ [_]F{0.0} ** 7,
            },
        },
    };

    const x_ideal = 0.12;
    const y_ideal = -0.18;
    const distorted = cm.forwardDistortionModelScal(
        model,
        x_ideal,
        y_ideal,
    );
    const recovered = try cm.invDistortionModelScal(
        model,
        distorted[0],
        distorted[1],
    );

    try std.testing.expectApproxEqAbs(x_ideal, recovered.x, unit_abs_tol);
    try std.testing.expectApproxEqAbs(y_ideal, recovered.y, unit_abs_tol);
}

test "Polynomial.invOnlyRoundTrip" {
    const model = cm.DistortionModel{
        .polynomial = .{
            .inv_map = .{
                .order = .linear,
                .coeffs_u = .{ 0.0, -0.03, 0.01 } ++ [_]F{0.0} ** 7,
                .coeffs_v = .{ 0.0, 0.02, -0.025 } ++ [_]F{0.0} ** 7,
            },
        },
    };

    const x_ideal = -0.16;
    const y_ideal = 0.11;
    const distorted = cm.forwardDistortionModelScal(
        model,
        x_ideal,
        y_ideal,
    );
    const recovered = try cm.invDistortionModelScal(
        model,
        distorted[0],
        distorted[1],
    );

    try std.testing.expectApproxEqAbs(x_ideal, recovered.x, unit_abs_tol);
    try std.testing.expectApproxEqAbs(y_ideal, recovered.y, unit_abs_tol);
}

test "BrownConradyPolynomial.forwardInv" {
    const model = cm.DistortionModel{
        .brown_conrady_polynomial = .{
            .brown_conrady = .{
                .k1 = -0.08,
                .k2 = 0.01,
                .k3 = -0.002,
                .p1 = 0.0004,
                .p2 = -0.0007,
            },
            .polynomial = .{
                .forward_map = .{
                    .order = .quadratic,
                    .coeffs_u = .{ 0.0, 0.01, -0.005, 0.002, 0.001, -0.001 } ++ [_]F{0.0} ** 4,
                    .coeffs_v = .{ 0.0, -0.004, 0.012, 0.001, -0.002, 0.0015 } ++ [_]F{0.0} ** 4,
                },
            },
        },
    };

    const x_ideal = 0.09;
    const y_ideal = -0.14;
    const distorted = cm.forwardDistortionModelScal(
        model,
        x_ideal,
        y_ideal,
    );
    const recovered = try cm.invDistortionModelScal(
        model,
        distorted[0],
        distorted[1],
    );

    try std.testing.expectApproxEqAbs(x_ideal, recovered.x, unit_abs_tol);
    try std.testing.expectApproxEqAbs(y_ideal, recovered.y, unit_abs_tol);
}

test "Polynomial.invOnlySIMDRoundTrip" {
    const VecSB = buildconfig.VecSB;
    const VecSF = buildconfig.VecSF;
    const lane_count = buildconfig.SimdWidth;
    const model = cm.DistortionModel{
        .polynomial = .{
            .inv_map = .{
                .order = .linear,
                .coeffs_u = .{ 0.0, -0.03, 0.01 } ++ [_]F{0.0} ** 7,
                .coeffs_v = .{ 0.0, 0.02, -0.025 } ++ [_]F{0.0} ** 7,
            },
        },
    };

    var x_ideal: [buildconfig.SimdWidth]F = [_]F{0.0} ** buildconfig.SimdWidth;
    var y_ideal: [buildconfig.SimdWidth]F = [_]F{0.0} ** buildconfig.SimdWidth;
    var x_dist: [buildconfig.SimdWidth]F = [_]F{0.0} ** buildconfig.SimdWidth;
    var y_dist: [buildconfig.SimdWidth]F = [_]F{0.0} ** buildconfig.SimdWidth;
    var active: [buildconfig.SimdWidth]bool = [_]bool{false} ** buildconfig.SimdWidth;

    for (0..lane_count) |ii| {
        x_ideal[ii] = -0.2 + 0.03 * @as(F, @floatFromInt(ii));
        y_ideal[ii] = 0.15 - 0.02 * @as(F, @floatFromInt(ii));
        const distorted = cm.forwardDistortionModelScal(
            model,
            x_ideal[ii],
            y_ideal[ii],
        );
        x_dist[ii] = distorted[0];
        y_dist[ii] = distorted[1];
        active[ii] = true;
    }

    const solved = try cm.invDistortionModelSIMD(
        model,
        @as(VecSF, x_dist),
        @as(VecSF, y_dist),
        @as(VecSB, active),
    );
    const x_solved: [buildconfig.SimdWidth]F = solved.x;
    const y_solved: [buildconfig.SimdWidth]F = solved.y;

    for (0..lane_count) |ii| {
        try std.testing.expectApproxEqAbs(x_ideal[ii], x_solved[ii], unit_abs_tol);
        try std.testing.expectApproxEqAbs(y_ideal[ii], y_solved[ii], unit_abs_tol);
    }
}

test "BrownConrady.gridInvRoundTrip" {
    const bc_mild = cm.BrownConrady{
        .k1 = -0.08,
        .k2 = 0.01,
        .k3 = -0.002,
        .p1 = 0.0004,
        .p2 = -0.0007,
    };
    const bc_strong = cm.BrownConrady{
        .k1 = -0.2,
        .k2 = 0.03,
        .k3 = -0.005,
        .p1 = 0.001,
        .p2 = -0.0015,
    };

    try checkDistortionGridInv(
        bc_mild,
        unit_rel_tol,
        unit_rel_tol,
    );
    try checkDistortionGridInv(
        bc_strong,
        unit_rel_tol,
        unit_rel_tol,
    );
}

test "BrownConradyExt.gridInvRoundTrip" {
    const bc_ext_mild = cm.BrownConradyExt{
        .k1 = -0.09,
        .k2 = 0.012,
        .k3 = -0.0015,
        .k4 = 0.004,
        .k5 = -0.0008,
        .k6 = 0.00015,
        .p1 = 0.0005,
        .p2 = -0.0006,
    };
    const bc_ext_strong = cm.BrownConradyExt{
        .k1 = -0.18,
        .k2 = 0.02,
        .k3 = -0.004,
        .k4 = 0.01,
        .k5 = -0.002,
        .k6 = 0.0004,
        .p1 = 0.0012,
        .p2 = -0.0018,
    };

    try checkDistortionGridInv(
        bc_ext_mild,
        unit_rel_tol,
        unit_rel_tol,
    );
    try checkDistortionGridInv(
        bc_ext_strong,
        unit_rel_tol,
        unit_rel_tol,
    );
}

test "CameraPrepared.distortionNone" {
    const input = CameraInput{
        .pixels_num = .{ 10, 10 },
        .pixels_size = .{ 0.01, 0.01 },
        .pos_world = vec.Vec3f.initZeros(),
        .rot_world = rotation.Rotation.init(0, 0, 0),
        .roi_cent_world = vec.Vec3f.initZeros(),
        .focal_length = 1.0,
        .sub_sample = 1,
        .distortion = .none,
        .subpixel_center_map = .full_in_mem,
    };

    const camera = try CameraPrepared.init(std.testing.allocator, input);
    defer camera.deinit(std.testing.allocator);

    const x_ideal = camera.ideal_pixel_centers.get(&[_]usize{ 0, 0, 0 });
    try std.testing.expectApproxEqAbs(0.5, x_ideal, unit_abs_tol);
}

test "CameraPrepared.brownConradyExtDistortionApplied" {
    const distortion = cm.BrownConradyExt{
        .k1 = -0.18,
        .k2 = 0.02,
        .k3 = -0.004,
        .k4 = 0.01,
        .k5 = -0.002,
        .k6 = 0.0004,
        .p1 = 0.0012,
        .p2 = -0.0018,
    };
    const input = CameraInput{
        .pixels_num = .{ 10, 10 },
        .pixels_size = .{ 0.01, 0.01 },
        .pos_world = vec.Vec3f.initZeros(),
        .rot_world = rotation.Rotation.init(0, 0, 0),
        .roi_cent_world = vec.Vec3f.initZeros(),
        .focal_length = 1.0,
        .sub_sample = 1,
        .distortion = .{ .brown_conrady_ext = distortion },
        .subpixel_center_map = .full_in_mem,
    };

    const camera = try CameraPrepared.init(std.testing.allocator, input);
    defer camera.deinit(std.testing.allocator);

    const x_off = 0.5 * @as(F, @floatFromInt(input.pixels_num[0]));
    const y_off = 0.5 * @as(F, @floatFromInt(input.pixels_num[1]));
    const fx = input.focal_length / input.pixels_size[0];
    const fy = input.focal_length / input.pixels_size[1];
    const x_d = (0.5 - x_off) / fx;
    const y_d = (0.5 - y_off) / fy;
    const solved = try distortion.inv(x_d, y_d);

    const x_ideal = camera.ideal_pixel_centers.get(&[_]usize{ 0, 0, 0 });
    const y_ideal = camera.ideal_pixel_centers.get(&[_]usize{ 0, 0, 1 });
    try expectApproxEqRelAbs(
        solved.x * fx + x_off,
        x_ideal,
        unit_rel_tol,
        unit_rel_tol,
    );
    try expectApproxEqRelAbs(
        solved.y * fy + y_off,
        y_ideal,
        unit_rel_tol,
        unit_rel_tol,
    );
}

test "PreparedPSF gaussian kernels normalize" {
    var prep = try cm.preparePSF(
        std.testing.allocator,
        .{ .gaussian = .{
            .sigma_px = 0.35,
            .supp_rad_px = 1.25,
            .separable = .yes,
        } },
        2,
    );
    defer prep.deinit(std.testing.allocator);

    var sum_x: F = 0.0;
    var sum_y: F = 0.0;
    for (prep.weights_x) |weight| sum_x += weight;
    for (prep.weights_y) |weight| sum_y += weight;

    try std.testing.expectApproxEqAbs(1.0, sum_x, sum_abs_tol);
    try std.testing.expectApproxEqAbs(1.0, sum_y, sum_abs_tol);
    try std.testing.expectEqual(@as(u16, 2), prep.halo_px);
}

test "PreparedPSF isotropic gaussian separable matches non-separable outer product" {
    var prepared_sep = try cm.preparePSF(
        std.testing.allocator,
        .{ .gaussian = .{
            .sigma_px = 0.35,
            .supp_rad_px = 1.25,
            .separable = .yes,
        } },
        2,
    );
    defer prepared_sep.deinit(std.testing.allocator);

    var prepared_nonsep = try cm.preparePSF(
        std.testing.allocator,
        .{ .gaussian = .{
            .sigma_px = 0.35,
            .supp_rad_px = 1.25,
            .separable = .no,
        } },
        2,
    );
    defer prepared_nonsep.deinit(std.testing.allocator);

    const width = 2 * prepared_sep.radius_x_subpx + 1;
    for (0..prepared_nonsep.weights_2d.len) |ii| {
        const yy = ii / width;
        const xx = ii % width;
        const expected = prepared_sep.weights_y[yy] *
            prepared_sep.weights_x[xx];
        try std.testing.expectApproxEqAbs(
            expected,
            prepared_nonsep.weights_2d[ii],
            sum_abs_tol,
        );
    }
}
