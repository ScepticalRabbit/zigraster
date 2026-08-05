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
const cameramodels = @import("cameramodels.zig");
const cfg = buildconfig.config;
const S = buildconfig.SimdWidth;
const VecSB = buildconfig.VecSB;
const VecSF = buildconfig.VecSF;
const VecSU = buildconfig.VecSU;
const common = @import("camera_common.zig");
const simdops = @import("simdops.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const CameraPrepared = common.CameraPreparedType(@This());

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn calcPinholeRasterPointSIMD(
    camera: *const CameraPrepared,
    v_observed_x_px: VecSF,
    v_observed_y_px: VecSF,
    v_lane_active: VecSB,
) !struct { x: VecSF, y: VecSF } {
    if (common.isNoDistortion(camera.distortion)) {
        return .{ .x = v_observed_x_px, .y = v_observed_y_px };
    }

    const focal_px = camera.calcFocalPx();
    const offsets = camera.calcRasterOffsets();
    const v_x_dist = (v_observed_x_px - @as(VecSF, @splat(offsets.x_off))) /
        @as(VecSF, @splat(focal_px.fx));
    const v_y_dist = (v_observed_y_px - @as(VecSF, @splat(offsets.y_off))) /
        @as(VecSF, @splat(focal_px.fy));

    const solved = try cameramodels.invDistortionModelSIMD(
        camera.distortion,
        v_x_dist,
        v_y_dist,
        v_lane_active,
    );
    return .{
        .x = solved.x * @as(VecSF, @splat(focal_px.fx)) +
            @as(VecSF, @splat(offsets.x_off)),
        .y = solved.y * @as(VecSF, @splat(focal_px.fy)) +
            @as(VecSF, @splat(offsets.y_off)),
    };
}

pub fn fillTileIdealCentersPerTile(
    camera: *const CameraPrepared,
    scratch_x_px_min: i32,
    scratch_x_px_max: i32,
    scratch_y_px_min: i32,
    scratch_y_px_max: i32,
    subpx_tile_size: usize,
    ideal_pixel_centers: []F,
) !void {
    const sub_samp: usize = @intCast(camera.sub_sample);
    const ideal_x_plane = common.getIdealXPlaneScratch(ideal_pixel_centers);
    const ideal_y_plane = common.getIdealYPlaneScratch(ideal_pixel_centers);
    const start_x = scratch_x_px_min * @as(i32, @intCast(sub_samp));
    const start_y = scratch_y_px_min * @as(i32, @intCast(sub_samp));
    const tile_w_px = scratch_x_px_max - scratch_x_px_min;
    const tile_h_px = scratch_y_px_max - scratch_y_px_min;
    const tile_w: usize = @intCast(tile_w_px * @as(i32, @intCast(sub_samp)));
    const tile_h: usize = @intCast(tile_h_px * @as(i32, @intCast(sub_samp)));

    const step = 1.0 / @as(F, @floatFromInt(camera.sub_sample));
    const off = 0.5 / @as(F, @floatFromInt(camera.sub_sample));

    if (common.isNoDistortion(camera.distortion)) {
        for (0..tile_h) |jj| {
            const global_y = start_y + @as(i32, @intCast(jj));
            const observed_y = @as(F, @floatFromInt(global_y)) * step + off;
            const scratch_row_off = jj * subpx_tile_size;
            @memset(
                ideal_y_plane[scratch_row_off .. scratch_row_off + tile_w],
                observed_y,
            );
        }
        for (0..tile_w) |ii| {
            const global_x = start_x + @as(i32, @intCast(ii));
            ideal_x_plane[ii] = @as(F, @floatFromInt(global_x)) * step + off;
        }
        for (1..tile_h) |jj| {
            const scratch_row_off = jj * subpx_tile_size;
            @memcpy(
                ideal_x_plane[scratch_row_off .. scratch_row_off + tile_w],
                ideal_x_plane[0..tile_w],
            );
        }
        return;
    }

    for (0..tile_h) |jj| {
        const global_y = start_y + @as(i32, @intCast(jj));
        const observed_y = @as(F, @floatFromInt(global_y)) * step + off;
        const v_observed_y: VecSF = @splat(observed_y);
        const scratch_row_off = jj * subpx_tile_size;

        var ii: usize = 0;
        while (ii < tile_w) : (ii += S) {
            const lane_count = @min(S, tile_w - ii);
            const v_active = calcActiveMask(lane_count);
            const global_x_start = start_x + @as(i32, @intCast(ii));
            const v_observed_x = calcObservedXVec(
                global_x_start,
                step,
                off,
            );

            const ideal = try calcPinholeRasterPointSIMD(
                camera,
                v_observed_x,
                v_observed_y,
                v_active,
            );
            storeIdealPairs(
                ideal_x_plane,
                ideal_y_plane,
                scratch_row_off + ii,
                lane_count,
                ideal.x,
                ideal.y,
            );
        }
    }
}

pub fn fillTileIdealCentersAffineJac(
    camera: *const CameraPrepared,
    scratch_x_px_min: i32,
    scratch_x_px_max: i32,
    scratch_y_px_min: i32,
    scratch_y_px_max: i32,
    subpx_tile_size: usize,
    ideal_pixel_centers: []F,
) void {
    fillTileIdealCentersPerTile(
        camera,
        scratch_x_px_min,
        scratch_x_px_max,
        scratch_y_px_min,
        scratch_y_px_max,
        subpx_tile_size,
        ideal_pixel_centers,
    ) catch unreachable;
}

pub fn initPixelCenterJac(camera: *CameraPrepared) !void {
    const jac = &camera.pixel_center_jac;
    const jac_slice = jac.slice;
    const jac_field_stride = jac.strides[2];
    const eps: F = 0.25;
    for (0..camera.pixels_num[1]) |jj| {
        const y_c = common.calcPixelCenterCoord(jj);
        const v_center_y: VecSF = @splat(y_c);
        const v_y_plus: VecSF = @splat(y_c + eps);
        const v_y_minus: VecSF = @splat(y_c - eps);

        var ii: usize = 0;
        while (ii < camera.pixels_num[0]) : (ii += S) {
            const lane_count = @min(S, @as(usize, camera.pixels_num[0]) - ii);
            const v_active = calcActiveMask(lane_count);

            var x_center_arr: [S]F = undefined;
            for (0..lane_count) |ll| {
                x_center_arr[ll] = common.calcPixelCenterCoord(ii + ll);
            }
            for (lane_count..S) |ll| x_center_arr[ll] = 0.0;
            const v_center_x: VecSF = x_center_arr;
            const v_x_plus = v_center_x + @as(VecSF, @splat(eps));
            const v_x_minus = v_center_x - @as(VecSF, @splat(eps));

            if (common.isNoDistortion(camera.distortion)) {
                const center_arr: [S]F = v_center_x;
                for (0..lane_count) |ll| {
                    const px = ii + ll;
                    const jac_px_base = jac.subBase2(jj, px);
                    jac_slice[jac_px_base + 0 * jac_field_stride] =
                        center_arr[ll];
                    jac_slice[jac_px_base + 1 * jac_field_stride] = y_c;
                    jac_slice[jac_px_base + 2 * jac_field_stride] = 1.0;
                    jac_slice[jac_px_base + 3 * jac_field_stride] = 0.0;
                    jac_slice[jac_px_base + 4 * jac_field_stride] = 0.0;
                    jac_slice[jac_px_base + 5 * jac_field_stride] = 1.0;
                }
                continue;
            }

            const center = try calcPinholeRasterPointSIMD(
                camera,
                v_center_x,
                v_center_y,
                v_active,
            );
            const x_p = try calcPinholeRasterPointSIMD(
                camera,
                v_x_plus,
                v_center_y,
                v_active,
            );
            const x_m = try calcPinholeRasterPointSIMD(
                camera,
                v_x_minus,
                v_center_y,
                v_active,
            );
            const y_p = try calcPinholeRasterPointSIMD(
                camera,
                v_center_x,
                v_y_plus,
                v_active,
            );
            const y_m = try calcPinholeRasterPointSIMD(
                camera,
                v_center_x,
                v_y_minus,
                v_active,
            );
            const inv_two_eps: VecSF = @splat(0.5 / eps);

            const center_x_arr: [S]F = center.x;
            const center_y_arr: [S]F = center.y;
            const j11_arr: [S]F = (x_p.x - x_m.x) * inv_two_eps;
            const j12_arr: [S]F = (y_p.x - y_m.x) * inv_two_eps;
            const j21_arr: [S]F = (x_p.y - x_m.y) * inv_two_eps;
            const j22_arr: [S]F = (y_p.y - y_m.y) * inv_two_eps;
            for (0..lane_count) |ll| {
                const px = ii + ll;
                const jac_px_base = jac.subBase2(jj, px);
                jac_slice[jac_px_base + 0 * jac_field_stride] =
                    center_x_arr[ll];
                jac_slice[jac_px_base + 1 * jac_field_stride] =
                    center_y_arr[ll];
                jac_slice[jac_px_base + 2 * jac_field_stride] = j11_arr[ll];
                jac_slice[jac_px_base + 3 * jac_field_stride] = j12_arr[ll];
                jac_slice[jac_px_base + 4 * jac_field_stride] = j21_arr[ll];
                jac_slice[jac_px_base + 5 * jac_field_stride] = j22_arr[ll];
            }
        }
    }
}

pub fn calcPinholeRasterPoint(
    camera: *const CameraPrepared,
    observed_x_px: F,
    observed_y_px: F,
) ![2]F {
    const focal_px = camera.calcFocalPx();
    const offsets = camera.calcRasterOffsets();
    const x_dist = (observed_x_px - offsets.x_off) / focal_px.fx;
    const y_dist = (observed_y_px - offsets.y_off) / focal_px.fy;

    const solved = try cameramodels.invDistortionModelScal(
        camera.distortion,
        x_dist,
        y_dist,
    );
    return .{
        solved.x * focal_px.fx + offsets.x_off,
        solved.y * focal_px.fy + offsets.y_off,
    };
}

fn calcActiveMask(lane_count: usize) VecSB {
    const v_lane_idx: VecSU = std.simd.iota(usize, S);
    return v_lane_idx < @as(VecSU, @splat(lane_count));
}

fn storeIdealPairs(
    ideal_x_plane: []F,
    ideal_y_plane: []F,
    scratch_base: usize,
    lane_count: usize,
    v_x: VecSF,
    v_y: VecSF,
) void {
    if (lane_count == S) {
        simdops.storeVecSF(ideal_x_plane, scratch_base, v_x);
        simdops.storeVecSF(ideal_y_plane, scratch_base, v_y);
        return;
    }

    const x_arr: [S]F = v_x;
    const y_arr: [S]F = v_y;
    for (0..lane_count) |ll| {
        ideal_x_plane[scratch_base + ll] = x_arr[ll];
        ideal_y_plane[scratch_base + ll] = y_arr[ll];
    }
}

fn calcObservedXVec(
    global_subx_start: i32,
    step: F,
    off: F,
) VecSF {
    const v_start: VecSF = @splat(@as(F, @floatFromInt(global_subx_start)));
    const v_lane: VecSF = @floatFromInt(std.simd.iota(i32, S));
    return (v_start + v_lane) * @as(VecSF, @splat(step)) +
        @as(VecSF, @splat(off));
}
