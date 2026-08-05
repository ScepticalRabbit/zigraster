// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const rops = @import("rasterops.zig");
const cam = @import("camera.zig");
const common = @import("scratchresolve_common.zig");
const matslice = @import("matslice.zig");
const ndarray = @import("ndarray.zig");
const buildconfig = @import("buildconfig.zig");
const F = buildconfig.F;

const cfg = buildconfig.config;

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const ScratchTileGeometry = common.ScratchTileGeometry;
pub const getScratchField = common.getScratchField;
pub const setScratchField = common.setScratchField;
pub const sampleScratchOrBackground = common.sampleScratchOrBackground;

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn resolveScratchDirectCore(
    tile: rops.ActiveTile,
    scratch_geom: ScratchTileGeometry,
    spx_stride: usize,
    fields_num: u8,
    spx_image_scratch: *const matslice.MatSlice(F),
    touched_min_x: []const usize,
    touched_max_x: []const usize,
    radius_x: usize,
    radius_y: usize,
    image_out_arr: *ndarray.NDArray(F),
) void {
    // Buff layout is planar/field-major.
    // Source: field ff is at spx_image_scratch.slice[ff * spx_stride].
    // Destination: field ff is at image_out_arr.slice[ff * image_out_arr.strides[0]].
    for (0..scratch_geom.core_h_px) |ii| {
        const image_px_y = tile.y_px_min + ii;
        const spx_start_y = scratch_geom.core_start_y_subpx + ii;

        var min_x = scratch_geom.scratch_w_subpx;
        var max_x: usize = 0;

        const start_y = if (spx_start_y >= radius_y)
            spx_start_y - radius_y
        else
            0;
        const end_y = @min(
            scratch_geom.scratch_h_subpx - 1,
            spx_start_y + radius_y,
        );

        for (start_y..end_y + 1) |nn| {
            const r_min = touched_min_x[nn];
            const r_max = touched_max_x[nn];
            if (r_min <= r_max) {
                if (r_min < min_x) min_x = r_min;
                if (r_max > max_x) max_x = r_max;
            }
        }

        if (min_x > max_x) {
            continue;
        }

        const active_subpx_min = if (min_x >= radius_x)
            min_x - radius_x
        else
            0;
        const active_subpx_max = max_x + radius_x;

        var tx_start: usize = 0;
        if (active_subpx_min > scratch_geom.core_start_x_subpx) {
            tx_start = active_subpx_min - scratch_geom.core_start_x_subpx;
        }
        var tx_end: usize = scratch_geom.core_w_px - 1;
        if (active_subpx_max >= scratch_geom.core_start_x_subpx) {
            const calc_end = active_subpx_max -
                scratch_geom.core_start_x_subpx;
            if (calc_end < tx_end) {
                tx_end = calc_end;
            }
        } else {
            continue;
        }

        const scratch_row_offset = spx_start_y * spx_stride;

        if (tx_start <= tx_end) {
            const len = tx_end - tx_start + 1;
            const src_base = scratch_row_offset +
                scratch_geom.core_start_x_subpx + tx_start;
            const dest_base = image_out_arr.offset3(
                0,
                image_px_y,
                tile.x_px_min + tx_start,
            );

            if (fields_num == 1) {
                @memcpy(
                    image_out_arr.slice[dest_base .. dest_base + len],
                    spx_image_scratch.slice[src_base .. src_base + len],
                );
            } else if (fields_num == 3) {
                const src0_base = spx_image_scratch.rowBase(0);
                const src1_base = spx_image_scratch.rowBase(1);
                const src2_base = spx_image_scratch.rowBase(2);
                @memcpy(
                    image_out_arr.slice[dest_base .. dest_base + len],
                    spx_image_scratch.slice[src0_base + src_base .. src0_base + src_base + len],
                );
                const dest1 = image_out_arr.offset3(
                    1,
                    image_px_y,
                    tile.x_px_min + tx_start,
                );
                @memcpy(
                    image_out_arr.slice[dest1 .. dest1 + len],
                    spx_image_scratch.slice[src1_base + src_base .. src1_base + src_base + len],
                );
                const dest2 = image_out_arr.offset3(
                    2,
                    image_px_y,
                    tile.x_px_min + tx_start,
                );
                @memcpy(
                    image_out_arr.slice[dest2 .. dest2 + len],
                    spx_image_scratch.slice[src2_base + src_base .. src2_base + src_base + len],
                );
            } else {
                for (0..fields_num) |ff| {
                    const src_offset = spx_image_scratch.rowBase(ff) + src_base;
                    const dest_offset = image_out_arr.offset3(
                        ff,
                        image_px_y,
                        tile.x_px_min + tx_start,
                    );
                    @memcpy(
                        image_out_arr.slice[dest_offset .. dest_offset + len],
                        spx_image_scratch.slice[src_offset .. src_offset + len],
                    );
                }
            }
        }
    }
}

pub fn avgScratchCore(
    tile: rops.ActiveTile,
    scratch_geom: ScratchTileGeometry,
    sub_samp: usize,
    spx_stride: usize,
    fields_num: u8,
    spx_image_scratch: *const matslice.MatSlice(F),
    touched_min_x: []const usize,
    touched_max_x: []const usize,
    radius_x: usize,
    radius_y: usize,
    image_out_arr: *ndarray.NDArray(F),
) void {
    const sub_samp_f = @as(F, @floatFromInt(sub_samp));
    const inv_sub_samp_sq = 1.0 / (sub_samp_f * sub_samp_f);

    for (0..scratch_geom.core_h_px) |ii| {
        const image_px_y = tile.y_px_min + ii;
        const spx_start_y = scratch_geom.core_start_y_subpx + sub_samp * ii;

        var min_x = scratch_geom.scratch_w_subpx;
        var max_x: usize = 0;

        const start_y = if (spx_start_y >= radius_y)
            spx_start_y - radius_y
        else
            0;
        const end_y = @min(
            scratch_geom.scratch_h_subpx - 1,
            spx_start_y + sub_samp - 1 + radius_y,
        );

        for (start_y..end_y + 1) |nn| {
            const r_min = touched_min_x[nn];
            const r_max = touched_max_x[nn];
            if (r_min <= r_max) {
                if (r_min < min_x) min_x = r_min;
                if (r_max > max_x) max_x = r_max;
            }
        }

        if (min_x > max_x) {
            continue;
        }

        const active_subpx_min = if (min_x >= radius_x)
            min_x - radius_x
        else
            0;
        const active_subpx_max = max_x + radius_x;

        var tx_start: usize = 0;
        if (active_subpx_min > scratch_geom.core_start_x_subpx) {
            tx_start = (active_subpx_min - scratch_geom.core_start_x_subpx) /
                sub_samp;
        }
        var tx_end: usize = scratch_geom.core_w_px - 1;
        if (active_subpx_max >= scratch_geom.core_start_x_subpx) {
            const calc_end =
                (active_subpx_max - scratch_geom.core_start_x_subpx) / sub_samp;
            if (calc_end < tx_end) {
                tx_end = calc_end;
            }
        } else {
            continue;
        }

        if (tx_start > tx_end) {
            continue;
        }

        for (tx_start..tx_end + 1) |jj| {
            const image_px_x = tile.x_px_min + jj;
            const spx_start_x = scratch_geom.core_start_x_subpx + sub_samp * jj;
            const image_base_0 = image_out_arr.offset3(0, image_px_y, image_px_x);

            // Direct resolve and averaging only visit core pixel subpx blocks, so the
            // full `sub_samp x sub_samp` region is guaranteed to be inside scratch.
            std.debug.assert(spx_start_x + sub_samp <= scratch_geom.scratch_w_subpx);
            std.debug.assert(spx_start_y + sub_samp <= scratch_geom.scratch_h_subpx);

            if (fields_num == 1) {
                var field_sum_0: F = 0.0;
                for (0..sub_samp) |row_idx| {
                    const scratch_row_offset = (spx_start_y + row_idx) *
                        spx_stride;
                    for (0..sub_samp) |col_idx| {
                        const scratch_flat_idx = scratch_row_offset +
                            spx_start_x + col_idx;
                        field_sum_0 += getScratchField(
                            spx_image_scratch,
                            scratch_flat_idx,
                            0,
                        );
                    }
                }
                image_out_arr.slice[image_base_0] = field_sum_0 * inv_sub_samp_sq;
            } else if (fields_num == 3) {
                var field_sum_0: F = 0.0;
                var field_sum_1: F = 0.0;
                var field_sum_2: F = 0.0;
                for (0..sub_samp) |row_idx| {
                    const scratch_row_offset = (spx_start_y + row_idx) *
                        spx_stride;
                    for (0..sub_samp) |col_idx| {
                        const scratch_flat_idx = scratch_row_offset +
                            spx_start_x + col_idx;
                        field_sum_0 += getScratchField(
                            spx_image_scratch,
                            scratch_flat_idx,
                            0,
                        );
                        field_sum_1 += getScratchField(
                            spx_image_scratch,
                            scratch_flat_idx,
                            1,
                        );
                        field_sum_2 += getScratchField(
                            spx_image_scratch,
                            scratch_flat_idx,
                            2,
                        );
                    }
                }
                image_out_arr.slice[image_base_0] = field_sum_0 * inv_sub_samp_sq;
                image_out_arr.slice[image_out_arr.offset3(1, image_px_y, image_px_x)] =
                    field_sum_1 * inv_sub_samp_sq;
                image_out_arr.slice[image_out_arr.offset3(2, image_px_y, image_px_x)] =
                    field_sum_2 * inv_sub_samp_sq;
            } else {
                var field_avg_buff = [_]F{0.0} ** cfg.max_nodal_fields;
                const spx_field_avg = field_avg_buff[0..@as(usize, fields_num)];
                @memset(spx_field_avg, 0.0);

                for (0..sub_samp) |row_idx| {
                    const scratch_row_offset = (spx_start_y + row_idx) *
                        spx_stride;
                    for (0..sub_samp) |col_idx| {
                        const scratch_flat_idx = scratch_row_offset +
                            spx_start_x + col_idx;
                        for (0..fields_num) |ff| {
                            spx_field_avg[ff] += getScratchField(
                                spx_image_scratch,
                                scratch_flat_idx,
                                ff,
                            );
                        }
                    }
                }

                for (0..fields_num) |ff| {
                    image_out_arr.slice[image_out_arr.offset3(ff, image_px_y, image_px_x)] =
                        spx_field_avg[ff] * inv_sub_samp_sq;
                }
            }
        }
    }
}

pub fn filterScratchSeparable(
    fields_num: u8,
    background_value: F,
    psf: cam.PreparedPSF,
    scratch_geom: ScratchTileGeometry,
    sub_samp: usize,
    spx_stride: usize,
    src: *const matslice.MatSlice(F),
    tmp: *matslice.MatSlice(F),
    dst: *matslice.MatSlice(F),
    touched_min_x: []const usize,
    touched_max_x: []const usize,
) void {
    const radius_x = psf.radius_x_subpx;
    const radius_y = psf.radius_y_subpx;

    // Horizontal pass
    for (0..scratch_geom.scratch_h_subpx) |yy| {
        const min_x = touched_min_x[yy];
        const max_x = touched_max_x[yy];
        if (min_x > max_x) continue;

        const active_min = if (min_x >= radius_x) min_x - radius_x else 0;
        const active_max = max_x + radius_x;

        const xx_start = @max(scratch_geom.core_start_x_subpx, active_min);
        const xx_end = @min(
            scratch_geom.core_start_x_subpx +
                scratch_geom.core_w_px * sub_samp - 1,
            active_max,
        );
        if (xx_start > xx_end) continue;

        for (0..fields_num) |ff| {
            // Scalar path
            for (xx_start..xx_end + 1) |xx| {
                var sum_h: F = 0.0;
                for (0..psf.weights_x.len) |kk| {
                    const x_off = @as(
                        isize,
                        @intCast(kk),
                    ) - @as(isize, @intCast(radius_x));
                    sum_h += psf.weights_x[kk] * sampleScratchOrBackground(
                        src,
                        @as(isize, @intCast(xx)) + x_off,
                        @as(isize, @intCast(yy)),
                        scratch_geom,
                        spx_stride,
                        ff,
                        background_value,
                    );
                }

                setScratchField(
                    tmp,
                    yy * spx_stride + xx,
                    ff,
                    sum_h,
                );
            }
        }
    }

    // Vertical pass
    const core_start_y = scratch_geom.core_start_y_subpx;
    const core_end_y = scratch_geom.core_start_y_subpx +
        scratch_geom.core_h_px * sub_samp - 1;

    for (core_start_y..core_end_y + 1) |yy| {
        var min_x = scratch_geom.scratch_w_subpx;
        var max_x: usize = 0;
        const start_y = if (yy >= radius_y) yy - radius_y else 0;
        const end_y = @min(scratch_geom.scratch_h_subpx - 1, yy + radius_y);

        for (start_y..end_y + 1) |nn| {
            const r_min = touched_min_x[nn];
            const r_max = touched_max_x[nn];
            if (r_min <= r_max) {
                if (r_min < min_x) min_x = r_min;
                if (r_max > max_x) max_x = r_max;
            }
        }

        if (min_x > max_x) continue;

        const active_min = if (min_x >= radius_x) min_x - radius_x else 0;
        const active_max = max_x + radius_x;

        const xx_start = @max(scratch_geom.core_start_x_subpx, active_min);
        const xx_end = @min(
            scratch_geom.core_start_x_subpx +
                scratch_geom.core_w_px * sub_samp - 1,
            active_max,
        );
        if (xx_start > xx_end) continue;

        for (0..fields_num) |ff| {
            // Scalar path
            for (xx_start..xx_end + 1) |xx| {
                var sum_v: F = 0.0;
                for (0..psf.weights_y.len) |kk| {
                    const y_off = @as(
                        isize,
                        @intCast(kk),
                    ) - @as(isize, @intCast(radius_y));

                    sum_v += psf.weights_y[kk] * sampleScratchOrBackground(
                        tmp,
                        @as(isize, @intCast(xx)),
                        @as(isize, @intCast(yy)) + y_off,
                        scratch_geom,
                        spx_stride,
                        ff,
                        background_value,
                    );
                }

                setScratchField(
                    dst,
                    yy * spx_stride + xx,
                    ff,
                    sum_v,
                );
            }
        }
    }
}

pub fn filterScratchNonSeparable(
    fields_num: u8,
    background_value: F,
    psf: cam.PreparedPSF,
    scratch_geom: ScratchTileGeometry,
    sub_samp: usize,
    spx_stride: usize,
    src: *const matslice.MatSlice(F),
    dst: *matslice.MatSlice(F),
    touched_min_x: []const usize,
    touched_max_x: []const usize,
) void {
    const radius_x = psf.radius_x_subpx;
    const radius_y = psf.radius_y_subpx;
    const kernel_w = 2 * radius_x + 1;

    const core_start_y = scratch_geom.core_start_y_subpx;
    const core_end_y = scratch_geom.core_start_y_subpx +
        scratch_geom.core_h_px * sub_samp - 1;

    for (core_start_y..core_end_y + 1) |yy| {
        var min_x = scratch_geom.scratch_w_subpx;
        var max_x: usize = 0;
        const start_y = if (yy >= radius_y) yy - radius_y else 0;
        const end_y = @min(scratch_geom.scratch_h_subpx - 1, yy + radius_y);

        for (start_y..end_y + 1) |nn| {
            const r_min = touched_min_x[nn];
            const r_max = touched_max_x[nn];
            if (r_min <= r_max) {
                if (r_min < min_x) min_x = r_min;
                if (r_max > max_x) max_x = r_max;
            }
        }

        if (min_x > max_x) continue;

        const active_min = if (min_x >= radius_x) min_x - radius_x else 0;
        const active_max = max_x + radius_x;

        const xx_start = @max(scratch_geom.core_start_x_subpx, active_min);
        const xx_end = @min(
            scratch_geom.core_start_x_subpx + scratch_geom.core_w_px * sub_samp - 1,
            active_max,
        );
        if (xx_start > xx_end) continue;

        for (0..fields_num) |ff| {
            // Scalar path
            for (xx_start..xx_end + 1) |xx| {
                var sum: F = 0.0;
                for (0..psf.weights_2d.len) |kk| {
                    const ky = kk / kernel_w;
                    const kx = kk % kernel_w;

                    const x_off = @as(
                        isize,
                        @intCast(kx),
                    ) - @as(isize, @intCast(radius_x));

                    const y_off = @as(
                        isize,
                        @intCast(ky),
                    ) - @as(isize, @intCast(radius_y));

                    sum += psf.weights_2d[kk] * sampleScratchOrBackground(
                        src,
                        @as(isize, @intCast(xx)) + x_off,
                        @as(isize, @intCast(yy)) + y_off,
                        scratch_geom,
                        spx_stride,
                        ff,
                        background_value,
                    );
                }

                setScratchField(
                    dst,
                    yy * spx_stride + xx,
                    ff,
                    sum,
                );
            }
        }
    }
}
