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
const S = buildconfig.SimdWidth;
const VecSF = buildconfig.VecSF;

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const ScratchTileGeometry = common.ScratchTileGeometry;
pub const getScratchField = common.getScratchField;
pub const setScratchField = common.setScratchField;
pub const sampleScratchOrBackground = common.sampleScratchOrBackground;

// --------------------------------------------------------------------------------------
// Public Entry-Point Functions
// --------------------------------------------------------------------------------------

pub fn avgScratchCoreSIMD(
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
    const cols_num = spx_image_scratch.cols_num;
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

            // Direct resolve and averaging only visit core pixel subpx blocks, so the
            // full `sub_samp x sub_samp` region is guaranteed to be inside scratch.
            std.debug.assert(spx_start_x + sub_samp <= scratch_geom.scratch_w_subpx);
            std.debug.assert(spx_start_y + sub_samp <= scratch_geom.scratch_h_subpx);

            for (0..fields_num) |ff| {
                var field_sum: F = 0.0;

                if (sub_samp == 8 and S == 8) {
                    var sum_vec = @as(VecSF, @splat(0.0));

                    for (0..8) |row_idx| {
                        const scratch_row_offset = (spx_start_y + row_idx) * spx_stride;
                        const scratch_flat_idx = scratch_row_offset + spx_start_x;
                        const offset = ff * cols_num + scratch_flat_idx;
                        const ptr = @as(
                            *const [8]F,
                            @ptrCast(&spx_image_scratch.slice[offset]),
                        );
                        sum_vec += ptr.*;
                    }

                    field_sum = @reduce(.Add, sum_vec);
                } else if (sub_samp == 4) {
                    var sum_vec = @as(@Vector(4, F), @splat(0.0));

                    for (0..4) |row_idx| {
                        const scratch_row_offset = (spx_start_y + row_idx) * spx_stride;
                        const scratch_flat_idx = scratch_row_offset + spx_start_x;
                        const offset = ff * cols_num + scratch_flat_idx;
                        const ptr = @as(
                            *const [4]F,
                            @ptrCast(&spx_image_scratch.slice[offset]),
                        );
                        sum_vec += ptr.*;
                    }

                    field_sum = @reduce(.Add, sum_vec);
                } else if (sub_samp == 2) {
                    var sum_vec = @as(@Vector(2, F), @splat(0.0));

                    for (0..2) |row_idx| {
                        const scratch_row_offset = (spx_start_y + row_idx) * spx_stride;
                        const scratch_flat_idx = scratch_row_offset + spx_start_x;
                        const offset = ff * cols_num + scratch_flat_idx;
                        const ptr = @as(
                            *const [2]F,
                            @ptrCast(&spx_image_scratch.slice[offset]),
                        );
                        sum_vec += ptr.*;
                    }

                    field_sum = @reduce(.Add, sum_vec);
                } else {
                    for (0..sub_samp) |row_idx| {
                        const scratch_row_offset = (spx_start_y + row_idx) * spx_stride;
                        var col_idx: usize = 0;

                        while (col_idx < sub_samp) {
                            if (col_idx + S <= sub_samp) {
                                const scratch_flat_idx = scratch_row_offset +
                                    spx_start_x + col_idx;
                                const offset = ff * cols_num + scratch_flat_idx;

                                const val_vec = @as(
                                    VecSF,
                                    @as(
                                        *const [S]F,
                                        @ptrCast(
                                            &spx_image_scratch.slice[offset],
                                        ),
                                    ).*,
                                );

                                field_sum += @reduce(.Add, val_vec);
                                col_idx += S;
                            } else {
                                const scratch_flat_idx = scratch_row_offset +
                                    spx_start_x + col_idx;
                                field_sum += getScratchField(
                                    spx_image_scratch,
                                    scratch_flat_idx,
                                    ff,
                                );
                                col_idx += 1;
                            }
                        }
                    }
                }
                const write_idx = image_out_arr.offset3(ff, image_px_y, image_px_x);
                image_out_arr.slice[write_idx] = field_sum * inv_sub_samp_sq;
            }
        }
    }
}

pub fn filterScratchSeparableSIMD(
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
            var xx = xx_start;

            while (xx + S <= xx_end + 1) : (xx += S) {
                var sum_h_vec = @as(VecSF, @splat(0.0));

                for (0..psf.weights_x.len) |kk| {
                    const x_off = @as(
                        isize,
                        @intCast(kk),
                    ) - @as(isize, @intCast(radius_x));

                    const src_row_vec = loadScratchRowSIMD(
                        src,
                        @as(isize, @intCast(xx)) + x_off,
                        @as(isize, @intCast(yy)),
                        scratch_geom,
                        spx_stride,
                        ff,
                        background_value,
                    );

                    sum_h_vec += src_row_vec * @as(
                        VecSF,
                        @splat(psf.weights_x[kk]),
                    );
                }

                storeScratchRowSIMD(
                    tmp,
                    xx,
                    yy,
                    spx_stride,
                    ff,
                    sum_h_vec,
                );
            }

            while (xx <= xx_end) : (xx += 1) {
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
            var xx = xx_start;

            while (xx + S <= xx_end + 1) : (xx += S) {
                var sum_v_vec = @as(VecSF, @splat(0.0));

                for (0..psf.weights_y.len) |kk| {
                    const y_off = @as(
                        isize,
                        @intCast(kk),
                    ) - @as(isize, @intCast(radius_y));

                    const src_row_vec = loadScratchRowSIMD(
                        tmp,
                        @as(isize, @intCast(xx)),
                        @as(isize, @intCast(yy)) + y_off,
                        scratch_geom,
                        spx_stride,
                        ff,
                        background_value,
                    );

                    sum_v_vec += src_row_vec * @as(
                        VecSF,
                        @splat(psf.weights_y[kk]),
                    );
                }

                storeScratchRowSIMD(
                    dst,
                    xx,
                    yy,
                    spx_stride,
                    ff,
                    sum_v_vec,
                );
            }

            while (xx <= xx_end) : (xx += 1) {
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

pub fn filterScratchNonSeparableSIMD(
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
            scratch_geom.core_start_x_subpx +
                scratch_geom.core_w_px * sub_samp - 1,
            active_max,
        );

        if (xx_start > xx_end) continue;

        for (0..fields_num) |ff| {
            var xx = xx_start;

            while (xx + S <= xx_end + 1) : (xx += S) {
                var sum_vec = @as(VecSF, @splat(0.0));

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

                    const src_row_vec = loadScratchRowSIMD(
                        src,
                        @as(isize, @intCast(xx)) + x_off,
                        @as(isize, @intCast(yy)) + y_off,
                        scratch_geom,
                        spx_stride,
                        ff,
                        background_value,
                    );

                    sum_vec += src_row_vec * @as(
                        VecSF,
                        @splat(psf.weights_2d[kk]),
                    );
                }

                storeScratchRowSIMD(
                    dst,
                    xx,
                    yy,
                    spx_stride,
                    ff,
                    sum_vec,
                );
            }
            while (xx <= xx_end) : (xx += 1) {
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

// --------------------------------------------------------------------------------------
// Private Helpers
// --------------------------------------------------------------------------------------

inline fn loadScratchRowSIMD(
    src: *const matslice.MatSlice(F),
    x: isize,
    y: isize,
    scratch_geom: ScratchTileGeometry,
    spx_stride: usize,
    field_idx: usize,
    background_value: F,
) VecSF {
    const cols_num = src.cols_num;
    if (y < 0 or y >= @as(isize, @intCast(scratch_geom.scratch_h_subpx))) {
        return @as(VecSF, @splat(background_value));
    }
    const uy = @as(usize, @intCast(y));

    const chunk_in_bounds = (x >= 0 and
        x + @as(isize, @intCast(S)) <=
            @as(isize, @intCast(scratch_geom.scratch_w_subpx)));

    if (chunk_in_bounds) {
        const ux = @as(usize, @intCast(x));
        const flat_idx = uy * spx_stride + ux;
        const offset = field_idx * cols_num + flat_idx;
        return @as(*const [S]F, @ptrCast(&src.slice[offset])).*;
    }

    var result: [S]F = undefined;
    for (0..S) |ii| {
        result[ii] = sampleScratchOrBackground(
            src,
            x + @as(isize, @intCast(ii)),
            y,
            scratch_geom,
            spx_stride,
            field_idx,
            background_value,
        );
    }
    return result;
}

inline fn storeScratchRowSIMD(
    dst: *matslice.MatSlice(F),
    x: usize,
    y: usize,
    spx_stride: usize,
    field_idx: usize,
    val_vec: VecSF,
) void {
    const cols_num = dst.cols_num;
    const flat_idx = y * spx_stride + x;
    const offset = field_idx * cols_num + flat_idx;
    @as(*[S]F, @ptrCast(&dst.slice[offset])).* = val_vec;
}
