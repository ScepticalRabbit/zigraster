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
const rops = @import("rasterops.zig");
const ndarray = @import("ndarray.zig");
const matslice = @import("matslice.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const ScratchTileGeometry = struct {
    core_w_px: usize,
    core_h_px: usize,
    scratch_w_px: usize,
    scratch_h_px: usize,
    scratch_w_subpx: usize,
    scratch_h_subpx: usize,
    core_start_x_subpx: usize,
    core_start_y_subpx: usize,

    pub fn init(tile: rops.ActiveTile, sub_sample: usize) ScratchTileGeometry {
        const core_w_px = @as(usize, tile.x_px_max - tile.x_px_min);
        const core_h_px = @as(usize, tile.y_px_max - tile.y_px_min);

        const scratch_w_px = @as(
            usize,
            @intCast(tile.scratch_x_px_max - tile.scratch_x_px_min),
        );

        const scratch_h_px = @as(
            usize,
            @intCast(tile.scratch_y_px_max - tile.scratch_y_px_min),
        );

        return .{
            .core_w_px = core_w_px,
            .core_h_px = core_h_px,
            .scratch_w_px = scratch_w_px,
            .scratch_h_px = scratch_h_px,
            .scratch_w_subpx = scratch_w_px * sub_sample,
            .scratch_h_subpx = scratch_h_px * sub_sample,
            .core_start_x_subpx = (@as(usize, @intCast(
                @as(i32, tile.x_px_min) - tile.scratch_x_px_min,
            ))) * sub_sample,
            .core_start_y_subpx = (@as(usize, @intCast(
                @as(i32, tile.y_px_min) - tile.scratch_y_px_min,
            ))) * sub_sample,
        };
    }

    pub fn initCoreOnly(
        tile: rops.ActiveTile,
        sub_sample: usize,
    ) ScratchTileGeometry {
        const core_w_px = @as(usize, tile.x_px_max - tile.x_px_min);
        const core_h_px = @as(usize, tile.y_px_max - tile.y_px_min);

        return .{
            .core_w_px = core_w_px,
            .core_h_px = core_h_px,
            .scratch_w_px = core_w_px,
            .scratch_h_px = core_h_px,
            .scratch_w_subpx = core_w_px * sub_sample,
            .scratch_h_subpx = core_h_px * sub_sample,
            .core_start_x_subpx = 0,
            .core_start_y_subpx = 0,
        };
    }
};

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub inline fn getScratchField(
    spx_image_scratch: *const matslice.MatSlice(F),
    scratch_flat_idx: usize,
    field_idx: usize,
) F {
    const scratch_base = spx_image_scratch.rowBase(field_idx);
    return spx_image_scratch.getFlat(scratch_base + scratch_flat_idx);
}

pub inline fn setScratchField(
    spx_image_scratch: *matslice.MatSlice(F),
    scratch_flat_idx: usize,
    field_idx: usize,
    val: F,
) void {
    const scratch_base = spx_image_scratch.rowBase(field_idx);
    spx_image_scratch.setFlat(scratch_base + scratch_flat_idx, val);
}

pub fn sampleScratchOrBackground(
    src: *const matslice.MatSlice(F),
    x: isize,
    y: isize,
    scratch_geom: ScratchTileGeometry,
    spx_stride: usize,
    field_idx: usize,
    background_value: F,
) F {
    if (x < 0 or y < 0 or
        x >= @as(isize, @intCast(scratch_geom.scratch_w_subpx)) or
        y >= @as(isize, @intCast(scratch_geom.scratch_h_subpx)))
    {
        return background_value;
    }

    const flat_idx = @as(usize, @intCast(y)) * spx_stride +
        @as(usize, @intCast(x));

    return getScratchField(src, flat_idx, field_idx);
}

pub fn resolveTileWithPSF(
    comptime ResolveDirectFn: type,
    comptime AvgFn: type,
    comptime FilterSeparableFn: type,
    comptime FilterNonSeparableFn: type,
    resolve_direct_fn: ResolveDirectFn,
    avg_fn: AvgFn,
    filter_separable_fn: FilterSeparableFn,
    filter_nonseparable_fn: FilterNonSeparableFn,
    tile: rops.ActiveTile,
    sub_samp: usize,
    spx_stride: usize,
    fields_num: u8,
    background_value: F,
    prep_psf: @import("camera.zig").PreparedPSF,
    scratch_geom: ScratchTileGeometry,
    spx_image_scratch: *matslice.MatSlice(F),
    filter_tmp: *matslice.MatSlice(F),
    touched_min_x: []const usize,
    touched_max_x: []const usize,
    image_out_arr: *ndarray.NDArray(F),
) void {
    switch (prep_psf.mode) {
        .identity_fast => {
            if (sub_samp > 1) {
                avg_fn(
                    tile,
                    scratch_geom,
                    sub_samp,
                    spx_stride,
                    fields_num,
                    spx_image_scratch,
                    touched_min_x,
                    touched_max_x,
                    0,
                    0,
                    image_out_arr,
                );
            } else {
                resolve_direct_fn(
                    tile,
                    scratch_geom,
                    spx_stride,
                    fields_num,
                    spx_image_scratch,
                    touched_min_x,
                    touched_max_x,
                    0,
                    0,
                    image_out_arr,
                );
            }
        },
        .separable => {
            filter_separable_fn(
                fields_num,
                background_value,
                prep_psf,
                scratch_geom,
                sub_samp,
                spx_stride,
                spx_image_scratch,
                filter_tmp,
                spx_image_scratch,
                touched_min_x,
                touched_max_x,
            );
            if (sub_samp > 1) {
                avg_fn(
                    tile,
                    scratch_geom,
                    sub_samp,
                    spx_stride,
                    fields_num,
                    spx_image_scratch,
                    touched_min_x,
                    touched_max_x,
                    prep_psf.radius_x_subpx,
                    prep_psf.radius_y_subpx,
                    image_out_arr,
                );
            } else {
                resolve_direct_fn(
                    tile,
                    scratch_geom,
                    spx_stride,
                    fields_num,
                    spx_image_scratch,
                    touched_min_x,
                    touched_max_x,
                    prep_psf.radius_x_subpx,
                    prep_psf.radius_y_subpx,
                    image_out_arr,
                );
            }
        },
        .nonseparable => {
            filter_nonseparable_fn(
                fields_num,
                background_value,
                prep_psf,
                scratch_geom,
                sub_samp,
                spx_stride,
                spx_image_scratch,
                filter_tmp,
                touched_min_x,
                touched_max_x,
            );
            if (sub_samp > 1) {
                avg_fn(
                    tile,
                    scratch_geom,
                    sub_samp,
                    spx_stride,
                    fields_num,
                    filter_tmp,
                    touched_min_x,
                    touched_max_x,
                    prep_psf.radius_x_subpx,
                    prep_psf.radius_y_subpx,
                    image_out_arr,
                );
            } else {
                resolve_direct_fn(
                    tile,
                    scratch_geom,
                    spx_stride,
                    fields_num,
                    filter_tmp,
                    touched_min_x,
                    touched_max_x,
                    prep_psf.radius_x_subpx,
                    prep_psf.radius_y_subpx,
                    image_out_arr,
                );
            }
        },
    }
}
