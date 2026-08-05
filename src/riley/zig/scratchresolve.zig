// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const buildconfig = @import("buildconfig.zig");
const F = buildconfig.F;
const rops = @import("rasterops.zig");
const cam = @import("camera.zig");
const matslice = @import("matslice.zig");
const ndarray = @import("ndarray.zig");
const common = @import("scratchresolve_common.zig");
const scal = @import("scratchresolve_scalar.zig");
const simd = @import("scratchresolve_simd.zig");

const cfg = buildconfig.config;
const resolve_scratch_direct_impl = scal.resolveScratchDirectCore;
const avg_scratch_impl = if (cfg.simd == .on)
    simd.avgScratchCoreSIMD
else
    scal.avgScratchCore;
const filter_scratch_separable_impl = if (cfg.simd == .on)
    simd.filterScratchSeparableSIMD
else
    scal.filterScratchSeparable;
const filter_scratch_nonseparable_impl = if (cfg.simd == .on)
    simd.filterScratchNonSeparableSIMD
else
    scal.filterScratchNonSeparable;

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const ScratchTileGeometry = common.ScratchTileGeometry;

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn resolveScratchDirect(
    tile: rops.ActiveTile,
    scratch_geom: ScratchTileGeometry,
    spx_tile_size: usize,
    fields_num: u8,
    spx_image_scratch: *const matslice.MatSlice(F),
    touched_min_x: []const usize,
    touched_max_x: []const usize,
    image_out_arr: *ndarray.NDArray(F),
) void {
    resolve_scratch_direct_impl(
        tile,
        scratch_geom,
        spx_tile_size,
        fields_num,
        spx_image_scratch,
        touched_min_x,
        touched_max_x,
        0,
        0,
        image_out_arr,
    );
}

pub fn avgScratch(
    tile: rops.ActiveTile,
    scratch_geom: ScratchTileGeometry,
    sub_samp: usize,
    spx_tile_size: usize,
    fields_num: u8,
    spx_image_scratch: *const matslice.MatSlice(F),
    touched_min_x: []const usize,
    touched_max_x: []const usize,
    image_out_arr: *ndarray.NDArray(F),
) void {
    avg_scratch_impl(
        tile,
        scratch_geom,
        sub_samp,
        spx_tile_size,
        fields_num,
        spx_image_scratch,
        touched_min_x,
        touched_max_x,
        0,
        0,
        image_out_arr,
    );
}

pub fn resolveTileWithPSF(
    tile: rops.ActiveTile,
    sub_samp: usize,
    spx_stride: usize,
    fields_num: u8,
    background_value: F,
    prep_psf: cam.PreparedPSF,
    scratch_geom: ScratchTileGeometry,
    spx_image_scratch: *matslice.MatSlice(F),
    filter_tmp: *matslice.MatSlice(F),
    touched_min_x: []const usize,
    touched_max_x: []const usize,
    image_out_arr: *ndarray.NDArray(F),
) void {
    common.resolveTileWithPSF(
        @TypeOf(resolve_scratch_direct_impl),
        @TypeOf(avg_scratch_impl),
        @TypeOf(filter_scratch_separable_impl),
        @TypeOf(filter_scratch_nonseparable_impl),
        resolve_scratch_direct_impl,
        avg_scratch_impl,
        filter_scratch_separable_impl,
        filter_scratch_nonseparable_impl,
        tile,
        sub_samp,
        spx_stride,
        fields_num,
        background_value,
        prep_psf,
        scratch_geom,
        spx_image_scratch,
        filter_tmp,
        touched_min_x,
        touched_max_x,
        image_out_arr,
    );
}
