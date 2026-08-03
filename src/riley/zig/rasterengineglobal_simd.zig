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
const ndarray = @import("ndarray.zig");
const rops = @import("rasterops.zig");
const report = @import("report.zig");
const backend = @import("rasterengine_simd.zig");
const common = @import("rasterengineglobal_common.zig");
const mo = @import("meshpipeline.zig");
const subpxframe = @import("subpxframe.zig");

const GlobalSubpxScratchBuffs = struct {
    stride_subpx: usize,
    inv_z: []align(64) F,
    image: @import("matslice.zig").MatSlice(F),
    filter_tmp: @import("matslice.zig").MatSlice(F),
    simd_chunks: []backend.SubpxSimdChunk,
    mask: []align(64) bool,
    xi: []align(64) F,
    eta: []align(64) F,
    touched_min_x: []usize,
    touched_max_x: []usize,
    ideal_pix_cent: []align(64) F,
    target_stride_subpx: usize = 0,
    target_subx_min: i32 = 0,
    target_suby_min: i32 = 0,
    tile_subx_min: i32 = 0,
    tile_suby_min: i32 = 0,

    pub inline fn imageIndex(
        self: *const GlobalSubpxScratchBuffs,
        local_idx: usize,
    ) usize {
        const local_subx = local_idx % self.stride_subpx;
        const local_suby = local_idx / self.stride_subpx;
        const global_subx = self.tile_subx_min + @as(i32, @intCast(local_subx));
        const global_suby = self.tile_suby_min + @as(i32, @intCast(local_suby));
        const target_subx = global_subx - self.target_subx_min;
        const target_suby = global_suby - self.target_suby_min;
        return @as(usize, @intCast(target_suby)) * self.target_stride_subpx +
            @as(usize, @intCast(target_subx));
    }
};

const GlobalBackend = struct {
    pub const SubpxScratchBuffs = GlobalSubpxScratchBuffs;

    pub fn initSubpxScratch(
        arena_alloc: std.mem.Allocator,
        fields_num: u8,
        subpx_tile_size: usize,
    ) !GlobalSubpxScratchBuffs {
        const local = try backend.initSubpxScratch(
            arena_alloc,
            fields_num,
            subpx_tile_size,
        );
        return .{
            .stride_subpx = local.stride_subpx,
            .inv_z = local.inv_z,
            .image = local.image,
            .filter_tmp = local.filter_tmp,
            .simd_chunks = local.simd_chunks,
            .mask = local.mask,
            .xi = local.xi,
            .eta = local.eta,
            .touched_min_x = local.touched_min_x,
            .touched_max_x = local.touched_max_x,
            .ideal_pix_cent = local.ideal_pix_cent,
        };
    }

    pub fn resetSubpxScratch(
        scratch: *GlobalSubpxScratchBuffs,
        subpx_tile_size: usize,
        _: F,
    ) void {
        @memset(scratch.inv_z, -std.math.inf(F));
        @memset(scratch.touched_min_x, subpx_tile_size);
        @memset(scratch.touched_max_x, 0);
    }

    pub fn configureTarget(
        scratch: *GlobalSubpxScratchBuffs,
        target: *subpxframe.SubpxTarget,
        tile: rops.ActiveTile,
        sub_sample: u32,
    ) void {
        scratch.image = target.image;
        scratch.target_stride_subpx = target.domain.storage_w_subpx;
        scratch.target_subx_min = target.global_subx_min;
        scratch.target_suby_min = target.global_suby_min;
        scratch.tile_subx_min = tile.scratch_x_px_min * @as(i32, @intCast(sub_sample));
        scratch.tile_suby_min = tile.scratch_y_px_min * @as(i32, @intCast(sub_sample));
    }

    pub fn RasterEngine(
        comptime Geom: type,
        comptime ShaderKern: type,
        comptime ShaderData: type,
    ) type {
        return backend.RasterEngineFor(
            GlobalSubpxScratchBuffs,
            Geom,
            ShaderKern,
            ShaderData,
        );
    }
};

pub fn rasterScene(
    comptime report_mode: report.ReportMode,
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    ctx_rast: rops.RasterContext,
    ctx_report: report.ReportContext(report_mode),
    requested_workers: u16,
    tiling: rops.TilingOverlaps,
    meshes: []const mo.MeshPrepared,
    raster_hulls: []const ?ndarray.NDArray(F),
    target: *subpxframe.SubpxTarget,
    image_out_arr: *ndarray.NDArray(F),
) !void {
    try common.rasterScene(
        GlobalBackend,
        report_mode,
        outer_alloc,
        io,
        ctx_rast,
        ctx_report,
        requested_workers,
        tiling,
        meshes,
        raster_hulls,
        target,
        image_out_arr,
    );
}
