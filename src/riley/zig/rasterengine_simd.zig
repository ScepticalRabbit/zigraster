// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const cam = @import("camera.zig");
const CameraPrepared = cam.CameraPrepared;
const buildconfig = @import("buildconfig.zig");
const F = buildconfig.F;
const cfg = buildconfig.config;
const S = buildconfig.SimdWidth;
const VecSB = buildconfig.VecSB;
const VecSF = buildconfig.VecSF;
const VecSU = buildconfig.VecSU;
const VecSU8 = buildconfig.VecSU8;

const rastcfg = @import("rasterconfig.zig");
const ReportMode = rastcfg.ReportMode;
const tol = cfg.tol;
const MatSlice = @import("matslice.zig").MatSlice;
const NDArray = @import("ndarray.zig").NDArray;
const hull = @import("hull.zig");
const HullResultSIMD = hull.HullResultSIMD;
const shapefun = @import("shapefun.zig");
const rops = @import("rasterops.zig");
const ElemBBox = rops.ElemBBox;
const OverlapBBox = rops.OverlapBBox;
const ActiveTile = rops.ActiveTile;
const Vec3Slices = rops.Vec3Slices;
const report = @import("report.zig");
const Timestamp = std.Io.Clock.Timestamp;
const comm = @import("rasterengine_common.zig");
const rasterreport = @import("rasterreport.zig");
const simdops = @import("simdops.zig");

const spec = @import("riley.zig");
const mo = @import("meshpipeline.zig");
const MeshPrepared = mo.MeshPrepared;
const MeshType = mo.MeshType;
const Shader = mo.Shader;
const shaderops = @import("shaderops.zig");
const NodalPrepared = shaderops.NodalPrepared;
const TexPrepared = shaderops.TexPrepared;
const geomkerns = @import("geometrykernels.zig");
const newton = @import("newton.zig");
const shadekerns = @import("shaderkernels.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------
pub const SubpxScratchBuffs = struct {
    stride_subpx: usize,
    inv_z: []align(64) F,
    image: MatSlice(F),
    filter_tmp: MatSlice(F),
    simd_chunks: []SubpxSimdChunk,
    mask: []align(64) bool,
    xi: []align(64) F,
    eta: []align(64) F,
    touched_min_x: []usize,
    touched_max_x: []usize,
    ideal_pix_cent: []align(64) F,
};

const SubpxSimdChunk = struct {
    scratch_x_u: [S]usize,
    scratch_y_u: [S]usize,
    px_f: [S]F,
    py_f: [S]F,
    seed_xi: [S]F,
    seed_eta: [S]F,
    count: usize,
};

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn initSubpxScratch(
    arena_alloc: std.mem.Allocator,
    fields_num: u8,
    subpx_tile_size: usize,
) !SubpxScratchBuffs {
    const subpx_tile_total = subpx_tile_size * subpx_tile_size;
    // Rounds up to the nearest multiple of S for align
    const subpx_tile_total_padded = std.mem.alignForward(usize, subpx_tile_total, S);
    const mem_align = std.mem.Alignment.@"64";

    const subpx_inv_z_scratch = try arena_alloc.alignedAlloc(
        F,
        mem_align,
        subpx_tile_total_padded + S,
    );

    const subpx_mask_scratch = try arena_alloc.alignedAlloc(
        bool,
        mem_align,
        subpx_tile_total_padded + S,
    );

    const subpx_xi_scratch = try arena_alloc.alignedAlloc(
        F,
        mem_align,
        subpx_tile_total_padded + S,
    );

    const subpx_eta_scratch = try arena_alloc.alignedAlloc(
        F,
        mem_align,
        subpx_tile_total_padded + S,
    );

    const subpx_img_mem = try arena_alloc.alignedAlloc(
        F,
        mem_align,
        (subpx_tile_total_padded + S) * @as(usize, fields_num),
    );
    const subpx_image_scratch = MatSlice(F).init(
        subpx_img_mem,
        @as(usize, fields_num),
        subpx_tile_total_padded + S,
    );
    const filter_tmp_mem = try arena_alloc.alignedAlloc(
        F,
        mem_align,
        (subpx_tile_total_padded + S) * @as(usize, fields_num),
    );
    const filter_tmp = MatSlice(F).init(
        filter_tmp_mem,
        @as(usize, fields_num),
        subpx_tile_total_padded + S,
    );

    const subpx_simd_chunk_count = @divFloor(subpx_tile_total_padded + (S - 1), S) + 1;
    const subpx_simd_chunks = try arena_alloc.alloc(
        SubpxSimdChunk,
        subpx_simd_chunk_count,
    );

    const ideal_pix_cent = try arena_alloc.alignedAlloc(
        F,
        mem_align,
        (subpx_tile_total_padded + S) * 2,
    );

    return .{
        .stride_subpx = subpx_tile_size,
        .inv_z = subpx_inv_z_scratch,
        .image = subpx_image_scratch,
        .filter_tmp = filter_tmp,
        .simd_chunks = subpx_simd_chunks,
        .mask = subpx_mask_scratch,
        .xi = subpx_xi_scratch,
        .eta = subpx_eta_scratch,
        .touched_min_x = try arena_alloc.alloc(usize, subpx_tile_size),
        .touched_max_x = try arena_alloc.alloc(usize, subpx_tile_size),
        .ideal_pix_cent = ideal_pix_cent,
    };
}

pub fn resetSubpxScratch(
    subpx_scratch: *SubpxScratchBuffs,
    subpx_tile_size: usize,
    background_value: F,
) void {
    @memset(subpx_scratch.inv_z, -std.math.inf(F));
    @memset(subpx_scratch.image.slice, background_value);
    @memset(subpx_scratch.filter_tmp.slice, background_value);
    @memset(subpx_scratch.touched_min_x, subpx_tile_size);
    @memset(subpx_scratch.touched_max_x, 0);
}

pub fn rasterScene(
    comptime report_mode: ReportMode,
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    ctx_rast: rops.RasterContext,
    ctx_report: report.ReportContext(report_mode),
    requested_workers: u16,
    tiling: rops.TilingOverlaps,
    meshes: []const MeshPrepared,
    raster_hulls: []const ?NDArray(F),
    image_out_arr: *NDArray(F),
) !void {
    try comm.rasterSceneComm(
        @This(),
        report_mode,
        outer_alloc,
        io,
        ctx_rast,
        ctx_report,
        requested_workers,
        tiling,
        meshes,
        raster_hulls,
        image_out_arr,
    );
}

//------------------------------------------------------------------------------------------
// Raster Engine Builder
//------------------------------------------------------------------------------------------
const SubpxDom = comm.SubpxDom;
const RasterBounds = comm.RasterBounds;

pub fn RasterEngine(
    comptime Geom: type, // geometrykernels.zig
    comptime ShaderKern: type, // shaderkernels.zig
    comptime ShaderData: type, // shaderops_common.zig, ShaderPrepared
) type {
    return struct {
        pub fn render(
            comptime report_mode: ReportMode,
            ctx_rast: rops.RasterContext,
            ctx_report: report.ReportContext(report_mode),
            tile: rops.ActiveTile,
            overlap: rops.OverlapBBox,
            coords: *const NDArray(F),
            raster_hull: ?*const NDArray(F),
            shader: *const ShaderData,
            shader_buf: *const shaderops.LocalShaderBuff(Geom.nodes_num),
            subpx_scratch: *SubpxScratchBuffs,
        ) !u64 {
            const sub_samp_u: usize = @intCast(ctx_rast.camera.sub_sample);
            const sub_samp_f: F = @as(F, @floatFromInt(ctx_rast.camera.sub_sample));

            const subpx_dom = SubpxDom{
                .step = 1.0 / sub_samp_f,
                .offset = 1.0 / (2.0 * sub_samp_f),
                .tile_size = subpx_scratch.stride_subpx,
                .x_off = 0.5 * @as(F, @floatFromInt(ctx_rast.camera.pixels_num[0])),
                .y_off = 0.5 * @as(F, @floatFromInt(ctx_rast.camera.pixels_num[1])),
            };

            const overlap_start_x_px = overlap.x_min - tile.scratch_x_px_min;
            const overlap_end_x_px = overlap.x_max - tile.scratch_x_px_min;
            const overlap_start_y_px = overlap.y_min - tile.scratch_y_px_min;
            const overlap_end_y_px = overlap.y_max - tile.scratch_y_px_min;
            const scratch_start_x_u = sub_samp_u * @as(usize, @intCast(overlap_start_x_px));
            const scratch_end_x_u = sub_samp_u * @as(usize, @intCast(overlap_end_x_px));
            const scratch_start_y_u = sub_samp_u * @as(usize, @intCast(overlap_start_y_px));
            const scratch_end_y_u = sub_samp_u * @as(usize, @intCast(overlap_end_y_px));

            const rast_bounds = RasterBounds{
                .start_x_u = scratch_start_x_u,
                .end_x_u = scratch_end_x_u,
                .start_y_u = scratch_start_y_u,
                .end_y_u = scratch_end_y_u,
                .x_min_f = @as(F, @floatFromInt(overlap.x_min)),
                .y_min_f = @as(F, @floatFromInt(overlap.y_min)),
            };

            const nodes_coords = try rops.loadElemVec3Slices(
                Geom.nodes_num,
                F,
                coords,
                overlap.elem_idx,
            );

            const shaded_px = if (comptime Geom == geomkerns.Tri3OptKernel())
                try rasterSteppedSIMD(
                    Geom,
                    ShaderKern,
                    report_mode,
                    ctx_rast,
                    ctx_report,
                    tile,
                    overlap,
                    subpx_dom,
                    rast_bounds,
                    scratch_start_x_u,
                    nodes_coords,
                    shader,
                    shader_buf,
                    subpx_scratch,
                )
            else if (Geom.solver_kind == .hyperb)
                try rasterDirectSIMD(
                    report_mode,
                    ctx_rast,
                    ctx_report,
                    tile,
                    overlap,
                    subpx_dom,
                    rast_bounds,
                    scratch_start_x_u,
                    nodes_coords,
                    shader,
                    shader_buf,
                    subpx_scratch,
                )
            else if (Geom.solver_kind == .inv_bi)
                // NOTE: SIMD is very inefficient for highly branched inv bilinear
                // solve fallback to scalar
                try rasterDirect(
                    report_mode,
                    ctx_rast,
                    ctx_report,
                    tile,
                    overlap,
                    raster_hull,
                    subpx_dom,
                    rast_bounds,
                    nodes_coords,
                    shader,
                    shader_buf,
                    subpx_scratch,
                )
            else if (Geom.solver_kind == .newton)
                try rasterNewtonSIMD(
                    report_mode,
                    ctx_rast,
                    ctx_report,
                    tile,
                    overlap,
                    raster_hull,
                    subpx_dom,
                    rast_bounds,
                    scratch_start_x_u,
                    nodes_coords,
                    shader,
                    shader_buf,
                    subpx_scratch,
                )
            else
                @compileError("Unsupped geometry in rasterengine_simd");

            return shaded_px;
        }

        fn rasterDirectSIMD(
            comptime report_mode: ReportMode,
            ctx_rast: rops.RasterContext,
            ctx_report: report.ReportContext(report_mode),
            tile: rops.ActiveTile,
            overlap: rops.OverlapBBox,
            subpx_dom: SubpxDom,
            rast_bounds: RasterBounds,
            orig_start_x_u: usize,
            nodes_coords: Vec3Slices(F),
            shader: anytype,
            shader_buf: *const shaderops.LocalShaderBuff(Geom.nodes_num),
            subpx_scratch: *SubpxScratchBuffs,
        ) !u64 {
            return rasterDirectSIMDImpl(
                Geom,
                ShaderKern,
                report_mode,
                ctx_rast,
                ctx_report,
                tile,
                overlap,
                subpx_dom,
                rast_bounds,
                orig_start_x_u,
                nodes_coords,
                shader,
                shader_buf,
                subpx_scratch,
            );
        }

        /// To use our S wide SIMD effectively for Newton we need to process in
        /// multiple passes to fill the S lanes where possible.
        fn rasterNewtonSIMD(
            comptime report_mode: ReportMode,
            ctx_rast: rops.RasterContext,
            ctx_report: report.ReportContext(report_mode),
            tile: rops.ActiveTile,
            overlap: rops.OverlapBBox,
            raster_hull: ?*const NDArray(F),
            subpx_dom: SubpxDom,
            rast_bounds: RasterBounds,
            orig_start_x_u: usize,
            nodes_coords: Vec3Slices(F),
            shader: anytype,
            shader_buf: *const shaderops.LocalShaderBuff(Geom.nodes_num),
            subpx_scratch: *SubpxScratchBuffs,
        ) !u64 {
            return rasterNewtonSIMDImpl(
                Geom,
                ShaderKern,
                report_mode,
                ctx_rast,
                ctx_report,
                tile,
                overlap,
                raster_hull,
                subpx_dom,
                rast_bounds,
                orig_start_x_u,
                nodes_coords,
                shader,
                shader_buf,
                subpx_scratch,
            );
        }

        /// Scalar fallback for the quad4ibi kernel which didn't work well in SIMD due to
        /// the large amount of branching logic required to handle all cases.
        fn rasterDirect(
            comptime report_mode: ReportMode,
            ctx_rast: rops.RasterContext,
            ctx_report: report.ReportContext(report_mode),
            tile: rops.ActiveTile,
            overlap: rops.OverlapBBox,
            raster_hull: ?*const NDArray(F),
            subpx_dom: SubpxDom,
            rast_bounds: RasterBounds,
            nodes_coords: Vec3Slices(F),
            shader: *const ShaderData,
            shader_buf: *const shaderops.LocalShaderBuff(Geom.nodes_num),
            subpx_scratch: *SubpxScratchBuffs,
        ) !u64 {
            return rasterDirectImpl(
                Geom,
                ShaderKern,
                ShaderData,
                report_mode,
                ctx_rast,
                ctx_report,
                tile,
                overlap,
                raster_hull,
                subpx_dom,
                rast_bounds,
                nodes_coords,
                shader,
                shader_buf,
                subpx_scratch,
            );
        }
    };
}

fn rasterDirectSIMDImpl(
    comptime Geom: type,
    comptime ShaderKern: type,
    comptime report_mode: ReportMode,
    ctx_rast: rops.RasterContext,
    ctx_report: report.ReportContext(report_mode),
    tile: rops.ActiveTile,
    overlap: rops.OverlapBBox,
    subpx_dom: SubpxDom,
    rast_bounds: RasterBounds,
    orig_start_x_u: usize,
    nodes_coords: Vec3Slices(F),
    shader: anytype,
    shader_buf: *const shaderops.LocalShaderBuff(Geom.nodes_num),
    subpx_scratch: *SubpxScratchBuffs,
) !u64 {
    const N = Geom.nodes_num;
    var shaded_px: u64 = 0;
    const sub_samp: usize = @intCast(ctx_rast.camera.sub_sample);
    std.debug.assert(subpx_scratch.image.rows_num <= std.math.maxInt(u8));
    const fields_num: u8 = @intCast(subpx_scratch.image.rows_num);

    const inv_area = Geom.getInvElemArea(nodes_coords);
    const v_inv_area: VecSF = @splat(inv_area);
    var nodes_inv_z: [N]F = undefined;

    inline for (0..N) |nn| nodes_inv_z[nn] = 1.0 / nodes_coords.z[nn];
    const v_nodes_inv_z = Geom.getSIMDInvZ(nodes_coords);

    const v_orig_start_x_u: VecSU = @splat(orig_start_x_u);
    const v_end_x_u: VecSU = @splat(rast_bounds.end_x_u);
    const ideal_x_plane = cam.getIdealXPlaneScratch(
        subpx_scratch.ideal_pix_cent,
    );
    const ideal_y_plane = cam.getIdealYPlaneScratch(
        subpx_scratch.ideal_pix_cent,
    );

    for (rast_bounds.start_y_u..rast_bounds.end_y_u) |scratch_y_u| {
        const row_offset = scratch_y_u * subpx_dom.tile_size;

        var scratch_x_u = rast_bounds.start_x_u;
        while (scratch_x_u < rast_bounds.end_x_u) : (scratch_x_u += S) {
            const v_lane_idx_u: VecSU = std.simd.iota(usize, S);
            const v_scratch_x_u: VecSU = @splat(scratch_x_u);
            const v_subpx_x_u = v_scratch_x_u + v_lane_idx_u;
            const v_x_mask = (v_subpx_x_u >= v_orig_start_x_u) &
                (v_subpx_x_u < v_end_x_u);

            const scratch_idx = row_offset + scratch_x_u;
            const v_ideal_x_pix = simdops.loadVecSF(ideal_x_plane, scratch_idx);
            const v_ideal_y_pix = simdops.loadVecSF(ideal_y_plane, scratch_idx);

            ctx_report.recordSolverCalls(S);
            const res = Geom.solveWeightsHyperbSIMD(
                nodes_coords,
                v_ideal_x_pix,
                v_ideal_y_pix,
                v_inv_area,
            );
            const v_mask_active = v_x_mask & res.v_mask;

            if (comptime report_mode == .full_stats) {
                const v_inv_z_stats = Geom.calcInvZSIMD(
                    v_nodes_inv_z,
                    res.v_weights,
                );
                rasterreport.recordTri3SIMDConvStats(
                    ctx_report,
                    tile,
                    sub_samp,
                    scratch_x_u,
                    scratch_y_u,
                    v_x_mask,
                    v_mask_active,
                    res.v_weights,
                    v_inv_z_stats,
                    nodes_inv_z,
                    nodes_coords.x,
                    nodes_coords.y,
                );
            }

            if (!@reduce(.Or, v_mask_active)) continue;

            const v_inv_z = Geom.calcInvZSIMD(
                v_nodes_inv_z,
                res.v_weights,
            );
            const v_old_inv_z = simdops.loadVecSF(
                subpx_scratch.inv_z,
                scratch_idx,
            );

            const v_depth_tol: VecSF =
                @splat(tol.geometry.depth_buff_inv_z_cmp);
            const v_depth_mask =
                v_mask_active & (v_inv_z + v_depth_tol >= v_old_inv_z);

            if (!@reduce(.Or, v_depth_mask)) continue;

            const v_new_inv_z = @select(F, v_depth_mask, v_inv_z, v_old_inv_z);

            simdops.storeVecSF(subpx_scratch.inv_z, scratch_idx, v_new_inv_z);

            const v_subpx_z: VecSF = @as(VecSF, @splat(1.0)) / v_inv_z;
            const v_xi = res.v_weights[1] * v_nodes_inv_z[1] / v_inv_z;
            const v_eta = res.v_weights[2] * v_nodes_inv_z[2] / v_inv_z;

            const v_depth_mask_arr: [S]bool = v_depth_mask;
            inline for (0..S) |ll| {
                if (v_depth_mask_arr[ll]) {
                    const touched_x_u = scratch_x_u + ll;
                    if (touched_x_u < subpx_scratch.touched_min_x[scratch_y_u]) {
                        subpx_scratch.touched_min_x[scratch_y_u] = touched_x_u;
                    }
                    if (touched_x_u > subpx_scratch.touched_max_x[scratch_y_u]) {
                        subpx_scratch.touched_max_x[scratch_y_u] = touched_x_u;
                    }
                }
            }

            const v_hit_one: VecSU8 = @splat(1);
            const v_hit_zero: VecSU8 = @splat(0);
            const v_hit_count = @select(u8, v_depth_mask, v_hit_one, v_hit_zero);
            shaded_px += @intCast(@reduce(.Add, v_hit_count));

            const ctx_shade = shaderops.ShadeContext{
                .frame_idx = ctx_rast.frame_idx,
                .elem_idx = overlap.elem_idx,
                .fields_num = fields_num,
                .actual_fields = fields_num,
                .scratch_idx = scratch_idx,
                .global_subx = comm.globalSubpxForReport(
                    tile.scratch_x_px_min,
                    sub_samp,
                    scratch_x_u,
                ),
                .global_suby = comm.globalSubpxForReport(
                    tile.scratch_y_px_min,
                    sub_samp,
                    scratch_y_u,
                ),
                .v_mask_active = v_depth_mask,
            };

            ShaderKern.shadeSIMD(
                Geom.coord_space,
                ctx_shade,
                ctx_report,
                v_depth_mask,
                res.v_weights,
                v_xi,
                v_eta,
                v_nodes_inv_z,
                v_subpx_z,
                shader_buf,
                shader,
                &subpx_scratch.image,
            );
        }
    }
    return shaded_px;
}

fn rasterNewtonSIMDImpl(
    comptime Geom: type,
    comptime ShaderKern: type,
    comptime report_mode: ReportMode,
    ctx_rast: rops.RasterContext,
    ctx_report: report.ReportContext(report_mode),
    tile: rops.ActiveTile,
    overlap: rops.OverlapBBox,
    raster_hull: ?*const NDArray(F),
    subpx_dom: SubpxDom,
    rast_bounds: RasterBounds,
    orig_start_x_u: usize,
    nodes_coords: Vec3Slices(F),
    shader: anytype,
    shader_buf: *const shaderops.LocalShaderBuff(Geom.nodes_num),
    subpx_scratch: *SubpxScratchBuffs,
) !u64 {
    const N = Geom.nodes_num;
    var shaded_px: u64 = 0;

    const sub_samp: usize = @intCast(ctx_rast.camera.sub_sample);
    std.debug.assert(subpx_scratch.image.rows_num <= std.math.maxInt(u8));
    const fields_num: u8 = @intCast(subpx_scratch.image.rows_num);

    var nodes_inv_z: [N]F = undefined;
    var v_nodes_z: [N]VecSF = undefined;
    var v_nodes_inv_z: [N]VecSF = undefined;
    inline for (0..N) |nn| {
        nodes_inv_z[nn] = 1.0 / nodes_coords.z[nn];
        v_nodes_z[nn] = @splat(nodes_coords.z[nn]);
        v_nodes_inv_z[nn] = @splat(nodes_inv_z[nn]);
    }

    const maybe_raster_hull = raster_hull;
    for (rast_bounds.start_y_u..rast_bounds.end_y_u) |scratch_y_u| {
        const row_offset = scratch_y_u * subpx_dom.tile_size;
        const mask_start = row_offset + rast_bounds.start_x_u;
        const mask_end = row_offset + rast_bounds.end_x_u;
        @memset(subpx_scratch.mask[mask_start..mask_end], false);
    }

    var subpx_tess_pass_count: usize = 0;
    const v_lane_idx: VecSU = std.simd.iota(usize, S);
    const v_orig_start_x_u: VecSU = @splat(orig_start_x_u);
    const v_bounds_end_x_u: VecSU = @splat(rast_bounds.end_x_u);

    const ideal_x_plane = cam.getIdealXPlaneScratch(
        subpx_scratch.ideal_pix_cent,
    );
    const ideal_y_plane = cam.getIdealYPlaneScratch(
        subpx_scratch.ideal_pix_cent,
    );

    for (rast_bounds.start_y_u..rast_bounds.end_y_u) |scratch_y_u| {
        const row_offset = scratch_y_u * subpx_dom.tile_size;

        var scratch_x_u: usize = rast_bounds.start_x_u;
        while (scratch_x_u < rast_bounds.end_x_u) : (scratch_x_u += S) {
            const v_scratch_x_u: VecSU = @splat(scratch_x_u);
            const v_subpx_x_u = v_scratch_x_u + v_lane_idx;
            const v_x_mask = (v_subpx_x_u >= v_orig_start_x_u) &
                (v_subpx_x_u < v_bounds_end_x_u);

            const scratch_idx = row_offset + scratch_x_u;
            const v_ideal_x_pix = simdops.loadVecSF(ideal_x_plane, scratch_idx);
            const v_ideal_y_pix = simdops.loadVecSF(ideal_y_plane, scratch_idx);

            var v_mask_active = v_x_mask;
            var xi_arr = [_]F{0.0} ** S;
            var eta_arr = [_]F{0.0} ** S;

            if (maybe_raster_hull) |rh| {
                const hx = rh.getSlice(&[_]usize{ overlap.elem_idx, 0, 0 }, 1);
                const hy = rh.getSlice(&[_]usize{ overlap.elem_idx, 1, 0 }, 1);
                const elem_tess = hull.getTessellation(
                    N,
                    Geom.hull_nodes_num,
                    Geom.tess_triangles_num,
                    hx,
                    hy,
                );
                const v_hull_res: HullResultSIMD = elem_tess.isInSIMD(
                    v_ideal_x_pix,
                    v_ideal_y_pix,
                );
                const init_seed = Geom.initSeedSIMD(
                    ctx_rast.config.newton_seed_mode,
                    .{
                        .v_xi = v_hull_res.v_seed_xi,
                        .v_eta = v_hull_res.v_seed_eta,
                    },
                );
                xi_arr = init_seed.v_xi;
                eta_arr = init_seed.v_eta;

                const v_mask_one_u8: VecSU8 = @splat(1);
                const v_mask_zero_u8: VecSU8 = @splat(0);
                const v_tess_check_u8 = @select(
                    u8,
                    v_x_mask,
                    v_mask_one_u8,
                    v_mask_zero_u8,
                );
                ctx_report.recordTessChecks(@intCast(@reduce(.Add, v_tess_check_u8)));

                v_mask_active = v_x_mask & v_hull_res.v_is_in;
                const v_tess_pass_u8 = @select(
                    u8,
                    v_mask_active,
                    v_mask_one_u8,
                    v_mask_zero_u8,
                );
                ctx_report.recordTessPasses(@intCast(@reduce(.Add, v_tess_pass_u8)));
            } else {
                const init_seed = Geom.initSeed(ctx_rast.config.newton_seed_mode, null);
                @memset(&xi_arr, init_seed.xi);
                @memset(&eta_arr, init_seed.eta);
            }

            if (!@reduce(.Or, v_mask_active)) continue;

            const mask_arr: [S]bool = v_mask_active;
            const x_arr_f: [S]F = v_ideal_x_pix;
            const y_arr_f: [S]F = v_ideal_y_pix;

            for (0..S) |ss| {
                if (!mask_arr[ss]) continue;

                var seed_xi = xi_arr[ss];
                var seed_eta = eta_arr[ss];
                if (ctx_rast.config.newton_seed_mode == .hull) {
                    const hull_seed = newton.NewtonSeed{ .xi = seed_xi, .eta = seed_eta };
                    const seed_quality = newton.evaluateSeedQuality(
                        Geom.nodes_num,
                        Geom.domViolation,
                        x_arr_f[ss] - subpx_dom.x_off,
                        y_arr_f[ss] - subpx_dom.y_off,
                        nodes_coords.x,
                        nodes_coords.y,
                        nodes_coords.z,
                        hull_seed,
                    );
                    if (!seed_quality.is_usable) {
                        const centroid_seed = Geom.initSeed(
                            ctx_rast.config.newton_seed_mode,
                            null,
                        );
                        seed_xi = centroid_seed.xi;
                        seed_eta = centroid_seed.eta;
                    }
                }

                const chunk_idx = subpx_tess_pass_count / S;
                const lane_idx = subpx_tess_pass_count % S;
                if (lane_idx == 0) {
                    subpx_scratch.simd_chunks[chunk_idx] = .{
                        .scratch_x_u = [_]usize{0} ** S,
                        .scratch_y_u = [_]usize{0} ** S,
                        .px_f = [_]F{0.0} ** S,
                        .py_f = [_]F{0.0} ** S,
                        .seed_xi = [_]F{0.0} ** S,
                        .seed_eta = [_]F{0.0} ** S,
                        .count = 0,
                    };
                }

                subpx_scratch.simd_chunks[chunk_idx].scratch_x_u[lane_idx] =
                    scratch_x_u + ss;
                subpx_scratch.simd_chunks[chunk_idx].scratch_y_u[lane_idx] =
                    scratch_y_u;
                subpx_scratch.simd_chunks[chunk_idx].px_f[lane_idx] =
                    x_arr_f[ss];
                subpx_scratch.simd_chunks[chunk_idx].py_f[lane_idx] =
                    y_arr_f[ss];
                subpx_scratch.simd_chunks[chunk_idx].seed_xi[lane_idx] =
                    seed_xi;
                subpx_scratch.simd_chunks[chunk_idx].seed_eta[lane_idx] =
                    seed_eta;
                subpx_scratch.simd_chunks[chunk_idx].count = lane_idx + 1;
                subpx_tess_pass_count += 1;
            }
        }
    }

    const subpx_simd_chunk_count = @divFloor(subpx_tess_pass_count + (S - 1), S);
    var seed_state = newton.NewtonSeedState{};
    const v_full_mask: VecSB = @splat(true);

    for (0..subpx_simd_chunk_count) |chunk_idx| {
        var subpx_simd_chunk = subpx_scratch.simd_chunks[chunk_idx];
        if (ctx_rast.config.newton_seed_reuse == .last_conv) {
            newton.applySeedReuseInPlace(
                subpx_simd_chunk.count,
                seed_state,
                subpx_simd_chunk.seed_xi[0..subpx_simd_chunk.count],
                subpx_simd_chunk.seed_eta[0..subpx_simd_chunk.count],
            );
        }

        const v_targ_x_f: VecSF = subpx_simd_chunk.px_f;
        const v_targ_y_f: VecSF = subpx_simd_chunk.py_f;
        const v_xi_seed: VecSF = subpx_simd_chunk.seed_xi;
        const v_eta_seed: VecSF = subpx_simd_chunk.seed_eta;
        const v_chunk_mask: VecSB = if (subpx_simd_chunk.count == S)
            v_full_mask
        else
            v_lane_idx < @as(VecSU, @splat(subpx_simd_chunk.count));

        const result = Geom.solveWeightsNewtonSIMD(
            nodes_coords,
            v_targ_x_f,
            v_targ_y_f,
            v_xi_seed,
            v_eta_seed,
            subpx_dom.x_off,
            subpx_dom.y_off,
        );

        const v_solver_iters = @select(
            u8,
            v_chunk_mask,
            result.v_iters,
            @as(VecSU8, @splat(0)),
        );
        ctx_report.recordSolverIters(@intCast(@reduce(.Add, v_solver_iters)));
        ctx_report.recordSolverCalls(subpx_simd_chunk.count);

        const v_fail_mask = v_chunk_mask & !result.v_mask;
        if (@reduce(.Or, v_fail_mask)) {
            const v_fail_one: VecSU8 = @splat(1);
            const v_fail_zero: VecSU8 = @splat(0);
            const v_fail_count = @select(u8, v_fail_mask, v_fail_one, v_fail_zero);
            ctx_report.recordSolverDivergedCount(@intCast(@reduce(.Add, v_fail_count)));
        }

        if (comptime report_mode == .full_stats) {
            rasterreport.recordNewtonSIMDChunkStats(
                N,
                Geom.domViolation,
                ctx_report,
                tile,
                sub_samp,
                subpx_simd_chunk,
                v_chunk_mask,
                result.v_mask,
                result.v_iters,
                result.v_status,
                result.v_pre_dom_conv,
                result.v_xi_final,
                result.v_eta_final,
                subpx_dom.x_off,
                subpx_dom.y_off,
                nodes_coords.x,
                nodes_coords.y,
                nodes_coords.z,
            );
        }

        const v_conv_mask = v_chunk_mask & result.v_mask;
        if (!@reduce(.Or, v_conv_mask)) continue;

        const v_scratch_idx =
            @as(VecSU, subpx_simd_chunk.scratch_y_u) *
            @as(VecSU, @splat(subpx_dom.tile_size)) +
            @as(VecSU, subpx_simd_chunk.scratch_x_u);
        const write_mask_arr: [S]bool = v_conv_mask;
        const write_xi_arr: [S]F = result.v_xi_out;
        const write_eta_arr: [S]F = result.v_eta_out;
        const scratch_idx_arr: [S]usize = v_scratch_idx;
        for (0..S) |jj| {
            if (!write_mask_arr[jj]) continue;
            const scratch_idx = scratch_idx_arr[jj];
            subpx_scratch.xi[scratch_idx] = write_xi_arr[jj];
            subpx_scratch.eta[scratch_idx] = write_eta_arr[jj];
            subpx_scratch.mask[scratch_idx] = true;
        }

        if (ctx_rast.config.newton_seed_reuse == .last_conv) {
            newton.updateSeedStateFromSIMDResult(
                &seed_state,
                v_chunk_mask,
                result.v_mask,
                result.v_xi_out,
                result.v_eta_out,
                result.v_resid_x,
                result.v_resid_y,
            );
        }
    }

    for (rast_bounds.start_y_u..rast_bounds.end_y_u) |scratch_y_u| {
        const row_offset = scratch_y_u * subpx_dom.tile_size;

        var scratch_x_u = rast_bounds.start_x_u;
        while (scratch_x_u < rast_bounds.end_x_u) : (scratch_x_u += S) {
            const scratch_idx = row_offset + scratch_x_u;
            var mask_arr: [S]bool = undefined;
            @memcpy(
                &mask_arr,
                subpx_scratch.mask[scratch_idx .. scratch_idx + S],
            );
            const v_mask_full: VecSB = mask_arr;

            const v_scratch_x_u: VecSU = @splat(scratch_x_u);
            const v_x_mask = (v_scratch_x_u + v_lane_idx >= v_orig_start_x_u) &
                (v_scratch_x_u + v_lane_idx < v_bounds_end_x_u);
            const v_mask_active = v_mask_full & v_x_mask;
            if (!@reduce(.Or, v_mask_active)) continue;

            var xi_arr: [S]F = undefined;
            var eta_arr: [S]F = undefined;
            const xi_slice = subpx_scratch.xi[scratch_idx .. scratch_idx + S];
            const eta_slice = subpx_scratch.eta[scratch_idx .. scratch_idx + S];
            @memcpy(&xi_arr, xi_slice);
            @memcpy(&eta_arr, eta_slice);
            const v_xi: VecSF = xi_arr;
            const v_eta: VecSF = eta_arr;

            var v_weights: [N]VecSF = undefined;
            var v_dNu: [N]VecSF = undefined;
            var v_dNv: [N]VecSF = undefined;
            shapefun.shapeFuncSIMD(N, v_xi, v_eta, &v_weights, &v_dNu, &v_dNv);

            var v_sum_z: VecSF = @splat(0.0);
            inline for (0..N) |nn| v_sum_z += v_weights[nn] * v_nodes_z[nn];
            const v_inv_z: VecSF = @as(VecSF, @splat(1.0)) / v_sum_z;

            const v_old_inv_z = simdops.loadVecSF(
                subpx_scratch.inv_z,
                scratch_idx,
            );
            const v_depth_tol: VecSF = @splat(tol.geometry.depth_buff_inv_z_cmp);
            const v_depth_mask = v_mask_active & (v_inv_z + v_depth_tol >= v_old_inv_z);

            if (!@reduce(.Or, v_depth_mask)) continue;

            const v_new_inv_z = @select(F, v_depth_mask, v_inv_z, v_old_inv_z);

            simdops.storeVecSF(subpx_scratch.inv_z, scratch_idx, v_new_inv_z);
            const v_subpx_z: VecSF = @as(VecSF, @splat(1.0)) / v_inv_z;

            const lane_depth_mask: [S]bool = v_depth_mask;
            inline for (0..S) |ll| {
                if (lane_depth_mask[ll]) {
                    const touched_x_u = scratch_x_u + ll;
                    if (touched_x_u < subpx_scratch.touched_min_x[scratch_y_u]) {
                        subpx_scratch.touched_min_x[scratch_y_u] = touched_x_u;
                    }
                    if (touched_x_u > subpx_scratch.touched_max_x[scratch_y_u]) {
                        subpx_scratch.touched_max_x[scratch_y_u] = touched_x_u;
                    }
                }
            }

            const v_hit_one: VecSU8 = @splat(1);
            const v_hit_zero: VecSU8 = @splat(0);
            const v_hit_count = @select(u8, v_depth_mask, v_hit_one, v_hit_zero);
            shaded_px += @intCast(@reduce(.Add, v_hit_count));

            const ctx_shade = shaderops.ShadeContext{
                .frame_idx = ctx_rast.frame_idx,
                .elem_idx = overlap.elem_idx,
                .fields_num = fields_num,
                .actual_fields = fields_num,
                .scratch_idx = scratch_idx,
                .global_subx = comm.globalSubpxForReport(
                    tile.scratch_x_px_min,
                    sub_samp,
                    scratch_x_u,
                ),
                .global_suby = comm.globalSubpxForReport(
                    tile.scratch_y_px_min,
                    sub_samp,
                    scratch_y_u,
                ),
                .v_mask_active = v_depth_mask,
            };

            ShaderKern.shadeSIMD(
                Geom.coord_space,
                ctx_shade,
                ctx_report,
                v_depth_mask,
                v_weights,
                v_xi,
                v_eta,
                v_nodes_inv_z,
                v_subpx_z,
                shader_buf,
                shader,
                &subpx_scratch.image,
            );
        }
    }

    return shaded_px;
}

fn rasterDirectImpl(
    comptime Geom: type,
    comptime ShaderKern: type,
    comptime ShaderData: type,
    comptime report_mode: ReportMode,
    ctx_rast: rops.RasterContext,
    ctx_report: report.ReportContext(report_mode),
    tile: rops.ActiveTile,
    overlap: rops.OverlapBBox,
    _: ?*const NDArray(F),
    subpx_dom: SubpxDom,
    rast_bounds: RasterBounds,
    nodes_coords: Vec3Slices(F),
    shader: *const ShaderData,
    shader_buf: *const shaderops.LocalShaderBuff(Geom.nodes_num),
    subpx_scratch: *SubpxScratchBuffs,
) !u64 {
    std.debug.assert(subpx_scratch.image.rows_num <= std.math.maxInt(u8));
    const fields_num: u8 = @intCast(subpx_scratch.image.rows_num);
    return comm.rasterDirectScalComm(
        Geom,
        ShaderKern,
        ShaderData,
        report_mode,
        SubpxScratchBuffs,
        ctx_rast,
        ctx_report,
        tile,
        overlap,
        subpx_dom,
        rast_bounds,
        fields_num,
        nodes_coords,
        shader,
        shader_buf,
        subpx_scratch,
    );
}

fn rasterSteppedSIMD(
    comptime Geom: type,
    comptime ShaderKern: type,
    comptime report_mode: ReportMode,
    ctx_rast: rops.RasterContext,
    ctx_report: report.ReportContext(report_mode),
    tile: rops.ActiveTile,
    overlap: rops.OverlapBBox,
    subpx_dom: SubpxDom,
    rast_bounds: RasterBounds,
    orig_start_x_u: usize,
    nodes_coords: Vec3Slices(F),
    shader: anytype,
    shader_buf: *const shaderops.LocalShaderBuff(Geom.nodes_num),
    subpx_scratch: *SubpxScratchBuffs,
) !u64 {
    const sub_samp: usize = @intCast(ctx_rast.camera.sub_sample);
    const tile_subpx_x = @as(isize, tile.scratch_x_px_min) *
        @as(isize, @intCast(sub_samp));
    const tile_subpx_y = @as(isize, tile.scratch_y_px_min) *
        @as(isize, @intCast(sub_samp));
    const start_subx_global = tile_subpx_x + @as(isize, @intCast(rast_bounds.start_x_u));
    const start_suby_global = tile_subpx_y + @as(isize, @intCast(rast_bounds.start_y_u));
    const width = rast_bounds.end_x_u - orig_start_x_u;
    const height = rast_bounds.end_y_u - rast_bounds.start_y_u;
    const aligned_width = std.mem.alignForward(usize, width, S);
    const max_x_steps = if (aligned_width > 0) aligned_width - 1 else 0;
    const max_y_steps = if (height > 0) height - 1 else 0;

    if (comm.Tri3FixedEdges.init(
        nodes_coords,
        sub_samp,
        start_subx_global,
        start_suby_global,
        max_x_steps,
        max_y_steps,
    )) |fixed| {
        return rasterSteppedSIMDFixP(
            Geom,
            ShaderKern,
            report_mode,
            ctx_rast,
            ctx_report,
            tile,
            overlap,
            subpx_dom,
            rast_bounds,
            orig_start_x_u,
            nodes_coords,
            shader,
            shader_buf,
            subpx_scratch,
            fixed,
        );
    }

    return rasterSteppedSIMDFloat(
        Geom,
        ShaderKern,
        report_mode,
        ctx_rast,
        ctx_report,
        tile,
        overlap,
        subpx_dom,
        rast_bounds,
        orig_start_x_u,
        nodes_coords,
        shader,
        shader_buf,
        subpx_scratch,
    );
}

fn rasterSteppedSIMDFixP(
    comptime Geom: type,
    comptime ShaderKern: type,
    comptime report_mode: ReportMode,
    ctx_rast: rops.RasterContext,
    ctx_report: report.ReportContext(report_mode),
    tile: rops.ActiveTile,
    overlap: rops.OverlapBBox,
    subpx_dom: SubpxDom,
    rast_bounds: RasterBounds,
    orig_start_x_u: usize,
    nodes_coords: Vec3Slices(F),
    shader: anytype,
    shader_buf: *const shaderops.LocalShaderBuff(Geom.nodes_num),
    subpx_scratch: *SubpxScratchBuffs,
    fixed: comm.Tri3FixedEdges,
) !u64 {
    var shaded_px: u64 = 0;
    const sub_samp: usize = @intCast(ctx_rast.camera.sub_sample);
    std.debug.assert(subpx_scratch.image.rows_num <= std.math.maxInt(u8));
    const fields_num: u8 = @intCast(subpx_scratch.image.rows_num);

    var x: [3]F = undefined;
    var y: [3]F = undefined;
    var z: [3]F = undefined;
    var inv_z_node: [3]F = undefined;
    inline for (0..3) |nn| {
        x[nn] = nodes_coords.x[nn];
        y[nn] = nodes_coords.y[nn];
        z[nn] = nodes_coords.z[nn];
        inv_z_node[nn] = 1.0 / z[nn];
    }
    const dx: [2]F = .{ x[2] - x[0], x[1] - x[0] };
    const dy: [2]F = .{ y[1] - y[0], y[2] - y[0] };
    const area = @mulAdd(F, dx[0], dy[0], -(dy[1] * dx[1]));

    const is_const_depth = z[0] == z[1] and z[1] == z[2];
    const v_nodes_inv_z = Geom.getSIMDInvZ(nodes_coords);

    const v_orig_start_x_u: VecSU = @splat(orig_start_x_u);
    const v_end_x_u: VecSU = @splat(rast_bounds.end_x_u);

    var v_lane_i: buildconfig.VecSTri3FixedEdge = undefined;
    inline for (0..S) |ii| {
        v_lane_i[ii] = @intCast(ii);
    }
    const step_group: buildconfig.Tri3FixedEdge = @intCast(S);
    var v_step_x_s: [3]buildconfig.VecSTri3FixedEdge = undefined;
    inline for (0..3) |nn| {
        v_step_x_s[nn] = @splat(fixed.step_x[nn] * step_group);
    }
    const v_fixed_inv_area: VecSF = @splat(fixed.inv_area);
    const v_edge_min: buildconfig.VecSTri3FixedEdge =
        @splat(-fixed.edge_tol);

    for (rast_bounds.start_y_u..rast_bounds.end_y_u) |scratch_y_u| {
        const row_offset = scratch_y_u * subpx_dom.tile_size;
        const global_suby = comm.globalSubpxForReport(
            tile.scratch_y_px_min,
            sub_samp,
            scratch_y_u,
        );
        const y_steps: buildconfig.Tri3FixedEdge = @intCast(
            scratch_y_u - rast_bounds.start_y_u,
        );

        var edge_row: [3]buildconfig.Tri3FixedEdge = undefined;
        var v_edge: [3]buildconfig.VecSTri3FixedEdge = undefined;
        inline for (0..3) |nn| {
            edge_row[nn] = fixed.start[nn] + y_steps * fixed.step_y[nn];
            v_edge[nn] =
                @as(buildconfig.VecSTri3FixedEdge, @splat(edge_row[nn])) +
                v_lane_i * @as(
                    buildconfig.VecSTri3FixedEdge,
                    @splat(fixed.step_x[nn]),
                );
        }

        var scratch_x_u = rast_bounds.start_x_u;
        while (scratch_x_u < rast_bounds.end_x_u) : ({
            scratch_x_u += S;
            inline for (0..3) |nn| {
                v_edge[nn] += v_step_x_s[nn];
            }
        }) {
            const v_lane_idx_u: VecSU = std.simd.iota(usize, S);
            const v_scratch_x_u: VecSU = @splat(scratch_x_u);
            const v_subpx_x_u = v_scratch_x_u + v_lane_idx_u;
            const v_x_mask = (v_subpx_x_u >= v_orig_start_x_u) & (v_subpx_x_u < v_end_x_u);
            const v_in_tri = (v_edge[0] >= v_edge_min) &
                (v_edge[1] >= v_edge_min) &
                (v_edge[2] >= v_edge_min);
            const v_mask_active = v_x_mask & v_in_tri;

            const scratch_idx = row_offset + scratch_x_u;

            ctx_report.recordSolverCalls(S);
            var v_w0: VecSF = undefined;
            var v_w1: VecSF = undefined;
            var v_w2: VecSF = undefined;
            var v_inv_z: VecSF = undefined;

            if (comptime report_mode == .full_stats) {
                if (@reduce(.Or, v_mask_active)) {
                    v_w1 = @as(VecSF, @floatFromInt(v_edge[1])) * v_fixed_inv_area;
                    v_w2 = @as(VecSF, @floatFromInt(v_edge[2])) * v_fixed_inv_area;
                    v_w0 = @as(VecSF, @splat(1.0)) - v_w1 - v_w2;
                    const v_inv_z_tail = @mulAdd(
                        VecSF,
                        v_w1,
                        @as(VecSF, @splat(inv_z_node[1])),
                        v_w2 * @as(VecSF, @splat(inv_z_node[2])),
                    );
                    v_inv_z = if (is_const_depth)
                        @as(VecSF, @splat(inv_z_node[0]))
                    else
                        @mulAdd(
                            VecSF,
                            v_w0,
                            @as(VecSF, @splat(inv_z_node[0])),
                            v_inv_z_tail,
                        );
                }

                const v_zero: VecSF = @splat(0.0);
                const v_stats_w0 = if (@reduce(.Or, v_mask_active)) v_w0 else v_zero;
                const v_stats_w1 = if (@reduce(.Or, v_mask_active)) v_w1 else v_zero;
                const v_stats_w2 = if (@reduce(.Or, v_mask_active)) v_w2 else v_zero;
                const v_stats_inv_z = if (@reduce(.Or, v_mask_active)) v_inv_z else v_zero;

                rasterreport.recordTri3SteppedSIMDConvStats(
                    ctx_report,
                    tile,
                    sub_samp,
                    scratch_x_u,
                    global_suby,
                    v_x_mask,
                    v_in_tri,
                    .{ v_stats_w0, v_stats_w1, v_stats_w2 },
                    v_stats_inv_z,
                    is_const_depth,
                    inv_z_node[1],
                    inv_z_node[2],
                    area,
                );
            }

            if (!@reduce(.Or, v_mask_active)) continue;

            if (comptime report_mode != .full_stats) {
                v_w1 = @as(VecSF, @floatFromInt(v_edge[1])) * v_fixed_inv_area;
                v_w2 = @as(VecSF, @floatFromInt(v_edge[2])) * v_fixed_inv_area;
                v_w0 = @as(VecSF, @splat(1.0)) - v_w1 - v_w2;
                const v_inv_z_tail = @mulAdd(
                    VecSF,
                    v_w1,
                    @as(VecSF, @splat(inv_z_node[1])),
                    v_w2 * @as(VecSF, @splat(inv_z_node[2])),
                );
                v_inv_z = if (is_const_depth)
                    @as(VecSF, @splat(inv_z_node[0]))
                else
                    @mulAdd(
                        VecSF,
                        v_w0,
                        @as(VecSF, @splat(inv_z_node[0])),
                        v_inv_z_tail,
                    );
            }

            const v_old_inv_z = simdops.loadVecSF(subpx_scratch.inv_z, scratch_idx);
            const v_depth_tol: VecSF = @splat(tol.geometry.depth_buff_inv_z_cmp);
            const v_depth_mask = v_mask_active & (v_inv_z + v_depth_tol >= v_old_inv_z);
            if (!@reduce(.Or, v_depth_mask)) continue;

            const v_new_inv_z = @select(F, v_depth_mask, v_inv_z, v_old_inv_z);
            simdops.storeVecSF(subpx_scratch.inv_z, scratch_idx, v_new_inv_z);

            const v_subpx_z = @as(VecSF, @splat(1.0)) / v_inv_z;

            const v_xi_num = @mulAdd(
                VecSF,
                v_w1,
                @as(VecSF, @splat(inv_z_node[1])),
                @as(VecSF, @splat(0.0)),
            );
            const v_eta_num = @mulAdd(
                VecSF,
                v_w2,
                @as(VecSF, @splat(inv_z_node[2])),
                @as(VecSF, @splat(0.0)),
            );
            const v_xi = if (is_const_depth) v_w1 else v_xi_num / v_inv_z;
            const v_eta = if (is_const_depth) v_w2 else v_eta_num / v_inv_z;

            const v_depth_mask_arr: [S]bool = v_depth_mask;
            inline for (0..S) |ll| {
                if (v_depth_mask_arr[ll]) {
                    const touched_x_u = scratch_x_u + ll;
                    if (touched_x_u < subpx_scratch.touched_min_x[scratch_y_u]) {
                        subpx_scratch.touched_min_x[scratch_y_u] = touched_x_u;
                    }
                    if (touched_x_u > subpx_scratch.touched_max_x[scratch_y_u]) {
                        subpx_scratch.touched_max_x[scratch_y_u] = touched_x_u;
                    }
                }
            }

            const v_hit_one: VecSU8 = @splat(1);
            const v_hit_zero: VecSU8 = @splat(0);
            const v_hit_count = @select(u8, v_depth_mask, v_hit_one, v_hit_zero);
            shaded_px += @intCast(@reduce(.Add, v_hit_count));

            const ctx_shade = shaderops.ShadeContext{
                .frame_idx = ctx_rast.frame_idx,
                .elem_idx = overlap.elem_idx,
                .fields_num = fields_num,
                .actual_fields = fields_num,
                .scratch_idx = scratch_idx,
                .global_subx = comm.globalSubpxForReport(
                    tile.scratch_x_px_min,
                    sub_samp,
                    scratch_x_u,
                ),
                .global_suby = comm.globalSubpxForReport(
                    tile.scratch_y_px_min,
                    sub_samp,
                    scratch_y_u,
                ),
                .v_mask_active = v_depth_mask,
            };

            const v_weights = [3]VecSF{ v_w0, v_w1, v_w2 };

            ShaderKern.shadeSIMD(
                Geom.coord_space,
                ctx_shade,
                ctx_report,
                v_depth_mask,
                v_weights,
                v_xi,
                v_eta,
                v_nodes_inv_z,
                v_subpx_z,
                shader_buf,
                shader,
                &subpx_scratch.image,
            );
        }
    }

    return shaded_px;
}

fn rasterSteppedSIMDFloat(
    comptime Geom: type,
    comptime ShaderKern: type,
    comptime report_mode: ReportMode,
    ctx_rast: rops.RasterContext,
    ctx_report: report.ReportContext(report_mode),
    tile: rops.ActiveTile,
    overlap: rops.OverlapBBox,
    subpx_dom: SubpxDom,
    rast_bounds: RasterBounds,
    orig_start_x_u: usize,
    nodes_coords: Vec3Slices(F),
    shader: anytype,
    shader_buf: *const shaderops.LocalShaderBuff(Geom.nodes_num),
    subpx_scratch: *SubpxScratchBuffs,
) !u64 {
    var shaded_px: u64 = 0;
    const sub_samp: usize = @intCast(ctx_rast.camera.sub_sample);
    std.debug.assert(subpx_scratch.image.rows_num <= std.math.maxInt(u8));
    const fields_num: u8 = @intCast(subpx_scratch.image.rows_num);

    var x: [3]F = undefined;
    var y: [3]F = undefined;
    var z: [3]F = undefined;
    var inv_z_node: [3]F = undefined;
    inline for (0..3) |nn| {
        x[nn] = nodes_coords.x[nn];
        y[nn] = nodes_coords.y[nn];
        z[nn] = nodes_coords.z[nn];
        inv_z_node[nn] = 1.0 / z[nn];
    }

    const dx: [2]F = .{ x[2] - x[0], x[1] - x[0] };
    const dy: [2]F = .{ y[1] - y[0], y[2] - y[0] };
    const area = @mulAdd(F, dx[0], dy[0], -(dy[1] * dx[1]));
    const inv_area = 1.0 / area;

    var a: [3]F = undefined;
    var b: [3]F = undefined;
    var c: [3]F = undefined;
    a[0] = (y[2] - y[1]) * inv_area;
    a[1] = (y[0] - y[2]) * inv_area;
    a[2] = (y[1] - y[0]) * inv_area;
    b[0] = (x[1] - x[2]) * inv_area;
    b[1] = (x[2] - x[0]) * inv_area;
    b[2] = (x[0] - x[1]) * inv_area;
    c[0] = @mulAdd(F, x[2], y[1], -(x[1] * y[2])) * inv_area;
    c[1] = @mulAdd(F, x[0], y[2], -(x[2] * y[0])) * inv_area;
    c[2] = @mulAdd(F, x[1], y[0], -(x[0] * y[1])) * inv_area;

    const step = subpx_dom.step;
    const offset = subpx_dom.offset;

    var dw_dx: [3]F = undefined;
    var dw_dy: [3]F = undefined;
    inline for (0..3) |nn| {
        dw_dx[nn] = a[nn] * step;
        dw_dy[nn] = b[nn] * step;
    }

    const tile_subx_off = @as(isize, tile.scratch_x_px_min) *
        @as(isize, @intCast(sub_samp));
    const tile_suby_off = @as(isize, tile.scratch_y_px_min) *
        @as(isize, @intCast(sub_samp));

    const is_const_depth = z[0] == z[1] and z[1] == z[2];
    const v_nodes_inv_z = Geom.getSIMDInvZ(nodes_coords);

    const edge_tol = tol.edge.tri_weight_inclusion;

    const start_subx_global = tile_subx_off +
        @as(isize, @intCast(rast_bounds.start_x_u));
    const start_suby_global = tile_suby_off +
        @as(isize, @intCast(rast_bounds.start_y_u));

    const start_subx_f = @as(F, @floatFromInt(start_subx_global));
    const start_suby_f = @as(F, @floatFromInt(start_suby_global));
    const x_start_pix = @mulAdd(F, start_subx_f, step, offset);
    const y_start_pix = @mulAdd(F, start_suby_f, step, offset);

    var w_start_y: [3]F = undefined;
    var w_start: [3]F = undefined;
    inline for (0..3) |nn| {
        w_start_y[nn] = @mulAdd(F, b[nn], y_start_pix, c[nn]);
        w_start[nn] = @mulAdd(F, a[nn], x_start_pix, w_start_y[nn]);
    }

    const v_orig_start_x_u: VecSU = @splat(orig_start_x_u);
    const v_end_x_u: VecSU = @splat(rast_bounds.end_x_u);

    var v_lane_f: VecSF = undefined;
    inline for (0..S) |ii| {
        v_lane_f[ii] = @as(F, @floatFromInt(ii));
    }

    var v_dw_dx_s: [3]VecSF = undefined;
    inline for (0..3) |nn| {
        v_dw_dx_s[nn] = @as(VecSF, @splat(dw_dx[nn] * @as(F, @floatFromInt(S))));
    }

    for (rast_bounds.start_y_u..rast_bounds.end_y_u) |scratch_y_u| {
        const row_offset = scratch_y_u * subpx_dom.tile_size;
        const global_suby = comm.globalSubpxForReport(
            tile.scratch_y_px_min,
            sub_samp,
            scratch_y_u,
        );
        const y_steps = @as(F, @floatFromInt(scratch_y_u - rast_bounds.start_y_u));

        var w_row: [3]F = undefined;
        inline for (0..3) |nn| {
            w_row[nn] = @mulAdd(F, y_steps, dw_dy[nn], w_start[nn]);
        }

        var v_w: [3]VecSF = undefined;
        inline for (0..3) |nn| {
            v_w[nn] = @mulAdd(
                VecSF,
                @as(VecSF, @splat(dw_dx[nn])),
                v_lane_f,
                @as(VecSF, @splat(w_row[nn])),
            );
        }

        var scratch_x_u = rast_bounds.start_x_u;
        while (scratch_x_u < rast_bounds.end_x_u) : ({
            scratch_x_u += S;
            inline for (0..3) |nn| {
                v_w[nn] += v_dw_dx_s[nn];
            }
        }) {
            const v_lane_idx_u: VecSU = std.simd.iota(usize, S);
            const v_scratch_x_u: VecSU = @splat(scratch_x_u);
            const v_subpx_x_u = v_scratch_x_u + v_lane_idx_u;
            const v_x_mask = (v_subpx_x_u >= v_orig_start_x_u) &
                (v_subpx_x_u < v_end_x_u);

            const v_edge_tol: VecSF = @splat(-edge_tol);
            const v_in_tri = (v_w[0] >= v_edge_tol) &
                (v_w[1] >= v_edge_tol) &
                (v_w[2] >= v_edge_tol);
            const v_mask_active = v_x_mask & v_in_tri;

            const scratch_idx = row_offset + scratch_x_u;

            ctx_report.recordSolverCalls(S);

            const v_inv_z_tail = @mulAdd(
                VecSF,
                v_w[1],
                @as(VecSF, @splat(inv_z_node[1])),
                v_w[2] * @as(VecSF, @splat(inv_z_node[2])),
            );

            const v_inv_z = if (is_const_depth)
                @as(VecSF, @splat(inv_z_node[0]))
            else
                @mulAdd(
                    VecSF,
                    v_w[0],
                    @as(VecSF, @splat(inv_z_node[0])),
                    v_inv_z_tail,
                );

            if (comptime report_mode == .full_stats) {
                rasterreport.recordTri3SteppedSIMDConvStats(
                    ctx_report,
                    tile,
                    sub_samp,
                    scratch_x_u,
                    global_suby,
                    v_x_mask,
                    v_in_tri,
                    v_w,
                    v_inv_z,
                    is_const_depth,
                    inv_z_node[1],
                    inv_z_node[2],
                    area,
                );
            }

            if (!@reduce(.Or, v_mask_active)) continue;

            const v_old_inv_z = simdops.loadVecSF(subpx_scratch.inv_z, scratch_idx);
            const v_depth_tol: VecSF = @splat(tol.geometry.depth_buff_inv_z_cmp);
            const v_depth_mask = v_mask_active & (v_inv_z + v_depth_tol >= v_old_inv_z);

            if (!@reduce(.Or, v_depth_mask)) continue;

            const v_new_inv_z = @select(F, v_depth_mask, v_inv_z, v_old_inv_z);
            simdops.storeVecSF(subpx_scratch.inv_z, scratch_idx, v_new_inv_z);

            const v_subpx_z = @as(VecSF, @splat(1.0)) / v_inv_z;

            const v_xi_num = @mulAdd(
                VecSF,
                v_w[1],
                @as(VecSF, @splat(inv_z_node[1])),
                @as(VecSF, @splat(0.0)),
            );
            const v_eta_num = @mulAdd(
                VecSF,
                v_w[2],
                @as(VecSF, @splat(inv_z_node[2])),
                @as(VecSF, @splat(0.0)),
            );
            const v_xi = if (is_const_depth) v_w[1] else v_xi_num / v_inv_z;
            const v_eta = if (is_const_depth) v_w[2] else v_eta_num / v_inv_z;

            const v_depth_mask_arr: [S]bool = v_depth_mask;
            inline for (0..S) |ll| {
                if (v_depth_mask_arr[ll]) {
                    const touched_x_u = scratch_x_u + ll;
                    if (touched_x_u < subpx_scratch.touched_min_x[scratch_y_u]) {
                        subpx_scratch.touched_min_x[scratch_y_u] = touched_x_u;
                    }
                    if (touched_x_u > subpx_scratch.touched_max_x[scratch_y_u]) {
                        subpx_scratch.touched_max_x[scratch_y_u] = touched_x_u;
                    }
                }
            }

            const v_hit_one: VecSU8 = @splat(1);
            const v_hit_zero: VecSU8 = @splat(0);
            const v_hit_count = @select(u8, v_depth_mask, v_hit_one, v_hit_zero);
            shaded_px += @intCast(@reduce(.Add, v_hit_count));

            const ctx_shade = shaderops.ShadeContext{
                .frame_idx = ctx_rast.frame_idx,
                .elem_idx = overlap.elem_idx,
                .fields_num = fields_num,
                .actual_fields = fields_num,
                .scratch_idx = scratch_idx,
                .global_subx = comm.globalSubpxForReport(
                    tile.scratch_x_px_min,
                    sub_samp,
                    scratch_x_u,
                ),
                .global_suby = comm.globalSubpxForReport(
                    tile.scratch_y_px_min,
                    sub_samp,
                    scratch_y_u,
                ),
                .v_mask_active = v_depth_mask,
            };

            ShaderKern.shadeSIMD(
                Geom.coord_space,
                ctx_shade,
                ctx_report,
                v_depth_mask,
                v_w,
                v_xi,
                v_eta,
                v_nodes_inv_z,
                v_subpx_z,
                shader_buf,
                shader,
                &subpx_scratch.image,
            );
        }
    }

    return shaded_px;
}
