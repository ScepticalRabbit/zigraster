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
const tol = buildconfig.config.tol;
const cam = @import("camera.zig");
const CameraPrepared = cam.CameraPrepared;
const MatSlice = @import("matslice.zig").MatSlice;
const NDArray = @import("ndarray.zig").NDArray;
const hull = @import("hull.zig");
const rops = @import("rasterops.zig");
const ElemBBox = rops.ElemBBox;
const OverlapBBox = rops.OverlapBBox;
const ActiveTile = rops.ActiveTile;
const Vec3Slices = rops.Vec3Slices;
const report = @import("report.zig");
const ReportMode = report.ReportMode;
const Timestamp = std.Io.Clock.Timestamp;
const comm = @import("rasterengine_common.zig");
const rasterreport = @import("rasterreport.zig");

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
    inv_z: []F,
    image: MatSlice(F),
    filter_tmp: MatSlice(F),
    touched_min_x: []usize,
    touched_max_x: []usize,
    ideal_pix_cent: []F,
};

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn initSubpxScratch(
    arena_alloc: std.mem.Allocator,
    fields_num: u8,
    subpx_tile_size: usize,
) !SubpxScratchBuffs {
    const subpx_tile_total: usize = subpx_tile_size * subpx_tile_size;
    const subpx_inv_z_scratch = try arena_alloc.alloc(F, subpx_tile_total);
    const subpx_img_mem = try arena_alloc.alloc(
        F,
        subpx_tile_total * @as(usize, fields_num),
    );
    const subpx_image_scratch = MatSlice(F).init(
        subpx_img_mem,
        @as(usize, fields_num),
        subpx_tile_total,
    );
    const filter_tmp_mem = try arena_alloc.alloc(
        F,
        subpx_tile_total * @as(usize, fields_num),
    );
    const filter_tmp = MatSlice(F).init(
        filter_tmp_mem,
        @as(usize, fields_num),
        subpx_tile_total,
    );

    const ideal_pix_cent = try arena_alloc.alloc(F, subpx_tile_total * 2);

    return .{
        .stride_subpx = subpx_tile_size,
        .inv_z = subpx_inv_z_scratch,
        .image = subpx_image_scratch,
        .filter_tmp = filter_tmp,
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
    comptime Geom: type,
    comptime ShaderKern: type,
    comptime ShaderData: type,
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

            const shaded_px = try rasterDirect(
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

            return shaded_px;
        }

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
            if (comptime Geom == geomkerns.Tri3OptKernel()) {
                return rasterSteppedScal(
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
            if (comptime Geom.solver_kind != .newton) {
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

            return rasterNewtonImpl(
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

        fn rasterNewton(
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
            return rasterNewtonImpl(
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

fn rasterNewtonImpl(
    comptime Geom: type,
    comptime ShaderKern: type,
    comptime ShaderData: type,
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
    comptime {
        if (Geom.solver_kind != .newton) {
            @compileError("rasterNewton only supps Newton geometries");
        }
    }

    const N = Geom.nodes_num;
    var shaded_px: u64 = 0;
    const sub_samp: usize = @intCast(ctx_rast.camera.sub_sample);
    std.debug.assert(subpx_scratch.image.rows_num <= std.math.maxInt(u8));
    const fields_num: u8 = @intCast(subpx_scratch.image.rows_num);

    var nodes_inv_z: [N]F = undefined;
    inline for (0..N) |nn| {
        nodes_inv_z[nn] = 1.0 / nodes_coords.z[nn];
    }

    var elem_tess: hull.Tessellation(Geom.tess_triangles_num) = undefined;
    if (comptime Geom.hull_nodes_num > 0) {
        if (raster_hull) |rh| {
            const hx = rh.getSlice(
                &[_]usize{ overlap.elem_idx, 0, 0 },
                1,
            );
            const hy = rh.getSlice(
                &[_]usize{ overlap.elem_idx, 1, 0 },
                1,
            );
            elem_tess = hull.getTessellation(
                N,
                Geom.hull_nodes_num,
                Geom.tess_triangles_num,
                hx,
                hy,
            );
        }
    }

    var seed_state = newton.NewtonSeedState{};
    const ideal_x_plane = cam.getIdealXPlaneScratch(
        subpx_scratch.ideal_pix_cent,
    );
    const ideal_y_plane = cam.getIdealYPlaneScratch(
        subpx_scratch.ideal_pix_cent,
    );

    for (rast_bounds.start_y_u..rast_bounds.end_y_u) |scratch_y| {
        const row_offset = scratch_y * subpx_dom.tile_size;

        for (rast_bounds.start_x_u..rast_bounds.end_x_u) |scratch_x| {
            const scratch_idx = row_offset + scratch_x;
            const ideal_x_pix = ideal_x_plane[scratch_idx];
            const ideal_y_pix = ideal_y_plane[scratch_idx];

            const global_subx = comm.globalSubpxForReport(
                tile.scratch_x_px_min,
                sub_samp,
                scratch_x,
            );
            const global_suby = comm.globalSubpxForReport(
                tile.scratch_y_px_min,
                sub_samp,
                scratch_y,
            );

            var hull_seed: ?newton.NewtonSeed = null;
            if (comptime Geom.hull_nodes_num > 0) {
                ctx_report.recordTessChecks(1);
                const tess_res = elem_tess.isInScalar(ideal_x_pix, ideal_y_pix);
                if (tess_res.is_in) {
                    ctx_report.recordTessPasses(1);
                    hull_seed = .{
                        .xi = tess_res.seed_xi,
                        .eta = tess_res.seed_eta,
                    };
                }
                if (comptime report_mode == .full_stats) {
                    rasterreport.recordEarlyOut(
                        ctx_report,
                        global_subx,
                        global_suby,
                        tess_res.is_in,
                    );
                }
                if (!tess_res.is_in) continue;
            } else {
                if (comptime report_mode == .full_stats) {
                    rasterreport.recordEarlyOut(
                        ctx_report,
                        global_subx,
                        global_suby,
                        true,
                    );
                }
            }

            ctx_report.recordSolverCalls(1);
            const result = blk: {
                if (ctx_rast.config.newton_seed_mode == .hull) {
                    if (hull_seed) |seed| {
                        const seed_quality = newton.evaluateSeedQuality(
                            Geom.nodes_num,
                            Geom.domViolation,
                            ideal_x_pix - subpx_dom.x_off,
                            ideal_y_pix - subpx_dom.y_off,
                            nodes_coords.x,
                            nodes_coords.y,
                            nodes_coords.z,
                            seed,
                        );
                        if (!seed_quality.is_usable) {
                            hull_seed = null;
                        }
                    }
                }
                const base_seed = Geom.initSeed(
                    ctx_rast.config.newton_seed_mode,
                    hull_seed,
                );
                const selected_seed = newton.selectSeed(
                    ctx_rast.config.newton_seed_reuse,
                    base_seed,
                    seed_state,
                );
                break :blk Geom.solveWeightsNewton(
                    nodes_coords,
                    ideal_x_pix,
                    ideal_y_pix,
                    subpx_dom.x_off,
                    subpx_dom.y_off,
                    selected_seed.xi,
                    selected_seed.eta,
                );
            };

            ctx_report.recordSolverIters(result.iters);
            const solve_state = newton.evaluateSolveState(
                N,
                ideal_x_pix - subpx_dom.x_off,
                ideal_y_pix - subpx_dom.y_off,
                nodes_coords.x,
                nodes_coords.y,
                nodes_coords.z,
                result.xi_final,
                result.eta_final,
            );
            const dom_violation = Geom.domViolation(
                result.xi_final,
                result.eta_final,
            );
            const hit_iter_lim = newton.hitIterLimitStatus(result.status);
            const jac_det = newton.calcJacDet2D(
                N,
                result.xi_final,
                result.eta_final,
                nodes_coords.x,
                nodes_coords.y,
            );

            if (result.weights == null) {
                if (comptime report_mode == .full_stats) {
                    rasterreport.recordPixelConvStats(
                        ctx_report,
                        global_subx,
                        global_suby,
                        false,
                        result.xi_final,
                        result.eta_final,
                        jac_det,
                    );

                    rasterreport.recordPixelSolverDiagnostics(
                        ctx_report,
                        global_subx,
                        global_suby,
                        result.status,
                        result.pre_dom_conv,
                        hit_iter_lim,
                        solve_state.resid_x,
                        solve_state.resid_y,
                        solve_state.interp_w,
                        solve_state.resid_mag,
                        solve_state.norm_resid_mag,
                        dom_violation,
                    );
                }
                if (result.iters > 0) ctx_report.recordSolverDiverged();
                continue;
            }

            if (comptime report_mode == .full_stats) {
                rasterreport.recordPixelConvStats(
                    ctx_report,
                    global_subx,
                    global_suby,
                    true,
                    result.xi_out,
                    result.eta_out,
                    jac_det,
                );
                rasterreport.recordPixelSolverDiagnostics(
                    ctx_report,
                    global_subx,
                    global_suby,
                    result.status,
                    result.pre_dom_conv,
                    hit_iter_lim,
                    solve_state.resid_x,
                    solve_state.resid_y,
                    solve_state.interp_w,
                    solve_state.resid_mag,
                    solve_state.norm_resid_mag,
                    dom_violation,
                );
            }

            if (ctx_rast.config.newton_seed_reuse == .last_conv) {
                newton.updateSeedState(
                    &seed_state,
                    result.xi_out,
                    result.eta_out,
                );
            }

            const weights = result.weights.?;
            const inv_z = Geom.calcInvZ(nodes_coords, weights);
            if (inv_z + tol.geometry.depth_buff_inv_z_cmp <
                subpx_scratch.inv_z[scratch_idx]) continue;

            subpx_scratch.inv_z[scratch_idx] = inv_z;
            if (scratch_x < subpx_scratch.touched_min_x[scratch_y]) {
                subpx_scratch.touched_min_x[scratch_y] = scratch_x;
            }
            if (scratch_x > subpx_scratch.touched_max_x[scratch_y]) {
                subpx_scratch.touched_max_x[scratch_y] = scratch_x;
            }
            const subpx_z = 1.0 / inv_z;
            shaded_px += 1;

            if (comptime report_mode == .full_stats) {
                rasterreport.recordPixelIterAndOccupancy(
                    ctx_report,
                    global_subx,
                    global_suby,
                    result.iters,
                    tile.scratch_x_px_min + scratch_x / sub_samp,
                    tile.scratch_y_px_min + scratch_y / sub_samp,
                );
            }

            const ctx_shade = shaderops.ShadeContext{
                .frame_idx = ctx_rast.frame_idx,
                .elem_idx = overlap.elem_idx,
                .fields_num = fields_num,
                .actual_fields = fields_num,
                .scratch_idx = scratch_idx,
                .global_subx = global_subx,
                .global_suby = global_suby,
            };
            const interp_data = shaderops.InterpData(N){
                .weights = weights,
                .nodes_inv_z = nodes_inv_z,
                .sub_pixel_z = subpx_z,
                .xi = result.xi_out,
                .eta = result.eta_out,
            };

            ShaderKern.shade(
                Geom.coord_space,
                ctx_shade,
                interp_data,
                shader_buf,
                shader,
                ctx_report,
                &subpx_scratch.image,
            );
        }
    }
    return shaded_px;
}

fn rasterSteppedScal(
    comptime Geom: type,
    comptime ShaderKern: type,
    comptime ShaderData: type,
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
    const sub_samp: usize = @intCast(ctx_rast.camera.sub_sample);
    const tile_subpx_x = @as(isize, tile.scratch_x_px_min) *
        @as(isize, @intCast(sub_samp));
    const tile_subpx_y = @as(isize, tile.scratch_y_px_min) *
        @as(isize, @intCast(sub_samp));
    const start_subx_global = tile_subpx_x + @as(isize, @intCast(rast_bounds.start_x_u));
    const start_suby_global = tile_subpx_y + @as(isize, @intCast(rast_bounds.start_y_u));
    const width = rast_bounds.end_x_u - rast_bounds.start_x_u;
    const height = rast_bounds.end_y_u - rast_bounds.start_y_u;
    const max_x_steps = if (width > 0) width - 1 else 0;
    const max_y_steps = if (height > 0) height - 1 else 0;

    if (comm.Tri3FixedEdges.init(
        nodes_coords,
        sub_samp,
        start_subx_global,
        start_suby_global,
        max_x_steps,
        max_y_steps,
    )) |fixed| {
        return rasterSteppedScalFixP(
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
            fixed,
        );
    }

    return rasterSteppedScalFloat(
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

fn rasterSteppedScalFixP(
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
    fixed: comm.Tri3FixedEdges,
) !u64 {
    const N = Geom.nodes_num;
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
    const area_cross = -(dy[1] * dx[1]);
    const area = @mulAdd(F, dx[0], dy[0], area_cross);

    const is_const_depth = z[0] == z[1] and z[1] == z[2];
    const nodes_inv_z = inv_z_node;

    const scratch_stride = subpx_dom.tile_size;

    for (rast_bounds.start_y_u..rast_bounds.end_y_u) |scratch_y_u| {
        const row_offset = scratch_y_u * scratch_stride;
        const global_suby = comm.globalSubpxForReport(
            tile.scratch_y_px_min,
            sub_samp,
            scratch_y_u,
        );
        const y_steps: buildconfig.Tri3FixedEdge = @intCast(
            scratch_y_u - rast_bounds.start_y_u,
        );

        var edge: [3]buildconfig.Tri3FixedEdge = undefined;
        inline for (0..3) |nn| {
            edge[nn] = fixed.start[nn] + y_steps * fixed.step_y[nn];
        }

        for (rast_bounds.start_x_u..rast_bounds.end_x_u) |scratch_x_u| {
            const scratch_idx = row_offset + scratch_x_u;
            const global_subx = comm.globalSubpxForReport(
                tile.scratch_x_px_min,
                sub_samp,
                scratch_x_u,
            );

            if (comptime report_mode == .full_stats) {
                rasterreport.recordEarlyOut(
                    ctx_report,
                    global_subx,
                    global_suby,
                    true,
                );
            }

            ctx_report.recordSolverCalls(1);
            ctx_report.recordSolverIters(1);

            if (edge[0] >= -fixed.edge_tol and
                edge[1] >= -fixed.edge_tol and
                edge[2] >= -fixed.edge_tol)
            {
                var weights: [3]F = undefined;
                weights[1] = @as(F, @floatFromInt(edge[1])) * fixed.inv_area;
                weights[2] = @as(F, @floatFromInt(edge[2])) * fixed.inv_area;
                weights[0] = 1.0 - weights[1] - weights[2];
                const inv_z_sum = @mulAdd(
                    F,
                    weights[1],
                    inv_z_node[1],
                    weights[2] * inv_z_node[2],
                );
                const inv_z = if (is_const_depth)
                    inv_z_node[0]
                else
                    @mulAdd(F, weights[0], inv_z_node[0], inv_z_sum);

                if (inv_z + buildconfig.config.tol.geometry.depth_buff_inv_z_cmp >=
                    subpx_scratch.inv_z[scratch_idx])
                {
                    subpx_scratch.inv_z[scratch_idx] = inv_z;
                    if (scratch_x_u < subpx_scratch.touched_min_x[scratch_y_u]) {
                        subpx_scratch.touched_min_x[scratch_y_u] = scratch_x_u;
                    }
                    if (scratch_x_u > subpx_scratch.touched_max_x[scratch_y_u]) {
                        subpx_scratch.touched_max_x[scratch_y_u] = scratch_x_u;
                    }
                    const subpx_z = 1.0 / inv_z;
                    shaded_px += 1;

                    if (comptime report_mode == .full_stats) {
                        rasterreport.recordPixelIterAndOccupancy(
                            ctx_report,
                            global_subx,
                            global_suby,
                            1,
                            tile.scratch_x_px_min + scratch_x_u / sub_samp,
                            tile.scratch_y_px_min + scratch_y_u / sub_samp,
                        );
                    }

                    const xi = if (is_const_depth)
                        weights[1]
                    else
                        @mulAdd(F, weights[1], inv_z_node[1], 0.0) / inv_z;

                    const eta = if (is_const_depth)
                        weights[2]
                    else
                        @mulAdd(F, weights[2], inv_z_node[2], 0.0) / inv_z;

                    if (comptime report_mode == .full_stats) {
                        rasterreport.recordPixelConvStats(
                            ctx_report,
                            global_subx,
                            global_suby,
                            true,
                            xi,
                            eta,
                            area,
                        );
                    }

                    const ctx_shade = shaderops.ShadeContext{
                        .frame_idx = ctx_rast.frame_idx,
                        .elem_idx = overlap.elem_idx,
                        .fields_num = fields_num,
                        .actual_fields = fields_num,
                        .scratch_idx = scratch_idx,
                        .global_subx = global_subx,
                        .global_suby = global_suby,
                    };

                    const interp_data = shaderops.InterpData(N){
                        .weights = weights,
                        .nodes_inv_z = nodes_inv_z,
                        .sub_pixel_z = subpx_z,
                        .xi = xi,
                        .eta = eta,
                    };

                    ShaderKern.shade(
                        Geom.coord_space,
                        ctx_shade,
                        interp_data,
                        shader_buf,
                        shader,
                        ctx_report,
                        &subpx_scratch.image,
                    );
                }
            } else {
                if (comptime report_mode == .full_stats) {
                    const nan = std.math.nan(F);
                    rasterreport.recordPixelConvStats(
                        ctx_report,
                        global_subx,
                        global_suby,
                        false,
                        nan,
                        nan,
                        nan,
                    );
                }
            }

            inline for (0..3) |nn| {
                edge[nn] += fixed.step_x[nn];
            }
        }
    }

    return shaded_px;
}

fn rasterSteppedScalFloat(
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
    const N = Geom.nodes_num;
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
    const area_cross = -(dy[1] * dx[1]);
    const area = @mulAdd(F, dx[0], dy[0], area_cross);
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
    const nodes_inv_z = inv_z_node;

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

    const scratch_stride = subpx_dom.tile_size;

    for (rast_bounds.start_y_u..rast_bounds.end_y_u) |scratch_y_u| {
        const row_offset = scratch_y_u * scratch_stride;
        const global_suby = comm.globalSubpxForReport(
            tile.scratch_y_px_min,
            sub_samp,
            scratch_y_u,
        );
        const y_steps = @as(F, @floatFromInt(scratch_y_u - rast_bounds.start_y_u));

        var weights: [3]F = undefined;
        inline for (0..3) |nn| {
            weights[nn] = @mulAdd(F, y_steps, dw_dy[nn], w_start[nn]);
        }

        for (rast_bounds.start_x_u..rast_bounds.end_x_u) |scratch_x_u| {
            const scratch_idx = row_offset + scratch_x_u;
            const global_subx = comm.globalSubpxForReport(
                tile.scratch_x_px_min,
                sub_samp,
                scratch_x_u,
            );

            if (comptime report_mode == .full_stats) {
                rasterreport.recordEarlyOut(
                    ctx_report,
                    global_subx,
                    global_suby,
                    true,
                );
            }

            ctx_report.recordSolverCalls(1);
            ctx_report.recordSolverIters(1);

            if (weights[0] >= -edge_tol and
                weights[1] >= -edge_tol and
                weights[2] >= -edge_tol)
            {
                const inv_z_tail = @mulAdd(
                    F,
                    weights[1],
                    inv_z_node[1],
                    weights[2] * inv_z_node[2],
                );
                const inv_z = if (is_const_depth)
                    inv_z_node[0]
                else
                    @mulAdd(F, weights[0], inv_z_node[0], inv_z_tail);

                if (inv_z + buildconfig.config.tol.geometry.depth_buff_inv_z_cmp >=
                    subpx_scratch.inv_z[scratch_idx])
                {
                    subpx_scratch.inv_z[scratch_idx] = inv_z;
                    if (scratch_x_u < subpx_scratch.touched_min_x[scratch_y_u]) {
                        subpx_scratch.touched_min_x[scratch_y_u] = scratch_x_u;
                    }
                    if (scratch_x_u > subpx_scratch.touched_max_x[scratch_y_u]) {
                        subpx_scratch.touched_max_x[scratch_y_u] = scratch_x_u;
                    }
                    const subpx_z = 1.0 / inv_z;
                    shaded_px += 1;

                    if (comptime report_mode == .full_stats) {
                        rasterreport.recordPixelIterAndOccupancy(
                            ctx_report,
                            global_subx,
                            global_suby,
                            1,
                            tile.scratch_x_px_min + scratch_x_u / sub_samp,
                            tile.scratch_y_px_min + scratch_y_u / sub_samp,
                        );
                    }

                    const xi = if (is_const_depth)
                        weights[1]
                    else
                        @mulAdd(F, weights[1], inv_z_node[1], 0.0) / inv_z;

                    const eta = if (is_const_depth)
                        weights[2]
                    else
                        @mulAdd(F, weights[2], inv_z_node[2], 0.0) / inv_z;

                    if (comptime report_mode == .full_stats) {
                        rasterreport.recordPixelConvStats(
                            ctx_report,
                            global_subx,
                            global_suby,
                            true,
                            xi,
                            eta,
                            area,
                        );
                    }

                    const ctx_shade = shaderops.ShadeContext{
                        .frame_idx = ctx_rast.frame_idx,
                        .elem_idx = overlap.elem_idx,
                        .fields_num = fields_num,
                        .actual_fields = fields_num,
                        .scratch_idx = scratch_idx,
                        .global_subx = global_subx,
                        .global_suby = global_suby,
                    };
                    const interp_data = shaderops.InterpData(N){
                        .weights = weights,
                        .nodes_inv_z = nodes_inv_z,
                        .sub_pixel_z = subpx_z,
                        .xi = xi,
                        .eta = eta,
                    };

                    ShaderKern.shade(
                        Geom.coord_space,
                        ctx_shade,
                        interp_data,
                        shader_buf,
                        shader,
                        ctx_report,
                        &subpx_scratch.image,
                    );
                }
            } else {
                if (comptime report_mode == .full_stats) {
                    const nan = std.math.nan(F);
                    rasterreport.recordPixelConvStats(
                        ctx_report,
                        global_subx,
                        global_suby,
                        false,
                        nan,
                        nan,
                        nan,
                    );
                }
            }

            inline for (0..3) |nn| {
                weights[nn] += dw_dx[nn];
            }
        }
    }

    return shaded_px;
}
