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
const cam = @import("camera.zig");
const ndarray = @import("ndarray.zig");
const subpxframe = @import("subpxframe.zig");
const pce = @import("parachunkexec.zig");

const S = buildconfig.SimdWidth;
const VecSF = buildconfig.VecSF;

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn resolve(
    target: *const subpxframe.SubpxTarget,
    camera: *const cam.CameraPrepared,
    background_value: F,
    image_out_arr: *ndarray.NDArray(F),
) void {
    resolveRows(
        target,
        camera,
        background_value,
        image_out_arr,
        0,
        target.domain.image_h_subpx / @as(usize, camera.sub_sample),
    );
}

pub fn resolveRows(
    target: *const subpxframe.SubpxTarget,
    camera: *const cam.CameraPrepared,
    background_value: F,
    image_out_arr: *ndarray.NDArray(F),
    image_y_min: usize,
    image_y_max: usize,
) void {
    const sub_samp: usize = @intCast(camera.sub_sample);
    const prep_psf = camera.prep_psf;
    const fields_num: usize = target.domain.fields_num;
    const image_w_px = target.domain.image_w_subpx / sub_samp;
    std.debug.assert(image_y_min <= image_y_max);
    std.debug.assert(image_out_arr.dims[0] == fields_num);
    std.debug.assert(image_out_arr.dims[1] >= image_y_max);
    std.debug.assert(image_out_arr.dims[2] >= image_w_px);

    for (image_y_min..image_y_max) |yy| {
        const global_suby_start = yy * sub_samp;
        for (0..image_w_px) |xx| {
            const global_subx_start = xx * sub_samp;
            for (0..fields_num) |ff| {
                var sum: F = 0.0;
                for (0..sub_samp) |ssy| {
                    for (0..sub_samp) |ssx| {
                        const global_subx: i32 = @intCast(global_subx_start + ssx);
                        const global_suby: i32 = @intCast(global_suby_start + ssy);
                        sum += sampleFiltered(
                            target,
                            prep_psf,
                            background_value,
                            global_subx,
                            global_suby,
                            ff,
                        );
                    }
                }
                const sub_samp_f: F = @floatFromInt(sub_samp);
                image_out_arr.slice[image_out_arr.offset3(ff, yy, xx)] =
                    sum / (sub_samp_f * sub_samp_f);
            }
        }
    }
}

/// Resolve a global target using independent output-row bands.  The direct
/// implementation remains the scalar reference for identity and non-separable
/// PSFs.  Separable PSFs use a compact horizontal intermediate at pixel x
/// resolution, then a vertical pass; this fuses the SSAA box sum into each
/// one-dimensional pass and avoids the former SSAA^2 x Kx x Ky stencil.
pub fn resolveParallel(
    comptime use_simd: bool,
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    target: *const subpxframe.SubpxTarget,
    camera: *const cam.CameraPrepared,
    background_value: F,
    image_out_arr: *ndarray.NDArray(F),
    image_y_min: usize,
    image_y_max: usize,
    requested_workers: u16,
) !usize {
    if (image_y_min == image_y_max) return 0;
    return switch (camera.prep_psf.mode) {
        .separable => resolveSeparableParallel(
            use_simd,
            outer_alloc,
            io,
            target,
            camera,
            image_out_arr,
            image_y_min,
            image_y_max,
            requested_workers,
        ),
        .identity_fast, .nonseparable => resolveDirectParallel(
            io,
            target,
            camera,
            background_value,
            image_out_arr,
            image_y_min,
            image_y_max,
            requested_workers,
        ),
    };
}

fn workersForRows(requested_workers: u16, rows: usize) usize {
    return @min(
        @max(@as(usize, 1), @as(usize, requested_workers)),
        @max(@as(usize, 1), rows),
    );
}

fn resolveDirectParallel(
    io: std.Io,
    target: *const subpxframe.SubpxTarget,
    camera: *const cam.CameraPrepared,
    background_value: F,
    image_out_arr: *ndarray.NDArray(F),
    image_y_min: usize,
    image_y_max: usize,
    requested_workers: u16,
) !usize {
    const workers_num = workersForRows(requested_workers, image_y_max - image_y_min);
    const Ctx = struct {
        target: *const subpxframe.SubpxTarget,
        camera: *const cam.CameraPrepared,
        background_value: F,
        image_out_arr: *ndarray.NDArray(F),
        image_y_min: usize,
    };
    const Adapter = struct {
        fn run(ctx_ptr: *anyopaque, _: usize, range_start: usize, range_end: usize) void {
            const ctx: *Ctx = @ptrCast(@alignCast(ctx_ptr));
            // This is intentionally the unchanged direct scalar oracle.  Each
            // task owns complete output rows, so no accumulation is shared.
            resolveRows(
                ctx.target,
                ctx.camera,
                ctx.background_value,
                ctx.image_out_arr,
                ctx.image_y_min + range_start,
                ctx.image_y_min + range_end,
            );
        }
    };
    var exec = pce.ParaChunkExecutor.init(io, @intCast(workers_num));
    var ctx = Ctx{
        .target = target,
        .camera = camera,
        .background_value = background_value,
        .image_out_arr = image_out_arr,
        .image_y_min = image_y_min,
    };
    try exec.runStaticRange(&ctx, Adapter.run, image_y_max - image_y_min, 1);
    return workers_num;
}

fn resolveSeparableParallel(
    comptime use_simd: bool,
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    target: *const subpxframe.SubpxTarget,
    camera: *const cam.CameraPrepared,
    image_out_arr: *ndarray.NDArray(F),
    image_y_min: usize,
    image_y_max: usize,
    requested_workers: u16,
) !usize {
    const sub_samp: usize = @intCast(camera.sub_sample);
    const image_w_px = target.domain.image_w_subpx / sub_samp;
    const storage_h = target.domain.storage_h_subpx;
    const fields_num: usize = target.domain.fields_num;
    const horizontal_len = try std.math.mul(usize, fields_num, try std.math.mul(usize, storage_h, image_w_px));
    const horizontal = try outer_alloc.alloc(F, horizontal_len);
    defer outer_alloc.free(horizontal);

    const HorizontalCtx = struct {
        target: *const subpxframe.SubpxTarget,
        camera: *const cam.CameraPrepared,
        horizontal: []F,
        image_w_px: usize,
        storage_h: usize,
    };
    const HorizontalAdapter = struct {
        fn run(ctx_ptr: *anyopaque, _: usize, range_start: usize, range_end: usize) void {
            const ctx: *HorizontalCtx = @ptrCast(@alignCast(ctx_ptr));
            horizontalRows(use_simd, ctx.*, range_start, range_end);
        }
    };

    const horizontal_workers = workersForRows(requested_workers, storage_h);
    var horizontal_exec = pce.ParaChunkExecutor.init(io, @intCast(horizontal_workers));
    var horizontal_ctx = HorizontalCtx{
        .target = target,
        .camera = camera,
        .horizontal = horizontal,
        .image_w_px = image_w_px,
        .storage_h = storage_h,
    };
    try horizontal_exec.runStaticRange(&horizontal_ctx, HorizontalAdapter.run, storage_h, 1);

    const VerticalCtx = struct {
        target: *const subpxframe.SubpxTarget,
        camera: *const cam.CameraPrepared,
        horizontal: []const F,
        image_out_arr: *ndarray.NDArray(F),
        image_w_px: usize,
        storage_h: usize,
        image_y_min: usize,
    };
    const VerticalAdapter = struct {
        fn run(ctx_ptr: *anyopaque, _: usize, range_start: usize, range_end: usize) void {
            const ctx: *VerticalCtx = @ptrCast(@alignCast(ctx_ptr));
            verticalRows(
                use_simd,
                ctx.*,
                ctx.image_y_min + range_start,
                ctx.image_y_min + range_end,
            );
        }
    };

    const vertical_workers = workersForRows(requested_workers, image_y_max - image_y_min);
    var vertical_exec = pce.ParaChunkExecutor.init(io, @intCast(vertical_workers));
    var vertical_ctx = VerticalCtx{
        .target = target,
        .camera = camera,
        .horizontal = horizontal,
        .image_out_arr = image_out_arr,
        .image_w_px = image_w_px,
        .storage_h = storage_h,
        .image_y_min = image_y_min,
    };
    try vertical_exec.runStaticRange(
        &vertical_ctx,
        VerticalAdapter.run,
        image_y_max - image_y_min,
        1,
    );
    return @min(horizontal_workers, vertical_workers);
}

fn horizontalRows(
    comptime use_simd: bool,
    ctx: anytype,
    local_y_start: usize,
    local_y_end: usize,
) void {
    const sub_samp: usize = @intCast(ctx.camera.sub_sample);
    const psf = ctx.camera.prep_psf;
    const local_core_x: usize = @intCast(-ctx.target.global_subx_min);
    const source_row_stride = ctx.target.domain.storage_w_subpx;
    const horizontal_field_stride = ctx.storage_h * ctx.image_w_px;

    for (0..ctx.target.domain.fields_num) |ff| {
        const source_field_base = ctx.target.image.rowBase(ff);
        const horizontal_field_base = ff * horizontal_field_stride;
        for (local_y_start..local_y_end) |local_y| {
            const source_row = source_field_base + local_y * source_row_stride;
            const horizontal_row = horizontal_field_base + local_y * ctx.image_w_px;
            var xx: usize = 0;
            if (comptime use_simd) {
                while (xx + S <= ctx.image_w_px) : (xx += S) {
                    var sum = @as(VecSF, @splat(0.0));
                    for (0..sub_samp) |sample_x| {
                        for (psf.weights_x, 0..) |weight, kk| {
                            var values: [S]F = undefined;
                            for (0..S) |lane| {
                                const source_x = local_core_x +
                                    (xx + lane) * sub_samp + sample_x + kk - psf.radius_x_subpx;
                                values[lane] = ctx.target.image.slice[source_row + source_x];
                            }
                            sum += @as(VecSF, values) * @as(VecSF, @splat(weight));
                        }
                    }
                    const out_ptr: *[S]F = @ptrCast(&ctx.horizontal[horizontal_row + xx]);
                    out_ptr.* = @bitCast(sum);
                }
            }
            while (xx < ctx.image_w_px) : (xx += 1) {
                var sum: F = 0.0;
                for (0..sub_samp) |sample_x| {
                    for (psf.weights_x, 0..) |weight, kk| {
                        const source_x = local_core_x + xx * sub_samp + sample_x + kk - psf.radius_x_subpx;
                        sum += weight * ctx.target.image.slice[source_row + source_x];
                    }
                }
                ctx.horizontal[horizontal_row + xx] = sum;
            }
        }
    }
}

fn verticalRows(
    comptime use_simd: bool,
    ctx: anytype,
    image_y_start: usize,
    image_y_end: usize,
) void {
    const sub_samp: usize = @intCast(ctx.camera.sub_sample);
    const psf = ctx.camera.prep_psf;
    const horizontal_field_stride = ctx.storage_h * ctx.image_w_px;
    const inv_sub_samp_sq = 1.0 / @as(F, @floatFromInt(sub_samp * sub_samp));

    for (0..ctx.target.domain.fields_num) |ff| {
        const horizontal_field_base = ff * horizontal_field_stride;
        for (image_y_start..image_y_end) |image_y| {
            const global_suby: i32 = @intCast(image_y * sub_samp);
            const local_core_y: usize = @intCast(global_suby - ctx.target.global_suby_min);
            const output_row = ctx.image_out_arr.offset3(ff, image_y, 0);
            var xx: usize = 0;
            if (comptime use_simd) {
                while (xx + S <= ctx.image_w_px) : (xx += S) {
                    var sum = @as(VecSF, @splat(0.0));
                    for (0..sub_samp) |sample_y| {
                        for (psf.weights_y, 0..) |weight, kk| {
                            const source_y = local_core_y + sample_y + kk - psf.radius_y_subpx;
                            const source_ptr: *const [S]F = @ptrCast(
                                &ctx.horizontal[horizontal_field_base + source_y * ctx.image_w_px + xx],
                            );
                            sum += @as(VecSF, source_ptr.*) * @as(VecSF, @splat(weight));
                        }
                    }
                    const output_ptr: *[S]F = @ptrCast(&ctx.image_out_arr.slice[output_row + xx]);
                    output_ptr.* = @bitCast(sum * @as(VecSF, @splat(inv_sub_samp_sq)));
                }
            }
            while (xx < ctx.image_w_px) : (xx += 1) {
                var sum: F = 0.0;
                for (0..sub_samp) |sample_y| {
                    for (psf.weights_y, 0..) |weight, kk| {
                        const source_y = local_core_y + sample_y + kk - psf.radius_y_subpx;
                        sum += weight * ctx.horizontal[
                            horizontal_field_base + source_y * ctx.image_w_px + xx
                        ];
                    }
                }
                ctx.image_out_arr.slice[output_row + xx] = sum * inv_sub_samp_sq;
            }
        }
    }
}

// --------------------------------------------------------------------------------------
// Private Func
// --------------------------------------------------------------------------------------

fn sampleFiltered(
    target: *const subpxframe.SubpxTarget,
    prep_psf: cam.PreparedPSF,
    background_value: F,
    global_subx: i32,
    global_suby: i32,
    field_idx: usize,
) F {
    return switch (prep_psf.mode) {
        .identity_fast => sampleTarget(
            target,
            background_value,
            global_subx,
            global_suby,
            field_idx,
        ),
        .separable => sampleSeparable(
            target,
            prep_psf,
            background_value,
            global_subx,
            global_suby,
            field_idx,
        ),
        .nonseparable => sampleNonSeparable(
            target,
            prep_psf,
            background_value,
            global_subx,
            global_suby,
            field_idx,
        ),
    };
}

fn sampleSeparable(
    target: *const subpxframe.SubpxTarget,
    prep_psf: cam.PreparedPSF,
    background_value: F,
    global_subx: i32,
    global_suby: i32,
    field_idx: usize,
) F {
    var sum: F = 0.0;
    for (prep_psf.weights_y, 0..) |weight_y, yy| {
        const y_off: i32 = @intCast(yy);
        for (prep_psf.weights_x, 0..) |weight_x, xx| {
            const x_off: i32 = @intCast(xx);
            const radius_x: i32 = @intCast(prep_psf.radius_x_subpx);
            const radius_y: i32 = @intCast(prep_psf.radius_y_subpx);
            sum += weight_x * weight_y * sampleTarget(
                target,
                background_value,
                global_subx + x_off - radius_x,
                global_suby + y_off - radius_y,
                field_idx,
            );
        }
    }
    return sum;
}

fn sampleNonSeparable(
    target: *const subpxframe.SubpxTarget,
    prep_psf: cam.PreparedPSF,
    background_value: F,
    global_subx: i32,
    global_suby: i32,
    field_idx: usize,
) F {
    const kernel_w = 2 * prep_psf.radius_x_subpx + 1;
    var sum: F = 0.0;
    for (prep_psf.weights_2d, 0..) |weight, kk| {
        const kernel_x = kk % kernel_w;
        const kernel_y = kk / kernel_w;
        const x_off: i32 = @intCast(kernel_x);
        const y_off: i32 = @intCast(kernel_y);
        const radius_x: i32 = @intCast(prep_psf.radius_x_subpx);
        const radius_y: i32 = @intCast(prep_psf.radius_y_subpx);
        sum += weight * sampleTarget(
            target,
            background_value,
            global_subx + x_off - radius_x,
            global_suby + y_off - radius_y,
            field_idx,
        );
    }
    return sum;
}

fn sampleTarget(
    target: *const subpxframe.SubpxTarget,
    background_value: F,
    global_subx: i32,
    global_suby: i32,
    field_idx: usize,
) F {
    const local_subx = global_subx - target.global_subx_min;
    const local_suby = global_suby - target.global_suby_min;
    if (local_subx < 0 or local_suby < 0 or
        local_subx >= @as(i32, @intCast(target.domain.storage_w_subpx)) or
        local_suby >= @as(i32, @intCast(target.domain.storage_h_subpx)))
    {
        return background_value;
    }

    const flat_idx = @as(usize, @intCast(local_suby)) * target.domain.storage_w_subpx +
        @as(usize, @intCast(local_subx));
    return target.image.slice[target.image.rowBase(field_idx) + flat_idx];
}

// --------------------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------------------

test "sampleTarget uses the target global origin" {
    const domain = try subpxframe.SubpxFrameDomain.init(1, .{ 2, 2 }, 1, 1);
    var target = try subpxframe.SubpxTarget.init(
        std.testing.allocator,
        domain,
        -1,
        -1,
        0.0,
    );
    defer target.deinit(std.testing.allocator);

    target.image.slice[target.image.rowBase(0) + target.flatIndex(0, 1)] = 7.0;
    try std.testing.expectEqual(@as(F, 7.0), sampleTarget(&target, -1.0, 0, 1, 0));
    try std.testing.expectEqual(@as(F, -1.0), sampleTarget(&target, -1.0, 3, 1, 0));
}
