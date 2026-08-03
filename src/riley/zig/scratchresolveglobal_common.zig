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
