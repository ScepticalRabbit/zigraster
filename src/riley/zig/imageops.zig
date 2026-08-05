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
const matslice = @import("matslice.zig");
const ndarray = @import("ndarray.zig");
const texops = @import("textureops.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const ScaleStrategy = union(enum) {
    none,
    auto,
    fixed: [2]F, // [min, max]
    frac: [2]F, // [min_frac, max_frac]
};

pub const ScalingParams = struct { min: F, range: F };
pub const ScaleFactors = struct { mul: F, add: F };

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn getScaleMax(bits: ?u8) F {
    if (bits) |b| {
        const max_u32 = (@as(u32, 1) << @as(u5, @intCast(b))) - 1;
        return @as(F, @floatFromInt(max_u32));
    }
    return 1.0;
}

pub fn getScaleFactors(
    strategy: ScaleStrategy,
    bits: ?u8,
    params: ScalingParams,
) ScaleFactors {
    if (strategy == .none) return .{ .mul = 1.0, .add = 0.0 };

    var mul = 1.0 / params.range;
    var add = -params.min / params.range;

    switch (strategy) {
        .frac => |f| {
            mul = (f[1] - f[0]) / params.range;
            add = f[0] - params.min * (f[1] - f[0]) / params.range;
        },
        else => {},
    }

    if (bits) |b| {
        const m = getScaleMax(b);
        mul *= m;
        add *= m;
    }

    return .{ .mul = mul, .add = add };
}

pub fn getScalingParamsTex(
    comptime T: type,
    comptime channels: usize,
    tex: *const texops.Tex(T, channels),
    strategy: ScaleStrategy,
) ScalingParams {
    switch (strategy) {
        .none => return .{ .min = 0.0, .range = 1.0 },
        .auto, .frac => {
            var px_min: F = std.math.inf(F);
            var px_max: F = -std.math.inf(F);
            for (tex.array.slice) |val| {
                const val_f = texops.texelToFloat(T, val);
                if (val_f < px_min) px_min = val_f;
                if (val_f > px_max) px_max = val_f;
            }
            const range = if (@abs(px_max - px_min) < tol.image.auto_scale_range)
                1.0
            else
                px_max - px_min;
            return .{ .min = px_min, .range = range };
        },
        .fixed => |range| {
            const px_rng = if (range[1] > range[0]) range[1] - range[0] else 1.0;
            return .{ .min = range[0], .range = px_rng };
        },
    }
}

pub fn getScalingParams(
    image: *const matslice.MatSlice(F),
    strategy: ScaleStrategy,
) ScalingParams {
    switch (strategy) {
        .none => return .{ .min = 0.0, .range = 1.0 },
        .auto, .frac => {
            const px_min = std.mem.min(F, image.slice);
            const px_max = std.mem.max(F, image.slice);
            const px_rng = if (px_max > px_min) px_max - px_min else 1.0;
            return .{ .min = px_min, .range = px_rng };
        },
        .fixed => |range| {
            const px_rng = if (range[1] > range[0]) range[1] - range[0] else 1.0;
            return .{ .min = range[0], .range = px_rng };
        },
    }
}

pub fn getScalingParamsNDArray(
    array: *const ndarray.NDArray(F),
    frame_idx: ?usize,
    strategy: ScaleStrategy,
) ScalingParams {
    switch (strategy) {
        .none => return .{ .min = 0.0, .range = 1.0 },
        .auto, .frac => {
            var px_min: F = std.math.inf(F);
            var px_max: F = -std.math.inf(F);

            if (frame_idx) |fi| {
                const safe_fi = @min(fi, array.dims[0] - 1);
                const stride = array.strides[0];
                const frame_mem = array.slice[safe_fi * stride .. (safe_fi + 1) * stride];
                px_min = std.mem.min(F, frame_mem);
                px_max = std.mem.max(F, frame_mem);
            } else {
                px_min = std.mem.min(F, array.slice);
                px_max = std.mem.max(F, array.slice);
            }

            const px_rng = if (px_max > px_min) px_max - px_min else 1.0;
            return .{ .min = px_min, .range = px_rng };
        },
        .fixed => |range| {
            const px_rng = if (range[1] > range[0]) range[1] - range[0] else 1.0;
            return .{ .min = range[0], .range = px_rng };
        },
    }
}

pub fn applyScaling(
    val: F,
    strategy: ScaleStrategy,
    bits: ?u8,
    params: ScalingParams,
) F {
    const norm = switch (strategy) {
        .none => return val,
        .auto, .fixed => (val - params.min) / params.range,
        .frac => |f| f[0] + ((val - params.min) / params.range) * (f[1] - f[0]),
    };

    if (bits) |b| {
        return norm * getScaleMax(b);
    }
    return norm;
}

pub fn applyClamping(val: F, bits: ?u8) F {
    if (bits) |b| {
        const max_v = getScaleMax(b);
        return @round(@max(0.0, @min(max_v, val)));
    }
    return val;
}

pub fn avgImage(
    image_subpx: *const matslice.MatSlice(F),
    sub_samp: u32,
    image_avg: *matslice.MatSlice(F),
) void {
    const num_px_x: usize = (image_subpx.cols_n) / @as(usize, sub_samp);
    const num_px_y: usize = (image_subpx.rows_n) / @as(usize, sub_samp);
    const sub_samp_us: usize = @as(usize, sub_samp);
    const sub_samp_f: F = @as(F, @floatFromInt(sub_samp));
    const subpx_per_px: F = sub_samp_f * sub_samp_f;

    var px_sum: F = 0.0;

    for (0..num_px_y) |iy| {
        for (0..num_px_x) |ix| {
            px_sum = 0.0;
            for (0..sub_samp_us) |sy| {
                for (0..sub_samp_us) |sx| {
                    px_sum += image_subpx.get(
                        sub_samp_us * iy + sy,
                        sub_samp_us * ix + sx,
                    );
                }
            }
            image_avg.set(iy, ix, px_sum / subpx_per_px);
        }
    }
}
