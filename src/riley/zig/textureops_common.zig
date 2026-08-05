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
const eval_branch_quota = buildconfig.comptime_eval_branch_quota;
const cfg = buildconfig.config;
const tol = cfg.tol;
const lut_size = cfg.interp_lut_size;
const NDArray = @import("ndarray.zig").NDArray;
const csvio = @import("csvio.zig");

// --------------------------------------------------------------------------------------
// Strategy Map:
//
// PIPELINE ENTRY POINTS (Dispatched by Shader/Kernel):
// │
// ├── PATH 1: sampScal (Purely Scal)
// │   "Used when .simd = .off or as fallback for complex elems (quad4ibi)"
// │   ├── getPx()           (Scalar Load)
// │   ├── sampLinear()      (Scalar Linear)
// │   └── sampConvo()       (Scalar Convolution)
// │
// ├── PATH 2: sampWide (Wide SIMD - Parallel over Pixels)
// │   "Each lane is a unique pixel; processes N pixels simultaneously"
// │   ├── getPxWide()       (Wide Load: N pixels)
// │   ├── sampLinearWide()  (Wide Linear: N pixels)
// │   └── sampConvoWide()   (Wide Convolution: N pixels)
// │
// └── PATH 3: sampLanes (Lane SIMD - Serial over Pixels, SIMD over Taps)
//     "Processes N lanes serially; math inside each lane uses SIMD for taps"
//     └── sampOneLane()     (Helper: Process 1 lane)
//         ├── getPx()       (Scalar Load: 1 pixel)
//         ├── sampLinearOneLane() (Scalar Linear: 1 pixel)
//         └── sampConvoOneLane()  (SIMD-Tap Convolution: 1 pixel)
//
// -------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const TexSamp = enum {
    nearest,
    linear,
    cubic_catmull_rom,
    cubic_mitchell_netravali,
    lanczos3,
    cubic_bspline,
    quintic_bspline,
    lanczos2,
};

pub const TexSampMode = enum {
    direct,
    lut,
    lut_lerp,
};

pub const TexSampConfig = struct {
    sample: TexSamp,
    mode: TexSampMode = .direct,

    pub fn isValid(self: TexSampConfig) bool {
        return switch (self.sample) {
            .nearest, .linear => self.mode == .direct,
            else => true,
        };
    }

    pub fn sanitize(self: TexSampConfig) TexSampConfig {
        return switch (self.sample) {
            .nearest, .linear => .{ .sample = self.sample, .mode = .direct },
            else => self,
        };
    }
};

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn Tex(comptime T: type, comptime C: usize) type {
    return struct {
        array: NDArray(T),
        rows_num: usize,
        cols_num: usize,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) !Self {
            const dims = &[_]usize{ C, rows, cols };
            const array = try NDArray(T).initFlat(allocator, dims);
            return .{
                .array = array,
                .rows_num = rows,
                .cols_num = cols,
            };
        }

        pub fn deinit(self: *const Self, allocator: std.mem.Allocator) void {
            allocator.free(self.array.slice);
            self.array.deinit(allocator);
        }

        pub fn setVal(self: *Self, ch: usize, row: usize, col: usize, val: T) void {
            const tex_flat_idx = self.array.subBase2(ch, row) + col;
            self.array.setFlat(tex_flat_idx, val);
        }

        pub fn getVal(self: *const Self, ch: usize, row: usize, col: usize) T {
            const tex_flat_idx = self.array.subBase2(ch, row) + col;
            return self.array.getFlat(tex_flat_idx);
        }

        pub fn saveCSV(
            self: *const Self,
            io: std.Io,
            out_dir: std.Io.Dir,
            file_name: []const u8,
        ) !void {
            const SaveCtx = struct {
                fn getVal(ctx: *const Self, row: usize, col: usize, ch: usize) F {
                    return texelToFloat(T, ctx.getVal(ch, row, col));
                }
            };

            try csvio.savePackedGridCSV(
                io,
                out_dir,
                file_name,
                self.rows_num,
                self.cols_num,
                C,
                self,
                SaveCtx.getVal,
            );
        }
    };
}

pub fn texelToFloat(comptime T: type, val: T) F {
    return switch (@typeInfo(T)) {
        .int => @as(F, @floatFromInt(val)),
        .float => @as(F, @floatCast(val)),
        else => @compileError("Unsupped tex storage type."),
    };
}

pub fn sampScal(
    comptime C: usize,
    comptime config: TexSampConfig,
    tex: anytype,
    u: F,
    v: F,
) [C]F {
    std.debug.assert(config.isValid());

    const cols_minus_1 = @as(isize, @intCast(tex.cols_num)) - 1;
    const rows_minus_1 = @as(isize, @intCast(tex.rows_num)) - 1;
    const cols_minus_1_f = @as(F, @floatFromInt(cols_minus_1));
    const rows_minus_1_f = @as(F, @floatFromInt(rows_minus_1));
    const tex_x_f = u * cols_minus_1_f;
    const tex_y_f = v * rows_minus_1_f;
    const tex_x_i = @as(isize, @intFromFloat(@floor(tex_x_f)));
    const tex_y_i = @as(isize, @intFromFloat(@floor(tex_y_f)));
    const tex_x_frac = tex_x_f - @as(F, @floatFromInt(tex_x_i));
    const tex_y_frac = tex_y_f - @as(F, @floatFromInt(tex_y_i));
    const tex_x_round = @as(isize, @intFromFloat(@round(tex_x_f)));
    const tex_y_round = @as(isize, @intFromFloat(@round(tex_y_f)));

    return switch (config.sample) {
        .nearest => getPx(
            C,
            tex,
            tex_x_round,
            tex_y_round,
        ),
        .linear => sampLinear(C, tex, tex_x_i, tex_y_i, tex_x_frac, tex_y_frac),
        .cubic_catmull_rom => sampTex4Tap(
            C,
            .cubic_catmull_rom,
            config.mode,
            tex,
            tex_x_i,
            tex_y_i,
            tex_x_frac,
            tex_y_frac,
        ),
        .cubic_mitchell_netravali => sampTex4Tap(
            C,
            .cubic_mitchell_netravali,
            config.mode,
            tex,
            tex_x_i,
            tex_y_i,
            tex_x_frac,
            tex_y_frac,
        ),
        .cubic_bspline => sampTex4Tap(
            C,
            .cubic_bspline,
            config.mode,
            tex,
            tex_x_i,
            tex_y_i,
            tex_x_frac,
            tex_y_frac,
        ),
        .lanczos2 => sampTex4Tap(
            C,
            .lanczos2,
            config.mode,
            tex,
            tex_x_i,
            tex_y_i,
            tex_x_frac,
            tex_y_frac,
        ),
        .lanczos3 => sampTex6Tap(
            C,
            .lanczos3,
            config.mode,
            tex,
            tex_x_i,
            tex_y_i,
            tex_x_frac,
            tex_y_frac,
        ),
        .quintic_bspline => sampTex6Tap(
            C,
            .quintic_bspline,
            config.mode,
            tex,
            tex_x_i,
            tex_y_i,
            tex_x_frac,
            tex_y_frac,
        ),
    };
}

pub fn sampGrey(
    comptime config: TexSampConfig,
    tex: anytype,
    u: F,
    v: F,
) F {
    return sampScal(1, config, tex, u, v)[0];
}

pub fn getPx(
    comptime C: usize,
    tex: anytype,
    x: isize,
    y: isize,
) [C]F {
    const tex_cols = @as(isize, @intCast(tex.cols_num));
    const tex_rows = @as(isize, @intCast(tex.rows_num));
    const tex_x_clamp = @max(0, @min(x, tex_cols - 1));
    const tex_y_clamp = @max(0, @min(y, tex_rows - 1));
    const tex_x_i = @as(usize, @intCast(tex_x_clamp));
    const tex_y_i = @as(usize, @intCast(tex_y_clamp));

    var samp_res: [C]F = undefined;
    inline for (0..C) |ch| {
        samp_res[ch] = tex.getVal(ch, tex_y_i, tex_x_i);
    }
    return samp_res;
}

pub fn sampLinear(
    comptime C: usize,
    tex: anytype,
    tex_x_i: isize,
    tex_y_i: isize,
    tex_x_frac: F,
    tex_y_frac: F,
) [C]F {
    const p00 = getPx(C, tex, tex_x_i, tex_y_i);
    const p10 = getPx(C, tex, tex_x_i + 1, tex_y_i);
    const p01 = getPx(C, tex, tex_x_i, tex_y_i + 1);
    const p11 = getPx(C, tex, tex_x_i + 1, tex_y_i + 1);
    var samp_res: [C]F = undefined;
    inline for (0..C) |ch| {
        samp_res[ch] = (1.0 - tex_x_frac) * (1.0 - tex_y_frac) * p00[ch] +
            tex_x_frac * (1.0 - tex_y_frac) * p10[ch] +
            (1.0 - tex_x_frac) * tex_y_frac * p01[ch] +
            tex_x_frac * tex_y_frac * p11[ch];
    }
    return samp_res;
}

pub fn sampConvo(
    comptime C: usize,
    comptime TAP: usize,
    tex: anytype,
    tex_x_i: isize,
    tex_y_i: isize,
    samp_coeff_x: [TAP]F,
    samp_coeff_y: [TAP]F,
) [C]F {
    const tap_offset = @as(isize, @intCast(TAP)) / 2 - 1;
    var samp_res: [C]F = [_]F{0.0} ** C;
    var samp_coeff_sum: F = 0.0;

    for (0..TAP) |jj| {
        for (0..TAP) |ii| {
            const tap_samp_coeff = samp_coeff_x[ii] * samp_coeff_y[jj];
            const px = getPx(
                C,
                tex,
                tex_x_i + @as(isize, @intCast(ii)) - tap_offset,
                tex_y_i + @as(isize, @intCast(jj)) - tap_offset,
            );
            inline for (0..C) |ch| {
                samp_res[ch] += px[ch] * tap_samp_coeff;
            }
            samp_coeff_sum += tap_samp_coeff;
        }
    }

    const inv_samp_coeff_sum = if (@abs(samp_coeff_sum) < tol.tex.samp_coeff_sum)
        1.0
    else
        1.0 / samp_coeff_sum;

    inline for (0..C) |ch| {
        samp_res[ch] *= inv_samp_coeff_sum;
    }
    return samp_res;
}

fn sampTex4Tap(
    comptime C: usize,
    comptime sample: TexSamp,
    comptime mode: TexSampMode,
    tex: anytype,
    tex_x_i: isize,
    tex_y_i: isize,
    tex_x_frac: F,
    tex_y_frac: F,
) [C]F {
    const TAP = 4;
    const coeff_fun = switch (sample) {
        .cubic_catmull_rom => cubicCoeffCatmullRom,
        .cubic_mitchell_netravali => cubicCoeffMitchellNetravali,
        .cubic_bspline => cubicBSplineCoeff,
        .lanczos2 => lanczos2Coeff,
        else => unreachable,
    };
    const lut = switch (sample) {
        .cubic_catmull_rom => catmull_rom_lut,
        .cubic_mitchell_netravali => mitchell_netravali_lut,
        .cubic_bspline => cubic_bspline_lut,
        .lanczos2 => lanczos2_lut,
        else => unreachable,
    };

    return switch (mode) {
        .direct => blk: {
            const coeffs_x = .{
                coeff_fun(tex_x_frac + 1),
                coeff_fun(tex_x_frac),
                coeff_fun(tex_x_frac - 1),
                coeff_fun(tex_x_frac - 2),
            };
            const coeffs_y = .{
                coeff_fun(tex_y_frac + 1),
                coeff_fun(tex_y_frac),
                coeff_fun(tex_y_frac - 1),
                coeff_fun(tex_y_frac - 2),
            };

            break :blk sampConvo(
                C,
                TAP,
                tex,
                tex_x_i,
                tex_y_i,
                coeffs_x,
                coeffs_y,
            );
        },
        .lut => blk: {
            const lut_size_f = @as(F, @floatFromInt(lut_size - 1));
            const idx_x = @as(usize, @intFromFloat(tex_x_frac * lut_size_f));
            const idx_y = @as(usize, @intFromFloat(tex_y_frac * lut_size_f));

            break :blk sampConvo(
                C,
                TAP,
                tex,
                tex_x_i,
                tex_y_i,
                lut[idx_x],
                lut[idx_y],
            );
        },
        .lut_lerp => blk: {
            const coeffs_x = getLerpSampCoeffs(TAP, lut, tex_x_frac);
            const coeffs_y = getLerpSampCoeffs(TAP, lut, tex_y_frac);

            break :blk sampConvo(
                C,
                TAP,
                tex,
                tex_x_i,
                tex_y_i,
                coeffs_x,
                coeffs_y,
            );
        },
    };
}

fn sampTex6Tap(
    comptime C: usize,
    comptime sample: TexSamp,
    comptime mode: TexSampMode,
    tex: anytype,
    tex_x_i: isize,
    tex_y_i: isize,
    tex_x_frac: F,
    tex_y_frac: F,
) [C]F {
    const TAP = 6;
    const coeff_fun = switch (sample) {
        .lanczos3 => lanczos3Coeff,
        .quintic_bspline => quinticBSplineCoeff,
        else => unreachable,
    };
    const lut = switch (sample) {
        .lanczos3 => lanczos3_lut,
        .quintic_bspline => quintic_bspline_lut,
        else => unreachable,
    };

    return switch (mode) {
        .direct => blk: {
            const coeffs_x = .{
                coeff_fun(tex_x_frac + 2),
                coeff_fun(tex_x_frac + 1),
                coeff_fun(tex_x_frac),
                coeff_fun(tex_x_frac - 1),
                coeff_fun(tex_x_frac - 2),
                coeff_fun(tex_x_frac - 3),
            };
            const coeffs_y = .{
                coeff_fun(tex_y_frac + 2),
                coeff_fun(tex_y_frac + 1),
                coeff_fun(tex_y_frac),
                coeff_fun(tex_y_frac - 1),
                coeff_fun(tex_y_frac - 2),
                coeff_fun(tex_y_frac - 3),
            };
            break :blk sampConvo(
                C,
                TAP,
                tex,
                tex_x_i,
                tex_y_i,
                coeffs_x,
                coeffs_y,
            );
        },
        .lut => blk: {
            const lut_size_f = @as(F, @floatFromInt(lut_size - 1));
            const idx_x = @as(usize, @intFromFloat(tex_x_frac * lut_size_f));
            const idx_y = @as(usize, @intFromFloat(tex_y_frac * lut_size_f));
            break :blk sampConvo(
                C,
                TAP,
                tex,
                tex_x_i,
                tex_y_i,
                lut[idx_x],
                lut[idx_y],
            );
        },
        .lut_lerp => blk: {
            const coeffs_x = getLerpSampCoeffs(TAP, lut, tex_x_frac);
            const coeffs_y = getLerpSampCoeffs(TAP, lut, tex_y_frac);
            break :blk sampConvo(
                C,
                TAP,
                tex,
                tex_x_i,
                tex_y_i,
                coeffs_x,
                coeffs_y,
            );
        },
    };
}

pub fn getLerpSampCoeffs(
    comptime TAP: usize,
    comptime table: [lut_size][TAP]F,
    t: F,
) [TAP]F {
    const scaled = t * (lut_size - 1);
    const idx = @as(usize, @intFromFloat(@floor(scaled)));
    const frac = scaled - @as(F, @floatFromInt(idx));
    var lerp_res: [TAP]F = undefined;

    const lut0 = table[idx];
    const lut1 = table[@min(idx + 1, lut_size - 1)];

    inline for (0..TAP) |ii| {
        lerp_res[ii] = lut0[ii] * (1.0 - frac) + lut1[ii] * frac;
    }
    return lerp_res;
}

pub fn getLerpSampCoeffsRuntime(
    comptime TAP: usize,
    table: [lut_size][TAP]F,
    t: F, // Between 0.0 and 1.0
) [TAP]F {
    const scaled = t * (lut_size - 1);
    const idx = @as(usize, @intFromFloat(@floor(scaled)));
    const frac = scaled - @as(F, @floatFromInt(idx));

    var lerp_res: [TAP]F = undefined;
    const lut0 = table[idx];
    const lut1 = table[@min(idx + 1, lut_size - 1)];

    inline for (0..TAP) |ii| {
        lerp_res[ii] = lut0[ii] * (1.0 - frac) + lut1[ii] * frac;
    }

    return lerp_res;
}

pub fn cubicCoeffCatmullRom(x: F) F {
    const abs_x = @abs(x);
    if (abs_x <= 1.0) {
        return ((1.5 * abs_x - 2.5) * abs_x + 0.0) * abs_x + 1.0;
    } else if (abs_x < 2.0) {
        return ((-0.5 * abs_x + 2.5) * abs_x - 4.0) * abs_x + 2.0;
    }
    return 0.0;
}

pub fn cubicCoeffMitchellNetravali(x: F) F {
    const r = @abs(x);
    const B = 1.0 / 3.0;
    const C = 1.0 / 3.0;
    if (r < 1.0) {
        return ((12.0 - 9.0 * B - 6.0 * C) * r * r * r +
            (-18.0 + 12.0 * B + 6.0 * C) * r * r +
            (6.0 - 2.0 * B)) / 6.0;
    } else if (r < 2.0) {
        return ((-B - 6.0 * C) * r * r * r +
            (6.0 * B + 30.0 * C) * r * r +
            (-12.0 * B - 48.0 * C) * r +
            (8.0 * B + 24.0 * C)) / 6.0;
    }
    return 0.0;
}

pub fn cubicBSplineCoeff(x: F) F {
    const r = @abs(x);
    if (r < 1.0) {
        return (3.0 * r * r * r - 6.0 * r * r + 4.0) / 6.0;
    } else if (r < 2.0) {
        const t = 2.0 - r;
        return t * t * t / 6.0;
    }
    return 0.0;
}

pub fn lanczos3Coeff(x: F) F {
    const abs_x = @abs(x);
    if (abs_x < tol.tex.lancsoz_centre_snap) return 1.0;
    if (abs_x >= 3.0) return 0.0;
    const pi_x = std.math.pi * x;
    const pi_x_3 = pi_x / 3.0;
    return (std.math.sin(pi_x) / pi_x) * (std.math.sin(pi_x_3) / pi_x_3);
}

pub fn lanczos2Coeff(x: F) F {
    const abs_x = @abs(x);
    if (abs_x < tol.tex.lancsoz_centre_snap) return 1.0;
    if (abs_x >= 2.0) return 0.0;
    const pi_x = std.math.pi * x;
    const pi_x_2 = pi_x / 2.0;
    return (std.math.sin(pi_x) / pi_x) * (std.math.sin(pi_x_2) / pi_x_2);
}

pub fn quinticBSplineCoeff(x: F) F {
    const r = @abs(x);

    if (r >= 3.0) return 0.0;

    if (r <= 1.0) {
        return ((((-(1.0 / 12.0) * r + (1.0 / 4.0)) * r + 0.0) * r - (1.0 / 2.0)) * r + 0.0) * r + (11.0 / 20.0);
    } else if (r <= 2.0) {
        const t = r - 1.0;
        return (((((1.0 / 24.0) * t - (1.0 / 6.0)) * t + (1.0 / 6.0)) * t + (1.0 / 6.0)) * t - (5.0 / 12.0)) * t + (13.0 / 60.0);
    } else {
        const u = r - 2.0;
        return (((((-(1.0 / 120.0) * u + (1.0 / 24.0)) * u - (1.0 / 12.0)) * u + (1.0 / 12.0)) * u - (1.0 / 24.0)) * u + (1.0 / 120.0));
    }
}

pub const catmull_rom_lut = blk: {
    @setEvalBranchQuota(eval_branch_quota);
    var table: [lut_size][4]F = undefined;
    const lut_size_f = @as(F, @floatFromInt(lut_size));
    for (0..lut_size) |ii| {
        const ii_f = @as(F, @floatFromInt(ii));
        const tt = ii_f / lut_size_f;
        for (0..4) |jj| {
            const jj_f = @as(F, @floatFromInt(jj));
            const xx = jj_f - 1.0 - tt;
            table[ii][jj] = cubicCoeffCatmullRom(xx);
        }
    }
    break :blk table;
};

pub const mitchell_netravali_lut = blk: {
    @setEvalBranchQuota(eval_branch_quota);
    var table: [lut_size][4]F = undefined;
    const lut_size_f = @as(F, @floatFromInt(lut_size));
    for (0..lut_size) |ii| {
        const ii_f = @as(F, @floatFromInt(ii));
        const tt = ii_f / lut_size_f;
        for (0..4) |jj| {
            const jj_f = @as(F, @floatFromInt(jj));
            const xx = jj_f - 1.0 - tt;
            table[ii][jj] = cubicCoeffMitchellNetravali(xx);
        }
    }
    break :blk table;
};

pub const cubic_bspline_lut = blk: {
    @setEvalBranchQuota(eval_branch_quota);
    var table: [lut_size][4]F = undefined;
    const lut_size_f = @as(F, @floatFromInt(lut_size));
    for (0..lut_size) |ii| {
        const ii_f = @as(F, @floatFromInt(ii));
        const tt = ii_f / lut_size_f;
        for (0..4) |jj| {
            const jj_f = @as(F, @floatFromInt(jj));
            const xx = jj_f - 1.0 - tt;
            table[ii][jj] = cubicBSplineCoeff(xx);
        }
    }
    break :blk table;
};

pub const lanczos2_lut = blk: {
    @setEvalBranchQuota(eval_branch_quota);
    var table: [lut_size][4]F = undefined;
    const lut_size_f = @as(F, @floatFromInt(lut_size));
    for (0..lut_size) |ii| {
        const ii_f = @as(F, @floatFromInt(ii));
        const tt = ii_f / lut_size_f;
        for (0..4) |jj| {
            const jj_f = @as(F, @floatFromInt(jj));
            const xx = jj_f - 1.0 - tt;
            table[ii][jj] = lanczos2Coeff(xx);
        }
    }
    break :blk table;
};

pub const lanczos3_lut = blk: {
    @setEvalBranchQuota(eval_branch_quota);
    var table: [lut_size][6]F = undefined;
    const lut_size_f = @as(F, @floatFromInt(lut_size));
    for (0..lut_size) |ii| {
        const ii_f = @as(F, @floatFromInt(ii));
        const tt = ii_f / lut_size_f;
        for (0..6) |jj| {
            const jj_f = @as(F, @floatFromInt(jj));
            const xx = jj_f - 2.0 - tt;
            table[ii][jj] = lanczos3Coeff(xx);
        }
    }
    break :blk table;
};

pub const quintic_bspline_lut = blk: {
    @setEvalBranchQuota(eval_branch_quota);
    var table: [lut_size][6]F = undefined;
    const lut_size_f = @as(F, @floatFromInt(lut_size));
    for (0..lut_size) |ii| {
        const ii_f = @as(F, @floatFromInt(ii));
        const tt = ii_f / lut_size_f;
        for (0..6) |jj| {
            const jj_f = @as(F, @floatFromInt(jj));
            const xx = jj_f - 2.0 - tt;
            table[ii][jj] = quinticBSplineCoeff(xx);
        }
    }
    break :blk table;
};
