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
const S = buildconfig.SimdWidth;
const VecSB = buildconfig.VecSB;
const VecSF = buildconfig.VecSF;
const VecSI = buildconfig.VecSI;
const VecSU = buildconfig.VecSU;
const lut_size = cfg.interp_lut_size;
const tol = cfg.tol;

const common = @import("textureops_common.zig");

// --------------------------------------------------------------------------
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
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const TexSamp = common.TexSamp;
pub const TexSampMode = common.TexSampMode;
pub const TexSampConfig = common.TexSampConfig;
pub const Tex = common.Tex;

const catmull_rom_lut = common.catmull_rom_lut;
const mitchell_netravali_lut = common.mitchell_netravali_lut;
const cubic_bspline_lut = common.cubic_bspline_lut;
const lanczos2_lut = common.lanczos2_lut;
const lanczos3_lut = common.lanczos3_lut;
const quintic_bspline_lut = common.quintic_bspline_lut;
const cubicCoeffCatmullRom = common.cubicCoeffCatmullRom;
const cubicCoeffMitchellNetravali = common.cubicCoeffMitchellNetravali;
const cubicBSplineCoeff = common.cubicBSplineCoeff;
const lanczos2Coeff = common.lanczos2Coeff;
const lanczos3Coeff = common.lanczos3Coeff;
const quinticBSplineCoeff = common.quinticBSplineCoeff;
const getLerpSampCoeffs = common.getLerpSampCoeffs;
const getLerpSampCoeffsRuntime = common.getLerpSampCoeffsRuntime;
const getPx = common.getPx;

pub const sampScal = common.sampScal;
pub const sampGrey = common.sampGrey;

// --------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------

pub inline fn sampWide(
    comptime C: usize,
    comptime config: TexSampConfig,
    tex: anytype,
    v_u: VecSF,
    v_v: VecSF,
) [C]VecSF {
    @setEvalBranchQuota(eval_branch_quota);
    std.debug.assert(config.isValid());
    const tex_cols_i = @as(isize, @intCast(tex.cols_num));
    const tex_rows_i = @as(isize, @intCast(tex.rows_num));
    const tex_cols_minus_1_i = tex_cols_i - 1;
    const tex_rows_minus_1_i = tex_rows_i - 1;
    const tex_cols_minus_1_f = @as(F, @floatFromInt(tex_cols_minus_1_i));
    const tex_rows_minus_1_f = @as(F, @floatFromInt(tex_rows_minus_1_i));

    const v_tex_x_f = v_u * @as(VecSF, @splat(tex_cols_minus_1_f));
    const v_tex_y_f = v_v * @as(VecSF, @splat(tex_rows_minus_1_f));

    var v_tex_x_i: [S]isize = undefined;
    var v_tex_y_i: [S]isize = undefined;
    const tex_x_f_arr: [S]F = v_tex_x_f;
    const tex_y_f_arr: [S]F = v_tex_y_f;

    for (0..S) |ii| {
        v_tex_x_i[ii] = @as(isize, @intFromFloat(@floor(tex_x_f_arr[ii])));
        v_tex_y_i[ii] = @as(isize, @intFromFloat(@floor(tex_y_f_arr[ii])));
    }

    const v_tex_x_i_vec = @as(VecSI, v_tex_x_i);
    const v_tex_y_i_vec = @as(VecSI, v_tex_y_i);
    const v_tex_x_i_f = @as(VecSF, @floatFromInt(v_tex_x_i_vec));
    const v_tex_y_i_f = @as(VecSF, @floatFromInt(v_tex_y_i_vec));
    const v_tex_x_frac = v_tex_x_f - v_tex_x_i_f;
    const v_tex_y_frac = v_tex_y_f - v_tex_y_i_f;

    return switch (config.sample) {
        .nearest => getPxWide(
            C,
            tex,
            @as(VecSI, @intFromFloat(@round(v_tex_x_f))),
            @as(VecSI, @intFromFloat(@round(v_tex_y_f))),
        ),
        .linear => sampLinearWide(
            C,
            tex,
            v_tex_x_i,
            v_tex_y_i,
            v_tex_x_frac,
            v_tex_y_frac,
        ),
        .cubic_catmull_rom, .cubic_mitchell_netravali, .cubic_bspline => samp4TapWide(
            C,
            config.sample,
            config.mode,
            tex,
            v_tex_x_i,
            v_tex_y_i,
            v_tex_x_frac,
            v_tex_y_frac,
        ),
        .lanczos2 => samp4TapWide(
            C,
            config.sample,
            config.mode,
            tex,
            v_tex_x_i,
            v_tex_y_i,
            v_tex_x_frac,
            v_tex_y_frac,
        ),
        .lanczos3, .quintic_bspline => samp6TapWide(
            C,
            config.sample,
            config.mode,
            tex,
            v_tex_x_i,
            v_tex_y_i,
            v_tex_x_frac,
            v_tex_y_frac,
        ),
    };
}

pub inline fn sampLanes(
    comptime C: usize,
    comptime config: TexSampConfig,
    v_mask_active: VecSB,
    tex: anytype,
    v_u: VecSF,
    v_v: VecSF,
) [C]VecSF {
    @setEvalBranchQuota(eval_branch_quota);
    var samp_res_arr: [C][S]F = [_][S]F{[_]F{0.0} ** S} ** C;
    const mask_arr: [S]bool = v_mask_active;
    const u_arr: [S]F = v_u;
    const v_arr: [S]F = v_v;

    for (0..S) |ii| {
        if (mask_arr[ii]) {
            const sampd = sampOneLane(
                C,
                config,
                tex,
                u_arr[ii],
                v_arr[ii],
            );
            inline for (0..C) |ch| {
                samp_res_arr[ch][ii] = sampd[ch];
            }
        }
    }

    var samp_res: [C]VecSF = undefined;
    inline for (0..C) |ch| {
        samp_res[ch] = samp_res_arr[ch];
    }

    return samp_res;
}

pub inline fn sampLanesTri3(
    comptime C: usize,
    comptime config: TexSampConfig,
    v_mask_active: VecSB,
    tex: anytype,
    v_u: VecSF,
    v_v: VecSF,
) [C]VecSF {
    @setEvalBranchQuota(eval_branch_quota);
    var samp_res_arr: [C][S]F = [_][S]F{[_]F{0.0} ** S} ** C;
    const mask_arr: [S]bool = v_mask_active;
    const u_arr: [S]F = v_u;
    const v_arr: [S]F = v_v;

    var active_lanes: [S]usize = undefined;
    var active_count: usize = 0;

    const tex_cols_i = @as(isize, @intCast(tex.cols_num));
    const tex_rows_i = @as(isize, @intCast(tex.rows_num));
    const tex_cols_minus_1_i = tex_cols_i - 1;
    const tex_rows_minus_1_i = tex_rows_i - 1;
    const tex_cols_minus_1_f = @as(F, @floatFromInt(tex_cols_minus_1_i));
    const tex_rows_minus_1_f = @as(F, @floatFromInt(tex_rows_minus_1_i));

    for (0..S) |ii| {
        if (mask_arr[ii]) {
            active_lanes[active_count] = ii;
            active_count += 1;
        }
    }

    if (active_count > 1) {
        var lane_keys: [S]u64 = undefined;
        for (0..active_count) |ii| {
            const lane = active_lanes[ii];
            const xf = u_arr[lane] * tex_cols_minus_1_f;
            const yf = v_arr[lane] * tex_rows_minus_1_f;

            const tex_x_i = @as(isize, @intFromFloat(@floor(xf)));
            const tex_y_i = @as(isize, @intFromFloat(@floor(yf)));

            const tex_x_key_clamp = @max(
                @as(isize, 0),
                @min(tex_x_i, tex_cols_minus_1_i),
            );
            const tex_y_key_clamp = @max(
                @as(isize, 0),
                @min(tex_y_i, tex_rows_minus_1_i),
            );

            const x_key = @as(usize, @intCast(tex_x_key_clamp));
            const y_key = @as(usize, @intCast(tex_y_key_clamp));

            lane_keys[ii] =
                (@as(u64, @intCast(y_key)) << 32) | @as(u64, @intCast(x_key));
        }

        for (1..active_count) |ii| {
            const lane = active_lanes[ii];
            const lane_key = lane_keys[ii];
            var jj = ii;
            while (jj > 0 and lane_key < lane_keys[jj - 1]) : (jj -= 1) {
                lane_keys[jj] = lane_keys[jj - 1];
                active_lanes[jj] = active_lanes[jj - 1];
            }
            lane_keys[jj] = lane_key;
            active_lanes[jj] = lane;
        }
    }

    for (0..active_count) |ii| {
        const lane = active_lanes[ii];
        const sampd = sampOneLane(
            C,
            config,
            tex,
            u_arr[lane],
            v_arr[lane],
        );
        inline for (0..C) |ch| {
            samp_res_arr[ch][lane] = sampd[ch];
        }
    }

    var samp_res: [C]VecSF = undefined;
    inline for (0..C) |ch| {
        samp_res[ch] = samp_res_arr[ch];
    }

    return samp_res;
}

pub inline fn sampOneLane(
    comptime C: usize,
    comptime config: TexSampConfig,
    tex: anytype,
    u: F,
    v: F,
) [C]F {
    const tex_cols_i = @as(isize, @intCast(tex.cols_num));
    const tex_rows_i = @as(isize, @intCast(tex.rows_num));
    const tex_cols_minus_1_i = tex_cols_i - 1;
    const tex_rows_minus_1_i = tex_rows_i - 1;
    const tex_cols_minus_1_f = @as(F, @floatFromInt(tex_cols_minus_1_i));
    const tex_rows_minus_1_f = @as(F, @floatFromInt(tex_rows_minus_1_i));

    const xf = u * tex_cols_minus_1_f;
    const yf = v * tex_rows_minus_1_f;

    const tex_x_i = @as(isize, @intFromFloat(@floor(xf)));
    const tex_y_i = @as(isize, @intFromFloat(@floor(yf)));

    const tex_x_frac = xf - @as(F, @floatFromInt(tex_x_i));
    const tex_y_frac = yf - @as(F, @floatFromInt(tex_y_i));
    const tex_x_round = @as(isize, @intFromFloat(@round(xf)));
    const tex_y_round = @as(isize, @intFromFloat(@round(yf)));

    return switch (config.sample) {
        .nearest => getPx(
            C,
            tex,
            tex_x_round,
            tex_y_round,
        ),
        .linear => sampLinearOneLane(
            C,
            tex,
            tex_x_i,
            tex_y_i,
            tex_x_frac,
            tex_y_frac,
        ),
        .cubic_catmull_rom, .cubic_mitchell_netravali, .cubic_bspline => samp4TapOneLane(
            C,
            config.sample,
            config.mode,
            tex,
            tex_x_i,
            tex_y_i,
            tex_x_frac,
            tex_y_frac,
        ),
        .lanczos2 => samp4TapOneLane(
            C,
            config.sample,
            config.mode,
            tex,
            tex_x_i,
            tex_y_i,
            tex_x_frac,
            tex_y_frac,
        ),
        .lanczos3, .quintic_bspline => samp6TapOneLane(
            C,
            config.sample,
            config.mode,
            tex,
            tex_x_i,
            tex_y_i,
            tex_x_frac,
            tex_y_frac,
        ),
    };
}

// --------------------------------------------------------------------------
// Wide SIMD Strategy
// --------------------------------------------------------------------------

fn getPxWide(
    comptime CH: usize,
    tex: anytype,
    v_tex_x_i: VecSI,
    v_tex_y_i: VecSI,
) [CH]VecSF {
    const T = @TypeOf(tex.array.slice[0]);
    const tex_cols = @as(isize, @intCast(tex.cols_num));
    const tex_rows = @as(isize, @intCast(tex.rows_num));

    const v_splat_zero: VecSI = @splat(0);
    const v_tex_cols_m1: VecSI = @splat(tex_cols - 1);
    const v_tex_rows_m1: VecSI = @splat(tex_rows - 1);

    const v_x_clamp = @max(v_splat_zero, @min(v_tex_x_i, v_tex_cols_m1));
    const v_y_clamp = @max(v_splat_zero, @min(v_tex_y_i, v_tex_rows_m1));
    const v_xu = @as(VecSU, @intCast(v_x_clamp));
    const v_yu = @as(VecSU, @intCast(v_y_clamp));

    const stride_y = tex.array.strides[1];
    const stride_x = @as(usize, 1);

    const v_pixel_offsets = v_yu * @as(VecSU, @splat(stride_y)) +
        v_xu * @as(VecSU, @splat(stride_x));

    var samp_res: [CH]VecSF = undefined;

    const first_off = v_pixel_offsets[0];
    const v_expected = @as(VecSU, @splat(first_off)) +
        std.simd.iota(usize, S);
    const is_contiguous = @reduce(.And, v_pixel_offsets == v_expected);

    inline for (0..CH) |cc| {
        const base_slice = tex.array.getPlaneSlice(cc);
        var vals: [S]F = undefined;
        if (is_contiguous) {
            const raw_vals: @Vector(S, T) = base_slice[first_off..][0..S].*;
            samp_res[cc] = texelVecToFloat(T, raw_vals);
        } else {
            const tap_offsets_arr: [S]usize = v_pixel_offsets;
            for (0..S) |ii| {
                vals[ii] = texelScalToFloat(T, base_slice[tap_offsets_arr[ii]]);
            }
            samp_res[cc] = vals;
        }
    }

    return samp_res;
}

pub inline fn sampLinearWide(
    comptime C: usize,
    tex: anytype,
    v_tex_x_i: VecSI,
    v_tex_y_i: VecSI,
    v_tex_x_frac: VecSF,
    v_tex_y_frac: VecSF,
) [C]VecSF {
    const T = @TypeOf(tex.array.slice[0]);
    const tex_cols = @as(isize, @intCast(tex.cols_num));
    const tex_rows = @as(isize, @intCast(tex.rows_num));

    const v_splat_zero: VecSI = @splat(0);
    const v_tex_cols_m1: VecSI = @splat(tex_cols - 1);
    const v_tex_rows_m1: VecSI = @splat(tex_rows - 1);

    const v_splat_1_i: VecSI = @splat(1);
    const v_x0_clamp = @max(v_splat_zero, @min(v_tex_x_i, v_tex_cols_m1));
    const v_x1_clamp = @max(v_splat_zero, @min(v_tex_x_i + v_splat_1_i, v_tex_cols_m1));
    const v_y0_clamp = @max(v_splat_zero, @min(v_tex_y_i, v_tex_rows_m1));
    const v_y1_clamp = @max(v_splat_zero, @min(v_tex_y_i + v_splat_1_i, v_tex_rows_m1));
    const v_x0 = @as(VecSU, @intCast(v_x0_clamp));
    const v_x1 = @as(VecSU, @intCast(v_x1_clamp));
    const v_y0 = @as(VecSU, @intCast(v_y0_clamp));
    const v_y1 = @as(VecSU, @intCast(v_y1_clamp));

    const stride_y = tex.array.strides[1];
    const stride_x = @as(usize, 1);

    const v_off00 = v_y0 * @as(VecSU, @splat(stride_y)) +
        v_x0 * @as(VecSU, @splat(stride_x));
    const v_off10 = v_y0 * @as(VecSU, @splat(stride_y)) +
        v_x1 * @as(VecSU, @splat(stride_x));
    const v_off01 = v_y1 * @as(VecSU, @splat(stride_y)) +
        v_x0 * @as(VecSU, @splat(stride_x));
    const v_off11 = v_y1 * @as(VecSU, @splat(stride_y)) +
        v_x1 * @as(VecSU, @splat(stride_x));

    var samp_res: [C]VecSF = undefined;

    inline for (0..C) |cc| {
        var p00: VecSF = undefined;
        var p10: VecSF = undefined;
        var p01: VecSF = undefined;
        var p11: VecSF = undefined;

        const base_slice = tex.array.getPlaneSlice(cc);
        p00 = loadTap(T, base_slice, v_off00);
        p10 = loadTap(T, base_slice, v_off10);
        p01 = loadTap(T, base_slice, v_off01);
        p11 = loadTap(T, base_slice, v_off11);

        const v_p00_10 = p00 + v_tex_x_frac * (p10 - p00);
        const v_p01_11 = p01 + v_tex_x_frac * (p11 - p01);
        samp_res[cc] = v_p00_10 + v_tex_y_frac * (v_p01_11 - v_p00_10);
    }

    return samp_res;
}

inline fn sampConvoWide(
    comptime C: usize,
    comptime TAP: usize,
    tex: anytype,
    v_tex_x_i: VecSI,
    v_tex_y_i: VecSI,
    tap_offset: isize,
    v_samp_coeff_x: [TAP]VecSF,
    v_samp_coeff_y: [TAP]VecSF,
    v_samp_coeff_sum: VecSF,
) [C]VecSF {
    @setEvalBranchQuota(eval_branch_quota);
    var samp_res: [C]VecSF = [_]VecSF{@splat(0.0)} ** C;
    var v_tap_samp_coeff_planes: [TAP * TAP]VecSF = undefined;

    inline for (0..TAP) |jj| {
        const v_samp_coeff_y_val = v_samp_coeff_y[jj];
        inline for (0..TAP) |ii| {
            const v_tap_samp_coeff = v_samp_coeff_x[ii] * v_samp_coeff_y_val;
            v_tap_samp_coeff_planes[jj * TAP + ii] = v_tap_samp_coeff;
        }
    }

    inline for (0..TAP) |jj| {
        inline for (0..TAP) |ii| {
            const v_tap_samp_coeff = v_tap_samp_coeff_planes[jj * TAP + ii];
            const ii_i = @as(isize, @intCast(ii));
            const jj_i = @as(isize, @intCast(jj));
            const x_off = ii_i - tap_offset;
            const y_off = jj_i - tap_offset;
            const v_x_off = @as(VecSI, @splat(x_off));
            const v_y_off = @as(VecSI, @splat(y_off));

            const v_px_vecs = getPxWide(
                C,
                tex,
                v_tex_x_i + v_x_off,
                v_tex_y_i + v_y_off,
            );

            inline for (0..C) |ch| {
                samp_res[ch] += v_px_vecs[ch] * v_tap_samp_coeff;
            }
        }
    }

    const v_splat_one: VecSF = @splat(1.0);
    const v_inv_w_sum = @select(
        F,
        @abs(v_samp_coeff_sum) < @as(VecSF, @splat(tol.tex.samp_coeff_sum)),
        v_splat_one,
        v_splat_one / v_samp_coeff_sum,
    );
    inline for (0..C) |ch| {
        samp_res[ch] *= v_inv_w_sum;
    }
    return samp_res;
}

fn samp4TapWide(
    comptime C: usize,
    comptime sample: TexSamp,
    comptime mode: TexSampMode,
    tex: anytype,
    v_tex_x_i: [S]isize,
    v_tex_y_i: [S]isize,
    v_tex_x_frac: VecSF,
    v_tex_y_frac: VecSF,
) [C]VecSF {
    const TAP = 4;
    const tap_offset = @divTrunc(@as(isize, @intCast(TAP)), 2) - 1;

    const tex_x_frac_arr: [S]F = v_tex_x_frac;
    const tex_y_frac_arr: [S]F = v_tex_y_frac;

    const lut = switch (sample) {
        .cubic_catmull_rom => catmull_rom_lut,
        .cubic_mitchell_netravali => mitchell_netravali_lut,
        .cubic_bspline => cubic_bspline_lut,
        .lanczos2 => lanczos2_lut,
        else => unreachable,
    };

    var v_samp_coeff_sum: VecSF = @splat(0.0);
    const v_samp_coeffs = switch (mode) {
        .direct => blk: {
            const v_kernel: *const fn (VecSF) VecSF = switch (sample) {
                .cubic_catmull_rom => cubicCoeffCatmullRomSIMD,
                .cubic_mitchell_netravali => cubicCoeffMitchellNetravaliSIMD,
                .cubic_bspline => cubicBSplineCoeffSIMD,
                .lanczos2 => lanczos2CoeffSIMD,
                else => unreachable,
            };

            const coeffs_x = [TAP]VecSF{
                v_kernel(v_tex_x_frac + @as(VecSF, @splat(1.0))),
                v_kernel(v_tex_x_frac),
                v_kernel(v_tex_x_frac - @as(VecSF, @splat(1.0))),
                v_kernel(v_tex_x_frac - @as(VecSF, @splat(2.0))),
            };

            const coeffs_y = [TAP]VecSF{
                v_kernel(v_tex_y_frac + @as(VecSF, @splat(1.0))),
                v_kernel(v_tex_y_frac),
                v_kernel(v_tex_y_frac - @as(VecSF, @splat(1.0))),
                v_kernel(v_tex_y_frac - @as(VecSF, @splat(2.0))),
            };

            inline for (0..TAP) |jj| {
                inline for (0..TAP) |ii| {
                    v_samp_coeff_sum += coeffs_x[ii] * coeffs_y[jj];
                }
            }

            break :blk [2][TAP]VecSF{ coeffs_x, coeffs_y };
        },
        .lut => blk: {
            var coeffs_x_arr: [TAP][S]F = undefined;
            var coeffs_y_arr: [TAP][S]F = undefined;
            const lut_size_f = @as(F, @floatFromInt(lut_size - 1));

            for (0..S) |ii| {
                const tex_x_lut = tex_x_frac_arr[ii] * lut_size_f;
                const tex_y_lut = tex_y_frac_arr[ii] * lut_size_f;
                const ix = @as(usize, @intFromFloat(tex_x_lut));
                const iy = @as(usize, @intFromFloat(tex_y_lut));

                inline for (0..TAP) |kk| {
                    coeffs_x_arr[kk][ii] = lut[ix][kk];
                    coeffs_y_arr[kk][ii] = lut[iy][kk];
                }
            }

            var samp_coeff_sum_arr: [S]F = @splat(0.0);

            for (0..S) |ii| {
                var sum: F = 0.0;
                const tex_x_lut = tex_x_frac_arr[ii] * lut_size_f;
                const tex_y_lut = tex_y_frac_arr[ii] * lut_size_f;
                const ix = @as(usize, @intFromFloat(tex_x_lut));
                const iy = @as(usize, @intFromFloat(tex_y_lut));

                for (0..TAP) |jj| {
                    for (0..TAP) |kk| {
                        sum += lut[ix][kk] * lut[iy][jj];
                    }
                }
                samp_coeff_sum_arr[ii] = sum;
            }
            v_samp_coeff_sum = samp_coeff_sum_arr;

            var coeffs_x: [TAP]VecSF = undefined;
            var coeffs_y: [TAP]VecSF = undefined;

            inline for (0..TAP) |kk| {
                coeffs_x[kk] = coeffs_x_arr[kk];
                coeffs_y[kk] = coeffs_y_arr[kk];
            }

            break :blk [2][TAP]VecSF{ coeffs_x, coeffs_y };
        },
        .lut_lerp => blk: {
            var coeffs_x_arr: [TAP][S]F = undefined;
            var coeffs_y_arr: [TAP][S]F = undefined;
            var samp_coeff_sum_arr: [S]F = @splat(0.0);

            for (0..S) |ii| {
                const coeffs_x_lane = getLerpSampCoeffs(
                    TAP,
                    lut,
                    tex_x_frac_arr[ii],
                );
                const coeffs_y_lane = getLerpSampCoeffs(
                    TAP,
                    lut,
                    tex_y_frac_arr[ii],
                );

                var sum: F = 0.0;
                inline for (0..TAP) |kk| {
                    coeffs_x_arr[kk][ii] = coeffs_x_lane[kk];
                    coeffs_y_arr[kk][ii] = coeffs_y_lane[kk];
                }

                for (0..TAP) |jj| {
                    for (0..TAP) |kk| {
                        sum += coeffs_x_lane[kk] * coeffs_y_lane[jj];
                    }
                }
                samp_coeff_sum_arr[ii] = sum;
            }
            v_samp_coeff_sum = samp_coeff_sum_arr;

            var coeffs_x: [TAP]VecSF = undefined;
            var coeffs_y: [TAP]VecSF = undefined;
            inline for (0..TAP) |kk| {
                coeffs_x[kk] = coeffs_x_arr[kk];
                coeffs_y[kk] = coeffs_y_arr[kk];
            }

            break :blk [2][TAP]VecSF{ coeffs_x, coeffs_y };
        },
    };

    return sampConvoWide(
        C,
        TAP,
        tex,
        v_tex_x_i,
        v_tex_y_i,
        tap_offset,
        v_samp_coeffs[0],
        v_samp_coeffs[1],
        v_samp_coeff_sum,
    );
}

fn samp6TapWide(
    comptime C: usize,
    comptime sample: TexSamp,
    comptime mode: TexSampMode,
    tex: anytype,
    v_tex_x_i: [S]isize,
    v_tex_y_i: [S]isize,
    v_tex_x_frac: VecSF,
    v_tex_y_frac: VecSF,
) [C]VecSF {
    const TAP = 6;
    const tap_offset = @divTrunc(@as(isize, @intCast(TAP)), 2) - 1;

    const tex_x_frac_arr: [S]F = v_tex_x_frac;
    const tex_y_frac_arr: [S]F = v_tex_y_frac;

    const lut = switch (sample) {
        .lanczos3 => lanczos3_lut,
        .quintic_bspline => quintic_bspline_lut,
        else => unreachable,
    };

    var v_samp_coeff_sum: VecSF = @splat(0.0);
    const v_samp_coeffs = switch (mode) {
        .direct => blk: {
            const v_kernel: *const fn (VecSF) VecSF = switch (sample) {
                .lanczos3 => lanczos3CoeffSIMD,
                .quintic_bspline => quinticBSplineCoeffSIMD,
                else => unreachable,
            };

            const coeffs_x = [TAP]VecSF{
                v_kernel(v_tex_x_frac + @as(VecSF, @splat(2.0))),
                v_kernel(v_tex_x_frac + @as(VecSF, @splat(1.0))),
                v_kernel(v_tex_x_frac),
                v_kernel(v_tex_x_frac - @as(VecSF, @splat(1.0))),
                v_kernel(v_tex_x_frac - @as(VecSF, @splat(2.0))),
                v_kernel(v_tex_x_frac - @as(VecSF, @splat(3.0))),
            };

            const coeffs_y = [TAP]VecSF{
                v_kernel(v_tex_y_frac + @as(VecSF, @splat(2.0))),
                v_kernel(v_tex_y_frac + @as(VecSF, @splat(1.0))),
                v_kernel(v_tex_y_frac),
                v_kernel(v_tex_y_frac - @as(VecSF, @splat(1.0))),
                v_kernel(v_tex_y_frac - @as(VecSF, @splat(2.0))),
                v_kernel(v_tex_y_frac - @as(VecSF, @splat(3.0))),
            };

            inline for (0..TAP) |jj| {
                inline for (0..TAP) |ii| {
                    v_samp_coeff_sum += coeffs_x[ii] * coeffs_y[jj];
                }
            }

            break :blk [2][TAP]VecSF{ coeffs_x, coeffs_y };
        },
        .lut => blk: {
            var coeffs_x_arr: [TAP][S]F = undefined;
            var coeffs_y_arr: [TAP][S]F = undefined;
            var samp_coeff_sum_arr: [S]F = @splat(0.0);
            const lut_size_f = @as(F, @floatFromInt(lut_size - 1));

            for (0..S) |ii| {
                const tex_x_lut = tex_x_frac_arr[ii] * lut_size_f;
                const tex_y_lut = tex_y_frac_arr[ii] * lut_size_f;
                const ix = @as(usize, @intFromFloat(tex_x_lut));
                const iy = @as(usize, @intFromFloat(tex_y_lut));
                var sum: F = 0.0;

                inline for (0..TAP) |kk| {
                    coeffs_x_arr[kk][ii] = lut[ix][kk];
                    coeffs_y_arr[kk][ii] = lut[iy][kk];
                }

                for (0..TAP) |jj| {
                    for (0..TAP) |kk| {
                        sum += lut[ix][kk] * lut[iy][jj];
                    }
                }

                samp_coeff_sum_arr[ii] = sum;
            }
            v_samp_coeff_sum = samp_coeff_sum_arr;

            var coeffs_x: [TAP]VecSF = undefined;
            var coeffs_y: [TAP]VecSF = undefined;
            inline for (0..TAP) |kk| {
                coeffs_x[kk] = coeffs_x_arr[kk];
                coeffs_y[kk] = coeffs_y_arr[kk];
            }

            break :blk [2][TAP]VecSF{ coeffs_x, coeffs_y };
        },
        .lut_lerp => blk: {
            var coeffs_x_arr: [TAP][S]F = undefined;
            var coeffs_y_arr: [TAP][S]F = undefined;
            var samp_coeff_sum_arr: [S]F = @splat(0.0);

            for (0..S) |ii| {
                const coeffs_x_lane = getLerpSampCoeffs(
                    TAP,
                    lut,
                    tex_x_frac_arr[ii],
                );

                const coeffs_y_lane = getLerpSampCoeffs(
                    TAP,
                    lut,
                    tex_y_frac_arr[ii],
                );

                var sum: F = 0.0;
                inline for (0..TAP) |kk| {
                    coeffs_x_arr[kk][ii] = coeffs_x_lane[kk];
                    coeffs_y_arr[kk][ii] = coeffs_y_lane[kk];
                }

                for (0..TAP) |jj| {
                    for (0..TAP) |kk| {
                        sum += coeffs_x_lane[kk] * coeffs_y_lane[jj];
                    }
                }

                samp_coeff_sum_arr[ii] = sum;
            }

            v_samp_coeff_sum = samp_coeff_sum_arr;

            var coeffs_x: [TAP]VecSF = undefined;
            var coeffs_y: [TAP]VecSF = undefined;

            inline for (0..TAP) |kk| {
                coeffs_x[kk] = coeffs_x_arr[kk];
                coeffs_y[kk] = coeffs_y_arr[kk];
            }

            break :blk [2][TAP]VecSF{ coeffs_x, coeffs_y };
        },
    };

    return sampConvoWide(
        C,
        TAP,
        tex,
        v_tex_x_i,
        v_tex_y_i,
        tap_offset,
        v_samp_coeffs[0],
        v_samp_coeffs[1],
        v_samp_coeff_sum,
    );
}

// --------------------------------------------------------------------------
// Lane SIMD Strategy
// --------------------------------------------------------------------------

pub inline fn sampLinearOneLane(
    comptime C: usize,
    tex: anytype,
    tex_x_i: isize,
    tex_y_i: isize,
    tex_x_frac: F,
    tex_y_frac: F,
) [C]F {
    const samp_coeff_x = [2]F{ 1.0 - tex_x_frac, tex_x_frac };
    const samp_coeff_y = [2]F{ 1.0 - tex_y_frac, tex_y_frac };
    return sampConvoOneLane(
        C,
        2,
        tex,
        tex_x_i,
        tex_y_i,
        samp_coeff_x,
        samp_coeff_y,
    );
}

inline fn sampConvoOneLane(
    comptime C: usize,
    comptime TAP: usize,
    tex: anytype,
    tex_x_i: isize,
    tex_y_i: isize,
    samp_coeff_x: [TAP]F,
    samp_coeff_y: [TAP]F,
) [C]F {
    @setEvalBranchQuota(eval_branch_quota);
    const T = @TypeOf(tex.array.slice[0]);
    const tap_offset = @as(isize, @intCast(TAP)) / 2 - 1;
    const tex_start_x = tex_x_i - tap_offset;
    const tex_start_y = tex_y_i - tap_offset;

    const tex_cols = @as(isize, @intCast(tex.cols_num));
    const tex_rows = @as(isize, @intCast(tex.rows_num));

    var samp_res: [C]F = [_]F{0.0} ** C;

    if (tex_start_x >= 0 and tex_start_x + @as(isize, @intCast(TAP)) <= tex_cols and
        tex_start_y >= 0 and tex_start_y + @as(isize, @intCast(TAP)) <= tex_rows)
    {
        const v_samp_coeff_x: @Vector(TAP, F) = samp_coeff_x;
        const v_samp_coeff_y: @Vector(TAP, F) = samp_coeff_y;
        const stride_y = tex.array.strides[1];

        const samp_coeff_sum = @reduce(.Add, v_samp_coeff_x) * @reduce(.Add, v_samp_coeff_y);

        const inv_samp_coeff_sum = if (@abs(samp_coeff_sum) > tol.tex.samp_coeff_sum)
            1.0 / samp_coeff_sum
        else
            1.0;

        inline for (0..TAP) |jj| {
            const wy_val = v_samp_coeff_y[jj];
            const row_off =
                @as(usize, @intCast(tex_start_y + @as(isize, @intCast(jj)))) *
                stride_y +
                @as(usize, @intCast(tex_start_x));

            inline for (0..C) |ch| {
                const plane_slice = tex.array.getPlaneSlice(ch);
                const v_row_raw: @Vector(TAP, T) =
                    plane_slice[row_off..][0..TAP].*;

                const v_row: @Vector(TAP, F) = switch (@typeInfo(T)) {
                    .int => @floatFromInt(v_row_raw),
                    .float => @floatCast(v_row_raw),
                    else => @compileError("Unsupped tex storage type."),
                };
                const row_sum = @reduce(.Add, v_row * v_samp_coeff_x);
                samp_res[ch] += row_sum * wy_val;
            }
        }

        inline for (0..C) |ch| {
            samp_res[ch] *= inv_samp_coeff_sum;
        }
    } else {
        var samp_coeff_sum: F = 0.0;
        for (0..TAP) |jj| {
            for (0..TAP) |ii| {
                const w = samp_coeff_x[ii] * samp_coeff_y[jj];

                const px = getPx(
                    C,
                    tex,
                    tex_start_x + @as(isize, @intCast(ii)),
                    tex_start_y + @as(isize, @intCast(jj)),
                );

                inline for (0..C) |ch| {
                    samp_res[ch] += px[ch] * w;
                }
                samp_coeff_sum += w;
            }
        }
        if (@abs(samp_coeff_sum) > tol.tex.samp_coeff_sum) {
            inline for (0..C) |ch| {
                samp_res[ch] /= samp_coeff_sum;
            }
        }
    }

    return samp_res;
}

fn samp4TapOneLane(
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

            break :blk sampConvoOneLane(
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

            break :blk sampConvoOneLane(
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

            break :blk sampConvoOneLane(
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

fn samp6TapOneLane(
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

            break :blk sampConvoOneLane(
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

            break :blk sampConvoOneLane(
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

            break :blk sampConvoOneLane(
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

// --------------------------------------------------------------------------
// Shared Infra & Helpers
// --------------------------------------------------------------------------

inline fn texelVecToFloat(comptime T: type, v_vals: @Vector(S, T)) VecSF {
    return switch (@typeInfo(T)) {
        .int => @as(VecSF, @floatFromInt(v_vals)),
        .float => @as(VecSF, @floatCast(v_vals)),
        else => @compileError("Unsupped tex storage type."),
    };
}

inline fn texelScalToFloat(comptime T: type, val: T) F {
    return switch (@typeInfo(T)) {
        .int => @as(F, @floatFromInt(val)),
        .float => @as(F, @floatCast(val)),
        else => @compileError("Unsupped tex storage type."),
    };
}

inline fn loadTap(comptime T: type, slice: []const T, v_offsets: VecSU) VecSF {
    const first_off = v_offsets[0];
    const v_expected = @as(VecSU, @splat(first_off)) + std.simd.iota(usize, S);
    const is_contiguous = @reduce(.And, v_offsets == v_expected);

    var vals: [S]F = undefined;
    if (is_contiguous) {
        const raw_vals: @Vector(S, T) = slice[first_off..][0..S].*;
        return texelVecToFloat(T, raw_vals);
    } else {
        const offsets_arr: [S]usize = v_offsets;
        for (0..S) |ii| {
            vals[ii] = texelScalToFloat(T, slice[offsets_arr[ii]]);
        }
        return vals;
    }
}

fn cubicCoeffCatmullRomSIMD(v_x: VecSF) VecSF {
    const v_ax = @abs(v_x);
    const v_splat_one: VecSF = @splat(1.0);
    const v_splat_two: VecSF = @splat(2.0);

    const v_mask_inner = v_ax <= v_splat_one;
    const v_mask_outer = (v_ax < v_splat_two) & !v_mask_inner;

    const v_w1 =
        ((@as(VecSF, @splat(1.5)) * v_ax -
            @as(VecSF, @splat(2.5))) * v_ax +
            @as(VecSF, @splat(0.0))) * v_ax +
        v_splat_one;
    const v_w2 =
        ((-@as(VecSF, @splat(0.5)) * v_ax +
            @as(VecSF, @splat(2.5))) * v_ax -
            @as(VecSF, @splat(4.0))) * v_ax +
        v_splat_two;

    var samp_res = @select(
        F,
        v_mask_inner,
        v_w1,
        @as(VecSF, @splat(0.0)),
    );
    samp_res = @select(F, v_mask_outer, v_w2, samp_res);
    return samp_res;
}

fn cubicCoeffMitchellNetravaliSIMD(v_x: VecSF) VecSF {
    const v_r = @abs(v_x);
    const B = 1.0 / 3.0;
    const C = 1.0 / 3.0;
    const v_splat_one: VecSF = @splat(1.0);
    const v_splat_two: VecSF = @splat(2.0);

    const v_mask_inner = v_r < v_splat_one;
    const v_mask_outer = (v_r < v_splat_two) & !v_mask_inner;

    const v_w1 = ((@as(VecSF, @splat(12.0 - 9.0 * B - 6.0 * C)) * v_r * v_r * v_r +
        @as(VecSF, @splat(-18.0 + 12.0 * B + 6.0 * C)) * v_r * v_r +
        @as(VecSF, @splat(6.0 - 2.0 * B))) / @as(VecSF, @splat(6.0)));

    const v_w2 = ((@as(VecSF, @splat(-B - 6.0 * C)) * v_r * v_r * v_r +
        @as(VecSF, @splat(6.0 * B + 30.0 * C)) * v_r * v_r +
        @as(VecSF, @splat(-12.0 * B - 48.0 * C)) * v_r +
        @as(VecSF, @splat(8.0 * B + 24.0 * C))) / @as(VecSF, @splat(6.0)));

    var samp_res = @select(F, v_mask_inner, v_w1, @as(VecSF, @splat(0.0)));
    samp_res = @select(F, v_mask_outer, v_w2, samp_res);
    return samp_res;
}

fn cubicBSplineCoeffSIMD(v_x: VecSF) VecSF {
    const v_r = @abs(v_x);
    const v_splat_one: VecSF = @splat(1.0);
    const v_splat_two: VecSF = @splat(2.0);

    const v_mask_inner = v_r < v_splat_one;
    const v_mask_outer = (v_r < v_splat_two) & !v_mask_inner;

    const v_w1 = (@as(VecSF, @splat(3.0)) * v_r * v_r * v_r -
        @as(VecSF, @splat(6.0)) * v_r * v_r +
        @as(VecSF, @splat(4.0))) / @as(VecSF, @splat(6.0));

    const v_t = v_splat_two - v_r;
    const v_w2 = v_t * v_t * v_t / @as(VecSF, @splat(6.0));

    var samp_res = @select(F, v_mask_inner, v_w1, @as(VecSF, @splat(0.0)));
    samp_res = @select(F, v_mask_outer, v_w2, samp_res);
    return samp_res;
}

fn lanczos3CoeffSIMD(v_x: VecSF) VecSF {
    const v_ax = @abs(v_x);
    var samp_res_arr: [S]F = undefined;
    const ax_arr: [S]F = v_ax;
    for (0..S) |ii| {
        samp_res_arr[ii] = lanczos3Coeff(ax_arr[ii]);
    }
    return samp_res_arr;
}

fn lanczos2CoeffSIMD(v_x: VecSF) VecSF {
    const x_arr: [S]F = v_x;
    var samp_res_arr: [S]F = undefined;
    for (0..S) |ii| {
        samp_res_arr[ii] = lanczos2Coeff(x_arr[ii]);
    }
    return samp_res_arr;
}

fn quinticBSplineCoeffSIMD(v_x: VecSF) VecSF {
    const v_r = @abs(v_x);
    const v_splat_zero: VecSF = @splat(0.0);
    const v_splat_one: VecSF = @splat(1.0);
    const v_splat_two: VecSF = @splat(2.0);
    const v_splat_three: VecSF = @splat(3.0);

    const v_mask_inner = v_r <= v_splat_one;
    const v_mask_middle = (v_r <= v_splat_two) & !v_mask_inner;
    const v_mask_outer = (v_r < v_splat_three) & !v_mask_inner & !v_mask_middle;

    const v_w1 = ((((-@as(VecSF, @splat(1.0 / 12.0)) * v_r +
        @as(VecSF, @splat(1.0 / 4.0))) *
        v_r + v_splat_zero) * v_r - @as(VecSF, @splat(1.0 / 2.0))) * v_r +
        v_splat_zero) * v_r + @as(VecSF, @splat(11.0 / 20.0));

    const v_t = v_r - v_splat_one;
    const v_w2 = (((((@as(VecSF, @splat(1.0 / 24.0)) * v_t -
        @as(VecSF, @splat(1.0 / 6.0))) *
        v_t + @as(VecSF, @splat(1.0 / 6.0))) *
        v_t + @as(VecSF, @splat(1.0 / 6.0))) * v_t -
        @as(VecSF, @splat(5.0 / 12.0))) * v_t + @as(VecSF, @splat(13.0 / 60.0)));

    const v_u = v_r - v_splat_two;
    const v_w3 = (((((@as(VecSF, @splat(-1.0 / 120.0)) * v_u +
        @as(VecSF, @splat(1.0 / 24.0))) * v_u - @as(VecSF, @splat(1.0 / 12.0))) * v_u +
        @as(VecSF, @splat(1.0 / 12.0))) * v_u - @as(VecSF, @splat(1.0 / 24.0))) *
        v_u + @as(VecSF, @splat(1.0 / 120.0)));

    var samp_res = @select(F, v_mask_inner, v_w1, @as(VecSF, @splat(0.0)));
    samp_res = @select(F, v_mask_middle, v_w2, samp_res);
    samp_res = @select(F, v_mask_outer, v_w3, samp_res);
    return samp_res;
}
