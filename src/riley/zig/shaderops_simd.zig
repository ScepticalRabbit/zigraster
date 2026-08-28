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
const cfg = @import("buildconfig.zig").config;
const F = buildconfig.F;
const S = buildconfig.SimdWidth;
const VecSB = buildconfig.VecSB;
const VecSI = buildconfig.VecSI;
const VecSF = buildconfig.VecSF;

const MatSlice = @import("matslice.zig").MatSlice;
const maths_simd = @import("maths_simd.zig");
const texops = @import("textureops.zig");
const TexSampConfig = texops.TexSampConfig;
const comm = @import("shaderops_common.zig");
const scal = @import("shaderops_scalar.zig");
const simdops = @import("simdops.zig");

// --------------------------------------------------------------------------------------
// Nodal Interp Shader
// --------------------------------------------------------------------------------------
pub const fillNodalClipScal = scal.fillNodalClipScal;
pub const fillNodalPerspScal = scal.fillNodalPerspScal;

pub inline fn fillNodalClipSIMD(
    comptime N: usize,
    ctx_shade: comm.ShadeContext,
    shader_buf: *const comm.LocalShaderBuff(N),
    v_weights: [N]VecSF,
    shader: *const comm.NodalPrepared,
    spx_image_scratch: *MatSlice(F),
) void {
    const v_splat_mul: VecSF = @splat(shader.scale_mul);
    const v_splat_add: VecSF = @splat(shader.scale_add);
    const px_stride = spx_image_scratch.cols_num;

    inline for (0..cfg.max_nodal_fields) |ff| {
        if (ff >= @as(usize, ctx_shade.actual_fields)) break;

        const base = ff * N;
        var v_weighted_sum: VecSF = @splat(0.0);
        inline for (0..N) |nn| {
            v_weighted_sum += v_weights[nn] *
                @as(VecSF, @splat(shader_buf.data[base + nn]));
        }

        const v_final = v_weighted_sum * v_splat_mul + v_splat_add;
        const flat_idx = ff * px_stride + ctx_shade.scratch_idx;
        simdops.storeMaskedVecSF(
            spx_image_scratch.slice,
            flat_idx,
            ctx_shade.v_mask_active.?,
            v_final,
        );
    }
}

pub inline fn fillNodalPerspSIMD(
    comptime N: usize,
    ctx_shade: comm.ShadeContext,
    shader_buf: *const comm.LocalShaderBuff(N),
    v_weights: [N]VecSF,
    v_nodes_inv_z: [N]VecSF,
    v_subpx_z: VecSF,
    shader: *const comm.NodalPrepared,
    spx_image_scratch: *MatSlice(F),
) void {
    const v_splat_mul: VecSF = @splat(shader.scale_mul);
    const v_splat_add: VecSF = @splat(shader.scale_add);
    const px_stride = spx_image_scratch.cols_num;

    inline for (0..cfg.max_nodal_fields) |ff| {
        if (ff >= @as(usize, ctx_shade.actual_fields)) break;

        const base = ff * N;
        var v_weighted_sum: VecSF = @splat(0.0);
        inline for (0..N) |nn| {
            v_weighted_sum += v_weights[nn] * v_nodes_inv_z[nn] *
                @as(VecSF, @splat(shader_buf.data[base + nn]));
        }

        const v_final = (v_weighted_sum * v_subpx_z) * v_splat_mul + v_splat_add;
        const flat_idx = ff * px_stride + ctx_shade.scratch_idx;

        simdops.storeMaskedVecSF(
            spx_image_scratch.slice,
            flat_idx,
            ctx_shade.v_mask_active.?,
            v_final,
        );
    }
}

// --------------------------------------------------------------------------------------
// Texture Shader
// --------------------------------------------------------------------------------------

pub const fillTexClipScal = scal.fillTexClipScal;
pub const fillTexPerspScal = scal.fillTexPerspScal;

fn texSimdInterpMode(
    comptime C: comptime_int,
    comptime samp_cfg: TexSampConfig,
) buildconfig.SimdTexInterpMode {
    return buildconfig.TexSIMDPolicy.resolve(
        C,
        samp_cfg.sample == .linear,
        samp_cfg.mode == .lut or samp_cfg.mode == .lut_lerp,
    );
}

pub inline fn fillTexClipSIMD(
    comptime N: usize,
    comptime T: type,
    comptime C: usize,
    comptime samp_cfg: TexSampConfig,
    ctx_shade: comm.ShadeContext,
    v_mask_active: VecSB,
    v_weights: [N]VecSF,
    shader_buf: *const comm.LocalShaderBuff(N),
    shader: *const comm.TexPrepared(T, C),
    spx_image_scratch: *MatSlice(F),
) void {
    var v_tex_u: VecSF = @splat(0.0);
    var v_tex_v: VecSF = @splat(0.0);
    inline for (0..N) |nn| {
        v_tex_u += v_weights[nn] * @as(VecSF, @splat(shader_buf.data[nn]));
        v_tex_v += v_weights[nn] * @as(VecSF, @splat(shader_buf.data[N + nn]));
    }

    const px_stride = spx_image_scratch.cols_num;
    const sampled_vecs = switch (comptime texSimdInterpMode(C, samp_cfg)) {
        .inner => texops.sampLanes(
            C,
            samp_cfg,
            v_mask_active,
            shader.tex,
            v_tex_u,
            v_tex_v,
        ),
        .over_pixels => texops.sampWide(
            C,
            samp_cfg,
            shader.tex,
            v_tex_u,
            v_tex_v,
        ),
    };

    inline for (0..C) |ch| {
        const v_final = sampled_vecs[ch] *
            @as(VecSF, @splat(shader.scale_mul)) +
            @as(VecSF, @splat(shader.scale_add));

        const flat_idx = ch * px_stride + ctx_shade.scratch_idx;

        simdops.storeMaskedVecSF(
            spx_image_scratch.slice,
            flat_idx,
            v_mask_active,
            v_final,
        );
    }
}

pub inline fn fillTexPerspSIMD(
    comptime N: usize,
    comptime T: type,
    comptime C: usize,
    comptime samp_cfg: TexSampConfig,
    ctx_shade: comm.ShadeContext,
    v_mask_active: VecSB,
    v_weights: [N]VecSF,
    v_nodes_inv_z: [N]VecSF,
    v_subpx_z: VecSF,
    shader_buf: *const comm.LocalShaderBuff(N),
    shader: *const comm.TexPrepared(T, C),
    spx_image_scratch: *MatSlice(F),
) void {
    const v_splat_mul: VecSF = @splat(shader.scale_mul);
    const v_splat_add: VecSF = @splat(shader.scale_add);
    const px_stride = spx_image_scratch.cols_num;

    var v_tex_u: VecSF = @splat(0.0);
    var v_tex_v: VecSF = @splat(0.0);
    inline for (0..N) |nn| {
        const v_inv_z = v_nodes_inv_z[nn];
        v_tex_u += v_weights[nn] *
            @as(VecSF, @splat(shader_buf.data[nn])) * v_inv_z;
        v_tex_v += v_weights[nn] *
            @as(VecSF, @splat(shader_buf.data[N + nn])) * v_inv_z;
    }

    v_tex_u *= v_subpx_z;
    v_tex_v *= v_subpx_z;

    const sampled_vecs = switch (comptime texSimdInterpMode(
        C,
        samp_cfg,
    )) {
        .inner => if (comptime N == 3)
            texops.sampLanesTri3(
                C,
                samp_cfg,
                v_mask_active,
                shader.tex,
                v_tex_u,
                v_tex_v,
            )
        else
            texops.sampLanes(
                C,
                samp_cfg,
                v_mask_active,
                shader.tex,
                v_tex_u,
                v_tex_v,
            ),
        .over_pixels => texops.sampWide(
            C,
            samp_cfg,
            shader.tex,
            v_tex_u,
            v_tex_v,
        ),
    };

    inline for (0..C) |ch| {
        const v_final = sampled_vecs[ch] * v_splat_mul + v_splat_add;
        const flat_idx = ch * px_stride + ctx_shade.scratch_idx;
        simdops.storeMaskedVecSF(
            spx_image_scratch.slice,
            flat_idx,
            v_mask_active,
            v_final,
        );
    }
}

// --------------------------------------------------------------------------------------
// Function Shader
// --------------------------------------------------------------------------------------

pub const fillFuncClipScal = scal.fillFuncClipScal;
pub const fillFuncPerspScal = scal.fillFuncPerspScal;

pub inline fn evalFuncShaderGreyNormSIMD(
    builtin: comm.FuncShaderBuiltin,
    coord: comm.FuncCoordSIMD,
    params: comm.FuncShaderParams,
) VecSF {
    const eval_coord = comm.applyFuncShaderCoordParamsSIMD(coord, params);
    const v_value = switch (builtin) {
        .constant => blk: {
            const p = params.settings.constant;
            break :blk @as(VecSF, @splat(p.value));
        },
        .linear => blk: {
            const p = params.settings.linear;
            break :blk @as(VecSF, @splat(p.coeffs[0])) +
                @as(VecSF, @splat(p.coeffs[1])) * eval_coord.coord_0 +
                @as(VecSF, @splat(p.coeffs[2])) * eval_coord.coord_1;
        },
        .quadratic => blk: {
            const p = params.settings.quadratic;
            const coord_u = eval_coord.coord_0;
            const coord_v = eval_coord.coord_1;
            const c = p.coeffs;
            const term_u = coord_u * (@as(VecSF, @splat(c[1])) +
                @as(VecSF, @splat(c[3])) * coord_u);
            const term_v = coord_v * (@as(VecSF, @splat(c[2])) +
                @as(VecSF, @splat(c[4])) * coord_u +
                @as(VecSF, @splat(c[5])) * coord_v);
            break :blk @as(VecSF, @splat(c[0])) + term_u + term_v;
        },
        .sinusoidal => blk: {
            const p = params.settings.sinusoidal;
            const v_bias: VecSF = @splat(p.bias);
            const v_amp_0: VecSF = @splat(p.amplitudes[0]);
            const v_amp_1: VecSF = @splat(p.amplitudes[1]);
            const v_wave_num_0: VecSF = @splat(p.wave_num_scalar[0]);
            const v_wave_num_1: VecSF = @splat(p.wave_num_scalar[1]);
            break :blk v_bias +
                v_amp_0 * @sin(v_wave_num_0 * eval_coord.coord_0) +
                v_amp_1 * @cos(v_wave_num_1 * eval_coord.coord_1);
        },
        .sinusoidal_approx => blk: {
            const p = params.settings.sinusoidal_approx;
            const v_bias: VecSF = @splat(p.bias);
            const v_amp_0: VecSF = @splat(p.amplitudes[0]);
            const v_amp_1: VecSF = @splat(p.amplitudes[1]);
            const v_wave_num_0: VecSF = @splat(p.wave_num_scalar[0]);
            const v_wave_num_1: VecSF = @splat(p.wave_num_scalar[1]);
            break :blk v_bias +
                v_amp_0 * maths_simd.sinApproxSIMD(
                    S,
                    F,
                    v_wave_num_0 * eval_coord.coord_0,
                ) +
                v_amp_1 * maths_simd.cosApproxSIMD(
                    S,
                    F,
                    v_wave_num_1 * eval_coord.coord_1,
                );
        },
        .checker => blk: {
            const p = params.settings.checker;
            const v_cell_x: VecSI = @intFromFloat(@floor(eval_coord.coord_0));
            const v_cell_y: VecSI = @intFromFloat(@floor(eval_coord.coord_1));
            const v_parity = @mod(
                v_cell_x + v_cell_y,
                @as(VecSI, @splat(2)),
            ) == @as(VecSI, @splat(0));
            break :blk @select(
                F,
                @as(VecSB, v_parity),
                @as(VecSF, @splat(p.levels[0])),
                @as(VecSF, @splat(p.levels[1])),
            );
        },
        .checker_smooth => blk: {
            const p = params.settings.checker_smooth;
            const v_half: VecSF = @splat(0.5);
            const v_freq_pi: VecSF = @splat(p.frequency * std.math.pi);
            const v_phase_x = v_half +
                v_half * @sin(v_freq_pi * eval_coord.coord_0);
            const v_phase_y = v_half +
                v_half * @sin(v_freq_pi * eval_coord.coord_1);
            break :blk comm.cubicSmoothStepSIMD(v_phase_x * v_phase_y);
        },
        .lambertian_normal_z => blk: {
            const p = params.settings.lambertian_normal_z;
            break :blk @as(VecSF, @splat(p.coeffs[0])) +
                @as(VecSF, @splat(p.coeffs[1])) * eval_coord.normal_z;
        },
        .eggbox => blk: {
            const p = params.settings.eggbox;
            const v_two_pi: VecSF = @splat(2.0 * std.math.pi);
            const v_phase_0: VecSF = @splat(p.phase[0]);
            const v_phase_1: VecSF = @splat(p.phase[1]);
            const v_pitch_0: VecSF = @splat(p.pitch[0]);
            const v_pitch_1: VecSF = @splat(p.pitch[1]);
            const v_mean: VecSF = @splat(p.mean);
            const v_half_contrast: VecSF = @splat(0.5 * p.contrast);
            const v_contrast: VecSF = @splat(p.contrast);
            const v_one: VecSF = @splat(1.0);
            const v_phase_x = v_two_pi * (eval_coord.coord_0 + v_phase_0) / v_pitch_0;
            const v_phase_y = v_two_pi * (eval_coord.coord_1 + v_phase_1) / v_pitch_1;
            break :blk v_mean + v_half_contrast * (v_one + @cos(v_phase_x)) *
                (v_one + @cos(v_phase_y)) - v_contrast;
        },
        .speckle => blk: {
            const coord_0: [S]F = coord.coord_0;
            const coord_1: [S]F = coord.coord_1;
            var values: [S]F = undefined;
            for (0..S) |lane| {
                values[lane] = if (std.math.isFinite(coord_0[lane]) and
                    std.math.isFinite(coord_1[lane]))
                    comm.evalSpeckle2D(
                        .{ coord_0[lane], coord_1[lane] },
                        params.settings.speckle,
                    )
                else
                    params.settings.speckle.background;
            }
            break :blk @as(VecSF, values);
        },
    };
    return comm.applyFuncShaderOutputParamsSIMD(v_value, params);
}

fn evalFuncShaderGreyPreparedSIMD(
    shader: *const comm.FuncPrepared,
    coord: comm.FuncCoordSIMD,
    v_mask_active: VecSB,
) VecSF {
    if (comptime buildconfig.speckle_evaluator == .mask_1bit) {
        if (shader.speckle_mask) |*mask| {
            const v_zero: VecSF = @splat(0.0);
            const v_one: VecSF = @splat(1.0);
            const v_half: VecSF = @splat(0.5);
            const v_inf: VecSF = @splat(std.math.inf(F));
            const v_sample = v_mask_active & (@abs(coord.coord_0) < v_inf) &
                (@abs(coord.coord_1) < v_inf);
            const v_u = @select(F, v_sample, coord.coord_0, v_zero);
            const v_v = @select(F, v_sample, coord.coord_1, v_zero);
            const v_texel_x: VecSI = @intFromFloat(
                @max(v_zero, @min(v_one, v_u)) *
                    @as(VecSF, @splat(mask.uv_to_texel[0])) + v_half,
            );
            const v_texel_y: VecSI = @intFromFloat(
                @max(v_zero, @min(v_one, v_v)) *
                    @as(VecSF, @splat(mask.uv_to_texel[1])) + v_half,
            );
            const sample: [S]bool = v_sample;
            const texel_x: [S]isize = v_texel_x;
            const texel_y: [S]isize = v_texel_y;

            var is_foreground = [_]bool{false} ** S;
            inline for (0..S) |lane| {
                if (sample[lane]) {
                    const x: usize = @intCast(texel_x[lane]);
                    const y: usize = @intCast(texel_y[lane]);
                    const shift: u3 = @intCast(x & 7);
                    is_foreground[lane] = mask.bits[y * mask.row_stride + x / 8] &
                        (@as(u8, 1) << shift) != 0;
                }
            }
            const v_value = @select(
                F,
                v_sample & @as(VecSB, is_foreground),
                @as(VecSF, @splat(mask.params.foreground)),
                @as(VecSF, @splat(mask.params.background)),
            );
            return comm.applyFuncShaderOutputParamsSIMD(v_value, shader.params);
        }
    }

    const speckles = shader.speckle_list orelse
        return evalFuncShaderGreyNormSIMD(shader.builtin, coord, shader.params);
    const coord_0: [S]F = coord.coord_0;
    const coord_1: [S]F = coord.coord_1;
    var values: [S]F = undefined;
    for (0..S) |lane| {
        values[lane] = if (std.math.isFinite(coord_0[lane]) and
            std.math.isFinite(coord_1[lane]))
            comm.evalSpeckleList2D(.{ coord_0[lane], coord_1[lane] }, speckles)
        else
            speckles.params.background;
    }
    return comm.applyFuncShaderOutputParamsSIMD(values, shader.params);
}

pub inline fn evalFuncShaderRGBNormSIMD(
    builtin: comm.FuncShaderBuiltin,
    coord: comm.FuncCoordSIMD,
    params: comm.FuncShaderParams,
) [3]VecSF {
    const eval_coord = comm.applyFuncShaderCoordParamsSIMD(coord, params);

    const v_vals = switch (builtin) {
        .constant => blk: {
            const p = params.settings.constant;
            break :blk .{
                @as(VecSF, @splat(p.value_rgb[0])),
                @as(VecSF, @splat(p.value_rgb[1])),
                @as(VecSF, @splat(p.value_rgb[2])),
            };
        },
        .linear => blk: {
            const p = params.settings.linear;
            const c = p.coeffs_rgb;
            break :blk .{
                @as(VecSF, @splat(c[0][0])) +
                    @as(VecSF, @splat(c[0][1])) * eval_coord.coord_0 +
                    @as(VecSF, @splat(c[0][2])) * eval_coord.coord_1,
                @as(VecSF, @splat(c[1][0])) +
                    @as(VecSF, @splat(c[1][1])) * eval_coord.coord_0 +
                    @as(VecSF, @splat(c[1][2])) * eval_coord.coord_1,
                @as(VecSF, @splat(c[2][0])) +
                    @as(VecSF, @splat(c[2][1])) * eval_coord.coord_0 +
                    @as(VecSF, @splat(c[2][2])) * eval_coord.coord_1,
            };
        },
        .quadratic => blk: {
            const p = params.settings.quadratic;
            const coord_u = eval_coord.coord_0;
            const coord_v = eval_coord.coord_1;
            const c = p.coeffs_rgb;

            const val_r = @as(VecSF, @splat(c[0][0])) +
                coord_u * (@as(VecSF, @splat(c[0][1])) +
                    @as(VecSF, @splat(c[0][3])) * coord_u) +
                coord_v * (@as(VecSF, @splat(c[0][2])) +
                    @as(VecSF, @splat(c[0][4])) * coord_u +
                    @as(VecSF, @splat(c[0][5])) * coord_v);
            const val_g = @as(VecSF, @splat(c[1][0])) +
                coord_u * (@as(VecSF, @splat(c[1][1])) +
                    @as(VecSF, @splat(c[1][3])) * coord_u) +
                coord_v * (@as(VecSF, @splat(c[1][2])) +
                    @as(VecSF, @splat(c[1][4])) * coord_u +
                    @as(VecSF, @splat(c[1][5])) * coord_v);
            const val_b = @as(VecSF, @splat(c[2][0])) +
                coord_u * (@as(VecSF, @splat(c[2][1])) +
                    @as(VecSF, @splat(c[2][3])) * coord_u) +
                coord_v * (@as(VecSF, @splat(c[2][2])) +
                    @as(VecSF, @splat(c[2][4])) * coord_u +
                    @as(VecSF, @splat(c[2][5])) * coord_v);

            break :blk .{ val_r, val_g, val_b };
        },
        .sinusoidal => blk: {
            const p = params.settings.sinusoidal;
            const v_bias_0: VecSF = @splat(p.bias_rgb[0]);
            const v_bias_1: VecSF = @splat(p.bias_rgb[1]);
            const v_bias_2: VecSF = @splat(p.bias_rgb[2]);
            const v_amp_0: VecSF = @splat(p.amplitudes_rgb[0]);
            const v_amp_1: VecSF = @splat(p.amplitudes_rgb[1]);
            const v_amp_2: VecSF = @splat(p.amplitudes_rgb[2]);
            const v_wave_num_0: VecSF = @splat(p.wave_num_rgb[0]);
            const v_wave_num_1: VecSF = @splat(p.wave_num_rgb[1]);
            const v_wave_num_2: VecSF = @splat(p.wave_num_rgb[2]);
            const v_coord_sum = eval_coord.coord_0 + eval_coord.coord_1;

            break :blk .{
                v_bias_0 + v_amp_0 * @sin(v_wave_num_0 * eval_coord.coord_0),
                v_bias_1 + v_amp_1 * @cos(v_wave_num_1 * eval_coord.coord_1),
                v_bias_2 + v_amp_2 * @sin(v_wave_num_2 * v_coord_sum),
            };
        },
        .sinusoidal_approx => blk: {
            const p = params.settings.sinusoidal_approx;
            const v_bias_0: VecSF = @splat(p.bias_rgb[0]);
            const v_bias_1: VecSF = @splat(p.bias_rgb[1]);
            const v_bias_2: VecSF = @splat(p.bias_rgb[2]);

            const v_amp_0: VecSF = @splat(p.amplitudes_rgb[0]);
            const v_amp_1: VecSF = @splat(p.amplitudes_rgb[1]);
            const v_amp_2: VecSF = @splat(p.amplitudes_rgb[2]);

            const v_wave_num_0: VecSF = @splat(p.wave_num_rgb[0]);
            const v_wave_num_1: VecSF = @splat(p.wave_num_rgb[1]);
            const v_wave_num_2: VecSF = @splat(p.wave_num_rgb[2]);

            const v_coord_sum = eval_coord.coord_0 + eval_coord.coord_1;

            const v_wc0 = v_wave_num_0 * eval_coord.coord_0;
            const v_wc1 = v_wave_num_1 * eval_coord.coord_1;
            const v_wc2 = v_wave_num_2 * v_coord_sum;

            break :blk .{
                v_bias_0 + v_amp_0 * maths_simd.sinApproxSIMD(S, F, v_wc0),
                v_bias_1 + v_amp_1 * maths_simd.cosApproxSIMD(S, F, v_wc1),
                v_bias_2 + v_amp_2 * maths_simd.sinApproxSIMD(S, F, v_wc2),
            };
        },
        .checker => blk: {
            const p = params.settings.checker;
            const v_cell_x: VecSI = @intFromFloat(@floor(eval_coord.coord_0));
            const v_cell_y: VecSI = @intFromFloat(@floor(eval_coord.coord_1));

            const v_0: VecSI = @splat(0);
            const v_2: VecSI = @splat(2);

            const v_parity = @as(VecSB, @mod(v_cell_x + v_cell_y, v_2) == v_0);
            const v_p0 = @as(VecSF, @splat(p.levels[0]));
            const v_p1 = @as(VecSF, @splat(p.levels[1]));

            const v_value = @select(F, v_parity, v_p0, v_p1);
            break :blk .{ v_value, v_value, v_value };
        },
        .checker_smooth => blk: {
            const p = params.settings.checker_smooth;

            const v_half: VecSF = @splat(0.5);
            const v_one: VecSF = @splat(1.0);
            const v_two_pi: VecSF = @splat(2.0 * std.math.pi);
            const v_freq_pi: VecSF = @splat(p.frequency * std.math.pi);

            const v_phase_x = v_half + v_half * @sin(v_freq_pi * eval_coord.coord_0);
            const v_phase_y = v_half + v_half * @sin(v_freq_pi * eval_coord.coord_1);
            const v_base = comm.cubicSmoothStepSIMD(v_phase_x * v_phase_y);

            break :blk .{
                v_base,
                comm.cubicSmoothStepSIMD(v_one - v_base),
                v_half + v_half * @sin(v_two_pi * v_base),
            };
        },
        .lambertian_normal_z => blk: {
            const p = params.settings.lambertian_normal_z;
            const v_c00: VecSF = @as(VecSF, @splat(p.coeffs_rgb[0][0]));
            const v_c01: VecSF = @as(VecSF, @splat(p.coeffs_rgb[0][1]));
            const v_c10: VecSF = @as(VecSF, @splat(p.coeffs_rgb[1][0]));
            const v_c11: VecSF = @as(VecSF, @splat(p.coeffs_rgb[1][1]));
            const v_c20: VecSF = @as(VecSF, @splat(p.coeffs_rgb[2][0]));
            const v_c21: VecSF = @as(VecSF, @splat(p.coeffs_rgb[2][1]));

            break :blk .{
                v_c00 + v_c01 * eval_coord.normal_z,
                v_c10 + v_c11 * eval_coord.normal_z,
                v_c20 + v_c21 * eval_coord.normal_z,
            };
        },
        .eggbox => blk: {
            const p = params.settings.eggbox;
            const v_two_pi: VecSF = @splat(2.0 * std.math.pi);

            const v_phase_0: VecSF = @splat(p.phase[0]);
            const v_phase_1: VecSF = @splat(p.phase[1]);
            const v_pitch_0: VecSF = @splat(p.pitch[0]);
            const v_pitch_1: VecSF = @splat(p.pitch[1]);

            const v_mean: VecSF = @splat(p.mean);
            const v_half_contrast: VecSF = @splat(0.5 * p.contrast);
            const v_contrast: VecSF = @splat(p.contrast);
            const v_one: VecSF = @splat(1.0);

            const v_phase_x = v_two_pi * (eval_coord.coord_0 + v_phase_0) / v_pitch_0;
            const v_phase_y = v_two_pi * (eval_coord.coord_1 + v_phase_1) / v_pitch_1;

            const v_value = v_mean + v_half_contrast * (v_one + @cos(v_phase_x)) *
                (v_one + @cos(v_phase_y)) - v_contrast;

            break :blk .{ v_value, v_value, v_value };
        },
        .speckle => unreachable,
    };
    return .{
        comm.applyFuncShaderOutputParamsSIMD(v_vals[0], params),
        comm.applyFuncShaderOutputParamsSIMD(v_vals[1], params),
        comm.applyFuncShaderOutputParamsSIMD(v_vals[2], params),
    };
}

fn calcNormalLaneVecs(
    comptime N: usize,
    has_normals: bool,
    shader_buf: *const comm.LocalShaderBuff(N),
    v_weights: [N]VecSF,
) [3]VecSF {
    var normal_vecs = [3]VecSF{ @splat(0.0), @splat(0.0), @splat(0.0) };

    if (!has_normals) {
        normal_vecs[2] = @splat(1.0);
        return normal_vecs;
    }

    inline for (0..N) |nn| {
        const v_norm0 = @as(VecSF, @splat(shader_buf.normals[0 * N + nn]));
        const v_norm1 = @as(VecSF, @splat(shader_buf.normals[1 * N + nn]));
        const v_norm2 = @as(VecSF, @splat(shader_buf.normals[2 * N + nn]));

        normal_vecs[0] += v_weights[nn] * v_norm0;
        normal_vecs[1] += v_weights[nn] * v_norm1;
        normal_vecs[2] += v_weights[nn] * v_norm2;
    }

    return normal_vecs;
}

pub inline fn fillFuncClipSIMD(
    comptime N: usize,
    comptime C: usize,
    ctx_shade: comm.ShadeContext,
    v_mask_active: VecSB,
    v_weights: [N]VecSF,
    v_xi: VecSF,
    v_eta: VecSF,
    shader_buf: *const comm.LocalShaderBuff(N),
    shader: *const comm.FuncPrepared,
    spx_image_scratch: *MatSlice(F),
) void {
    var v_coord_0: VecSF = v_xi;
    var v_coord_1: VecSF = v_eta;

    switch (shader.coord_mode) {
        .uv, .world_reference, .world_deformed => {
            v_coord_0 = @splat(0.0);
            v_coord_1 = @splat(0.0);

            inline for (0..N) |nn| {
                const v_fc0 = @as(VecSF, @splat(shader_buf.func_coords[nn]));
                const v_fc1 = @as(VecSF, @splat(shader_buf.func_coords[N + nn]));
                v_coord_0 += v_weights[nn] * v_fc0;
                v_coord_1 += v_weights[nn] * v_fc1;
            }
        },
        .para => {},
    }

    const normal_vecs = if (shader.builtin == .speckle)
        [3]VecSF{ @splat(0.0), @splat(0.0), @splat(1.0) }
    else
        calcNormalLaneVecs(
            N,
            shader.elem_normals != null,
            shader_buf,
            v_weights,
        );

    const px_stride = spx_image_scratch.cols_num;
    const scratch_idx = ctx_shade.scratch_idx;

    const coord = comm.FuncCoordSIMD{
        .coord_0 = v_coord_0,
        .coord_1 = v_coord_1,
        .normal_x = normal_vecs[0],
        .normal_y = normal_vecs[1],
        .normal_z = normal_vecs[2],
    };
    const params = shader.params;

    if (comptime C == 1) {
        const v_eval = evalFuncShaderGreyPreparedSIMD(shader, coord, v_mask_active);
        const v_mul = @as(VecSF, @splat(shader.scale_mul));
        const v_add = @as(VecSF, @splat(shader.scale_add));
        const v_final = v_eval * v_mul + v_add;

        const flat_idx = scratch_idx;

        simdops.storeMaskedVecSF(
            spx_image_scratch.slice,
            flat_idx,
            v_mask_active,
            v_final,
        );
        return;
    }

    const v_vals = evalFuncShaderRGBNormSIMD(
        shader.builtin,
        coord,
        params,
    );

    inline for (0..C) |ch| {
        const v_mul = @as(VecSF, @splat(shader.scale_mul));
        const v_add = @as(VecSF, @splat(shader.scale_add));
        const v_final = v_vals[ch] * v_mul + v_add;
        const flat_idx = ch * px_stride + scratch_idx;

        simdops.storeMaskedVecSF(
            spx_image_scratch.slice,
            flat_idx,
            v_mask_active,
            v_final,
        );
    }
}

pub inline fn fillFuncPerspSIMD(
    comptime N: usize,
    comptime C: usize,
    ctx_shade: comm.ShadeContext,
    v_mask_active: VecSB,
    v_weights: [N]VecSF,
    v_xi: VecSF,
    v_eta: VecSF,
    v_nodes_inv_z: [N]VecSF,
    v_subpx_z: VecSF,
    shader_buf: *const comm.LocalShaderBuff(N),
    shader: *const comm.FuncPrepared,
    spx_image_scratch: *MatSlice(F),
) void {
    var v_coord_0: VecSF = v_xi;
    var v_coord_1: VecSF = v_eta;

    switch (shader.coord_mode) {
        .uv, .world_reference, .world_deformed => {
            v_coord_0 = @splat(0.0);
            v_coord_1 = @splat(0.0);

            inline for (0..N) |nn| {
                const v_inv_z = v_nodes_inv_z[nn];
                const v_fc0 = @as(VecSF, @splat(shader_buf.func_coords[nn]));
                const v_fc1 = @as(VecSF, @splat(shader_buf.func_coords[N + nn]));

                v_coord_0 += v_weights[nn] * v_fc0 * v_inv_z;
                v_coord_1 += v_weights[nn] * v_fc1 * v_inv_z;
            }

            v_coord_0 *= v_subpx_z;
            v_coord_1 *= v_subpx_z;
        },
        .para => {},
    }

    const normal_vecs = if (shader.builtin == .speckle)
        [3]VecSF{ @splat(0.0), @splat(0.0), @splat(1.0) }
    else
        calcNormalLaneVecs(
            N,
            shader.elem_normals != null,
            shader_buf,
            v_weights,
        );

    const px_stride = spx_image_scratch.cols_num;
    const scratch_idx = ctx_shade.scratch_idx;
    const coord = comm.FuncCoordSIMD{
        .coord_0 = v_coord_0,
        .coord_1 = v_coord_1,
        .normal_x = normal_vecs[0],
        .normal_y = normal_vecs[1],
        .normal_z = normal_vecs[2],
    };
    const params = shader.params;

    if (comptime C == 1) {
        const v_eval = evalFuncShaderGreyPreparedSIMD(shader, coord, v_mask_active);
        const v_mul = @as(VecSF, @splat(shader.scale_mul));
        const v_add = @as(VecSF, @splat(shader.scale_add));
        const v_final = v_eval * v_mul + v_add;

        const flat_idx = scratch_idx;
        simdops.storeMaskedVecSF(
            spx_image_scratch.slice,
            flat_idx,
            v_mask_active,
            v_final,
        );
        return;
    }

    const v_vals = evalFuncShaderRGBNormSIMD(
        shader.builtin,
        coord,
        params,
    );
    inline for (0..C) |ch| {
        const v_mul = @as(VecSF, @splat(shader.scale_mul));
        const v_add = @as(VecSF, @splat(shader.scale_add));
        const v_final = v_vals[ch] * v_mul + v_add;
        const flat_idx = ch * px_stride + scratch_idx;
        simdops.storeMaskedVecSF(
            spx_image_scratch.slice,
            flat_idx,
            v_mask_active,
            v_final,
        );
    }
}

test "1-bit speckle mask SIMD matches scalar and backgrounds invalid lanes" {
    if (comptime buildconfig.speckle_evaluator != .mask_1bit) return;

    const bits = [_]u8{ 0b10000011, 0b00000001, 0b01010101, 0b00000000 };
    const mask_params: comm.Speckle2DParams = .{
        .foreground = 0.75,
        .background = 0.25,
    };
    const mask: comm.SpeckleMask2D = .{
        .bits = &bits,
        .dims = .{ 9, 2 },
        .row_stride = 2,
        .uv_to_texel = .{ 8.0, 1.0 },
        .params = mask_params,
    };
    const shader: comm.FuncPrepared = .{
        .elem_uvs = null,
        .speckle_mask = mask,
        .builtin = .speckle,
        .params = .{
            .output_scale = 1.5,
            .output_offset = -0.125,
            .settings = .{ .speckle = mask_params },
        },
    };
    const u_cases = [_]F{ -2.0, 0.0, 0.0624, 0.0625, 0.99, 1.0, 3.0, 0.5 };
    var coord_0: [S]F = undefined;
    var coord_1: [S]F = undefined;
    for (0..S) |lane| {
        coord_0[lane] = u_cases[lane % u_cases.len];
        coord_1[lane] = @as(F, @floatFromInt(lane & 1));
    }
    var coord: comm.FuncCoordSIMD = .{
        .coord_0 = coord_0,
        .coord_1 = coord_1,
        .normal_x = @splat(0.0),
        .normal_y = @splat(0.0),
        .normal_z = @splat(0.0),
    };

    const finite: [S]F = evalFuncShaderGreyPreparedSIMD(&shader, coord, @splat(true));
    for (0..S) |lane| {
        const expected = comm.evalSpeckleMask2D(.{ coord_0[lane], coord_1[lane] }, mask) *
            shader.params.output_scale + shader.params.output_offset;
        try std.testing.expectEqual(expected, finite[lane]);
    }

    const expected_background = mask.params.background * shader.params.output_scale +
        shader.params.output_offset;
    for (0..S) |lane| {
        coord_0[lane] = if (lane & 1 == 0) std.math.nan(F) else 0.0;
        coord_1[lane] = if (lane & 1 == 0) 0.0 else std.math.inf(F);
    }
    coord.coord_0 = coord_0;
    coord.coord_1 = coord_1;
    const nonfinite: [S]F = evalFuncShaderGreyPreparedSIMD(&shader, coord, @splat(true));
    const inactive: [S]F = evalFuncShaderGreyPreparedSIMD(&shader, .{
        .coord_0 = @splat(0.0),
        .coord_1 = @splat(0.0),
        .normal_x = @splat(0.0),
        .normal_y = @splat(0.0),
        .normal_z = @splat(0.0),
    }, @splat(false));
    for (nonfinite, inactive) |nonfinite_value, inactive_value| {
        try std.testing.expectEqual(expected_background, nonfinite_value);
        try std.testing.expectEqual(expected_background, inactive_value);
    }
}
