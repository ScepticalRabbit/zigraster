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
const matslice = @import("matslice.zig");
const texops = @import("textureops.zig");
const comm = @import("shaderops_common.zig");

// --------------------------------------------------------------------------------------
// Nodal Interp Shader
// --------------------------------------------------------------------------------------

pub inline fn fillNodalClipScal(
    comptime N: usize,
    ctx_shade: comm.ShadeContext,
    interp: comm.InterpData(N),
    shader_buf: *const comm.LocalShaderBuff(N),
    shader: *const comm.NodalPrepared,
    spx_img_scratch: *matslice.MatSlice(F),
) void {
    for (0..@as(usize, ctx_shade.actual_fields)) |ff| {
        const value = shader_buf.interp(ff, interp.weights);
        const idx = ff * spx_img_scratch.cols_num + ctx_shade.scratch_idx;
        spx_img_scratch.slice[idx] = value * shader.scale_mul + shader.scale_add;
    }
}

pub inline fn fillNodalPerspScal(
    comptime N: usize,
    ctx_shade: comm.ShadeContext,
    interp: comm.InterpData(N),
    shader_buf: *const comm.LocalShaderBuff(N),
    shader: *const comm.NodalPrepared,
    spx_img_scratch: *matslice.MatSlice(F),
) void {
    for (0..@as(usize, ctx_shade.actual_fields)) |ff| {
        const base = ff * N;

        var value: F = 0.0;
        inline for (0..N) |nn| {
            const inv_z = interp.nodes_inv_z[nn];
            value += interp.weights[nn] * shader_buf.data[base + nn] * inv_z;
        }

        const final_val = value * interp.sub_pixel_z;
        const idx = ff * spx_img_scratch.cols_num + ctx_shade.scratch_idx;
        spx_img_scratch.slice[idx] = final_val * shader.scale_mul + shader.scale_add;
    }
}

// --------------------------------------------------------------------------------------
// Texture Shader
// --------------------------------------------------------------------------------------

pub inline fn fillTexClipScal(
    comptime N: usize,
    comptime T: type,
    comptime C: usize,
    comptime samp_cfg: texops.TexSampConfig,
    ctx_shade: comm.ShadeContext,
    interp: comm.InterpData(N),
    shader_buf: *const comm.LocalShaderBuff(N),
    shader: *const comm.TexPrepared(T, C),
    spx_img_scratch: *matslice.MatSlice(F),
) void {
    var tex_u: F = 0.0;
    var tex_v: F = 0.0;
    inline for (0..N) |nn| {
        tex_u += interp.weights[nn] * shader_buf.data[nn];
        tex_v += interp.weights[nn] * shader_buf.data[N + nn];
    }

    const sampled = texops.sampScal(
        C,
        samp_cfg,
        shader.tex,
        tex_u,
        tex_v,
    );

    inline for (0..C) |ch| {
        const idx = ch * spx_img_scratch.cols_num + ctx_shade.scratch_idx;
        spx_img_scratch.slice[idx] = sampled[ch] * shader.scale_mul + shader.scale_add;
    }
}

pub inline fn fillTexPerspScal(
    comptime N: usize,
    comptime T: type,
    comptime C: usize,
    comptime samp_cfg: texops.TexSampConfig,
    ctx_shade: comm.ShadeContext,
    interp: comm.InterpData(N),
    shader_buf: *const comm.LocalShaderBuff(N),
    shader: *const comm.TexPrepared(T, C),
    spx_img_scratch: *matslice.MatSlice(F),
) void {
    var tex_u: F = 0.0;
    var tex_v: F = 0.0;
    inline for (0..N) |nn| {
        const inv_z = interp.nodes_inv_z[nn];
        tex_u += interp.weights[nn] * shader_buf.data[nn] * inv_z;
        tex_v += interp.weights[nn] * shader_buf.data[N + nn] * inv_z;
    }

    const sampled = texops.sampScal(
        C,
        samp_cfg,
        shader.tex,
        tex_u * interp.sub_pixel_z,
        tex_v * interp.sub_pixel_z,
    );

    inline for (0..C) |ch| {
        const idx = ch * spx_img_scratch.cols_num + ctx_shade.scratch_idx;
        spx_img_scratch.slice[idx] = sampled[ch] * shader.scale_mul + shader.scale_add;
    }
}

// --------------------------------------------------------------------------------------
// Function Shader
// --------------------------------------------------------------------------------------

inline fn getFuncCoord(
    comptime N: usize,
    interp: comm.InterpData(N),
    shader_buf: *const comm.LocalShaderBuff(N),
    elem_normals: ?ndarray.MappedNDArray(F),
) comm.FuncCoord {
    if (elem_normals != null) {
        const normal = shader_buf.interpNormal(interp.weights);
        return .{
            .coord_0 = 0.0,
            .coord_1 = 0.0,
            .normal_x = normal[0],
            .normal_y = normal[1],
            .normal_z = normal[2],
        };
    }

    return .{
        .coord_0 = 0.0,
        .coord_1 = 0.0,
        .normal_x = 0.0,
        .normal_y = 0.0,
        .normal_z = 1.0,
    };
}

inline fn setCoordValues(
    coord: *comm.FuncCoord,
    coord_0: F,
    coord_1: F,
) void {
    coord.coord_0 = coord_0;
    coord.coord_1 = coord_1;
}

inline fn resolveFuncCoordsClip(
    comptime N: usize,
    interp: comm.InterpData(N),
    shader_buf: *const comm.LocalShaderBuff(N),
    shader: *const comm.FuncPrepared,
) struct { coord_0: F, coord_1: F } {
    return switch (shader.coord_mode) {
        .uv => .{
            .coord_0 = shader_buf.interpFuncCoord(0, interp.weights),
            .coord_1 = shader_buf.interpFuncCoord(1, interp.weights),
        },
        .para => .{
            .coord_0 = interp.xi,
            .coord_1 = interp.eta,
        },
        .world_reference, .world_deformed => .{
            .coord_0 = shader_buf.interpFuncCoord(0, interp.weights),
            .coord_1 = shader_buf.interpFuncCoord(1, interp.weights),
        },
    };
}

inline fn resolveFuncCoordsPersp(
    comptime N: usize,
    interp: comm.InterpData(N),
    shader_buf: *const comm.LocalShaderBuff(N),
    shader: *const comm.FuncPrepared,
) struct { coord_0: F, coord_1: F } {
    return switch (shader.coord_mode) {
        .uv, .world_reference, .world_deformed => blk: {
            var coord_0: F = 0.0;
            var coord_1: F = 0.0;

            inline for (0..N) |nn| {
                const inv_z = interp.nodes_inv_z[nn];
                coord_0 += interp.weights[nn] * shader_buf.func_coords[nn] * inv_z;
                coord_1 += interp.weights[nn] * shader_buf.func_coords[N + nn] * inv_z;
            }

            break :blk .{
                .coord_0 = coord_0 * interp.sub_pixel_z,
                .coord_1 = coord_1 * interp.sub_pixel_z,
            };
        },
        .para => .{
            .coord_0 = interp.xi,
            .coord_1 = interp.eta,
        },
    };
}

inline fn evalPreparedSpeckleMaskScal(
    uv: [2]F,
    mask: comm.SpeckleMask2D,
    params: comm.FuncShaderParams,
) F {
    return comm.evalSpeckleMask2D(uv, mask) * params.output_scale +
        params.output_offset;
}

pub inline fn evalFuncShaderGreyPreparedScal(
    shader: *const comm.FuncPrepared,
    coord: comm.FuncCoord,
) F {
    if (shader.builtin == .speckle) {
        const uv = [2]F{ coord.coord_0, coord.coord_1 };
        const params = shader.params;
        if (comptime buildconfig.speckle_classified_indexed) {
            if (shader.speckle_classified) |classified| {
                return comm.evalClassifiedIndexedSpeckle2D(uv, classified) *
                    params.output_scale + params.output_offset;
            }
        } else if (comptime buildconfig.speckle_direct_fixed) {
            if (shader.speckle_direct_fixed) |direct| {
                return comm.evalDirectFixedSpeckle2D(uv, direct) *
                    params.output_scale + params.output_offset;
            }
        } else switch (comptime buildconfig.speckle_evaluator) {
            .cell_hash => {},
            .classified_indexed, .direct_fixed => unreachable,
            .list_naive, .list_indexed => {
                if (shader.speckle_list) |speckles| {
                    return comm.evalSpeckleList2D(uv, speckles) *
                        params.output_scale + params.output_offset;
                }
            },
            .mask_1bit, .mask_u8 => {
                if (shader.speckle_mask) |mask| {
                    return evalPreparedSpeckleMaskScal(uv, mask, params);
                }
            },
        }
    }

    return comm.evalFuncShaderBuiltinGreyNorm(
        shader.builtin,
        coord,
        shader.params,
    );
}

pub inline fn fillFuncClipScal(
    comptime N: usize,
    comptime C: usize,
    ctx_shade: comm.ShadeContext,
    interp: comm.InterpData(N),
    shader_buf: *const comm.LocalShaderBuff(N),
    shader: *const comm.FuncPrepared,
    spx_img_scratch: *matslice.MatSlice(F),
) void {
    const coords = resolveFuncCoordsClip(N, interp, shader_buf, shader);
    var coord = getFuncCoord(N, interp, shader_buf, shader.elem_normals);
    setCoordValues(&coord, coords.coord_0, coords.coord_1);
    const params = shader.params;

    if (comptime C == 1) {
        const value = evalFuncShaderGreyPreparedScal(shader, coord);
        spx_img_scratch.slice[ctx_shade.scratch_idx] =
            value * shader.scale_mul + shader.scale_add;
    } else {
        const vals = comm.evalFuncShaderBuiltinRGBNorm(
            shader.builtin,
            coord,
            params,
        );

        inline for (0..C) |ch| {
            const idx = ch * spx_img_scratch.cols_num + ctx_shade.scratch_idx;
            spx_img_scratch.slice[idx] = vals[ch] * shader.scale_mul + shader.scale_add;
        }
    }
}

pub inline fn fillFuncPerspScal(
    comptime N: usize,
    comptime C: usize,
    ctx_shade: comm.ShadeContext,
    interp: comm.InterpData(N),
    shader_buf: *const comm.LocalShaderBuff(N),
    shader: *const comm.FuncPrepared,
    spx_img_scratch: *matslice.MatSlice(F),
) void {
    const coords = resolveFuncCoordsPersp(N, interp, shader_buf, shader);
    var coord = getFuncCoord(N, interp, shader_buf, shader.elem_normals);
    setCoordValues(&coord, coords.coord_0, coords.coord_1);
    const params = shader.params;

    if (comptime C == 1) {
        const value = evalFuncShaderGreyPreparedScal(shader, coord);
        spx_img_scratch.slice[ctx_shade.scratch_idx] =
            value * shader.scale_mul + shader.scale_add;
    } else {
        const vals = comm.evalFuncShaderBuiltinRGBNorm(
            shader.builtin,
            coord,
            params,
        );

        inline for (0..C) |ch| {
            const idx = ch * spx_img_scratch.cols_num + ctx_shade.scratch_idx;
            spx_img_scratch.slice[idx] = vals[ch] * shader.scale_mul + shader.scale_add;
        }
    }
}

test "direct fixed scalar handles active inactive nonfinite and scaling" {
    if (comptime !buildconfig.speckle_direct_fixed) return;

    const inactive: comm.DirectFixedSpeckleCell2D = 0;
    const cells = [_]comm.DirectFixedSpeckleCell2D{
        0x0000_8000_8000_0001,
        inactive,
        inactive,
        inactive,
        inactive,
        inactive,
    };
    const speckle_params: comm.Speckle2DParams = .{
        .cells_per_uv = .{ 2.0, 1.0 },
        .occupancy = 0.5,
        .radius_mean = 0.25,
        .foreground = 0.2,
        .background = 0.8,
    };
    const direct: comm.DirectFixedSpeckle2D = .{
        .params = speckle_params,
        .cells = &cells,
        .cell_origin = .{ 0, 0 },
        .cell_dims = .{ 3, 2 },
        .radius2 = 0.25 * 0.25,
    };
    const shader: comm.FuncPrepared = .{
        .elem_uvs = null,
        .speckle_direct_fixed = direct,
        .builtin = .speckle,
        .params = .{
            .output_scale = 1.75,
            .output_offset = -0.125,
            .settings = .{ .speckle = speckle_params },
        },
    };
    const foreground = speckle_params.foreground * shader.params.output_scale +
        shader.params.output_offset;
    const background = speckle_params.background * shader.params.output_scale +
        shader.params.output_offset;
    const coords = [_][2]F{
        .{ 0.25, 0.5 },
        .{ 0.1, 0.5 },
        .{ 0.75, 0.5 },
        .{ std.math.nan(F), 0.5 },
        .{ 0.25, std.math.inf(F) },
    };
    for (coords, 0..) |uv, index| {
        const actual = evalFuncShaderGreyPreparedScal(&shader, .{
            .coord_0 = uv[0],
            .coord_1 = uv[1],
            .normal_x = 0.0,
            .normal_y = 0.0,
            .normal_z = 0.0,
        });
        try std.testing.expectEqual(if (index == 0) foreground else background, actual);
    }
}

test "u8 speckle mask scalar sampling handles scaling and nonfinite UVs" {
    if (comptime buildconfig.speckle_evaluator != .mask_u8) return;

    const bits = [_]u8{ 0, 64, 128, 255 };
    const mask_params: comm.Speckle2DParams = .{
        .foreground = 0.8,
        .background = 0.2,
    };
    const mask: comm.SpeckleMask2D = .{
        .bits = &bits,
        .dims = .{ 2, 2 },
        .row_stride = 2,
        .uv_to_texel = .{ 1.0, 1.0 },
        .params = mask_params,
    };
    const params: comm.FuncShaderParams = .{
        .output_scale = 1.75,
        .output_offset = -0.125,
        .settings = .{ .speckle = mask_params },
    };

    const coverage: F = 64.0 / 255.0;
    const expected = (mask_params.background +
        coverage * (mask_params.foreground - mask_params.background)) *
        params.output_scale + params.output_offset;
    try std.testing.expectEqual(
        expected,
        evalPreparedSpeckleMaskScal(.{ 1.0, 0.0 }, mask, params),
    );
    const expected_background = mask_params.background * params.output_scale +
        params.output_offset;
    try std.testing.expectEqual(
        expected_background,
        evalPreparedSpeckleMaskScal(.{ std.math.nan(F), 0.0 }, mask, params),
    );
    try std.testing.expectEqual(
        expected_background,
        evalPreparedSpeckleMaskScal(.{ 0.0, std.math.inf(F) }, mask, params),
    );
}
