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
const VecSF = buildconfig.VecSF;

const ndarray = @import("ndarray.zig");
const matslice = @import("matslice.zig");

const imageops = @import("imageops.zig");
const texops = @import("textureops.zig");
const meshio = @import("meshio.zig");
const maths_simd = @import("maths_simd.zig");
const simd_impl = @import("shaderops_simd.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const ScaleOver = enum { within_frames, over_frames };
pub const NormalType = enum { none, exact, avg };

pub fn LocalShaderBuff(comptime N: usize) type {
    return struct {
        data: [cfg.max_nodal_fields * N]F = undefined,
        func_coords: [3 * N]F = undefined,
        normals: [3 * N]F = undefined,
        actual_fields: u8 = 0,
        actual_func_coords: u8 = 0,

        const Self = @This();

        pub inline fn load(
            self: *Self,
            array: ndarray.NDArray(F),
            start_idx: usize,
            fields_num: u8,
        ) void {
            std.debug.assert(fields_num <= cfg.max_nodal_fields);
            self.actual_fields = fields_num;
            const count = @as(usize, fields_num) * N;
            @memcpy(self.data[0..count], array.slice[start_idx .. start_idx + count]);
        }

        pub inline fn loadNormals(
            self: *Self,
            array: ndarray.NDArray(F),
            start_idx: usize,
        ) void {
            const count = 3 * N;
            @memcpy(self.normals[0..count], array.slice[start_idx .. start_idx + count]);
        }

        pub inline fn loadFuncCoords(
            self: *Self,
            array: ndarray.NDArray(F),
            start_idx: usize,
            coords_num: u8,
        ) void {
            std.debug.assert(coords_num <= 3);
            self.actual_func_coords = coords_num;
            const count = @as(usize, coords_num) * N;
            @memcpy(
                self.func_coords[0..count],
                array.slice[start_idx .. start_idx + count],
            );
        }

        pub inline fn interp(
            self: *const Self,
            field_idx: usize,
            weights: [N]F,
        ) F {
            const base = field_idx * N;
            var sum: F = 0.0;
            inline for (0..N) |nn| {
                sum += weights[nn] * self.data[base + nn];
            }
            return sum;
        }

        pub inline fn interpNormal(
            self: *const Self,
            weights: [N]F,
        ) [3]F {
            var norm = [3]F{ 0.0, 0.0, 0.0 };
            inline for (0..N) |nn| {
                norm[0] += weights[nn] * self.normals[0 * N + nn];
                norm[1] += weights[nn] * self.normals[1 * N + nn];
                norm[2] += weights[nn] * self.normals[2 * N + nn];
            }
            return norm;
        }

        pub inline fn interpFuncCoord(
            self: *const Self,
            coord_idx: usize,
            weights: [N]F,
        ) F {
            const base = coord_idx * N;
            var sum: F = 0.0;
            inline for (0..N) |nn| {
                sum += weights[nn] * self.func_coords[base + nn];
            }
            return sum;
        }
    };
}

// Input: Raw user shader data for all frames.
// Nodal Fields: Node-order [num_frames, total_nodes, num_fields]
// UVs: Node-order [total_nodes, 2]
pub const NodalInput = struct {
    field: meshio.Field,
    bits: ?u8 = 8,
    scaling: imageops.ScaleStrategy = .none,
    scale_over: ScaleOver = .over_frames,
    normal_type: NormalType = .none,
};

pub fn TexInput(comptime T: type, comptime C: usize) type {
    return struct {
        uvs: ndarray.NDArray(F),
        tex: texops.Tex(T, C),
        samp_cfg: texops.TexSampConfig = .{
            .sample = .cubic_catmull_rom,
            .mode = .lut_lerp,
        },
        bits: ?u8 = 8,
        scaling: imageops.ScaleStrategy = .none,
        normal_type: NormalType = .none,
    };
}

pub const FuncInput = struct {
    uvs: ?ndarray.NDArray(F) = null,
    coord_mode: FuncCoordMode = .para,
    builtin: FuncShaderBuiltin,
    params: FuncShaderParams = .{},
    bits: ?u8 = 8,
    scaling: imageops.ScaleStrategy = .none,
    normal_type: NormalType = .none,
};

pub const ShaderInput = union(enum) {
    nodal: NodalInput,
    tex_u8: TexInput(u8, 1),
    tex_u16: TexInput(u16, 1),
    tex_f: TexInput(F, 1),
    tex_rgb_u8: TexInput(u8, 3),
    tex_rgb_u16: TexInput(u16, 3),
    tex_rgb_f: TexInput(F, 3),
    func: FuncInput,
    func_rgb: FuncInput,
};

pub const FuncCoordMode = enum {
    uv,
    para,
    world_reference,
    world_deformed,
};

pub const FuncShaderBuiltin = enum {
    constant,
    linear,
    quadratic,
    sinusoidal,
    sinusoidal_approx,
    checker,
    checker_smooth,
    lambertian_normal_z,
    eggbox,
};

pub const ConstantParams = struct {
    value: F = 0.5,
    value_rgb: [3]F = .{ 0.2, 0.5, 0.8 },
};

pub const LinearParams = struct {
    coeffs: [3]F = .{ 0.5, 0.25, 0.2 },
    coeffs_rgb: [3][3]F = .{
        .{ 0.5, 0.25, 0.0 },
        .{ 0.5, 0.0, 0.25 },
        .{ 0.5, 0.15, -0.15 },
    },
};

pub const QuadraticParams = struct {
    coeffs: [6]F = .{ 0.35, 0.2, 0.15, 0.1, -0.08, 0.06 },
    coeffs_rgb: [3][6]F = .{
        .{ 0.3, 0.0, 0.0, 0.2, 0.0, 0.0 },
        .{ 0.3, 0.0, 0.0, 0.0, 0.0, 0.2 },
        .{ 0.3, 0.0, 0.0, 0.0, 0.12, 0.0 },
    },
};

pub const SinusoidalParams = struct {
    wave_num_scalar: [2]F = .{ 6.0, 5.0 },
    wave_num_rgb: [3]F = .{ 6.0, 6.0, 4.0 },
    bias: F = 0.5,
    amplitudes: [2]F = .{ 0.25, 0.2 },
    bias_rgb: [3]F = .{ 0.5, 0.5, 0.5 },
    amplitudes_rgb: [3]F = .{ 0.25, 0.25, 0.2 },
};

pub const CheckerParams = struct {
    levels: [2]F = .{ 0.0, 1.0 },
};

pub const CheckerSmoothParams = struct {
    frequency: F = 8.0,
};

pub const LambertianParams = struct {
    coeffs: [2]F = .{ 0.5, 0.5 },
    coeffs_rgb: [3][2]F = .{
        .{ 0.5, 0.5 },
        .{ 0.375, 0.375 },
        .{ 0.25, 0.25 },
    },
};

pub const EggboxParams = struct {
    mean: F = 0.5,
    contrast: F = 0.4,
    pitch: [2]F = .{ 1.0, 1.0 },
    phase: [2]F = .{ 0.0, 0.0 },
};

pub const FuncShaderParams = struct {
    coord_scale: [2]F = .{ 1.0, 1.0 },
    coord_offset: [2]F = .{ 0.0, 0.0 },
    output_scale: F = 1.0,
    output_offset: F = 0.0,
    settings: union(FuncShaderBuiltin) {
        constant: ConstantParams,
        linear: LinearParams,
        quadratic: QuadraticParams,
        sinusoidal: SinusoidalParams,
        sinusoidal_approx: SinusoidalParams,
        checker: CheckerParams,
        checker_smooth: CheckerSmoothParams,
        lambertian_normal_z: LambertianParams,
        eggbox: EggboxParams,
    } = .{ .constant = .{} },
};

pub const FuncCoordSIMD = struct {
    coord_0: VecSF,
    coord_1: VecSF,
    normal_x: VecSF,
    normal_y: VecSF,
    normal_z: VecSF,
};

// Static: Persistent multi-frame shader resources in engine memory.
// Nodal Fields: Node-order [num_frames, total_nodes, num_fields]
// UVs: Elem-order [total_elems, 2, nodes_per_elem]
pub const NodalStatic = struct {
    field: meshio.Field,
    bits: ?u8 = 8,
    scaling: imageops.ScaleStrategy = .none,
    scale_over: ScaleOver = .over_frames,
    normal_type: NormalType = .none,
};

pub fn TexStatic(comptime T: type, comptime C: usize) type {
    return struct {
        elem_uvs: ndarray.NDArray(F),
        tex: texops.Tex(T, C),
        samp_cfg: texops.TexSampConfig = .{
            .sample = .cubic_catmull_rom,
            .mode = .lut_lerp,
        },
        bits: ?u8 = 8,
        scaling: imageops.ScaleStrategy = .none,
        normal_type: NormalType = .none,
    };
}

pub const FuncStatic = struct {
    elem_uvs: ?ndarray.NDArray(F),
    coord_mode: FuncCoordMode = .para,
    builtin: FuncShaderBuiltin,
    params: FuncShaderParams = .{},
    bits: ?u8 = 8,
    scaling: imageops.ScaleStrategy = .none,
    normal_type: NormalType = .none,
};

pub const ShaderStatic = union(enum) {
    nodal: NodalStatic,
    tex_u8: TexStatic(u8, 1),
    tex_u16: TexStatic(u16, 1),
    tex_f: TexStatic(F, 1),
    tex_rgb_u8: TexStatic(u8, 3),
    tex_rgb_u16: TexStatic(u16, 3),
    tex_rgb_f: TexStatic(F, 3),
    func: FuncStatic,
    func_rgb: FuncStatic,
};

// Prep: Culled and expanded shader data for a SINGLE frame.
// Prep means culled elem-order ndarray.NDArray data ready for the raster loop.
// Nodal Fields: Elem-order [vis_elems, num_fields, nodes_per_elem]
// UVs: Elem-order [vis_elems, 2, nodes_per_elem]
pub const NodalPrepared = struct {
    elem_field: ndarray.NDArray(F),
    bits: ?u8 = 8,
    scaling: imageops.ScaleStrategy = .none,
    scale_over: ScaleOver = .over_frames,
    scale_mul: F = 1.0,
    scale_add: F = 0.0,
    normal_type: NormalType = .none,
    elem_normals: ?ndarray.MappedNDArray(F) = null,
};

pub fn TexPrepared(comptime T: type, comptime C: usize) type {
    return struct {
        elem_uvs: ndarray.NDArray(F),
        tex: texops.Tex(T, C),
        samp_cfg: texops.TexSampConfig = .{
            .sample = .cubic_catmull_rom,
            .mode = .lut_lerp,
        },
        bits: ?u8 = 8,
        scaling: imageops.ScaleStrategy = .none,
        scale_mul: F = 1.0,
        scale_add: F = 0.0,
        normal_type: NormalType = .none,
        elem_normals: ?ndarray.MappedNDArray(F) = null,
    };
}

pub const FuncPrepared = struct {
    elem_uvs: ?ndarray.NDArray(F),
    elem_world_ref: ?ndarray.NDArray(F) = null,
    elem_world_def: ?ndarray.NDArray(F) = null,
    coord_mode: FuncCoordMode = .para,
    builtin: FuncShaderBuiltin,
    params: FuncShaderParams = .{},
    bits: ?u8 = 8,
    scaling: imageops.ScaleStrategy = .none,
    scale_mul: F = 1.0,
    scale_add: F = 0.0,
    normal_type: NormalType = .none,
    elem_normals: ?ndarray.MappedNDArray(F) = null,
};

pub const ShaderPrepared = union(enum) {
    nodal: NodalPrepared,
    tex_u8: TexPrepared(u8, 1),
    tex_u16: TexPrepared(u16, 1),
    tex_f: TexPrepared(F, 1),
    tex_rgb_u8: TexPrepared(u8, 3),
    tex_rgb_u16: TexPrepared(u16, 3),
    tex_rgb_f: TexPrepared(F, 3),
    func: FuncPrepared,
    func_rgb: FuncPrepared,
};

pub const ShadeContext = struct {
    frame_idx: usize,
    elem_idx: usize,
    fields_num: u8,
    actual_fields: u8,
    scratch_idx: usize,
    global_subx: usize,
    global_suby: usize,
    v_mask_active: ?buildconfig.VecSB = null,
};

pub fn InterpData(comptime N: usize) type {
    return struct {
        weights: [N]F,
        nodes_inv_z: [N]F,
        sub_pixel_z: F,
        xi: F,
        eta: F,
    };
}

pub const FuncCoord = struct {
    coord_0: F,
    coord_1: F,
    normal_x: F,
    normal_y: F,
    normal_z: F,
};

pub inline fn normFuncShaderParams(
    builtin: FuncShaderBuiltin,
    params: FuncShaderParams,
) FuncShaderParams {
    var out = params;
    out.settings = switch (builtin) {
        .constant => .{
            .constant = if (params.settings == .constant)
                params.settings.constant
            else
                ConstantParams{},
        },
        .linear => .{
            .linear = if (params.settings == .linear)
                params.settings.linear
            else
                LinearParams{},
        },
        .quadratic => .{
            .quadratic = if (params.settings == .quadratic)
                params.settings.quadratic
            else
                QuadraticParams{},
        },
        .sinusoidal => .{
            .sinusoidal = if (params.settings == .sinusoidal)
                params.settings.sinusoidal
            else
                SinusoidalParams{},
        },
        .sinusoidal_approx => .{
            .sinusoidal_approx = if (params.settings == .sinusoidal_approx)
                params.settings.sinusoidal_approx
            else
                SinusoidalParams{},
        },
        .checker => .{
            .checker = if (params.settings == .checker)
                params.settings.checker
            else
                CheckerParams{},
        },
        .checker_smooth => .{
            .checker_smooth = if (params.settings == .checker_smooth)
                params.settings.checker_smooth
            else
                CheckerSmoothParams{},
        },
        .lambertian_normal_z => .{
            .lambertian_normal_z = if (params.settings == .lambertian_normal_z)
                params.settings.lambertian_normal_z
            else
                LambertianParams{},
        },
        .eggbox => .{
            .eggbox = if (params.settings == .eggbox)
                params.settings.eggbox
            else
                EggboxParams{},
        },
    };
    return out;
}

inline fn cubicSmoothStep(val: F) F {
    const clamped = @max(0.0, @min(1.0, val));
    return clamped * clamped * (3.0 - 2.0 * clamped);
}

pub inline fn cubicSmoothStepSIMD(v_val: VecSF) VecSF {
    const v_zero: VecSF = @splat(0.0);
    const v_one: VecSF = @splat(1.0);
    const clamped = @max(v_zero, @min(v_one, v_val));
    return clamped * clamped * (@as(VecSF, @splat(3.0)) -
        @as(VecSF, @splat(2.0)) * clamped);
}

inline fn applyFuncShaderCoordParams(
    coord: FuncCoord,
    params: FuncShaderParams,
) FuncCoord {
    var out = coord;
    out.coord_0 = params.coord_scale[0] * coord.coord_0 + params.coord_offset[0];
    out.coord_1 = params.coord_scale[1] * coord.coord_1 + params.coord_offset[1];
    return out;
}

pub inline fn applyFuncShaderCoordParamsSIMD(
    coord: FuncCoordSIMD,
    params: FuncShaderParams,
) FuncCoordSIMD {
    var out = coord;
    out.coord_0 = @as(VecSF, @splat(params.coord_scale[0])) * coord.coord_0 +
        @as(VecSF, @splat(params.coord_offset[0]));
    out.coord_1 = @as(VecSF, @splat(params.coord_scale[1])) * coord.coord_1 +
        @as(VecSF, @splat(params.coord_offset[1]));
    return out;
}

inline fn applyFuncShaderOutputParams(value: F, params: FuncShaderParams) F {
    return value * params.output_scale + params.output_offset;
}

pub inline fn applyFuncShaderOutputParamsSIMD(
    v_value: VecSF,
    params: FuncShaderParams,
) VecSF {
    return v_value * @as(VecSF, @splat(params.output_scale)) +
        @as(VecSF, @splat(params.output_offset));
}

inline fn sinApproxScalar(val: F) F {
    const vals: [1]F = maths_simd.sinApproxSIMD(1, F, .{val});
    return vals[0];
}

inline fn cosApproxScalar(val: F) F {
    const vals: [1]F = maths_simd.cosApproxSIMD(1, F, .{val});
    return vals[0];
}

pub inline fn evalFuncShaderBuiltinGreyNorm(
    builtin: FuncShaderBuiltin,
    coord: FuncCoord,
    params: FuncShaderParams,
) F {
    const eval_coord = applyFuncShaderCoordParams(coord, params);
    const value = switch (builtin) {
        .constant => blk: {
            const p = params.settings.constant;
            break :blk p.value;
        },
        .linear => blk: {
            const p = params.settings.linear;
            break :blk p.coeffs[0] +
                p.coeffs[1] * eval_coord.coord_0 +
                p.coeffs[2] * eval_coord.coord_1;
        },
        .quadratic => blk: {
            const p = params.settings.quadratic;
            const coord_u = eval_coord.coord_0;
            const coord_v = eval_coord.coord_1;
            const c = p.coeffs;
            const term_u = coord_u * (c[1] + c[3] * coord_u);
            const term_v = coord_v * (c[2] + c[4] * coord_u + c[5] * coord_v);
            break :blk c[0] + term_u + term_v;
        },
        .sinusoidal => blk: {
            const p = params.settings.sinusoidal;
            break :blk p.bias +
                p.amplitudes[0] * @sin(p.wave_num_scalar[0] * eval_coord.coord_0) +
                p.amplitudes[1] * @cos(p.wave_num_scalar[1] * eval_coord.coord_1);
        },
        .sinusoidal_approx => blk: {
            const p = params.settings.sinusoidal_approx;
            break :blk p.bias +
                p.amplitudes[0] *
                    sinApproxScalar(p.wave_num_scalar[0] * eval_coord.coord_0) +
                p.amplitudes[1] *
                    cosApproxScalar(p.wave_num_scalar[1] * eval_coord.coord_1);
        },
        .checker => blk: {
            const p = params.settings.checker;
            const cell_x: i64 = @intFromFloat(@floor(eval_coord.coord_0));
            const cell_y: i64 = @intFromFloat(@floor(eval_coord.coord_1));
            break :blk if (@mod(cell_x + cell_y, 2) == 0)
                p.levels[0]
            else
                p.levels[1];
        },
        .checker_smooth => blk: {
            const p = params.settings.checker_smooth;
            const phase_x = 0.5 + 0.5 * @sin(
                p.frequency * std.math.pi * eval_coord.coord_0,
            );
            const phase_y = 0.5 + 0.5 * @sin(
                p.frequency * std.math.pi * eval_coord.coord_1,
            );
            const prod = phase_x * phase_y;
            break :blk cubicSmoothStep(prod);
        },
        .lambertian_normal_z => blk: {
            const p = params.settings.lambertian_normal_z;
            break :blk p.coeffs[0] + p.coeffs[1] * eval_coord.normal_z;
        },
        .eggbox => blk: {
            const p = params.settings.eggbox;
            const phase_x = 2.0 * std.math.pi *
                (eval_coord.coord_0 + p.phase[0]) / p.pitch[0];
            const phase_y = 2.0 * std.math.pi *
                (eval_coord.coord_1 + p.phase[1]) / p.pitch[1];
            break :blk p.mean +
                0.5 * p.contrast * (1.0 + @cos(phase_x)) * (1.0 + @cos(phase_y)) -
                p.contrast;
        },
    };
    return applyFuncShaderOutputParams(value, params);
}

pub inline fn evalFuncShaderBuiltinRGBNorm(
    builtin: FuncShaderBuiltin,
    coord: FuncCoord,
    params: FuncShaderParams,
) [3]F {
    const eval_coord = applyFuncShaderCoordParams(coord, params);
    const vals = switch (builtin) {
        .constant => blk: {
            const p = params.settings.constant;
            break :blk p.value_rgb;
        },
        .linear => blk: {
            const p = params.settings.linear;
            const c = p.coeffs_rgb;
            break :blk .{
                c[0][0] + c[0][1] * eval_coord.coord_0 + c[0][2] * eval_coord.coord_1,
                c[1][0] + c[1][1] * eval_coord.coord_0 + c[1][2] * eval_coord.coord_1,
                c[2][0] + c[2][1] * eval_coord.coord_0 + c[2][2] * eval_coord.coord_1,
            };
        },
        .quadratic => blk: {
            const p = params.settings.quadratic;
            const coord_u = eval_coord.coord_0;
            const coord_v = eval_coord.coord_1;
            const c = p.coeffs_rgb;

            const val_r = c[0][0] + coord_u * (c[0][1] + c[0][3] * coord_u) +
                coord_v * (c[0][2] + c[0][4] * coord_u + c[0][5] * coord_v);
            const val_g = c[1][0] + coord_u * (c[1][1] + c[1][3] * coord_u) +
                coord_v * (c[1][2] + c[1][4] * coord_u + c[1][5] * coord_v);
            const val_b = c[2][0] + coord_u * (c[2][1] + c[2][3] * coord_u) +
                coord_v * (c[2][2] + c[2][4] * coord_u + c[2][5] * coord_v);
            break :blk .{ val_r, val_g, val_b };
        },
        .sinusoidal => blk: {
            const p = params.settings.sinusoidal;
            break :blk .{
                p.bias_rgb[0] + p.amplitudes_rgb[0] *
                    @sin(p.wave_num_rgb[0] * eval_coord.coord_0),
                p.bias_rgb[1] + p.amplitudes_rgb[1] *
                    @cos(p.wave_num_rgb[1] * eval_coord.coord_1),
                p.bias_rgb[2] + p.amplitudes_rgb[2] *
                    @sin(p.wave_num_rgb[2] * (eval_coord.coord_0 + eval_coord.coord_1)),
            };
        },
        .sinusoidal_approx => blk: {
            const p = params.settings.sinusoidal_approx;
            break :blk .{
                p.bias_rgb[0] + p.amplitudes_rgb[0] *
                    sinApproxScalar(p.wave_num_rgb[0] * eval_coord.coord_0),
                p.bias_rgb[1] + p.amplitudes_rgb[1] *
                    cosApproxScalar(p.wave_num_rgb[1] * eval_coord.coord_1),
                p.bias_rgb[2] + p.amplitudes_rgb[2] *
                    sinApproxScalar(
                        p.wave_num_rgb[2] *
                            (eval_coord.coord_0 + eval_coord.coord_1),
                    ),
            };
        },
        .checker => blk: {
            const p = params.settings.checker;
            const cell_x: i64 = @intFromFloat(@floor(eval_coord.coord_0));
            const cell_y: i64 = @intFromFloat(@floor(eval_coord.coord_1));
            const value = if (@mod(cell_x + cell_y, 2) == 0)
                p.levels[0]
            else
                p.levels[1];
            break :blk .{ value, value, value };
        },
        .checker_smooth => blk: {
            const p = params.settings.checker_smooth;
            const phase_x = 0.5 + 0.5 * @sin(
                p.frequency * std.math.pi * eval_coord.coord_0,
            );
            const phase_y = 0.5 + 0.5 * @sin(
                p.frequency * std.math.pi * eval_coord.coord_1,
            );
            const base = cubicSmoothStep(phase_x * phase_y);
            break :blk .{
                base,
                cubicSmoothStep(1.0 - base),
                0.5 + 0.5 * @sin(2.0 * std.math.pi * base),
            };
        },
        .lambertian_normal_z => blk: {
            const p = params.settings.lambertian_normal_z;
            break :blk .{
                p.coeffs_rgb[0][0] + p.coeffs_rgb[0][1] * eval_coord.normal_z,
                p.coeffs_rgb[1][0] + p.coeffs_rgb[1][1] * eval_coord.normal_z,
                p.coeffs_rgb[2][0] + p.coeffs_rgb[2][1] * eval_coord.normal_z,
            };
        },
        .eggbox => blk: {
            const p = params.settings.eggbox;
            const phase_x = 2.0 * std.math.pi *
                (eval_coord.coord_0 + p.phase[0]) / p.pitch[0];
            const phase_y = 2.0 * std.math.pi *
                (eval_coord.coord_1 + p.phase[1]) / p.pitch[1];
            const value = p.mean +
                0.5 * p.contrast * (1.0 + @cos(phase_x)) * (1.0 + @cos(phase_y)) -
                p.contrast;
            break :blk .{ value, value, value };
        },
    };
    return .{
        applyFuncShaderOutputParams(vals[0], params),
        applyFuncShaderOutputParams(vals[1], params),
        applyFuncShaderOutputParams(vals[2], params),
    };
}

// --------------------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------------------

const testing = std.testing;
const unit_tol: F = if (F == f32) 1e-5 else 1e-12;

test "FuncShaderParams defaults preserve constant shader" {
    const coord = FuncCoord{
        .coord_0 = 0.25,
        .coord_1 = -0.5,
        .normal_x = 0.0,
        .normal_y = 0.0,
        .normal_z = 1.0,
    };
    const value = evalFuncShaderBuiltinGreyNorm(
        .constant,
        coord,
        normFuncShaderParams(.constant, .{}),
    );
    try testing.expectApproxEqAbs(@as(F, 0.5), value, unit_tol);
}

test "FuncShaderParams control sinusoidal frequency and output scaling" {
    const coord = FuncCoord{
        .coord_0 = 0.25,
        .coord_1 = 0.0,
        .normal_x = 0.0,
        .normal_y = 0.0,
        .normal_z = 1.0,
    };
    const base = evalFuncShaderBuiltinGreyNorm(
        .sinusoidal,
        coord,
        normFuncShaderParams(.sinusoidal, .{}),
    );
    const shifted = evalFuncShaderBuiltinGreyNorm(
        .sinusoidal,
        coord,
        normFuncShaderParams(.sinusoidal, .{
            .coord_scale = .{ 2.0, 1.0 },
            .output_scale = 2.0,
            .output_offset = -0.25,
        }),
    );
    const expected_base = 0.5 + 0.25 * @sin(6.0 * 0.25) + 0.2 * @cos(0.0);
    const expected_shifted = (0.5 + 0.25 * @sin(6.0 * 0.5) + 0.2 * @cos(0.0)) * 2.0 - 0.25;
    try testing.expectApproxEqAbs(expected_base, base, unit_tol);
    try testing.expectApproxEqAbs(expected_shifted, shifted, unit_tol);
}

test "checker texfunc creates hard black white cells from coord scale" {
    const coord_black = FuncCoord{
        .coord_0 = 0.01,
        .coord_1 = 0.01,
        .normal_x = 0.0,
        .normal_y = 0.0,
        .normal_z = 1.0,
    };
    const coord_white = FuncCoord{
        .coord_0 = 0.05,
        .coord_1 = 0.01,
        .normal_x = 0.0,
        .normal_y = 0.0,
        .normal_z = 1.0,
    };
    const params = FuncShaderParams{
        .coord_scale = .{ 36.0, 36.0 },
    };

    const value_black = evalFuncShaderBuiltinGreyNorm(
        .checker,
        coord_black,
        normFuncShaderParams(.checker, params),
    );
    const value_white = evalFuncShaderBuiltinGreyNorm(
        .checker,
        coord_white,
        normFuncShaderParams(.checker, params),
    );

    try testing.expectEqual(@as(F, 0.0), value_black);
    try testing.expectEqual(@as(F, 1.0), value_white);
}

test "eggbox reaches mean plus contrast at cell center" {
    const coord = FuncCoord{
        .coord_0 = 0.0,
        .coord_1 = 0.0,
        .normal_x = 0.0,
        .normal_y = 0.0,
        .normal_z = 1.0,
    };
    const params = FuncShaderParams{
        .settings = .{
            .eggbox = .{
                .mean = 0.5,
                .contrast = 0.4,
                .pitch = .{ 1.0, 1.0 },
            },
        },
    };
    const value = evalFuncShaderBuiltinGreyNorm(
        .eggbox,
        coord,
        normFuncShaderParams(.eggbox, params),
    );
    try testing.expectApproxEqAbs(@as(F, 0.9), value, unit_tol);
}

test "eggbox reaches mean minus contrast on grid line" {
    const coord = FuncCoord{
        .coord_0 = 0.5,
        .coord_1 = 0.0,
        .normal_x = 0.0,
        .normal_y = 0.0,
        .normal_z = 1.0,
    };
    const params = FuncShaderParams{
        .settings = .{
            .eggbox = .{
                .mean = 0.5,
                .contrast = 0.4,
                .pitch = .{ 1.0, 1.0 },
            },
        },
    };
    const value = evalFuncShaderBuiltinGreyNorm(
        .eggbox,
        coord,
        normFuncShaderParams(.eggbox, params),
    );
    try testing.expectApproxEqAbs(@as(F, 0.1), value, unit_tol);
}

test "SIMD func builtin matches scalar builtin per lane" {
    const coord_scalar = [_]FuncCoord{
        .{
            .coord_0 = 0.1,
            .coord_1 = -0.2,
            .normal_x = 0.0,
            .normal_y = 0.0,
            .normal_z = 1.0,
        },
        .{
            .coord_0 = 0.35,
            .coord_1 = 0.125,
            .normal_x = 0.1,
            .normal_y = -0.2,
            .normal_z = 0.7,
        },
        .{
            .coord_0 = -0.45,
            .coord_1 = 0.8,
            .normal_x = -0.3,
            .normal_y = 0.2,
            .normal_z = 0.4,
        },
        .{
            .coord_0 = 1.2,
            .coord_1 = -0.9,
            .normal_x = 0.0,
            .normal_y = 0.0,
            .normal_z = 0.25,
        },
    };
    const coord_simd = FuncCoordSIMD{
        .coord_0 = .{
            coord_scalar[0].coord_0,
            coord_scalar[1].coord_0,
            coord_scalar[2].coord_0,
            coord_scalar[3].coord_0,
        } ++ [_]F{0.0} ** (S - 4),
        .coord_1 = .{
            coord_scalar[0].coord_1,
            coord_scalar[1].coord_1,
            coord_scalar[2].coord_1,
            coord_scalar[3].coord_1,
        } ++ [_]F{0.0} ** (S - 4),
        .normal_x = .{
            coord_scalar[0].normal_x,
            coord_scalar[1].normal_x,
            coord_scalar[2].normal_x,
            coord_scalar[3].normal_x,
        } ++ [_]F{0.0} ** (S - 4),
        .normal_y = .{
            coord_scalar[0].normal_y,
            coord_scalar[1].normal_y,
            coord_scalar[2].normal_y,
            coord_scalar[3].normal_y,
        } ++ [_]F{0.0} ** (S - 4),
        .normal_z = .{
            coord_scalar[0].normal_z,
            coord_scalar[1].normal_z,
            coord_scalar[2].normal_z,
            coord_scalar[3].normal_z,
        } ++ [_]F{0.0} ** (S - 4),
    };
    const params = FuncShaderParams{
        .coord_scale = .{ 1.7, 0.8 },
        .coord_offset = .{ -0.1, 0.3 },
        .output_scale = 1.25,
        .output_offset = -0.05,
    };

    const scalar_builtins = [_]FuncShaderBuiltin{
        .constant,
        .linear,
        .quadratic,
        .sinusoidal,
        .sinusoidal_approx,
        .checker,
        .checker_smooth,
        .lambertian_normal_z,
        .eggbox,
    };
    for (scalar_builtins) |builtin| {
        const v_vals = simd_impl.evalFuncShaderGreyNormSIMD(
            builtin,
            coord_simd,
            normFuncShaderParams(builtin, params),
        );
        const vals_arr: [S]F = v_vals;
        for (coord_scalar, 0..) |coord, ll| {
            const expected = evalFuncShaderBuiltinGreyNorm(
                builtin,
                coord,
                normFuncShaderParams(builtin, params),
            );
            try testing.expectApproxEqAbs(expected, vals_arr[ll], unit_tol);
        }

        const v_rgb = simd_impl.evalFuncShaderRGBNormSIMD(
            builtin,
            coord_simd,
            normFuncShaderParams(builtin, params),
        );
        inline for (0..3) |ch| {
            const vals_rgb_arr: [S]F = v_rgb[ch];
            for (coord_scalar, 0..) |coord, ll| {
                const expected = evalFuncShaderBuiltinRGBNorm(
                    builtin,
                    coord,
                    normFuncShaderParams(builtin, params),
                )[ch];
                try testing.expectApproxEqAbs(
                    expected,
                    vals_rgb_arr[ll],
                    unit_tol,
                );
            }
        }
    }
}
