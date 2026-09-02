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
const speckle_boundary_blur = buildconfig.speckle_boundary_blur;
const speckle_neighbor_count = buildconfig.speckle_neighbor_count;
const speckle_shape = buildconfig.speckle_shape;
const speckle_mask_samples_per_cell: F =
    @floatFromInt(buildconfig.speckle_mask_samples_per_cell);
const max_speckle_mask_bytes = 256 * 1024 * 1024;
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
    speckle,
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

pub const Speckle2DParams = struct {
    seed: u32 = 0xa511e9b3,
    cells_per_uv: [2]F = .{ 192.0, 160.0 },
    uv_offset: [2]F = .{ 0.0, 0.0 },
    occupancy: F = 0.9,
    radius_mean: F = 0.45,
    radius_jitter: F = 0.08,
    edge_softness: F = 0.035,
    perlin_coverage_threshold: F = 0.0,
    perlin_coverage_transition_width: F = 0.12,
    foreground: F = 0.0,
    background: F = 1.0,

    pub fn toFuncShaderParams(self: Speckle2DParams) FuncShaderParams {
        return .{ .settings = .{ .speckle = self } };
    }

    pub fn validate(self: Speckle2DParams) !void {
        for (self.cells_per_uv) |cell_count| {
            if (!std.math.isFinite(cell_count) or cell_count <= 0.0) {
                return error.InvalidSpeckleCellsPerUV;
            }
        }
        for (self.uv_offset) |offset| {
            if (!std.math.isFinite(offset)) {
                return error.InvalidSpeckleUVOffset;
            }
        }
        if (comptime speckle_shape == .perlin) {
            if (!std.math.isFinite(self.perlin_coverage_threshold)) {
                return error.InvalidSpecklePerlinCoverageThreshold;
            }
            if (!std.math.isFinite(self.perlin_coverage_transition_width) or
                self.perlin_coverage_transition_width < 0.0)
            {
                return error.InvalidSpecklePerlinCoverageTransitionWidth;
            }
        } else {
            if (!std.math.isFinite(self.occupancy) or
                self.occupancy < 0.0 or self.occupancy > 1.0)
            {
                return error.InvalidSpeckleOccupancy;
            }
            if (!std.math.isFinite(self.radius_mean) or self.radius_mean <= 0.0) {
                return error.InvalidSpeckleRadiusMean;
            }
            if (!std.math.isFinite(self.radius_jitter) or self.radius_jitter < 0.0) {
                return error.InvalidSpeckleRadiusJitter;
            }
            if (self.radius_jitter > self.radius_mean) {
                return error.InvalidSpeckleRadiusRange;
            }
            if (!std.math.isFinite(self.edge_softness) or self.edge_softness < 0.0) {
                return error.InvalidSpeckleEdgeSoftness;
            }
            const effective_softness = if (comptime speckle_boundary_blur)
                self.edge_softness
            else
                0.0;
            if (self.radius_mean + self.radius_jitter + effective_softness > 1.0) {
                return error.InvalidSpeckleNeighborhoodRadius;
            }
        }
        if (!std.math.isFinite(self.foreground) or
            self.foreground < 0.0 or self.foreground > 1.0)
        {
            return error.InvalidSpeckleForeground;
        }
        if (!std.math.isFinite(self.background) or
            self.background < 0.0 or self.background > 1.0)
        {
            return error.InvalidSpeckleBackground;
        }

        // Preserve at least eight bits of sub-cell precision in procedural coordinates.
        const cell_coord_lim: F = if (F == f32)
            65_536.0
        else
            35_184_372_088_832.0;
        for (0..2) |axis| {
            const coord_min = self.uv_offset[axis] - 1.0;
            const coord_max = self.uv_offset[axis] + self.cells_per_uv[axis] + 1.0;
            if (!std.math.isFinite(coord_max) or
                coord_min < -cell_coord_lim or coord_max > cell_coord_lim)
            {
                return error.SpeckleCellCoordinateOutOfRange;
            }
        }
    }
};

pub const SpeckleDisk2D = struct {
    center: [2]F,
    radius: F,
};

pub const SpeckleList2D = struct {
    pub const no_disk = std.math.maxInt(u32);

    params: Speckle2DParams,
    disks: []const SpeckleDisk2D,
    cell_origin: [2]i64,
    cell_dims: [2]usize,
    disk_by_cell: []const u32,
};

pub const SpeckleMask2D = struct {
    bits: []const u8,
    dims: [2]usize,
    row_stride: usize,
    uv_to_texel: [2]F,
    params: Speckle2DParams,
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
        speckle: Speckle2DParams,
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
    speckle_list: ?SpeckleList2D = null,
    speckle_mask: ?SpeckleMask2D = null,
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
    speckle_list: ?SpeckleList2D = null,
    speckle_mask: ?SpeckleMask2D = null,
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
        .speckle => .{
            .speckle = if (params.settings == .speckle)
                params.settings.speckle
            else
                Speckle2DParams{},
        },
    };
    return out;
}

pub fn validateSpeckleInput(
    input: FuncInput,
    is_rgb: bool,
    connect: *const meshio.Connect,
) !void {
    if (input.builtin != .speckle) return;
    if (is_rgb) return error.SpeckleRequiresGrayscale;
    if (input.coord_mode != .uv) return error.SpeckleRequiresUVCoordinates;
    const uvs = input.uvs orelse return error.MissingUVsForSpeckleShader;
    if (input.normal_type != .none) return error.SpeckleRequiresNoNormals;
    if (uvs.dims.len != 2 or uvs.dims[1] != 2) {
        return error.InvalidSpeckleUVShape;
    }
    for (connect.table_mem) |node_idx| {
        if (node_idx >= uvs.dims[0]) return error.InvalidSpeckleUVNodeIndex;
    }
    for (uvs.slice) |value| {
        if (!std.math.isFinite(value)) return error.InvalidSpeckleUVValue;
    }
    if (input.params.coord_scale[0] != 1.0 or
        input.params.coord_scale[1] != 1.0 or
        input.params.coord_offset[0] != 0.0 or
        input.params.coord_offset[1] != 0.0)
    {
        return error.SpeckleUsesTypedCoordinateParams;
    }

    const params = normFuncShaderParams(.speckle, input.params);
    try params.settings.speckle.validate();
}

// --------------------------------------------------------------------------------------
// Procedural Speckle Shader
// --------------------------------------------------------------------------------------

pub fn hashSpeckleCell(cell_x: i64, cell_y: i64, seed: u32) u64 {
    var key: [16]u8 = undefined;
    std.mem.writeInt(u64, key[0..8], @bitCast(cell_x), .little);
    std.mem.writeInt(u64, key[8..16], @bitCast(cell_y), .little);
    return std.hash.Wyhash.hash(seed, &key);
}

// Split one cell hash into four variates to avoid additional hash evaluations.
fn randomUnitFromHash(hash: u64, comptime shift: u6) F {
    const bits: u16 = @truncate(hash >> shift);
    return @as(F, @floatFromInt(bits)) / 65_536.0;
}

inline fn effectiveSpeckleSoftness(params: Speckle2DParams) F {
    return if (comptime speckle_shape == .disk and speckle_boundary_blur)
        params.edge_softness
    else
        0.0;
}

fn speckleDiskFromHash(
    cell_x: i64,
    cell_y: i64,
    hash: u64,
    params: Speckle2DParams,
) SpeckleDisk2D {
    const radius_variation = 2.0 * randomUnitFromHash(hash, 48) - 1.0;
    const edge_softness = effectiveSpeckleSoftness(params);
    var radius = params.radius_mean + params.radius_jitter * radius_variation;
    if (comptime speckle_neighbor_count == 1) {
        radius = @min(radius, 0.5 - edge_softness);
    }
    const center_min = if (comptime speckle_neighbor_count == 1)
        radius + edge_softness
    else
        0.0;
    const center_extent = switch (comptime speckle_neighbor_count) {
        1 => 1.0 - 2.0 * center_min,
        4 => 1.0 - radius - edge_softness,
        9 => 1.0,
        else => unreachable,
    };
    return .{
        .center = .{
            @as(F, @floatFromInt(cell_x)) + center_min +
                randomUnitFromHash(hash, 16) * center_extent,
            @as(F, @floatFromInt(cell_y)) + center_min +
                randomUnitFromHash(hash, 32) * center_extent,
        },
        .radius = radius,
    };
}

fn speckleDiskForCell(
    cell_x: i64,
    cell_y: i64,
    params: Speckle2DParams,
) ?SpeckleDisk2D {
    const hash = hashSpeckleCell(cell_x, cell_y, params.seed);
    if (randomUnitFromHash(hash, 0) >= params.occupancy) return null;
    return speckleDiskFromHash(cell_x, cell_y, hash, params);
}

pub fn generateSpeckleList2D(
    allocator: std.mem.Allocator,
    params: Speckle2DParams,
) !SpeckleList2D {
    try params.validate();

    const min_x = @as(i64, @intFromFloat(@floor(params.uv_offset[0]))) - 1;
    const min_y = @as(i64, @intFromFloat(@floor(params.uv_offset[1]))) - 1;
    const max_x = @as(i64, @intFromFloat(@floor(
        params.uv_offset[0] + params.cells_per_uv[0],
    ))) + 1;
    const max_y = @as(i64, @intFromFloat(@floor(
        params.uv_offset[1] + params.cells_per_uv[1],
    ))) + 1;
    const width = std.math.cast(usize, max_x - min_x + 1) orelse
        return error.SpeckleListTooLarge;
    const height = std.math.cast(usize, max_y - min_y + 1) orelse
        return error.SpeckleListTooLarge;
    const cell_count = std.math.mul(usize, width, height) catch
        return error.SpeckleListTooLarge;
    if (cell_count > 10_000_000) return error.SpeckleListTooLarge;

    var active_count: usize = 0;
    var cell_y = min_y;
    while (cell_y <= max_y) : (cell_y += 1) {
        var cell_x = min_x;
        while (cell_x <= max_x) : (cell_x += 1) {
            if (speckleDiskForCell(cell_x, cell_y, params) != null) active_count += 1;
        }
    }

    const disks = try allocator.alloc(SpeckleDisk2D, active_count);
    errdefer allocator.free(disks);
    const disk_by_cell_count = if (comptime buildconfig.speckle_evaluator == .list_naive)
        0
    else
        cell_count;
    const disk_by_cell = try allocator.alloc(u32, disk_by_cell_count);
    errdefer allocator.free(disk_by_cell);
    if (comptime buildconfig.speckle_evaluator != .list_naive) {
        @memset(disk_by_cell, SpeckleList2D.no_disk);
    }
    var disk_index: usize = 0;
    var cell_index: usize = 0;
    cell_y = min_y;
    while (cell_y <= max_y) : (cell_y += 1) {
        var cell_x = min_x;
        while (cell_x <= max_x) : (cell_x += 1) {
            if (speckleDiskForCell(cell_x, cell_y, params)) |disk| {
                disks[disk_index] = disk;
                if (comptime buildconfig.speckle_evaluator != .list_naive) {
                    disk_by_cell[cell_index] = @intCast(disk_index);
                }
                disk_index += 1;
            }
            cell_index += 1;
        }
    }
    return .{
        .params = params,
        .disks = disks,
        .cell_origin = .{ min_x, min_y },
        .cell_dims = .{ width, height },
        .disk_by_cell = disk_by_cell,
    };
}

fn speckleDiskMask(distance2: F, radius: F, edge_softness: F) F {
    if (comptime speckle_shape == .gaussian) {
        const radius2 = radius * radius;
        if (distance2 >= radius2) return 0.0;
        const sigma = radius / 3.0;
        return @exp(-0.5 * distance2 / (sigma * sigma));
    }
    if (comptime !speckle_boundary_blur) {
        return if (distance2 <= radius * radius) 1.0 else 0.0;
    }
    if (edge_softness == 0.0) return if (distance2 <= radius * radius) 1.0 else 0.0;

    const inner_radius = @max(0.0, radius - edge_softness);
    const outer_radius = radius + edge_softness;
    const inner2 = inner_radius * inner_radius;
    const outer2 = outer_radius * outer_radius;

    if (distance2 <= inner2) return 1.0;
    if (distance2 >= outer2) return 0.0;

    const transition = (distance2 - inner2) / (outer2 - inner2);
    return 1.0 - cubicSmoothStep(transition);
}

fn speckleListDiskAt(
    speckles: SpeckleList2D,
    cell_x: i64,
    cell_y: i64,
) ?SpeckleDisk2D {
    if (cell_x < speckles.cell_origin[0] or cell_y < speckles.cell_origin[1]) {
        return null;
    }
    const rel_x = std.math.cast(usize, cell_x - speckles.cell_origin[0]) orelse
        return null;
    const rel_y = std.math.cast(usize, cell_y - speckles.cell_origin[1]) orelse
        return null;
    if (rel_x >= speckles.cell_dims[0] or rel_y >= speckles.cell_dims[1]) return null;
    const encoded = speckles.disk_by_cell[rel_y * speckles.cell_dims[0] + rel_x];
    if (encoded == SpeckleList2D.no_disk) return null;
    return speckles.disks[encoded];
}

const SpeckleMaskCellBounds = struct {
    min: [2]i64,
    max: [2]i64,
    dims: [2]usize,
};

fn speckleMaskCellBounds(params: Speckle2DParams) !SpeckleMaskCellBounds {
    const min = [2]i64{
        @as(i64, @intFromFloat(@floor(params.uv_offset[0]))) - 1,
        @as(i64, @intFromFloat(@floor(params.uv_offset[1]))) - 1,
    };
    const max = [2]i64{
        @as(i64, @intFromFloat(@floor(
            params.uv_offset[0] + params.cells_per_uv[0],
        ))) + 1,
        @as(i64, @intFromFloat(@floor(
            params.uv_offset[1] + params.cells_per_uv[1],
        ))) + 1,
    };
    const width = std.math.cast(usize, max[0] - min[0] + 1) orelse
        return error.SpeckleMaskTooLarge;
    const height = std.math.cast(usize, max[1] - min[1] + 1) orelse
        return error.SpeckleMaskTooLarge;
    const cell_count = std.math.mul(usize, width, height) catch
        return error.SpeckleMaskTooLarge;
    if (cell_count > 10_000_000) return error.SpeckleMaskTooLarge;
    return .{ .min = min, .max = max, .dims = .{ width, height } };
}

inline fn quantizeSpeckleCoverage(coverage: F) u8 {
    return @intFromFloat(@round(@max(0.0, @min(1.0, coverage)) * 255.0));
}

inline fn quinticPerlinFade(value: F) F {
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0);
}

inline fn perlinGradientDot(gradient_index: u8, delta_x: F, delta_y: F) F {
    const diagonal = @sqrt(@as(F, 0.5));
    return switch (gradient_index & 7) {
        0 => delta_x,
        1 => diagonal * (delta_x + delta_y),
        2 => delta_y,
        3 => diagonal * (-delta_x + delta_y),
        4 => -delta_x,
        5 => -diagonal * (delta_x + delta_y),
        6 => -delta_y,
        7 => diagonal * (delta_x - delta_y),
        else => unreachable,
    };
}

inline fn perlinCoverage(noise: F, params: Speckle2DParams) F {
    const width = params.perlin_coverage_transition_width;
    if (width == 0.0) {
        return if (noise >= params.perlin_coverage_threshold) 1.0 else 0.0;
    }
    const transition = (noise - params.perlin_coverage_threshold) / width + 0.5;
    return cubicSmoothStep(transition);
}

pub fn evalPerlinSpeckle2D(uv: [2]F, params: Speckle2DParams) F {
    const proc_x = @max(0.0, @min(1.0, uv[0])) * params.cells_per_uv[0] +
        params.uv_offset[0];
    const proc_y = @max(0.0, @min(1.0, uv[1])) * params.cells_per_uv[1] +
        params.uv_offset[1];
    const cell_x = @as(i64, @intFromFloat(@floor(proc_x)));
    const cell_y = @as(i64, @intFromFloat(@floor(proc_y)));
    const frac_x = proc_x - @as(F, @floatFromInt(cell_x));
    const frac_y = proc_y - @as(F, @floatFromInt(cell_y));
    const fade_x = quinticPerlinFade(frac_x);
    const fade_y = quinticPerlinFade(frac_y);
    const noise_00 = perlinGradientDot(
        @truncate(hashSpeckleCell(cell_x, cell_y, params.seed)),
        frac_x,
        frac_y,
    );
    const noise_10 = perlinGradientDot(
        @truncate(hashSpeckleCell(cell_x + 1, cell_y, params.seed)),
        frac_x - 1.0,
        frac_y,
    );
    const noise_01 = perlinGradientDot(
        @truncate(hashSpeckleCell(cell_x, cell_y + 1, params.seed)),
        frac_x,
        frac_y - 1.0,
    );
    const noise_11 = perlinGradientDot(
        @truncate(hashSpeckleCell(cell_x + 1, cell_y + 1, params.seed)),
        frac_x - 1.0,
        frac_y - 1.0,
    );
    const noise_x0 = noise_00 + fade_x * (noise_10 - noise_00);
    const noise_x1 = noise_01 + fade_x * (noise_11 - noise_01);
    const noise = noise_x0 + fade_y * (noise_x1 - noise_x0);
    const coverage = perlinCoverage(noise, params);
    return params.background + coverage * (params.foreground - params.background);
}

fn rasterizePerlinSpeckleMask(
    allocator: std.mem.Allocator,
    bits: []u8,
    dims: [2]usize,
    row_stride: usize,
    uv_to_texel: [2]F,
    cell_bounds: SpeckleMaskCellBounds,
    params: Speckle2DParams,
) !void {
    const gradient_count = std.math.mul(
        usize,
        cell_bounds.dims[0],
        cell_bounds.dims[1],
    ) catch return error.SpeckleMaskTooLarge;
    const gradient_indices = try allocator.alloc(u8, gradient_count);
    defer allocator.free(gradient_indices);

    for (0..cell_bounds.dims[1]) |yy| {
        const cell_y = cell_bounds.min[1] + @as(i64, @intCast(yy));
        for (0..cell_bounds.dims[0]) |xx| {
            const cell_x = cell_bounds.min[0] + @as(i64, @intCast(xx));
            gradient_indices[yy * cell_bounds.dims[0] + xx] =
                @as(u8, @truncate(hashSpeckleCell(cell_x, cell_y, params.seed))) & 7;
        }
    }

    const texel_to_proc = [2]F{
        params.cells_per_uv[0] / uv_to_texel[0],
        params.cells_per_uv[1] / uv_to_texel[1],
    };
    for (0..dims[1]) |yy| {
        const proc_y = @as(F, @floatFromInt(yy)) * texel_to_proc[1] +
            params.uv_offset[1];
        const cell_y = @as(i64, @intFromFloat(@floor(proc_y)));
        const rel_y = std.math.cast(usize, cell_y - cell_bounds.min[1]) orelse
            unreachable;
        const frac_y = proc_y - @as(F, @floatFromInt(cell_y));
        const fade_y = quinticPerlinFade(frac_y);

        for (0..dims[0]) |xx| {
            const proc_x = @as(F, @floatFromInt(xx)) * texel_to_proc[0] +
                params.uv_offset[0];
            const cell_x = @as(i64, @intFromFloat(@floor(proc_x)));
            const rel_x = std.math.cast(usize, cell_x - cell_bounds.min[0]) orelse
                unreachable;
            const frac_x = proc_x - @as(F, @floatFromInt(cell_x));
            const fade_x = quinticPerlinFade(frac_x);
            const gradient_row_0 = rel_y * cell_bounds.dims[0];
            const gradient_row_1 = (rel_y + 1) * cell_bounds.dims[0];

            const noise_00 = perlinGradientDot(
                gradient_indices[gradient_row_0 + rel_x],
                frac_x,
                frac_y,
            );
            const noise_10 = perlinGradientDot(
                gradient_indices[gradient_row_0 + rel_x + 1],
                frac_x - 1.0,
                frac_y,
            );
            const noise_01 = perlinGradientDot(
                gradient_indices[gradient_row_1 + rel_x],
                frac_x,
                frac_y - 1.0,
            );
            const noise_11 = perlinGradientDot(
                gradient_indices[gradient_row_1 + rel_x + 1],
                frac_x - 1.0,
                frac_y - 1.0,
            );
            const noise_x0 = noise_00 + fade_x * (noise_10 - noise_00);
            const noise_x1 = noise_01 + fade_x * (noise_11 - noise_01);
            const noise = noise_x0 + fade_y * (noise_x1 - noise_x0);
            bits[yy * row_stride + xx] = quantizeSpeckleCoverage(
                perlinCoverage(noise, params),
            );
        }
    }
}

inline fn rasterizeSpeckleMaskDisk(
    bits: []u8,
    row_stride: usize,
    uv_to_texel: [2]F,
    params: Speckle2DParams,
    disk: SpeckleDisk2D,
) void {
    const edge_softness = effectiveSpeckleSoftness(params);
    const outer_radius = disk.radius + edge_softness;
    const proc_to_texel = [2]F{
        uv_to_texel[0] / params.cells_per_uv[0],
        uv_to_texel[1] / params.cells_per_uv[1],
    };
    const texel_to_proc = [2]F{
        params.cells_per_uv[0] / uv_to_texel[0],
        params.cells_per_uv[1] / uv_to_texel[1],
    };
    const box_min = [2]F{
        (disk.center[0] - outer_radius - params.uv_offset[0]) * proc_to_texel[0],
        (disk.center[1] - outer_radius - params.uv_offset[1]) * proc_to_texel[1],
    };
    const box_max = [2]F{
        (disk.center[0] + outer_radius - params.uv_offset[0]) * proc_to_texel[0],
        (disk.center[1] + outer_radius - params.uv_offset[1]) * proc_to_texel[1],
    };
    if (box_max[0] < 0.0 or box_max[1] < 0.0 or
        box_min[0] > uv_to_texel[0] or box_min[1] > uv_to_texel[1]) return;
    const min_x: usize = @intFromFloat(@max(0.0, @floor(box_min[0])));
    const min_y: usize = @intFromFloat(@max(0.0, @floor(box_min[1])));
    const max_x: usize = @intFromFloat(@min(uv_to_texel[0], @ceil(box_max[0])));
    const max_y: usize = @intFromFloat(@min(uv_to_texel[1], @ceil(box_max[1])));

    for (min_y..max_y + 1) |yy| {
        const proc_y = @as(F, @floatFromInt(yy)) * texel_to_proc[1] +
            params.uv_offset[1];
        const delta_y = proc_y - disk.center[1];
        for (min_x..max_x + 1) |xx| {
            const proc_x = @as(F, @floatFromInt(xx)) * texel_to_proc[0] +
                params.uv_offset[0];
            const delta_x = proc_x - disk.center[0];
            const coverage = speckleDiskMask(
                delta_x * delta_x + delta_y * delta_y,
                disk.radius,
                edge_softness,
            );
            if (comptime buildconfig.speckle_evaluator == .mask_u8) {
                const idx = yy * row_stride + xx;
                bits[idx] = @max(bits[idx], quantizeSpeckleCoverage(coverage));
            } else if (coverage != 0.0) {
                const shift: u3 = @intCast(xx & 7);
                bits[yy * row_stride + xx / 8] |= @as(u8, 1) << shift;
            }
        }
    }
}

pub fn generateSpeckleMask2D(
    allocator: std.mem.Allocator,
    params: Speckle2DParams,
) !SpeckleMask2D {
    try params.validate();
    const cell_bounds = try speckleMaskCellBounds(params);

    var intervals: [2]usize = undefined;
    for (params.cells_per_uv, 0..) |cells, axis| {
        const interval_f = @ceil(cells * speckle_mask_samples_per_cell);
        const max_interval: F = @floatFromInt(std.math.maxInt(usize) - 1);
        if (!std.math.isFinite(interval_f) or interval_f > max_interval) {
            return error.SpeckleMaskTooLarge;
        }
        intervals[axis] = @intFromFloat(interval_f);
    }
    const dims = [2]usize{
        std.math.add(usize, intervals[0], 1) catch return error.SpeckleMaskTooLarge,
        std.math.add(usize, intervals[1], 1) catch return error.SpeckleMaskTooLarge,
    };
    const row_stride = if (comptime buildconfig.speckle_evaluator == .mask_u8)
        dims[0]
    else
        std.math.divCeil(usize, dims[0], 8) catch
            return error.SpeckleMaskTooLarge;
    const byte_count = std.math.mul(usize, row_stride, dims[1]) catch
        return error.SpeckleMaskTooLarge;
    if (byte_count > max_speckle_mask_bytes) return error.SpeckleMaskTooLarge;
    const uv_to_texel = [2]F{
        @floatFromInt(intervals[0]),
        @floatFromInt(intervals[1]),
    };
    var mask_params = params;
    if (comptime speckle_shape == .perlin) mask_params.occupancy = 1.0;
    const empty_pattern = if (comptime speckle_shape == .perlin)
        false
    else
        params.occupancy == 0.0;
    if (empty_pattern or params.foreground == params.background) {
        return .{
            .bits = try allocator.alloc(u8, 0),
            .dims = dims,
            .row_stride = row_stride,
            .uv_to_texel = uv_to_texel,
            .params = mask_params,
        };
    }

    const bits = try allocator.alloc(u8, byte_count);
    errdefer allocator.free(bits);
    if (comptime speckle_shape == .perlin) {
        try rasterizePerlinSpeckleMask(
            allocator,
            bits,
            dims,
            row_stride,
            uv_to_texel,
            cell_bounds,
            params,
        );
    } else {
        @memset(bits, 0);
        var cell_y = cell_bounds.min[1];
        while (cell_y <= cell_bounds.max[1]) : (cell_y += 1) {
            var cell_x = cell_bounds.min[0];
            while (cell_x <= cell_bounds.max[0]) : (cell_x += 1) {
                const hash = hashSpeckleCell(cell_x, cell_y, params.seed);
                if (params.occupancy < 1.0 and
                    randomUnitFromHash(hash, 0) >= params.occupancy)
                {
                    continue;
                }
                rasterizeSpeckleMaskDisk(
                    bits,
                    row_stride,
                    uv_to_texel,
                    params,
                    speckleDiskFromHash(cell_x, cell_y, hash, params),
                );
            }
        }
    }

    return .{
        .bits = bits,
        .dims = dims,
        .row_stride = row_stride,
        .uv_to_texel = uv_to_texel,
        .params = mask_params,
    };
}

pub inline fn evalSpeckleMask2D(uv: [2]F, mask: SpeckleMask2D) F {
    const params = mask.params;
    if (params.occupancy == 0.0 or params.foreground == params.background or
        !std.math.isFinite(uv[0]) or !std.math.isFinite(uv[1]))
    {
        return params.background;
    }
    const texel_x: usize = @intFromFloat(
        @max(0.0, @min(1.0, uv[0])) * mask.uv_to_texel[0] + 0.5,
    );
    const texel_y: usize = @intFromFloat(
        @max(0.0, @min(1.0, uv[1])) * mask.uv_to_texel[1] + 0.5,
    );
    if (comptime buildconfig.speckle_evaluator == .mask_u8) {
        const coverage = @as(F, @floatFromInt(
            mask.bits[texel_y * mask.row_stride + texel_x],
        )) / 255.0;
        return params.background + coverage * (params.foreground - params.background);
    }
    const shift: u3 = @intCast(texel_x & 7);
    const is_foreground = mask.bits[texel_y * mask.row_stride + texel_x / 8] &
        (@as(u8, 1) << shift) != 0;
    return if (is_foreground) params.foreground else params.background;
}

pub fn evalSpeckleList2DNaive(uv: [2]F, speckles: SpeckleList2D) F {
    const params = speckles.params;
    if (speckles.disks.len == 0 or params.foreground == params.background) {
        return params.background;
    }
    const proc_x = @max(0.0, @min(1.0, uv[0])) * params.cells_per_uv[0] +
        params.uv_offset[0];
    const proc_y = @max(0.0, @min(1.0, uv[1])) * params.cells_per_uv[1] +
        params.uv_offset[1];
    const edge_softness = effectiveSpeckleSoftness(params);
    var coverage: F = 0.0;
    for (speckles.disks) |disk| {
        const delta_x = proc_x - disk.center[0];
        const delta_y = proc_y - disk.center[1];
        const distance2 = delta_x * delta_x + delta_y * delta_y;
        coverage = @max(
            coverage,
            speckleDiskMask(distance2, disk.radius, edge_softness),
        );
        if (coverage == 1.0) break;
    }
    return params.background + coverage * (params.foreground - params.background);
}

pub fn evalSpeckleList2DIndexed(uv: [2]F, speckles: SpeckleList2D) F {
    const params = speckles.params;
    if (speckles.disks.len == 0 or params.foreground == params.background) {
        return params.background;
    }
    const proc_x = @max(0.0, @min(1.0, uv[0])) * params.cells_per_uv[0] +
        params.uv_offset[0];
    const proc_y = @max(0.0, @min(1.0, uv[1])) * params.cells_per_uv[1] +
        params.uv_offset[1];
    const cell_x_f = @floor(proc_x);
    const cell_y_f = @floor(proc_y);
    const cell_x: i64 = @intFromFloat(cell_x_f);
    const cell_y: i64 = @intFromFloat(cell_y_f);
    const frac_x = proc_x - cell_x_f;
    const frac_y = proc_y - cell_y_f;
    const offsets = if (comptime speckle_neighbor_count == 1)
        [_]i64{0}
    else if (comptime speckle_neighbor_count == 4)
        [_]i64{ 0, 1 }
    else
        [_]i64{ -1, 0, 1 };
    const min_delta_x = if (comptime speckle_neighbor_count == 1)
        [_]F{0.0}
    else if (comptime speckle_neighbor_count == 4)
        [_]F{ 0.0, 1.0 - frac_x }
    else
        [_]F{ frac_x, 0.0, 1.0 - frac_x };
    const min_delta_y = if (comptime speckle_neighbor_count == 1)
        [_]F{0.0}
    else if (comptime speckle_neighbor_count == 4)
        [_]F{ 0.0, 1.0 - frac_y }
    else
        [_]F{ frac_y, 0.0, 1.0 - frac_y };
    const edge_softness = effectiveSpeckleSoftness(params);
    const max_outer_radius = params.radius_mean + params.radius_jitter +
        edge_softness;
    const max_outer_radius2 = max_outer_radius * max_outer_radius;

    var coverage: F = 0.0;
    neighbor_loop: for (offsets, min_delta_y) |offset_y, min_dy| {
        for (offsets, min_delta_x) |offset_x, min_dx| {
            if (min_dx * min_dx + min_dy * min_dy > max_outer_radius2) continue;
            const disk = speckleListDiskAt(
                speckles,
                cell_x + offset_x,
                cell_y + offset_y,
            ) orelse continue;
            const delta_x = proc_x - disk.center[0];
            const delta_y = proc_y - disk.center[1];
            const distance2 = delta_x * delta_x + delta_y * delta_y;
            coverage = @max(
                coverage,
                speckleDiskMask(distance2, disk.radius, edge_softness),
            );
            if (coverage == 1.0) break :neighbor_loop;
        }
    }
    return params.background + coverage * (params.foreground - params.background);
}

pub fn evalSpeckleList2D(uv: [2]F, speckles: SpeckleList2D) F {
    return switch (comptime buildconfig.speckle_evaluator) {
        .list_indexed, .mask_1bit, .mask_u8 => evalSpeckleList2DIndexed(uv, speckles),
        .cell_hash, .list_naive => evalSpeckleList2DNaive(uv, speckles),
    };
}

pub fn evalSpeckle2D(uv: [2]F, params: Speckle2DParams) F {
    if (comptime speckle_shape == .perlin) return evalPerlinSpeckle2D(uv, params);
    if (params.occupancy == 0.0 or params.foreground == params.background) {
        return params.background;
    }

    const proc_x = @max(0.0, @min(1.0, uv[0])) * params.cells_per_uv[0] +
        params.uv_offset[0];
    const proc_y = @max(0.0, @min(1.0, uv[1])) * params.cells_per_uv[1] +
        params.uv_offset[1];
    const cell_x_f = @floor(proc_x);
    const cell_y_f = @floor(proc_y);
    const cell_x: i64 = @intFromFloat(cell_x_f);
    const cell_y: i64 = @intFromFloat(cell_y_f);
    const frac_x = proc_x - cell_x_f;
    const frac_y = proc_y - cell_y_f;
    const neighbor_offsets = if (comptime speckle_neighbor_count == 1)
        [_]i64{0}
    else if (comptime speckle_neighbor_count == 4)
        [_]i64{ 0, 1 }
    else
        [_]i64{ -1, 0, 1 };
    const min_delta_x = if (comptime speckle_neighbor_count == 1)
        [_]F{0.0}
    else if (comptime speckle_neighbor_count == 4)
        [_]F{ 0.0, 1.0 - frac_x }
    else
        [_]F{ frac_x, 0.0, 1.0 - frac_x };
    const min_delta_y = if (comptime speckle_neighbor_count == 1)
        [_]F{0.0}
    else if (comptime speckle_neighbor_count == 4)
        [_]F{ 0.0, 1.0 - frac_y }
    else
        [_]F{ frac_y, 0.0, 1.0 - frac_y };
    const edge_softness = effectiveSpeckleSoftness(params);
    const max_outer_radius = params.radius_mean + params.radius_jitter +
        edge_softness;
    const max_outer_radius2 = max_outer_radius * max_outer_radius;

    var coverage: F = 0.0;
    neighbor_loop: for (neighbor_offsets, min_delta_y) |offset_y, min_dy| {
        for (neighbor_offsets, min_delta_x) |offset_x, min_dx| {
            const min_distance2 = min_dx * min_dx + min_dy * min_dy;
            if (min_distance2 > max_outer_radius2) continue;

            const candidate_x = cell_x + offset_x;
            const candidate_y = cell_y + offset_y;
            const hash = hashSpeckleCell(candidate_x, candidate_y, params.seed);
            if (params.occupancy < 1.0 and
                randomUnitFromHash(hash, 0) >= params.occupancy)
            {
                continue;
            }

            const disk = speckleDiskFromHash(
                candidate_x,
                candidate_y,
                hash,
                params,
            );
            const delta_x = proc_x - disk.center[0];
            const delta_y = proc_y - disk.center[1];
            const distance2 = delta_x * delta_x + delta_y * delta_y;
            coverage = @max(
                coverage,
                speckleDiskMask(distance2, disk.radius, edge_softness),
            );
            if (coverage == 1.0) break :neighbor_loop;
        }
    }

    return params.background + coverage * (params.foreground - params.background);
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
        .speckle => evalSpeckle2D(
            .{ coord.coord_0, coord.coord_1 },
            params.settings.speckle,
        ),
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
        .speckle => unreachable,
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

test "procedural speckle hash has stable known vectors" {
    try testing.expectEqual(@as(u64, 0x42cc592e95069169), hashSpeckleCell(0, 0, 0));
    try testing.expectEqual(@as(u64, 0xe4214bce0919ce5d), hashSpeckleCell(17, 29, 12345));
    try testing.expectEqual(
        @as(u64, 0xcbba3e407b1fa232),
        hashSpeckleCell(-7, -11, 0xa511e9b3),
    );
    try testing.expectEqual(
        @as(u64, 0x833cd6c57d01169f),
        hashSpeckleCell(-1, 5, std.math.maxInt(u32)),
    );
}

test "procedural speckle parameters validate shape-specific settings" {
    try (Speckle2DParams{}).validate();

    var invalid = Speckle2DParams{};
    if (comptime speckle_shape == .perlin) {
        invalid.perlin_coverage_transition_width = -0.01;
        try testing.expectError(
            error.InvalidSpecklePerlinCoverageTransitionWidth,
            invalid.validate(),
        );
        invalid = Speckle2DParams{};
        invalid.occupancy = -1.0;
        invalid.radius_mean = -1.0;
        invalid.radius_jitter = -1.0;
        invalid.edge_softness = -1.0;
        try invalid.validate();
    } else {
        invalid.radius_jitter = invalid.radius_mean + 0.01;
        try testing.expectError(error.InvalidSpeckleRadiusRange, invalid.validate());

        invalid = Speckle2DParams{};
        invalid.radius_mean = 0.9;
        invalid.radius_jitter = 0.15;
        invalid.edge_softness = 0.1;
        try testing.expectError(
            error.InvalidSpeckleNeighborhoodRadius,
            invalid.validate(),
        );
    }
}

test "procedural speckle shape has bounded support" {
    if (comptime speckle_shape == .gaussian) {
        try testing.expectEqual(@as(F, 1.0), speckleDiskMask(0.0, 0.5, 0.0));
        const transition = speckleDiskMask(0.0625, 0.5, 0.0);
        try testing.expect(transition > 0.0);
        try testing.expect(transition < 1.0);
        try testing.expectEqual(@as(F, 0.0), speckleDiskMask(0.25, 0.5, 0.0));
    } else if (comptime speckle_shape == .disk) {
        try testing.expectEqual(@as(F, 1.0), speckleDiskMask(0.25, 0.5, 0.0));
        try testing.expectEqual(@as(F, 0.0), speckleDiskMask(0.251, 0.5, 0.0));

        const transition = speckleDiskMask(0.25, 0.5, 0.1);
        if (comptime speckle_boundary_blur) {
            try testing.expect(transition > 0.0);
            try testing.expect(transition < 1.0);
        } else {
            try testing.expectEqual(@as(F, 1.0), transition);
        }
    } else {
        const params = Speckle2DParams{};
        try testing.expectEqual(@as(F, 0.0), perlinCoverage(-1.0, params));
        try testing.expectEqual(@as(F, 1.0), perlinCoverage(1.0, params));
        const transition = perlinCoverage(params.perlin_coverage_threshold, params);
        try testing.expect(transition > 0.0 and transition < 1.0);
    }
}

test "procedural speckle is deterministic bounded and UV clamped" {
    var params = Speckle2DParams{};
    params.cells_per_uv = .{ 12.0, 10.0 };

    const value = evalSpeckle2D(.{ 0.37, 0.61 }, params);
    try testing.expectEqual(value, evalSpeckle2D(.{ 0.37, 0.61 }, params));
    try testing.expectEqual(
        evalSpeckle2D(.{ 0.0, 1.0 }, params),
        evalSpeckle2D(.{ -2.0, 3.0 }, params),
    );

    var seed_changed = params;
    seed_changed.seed +%= 1;
    var found_seed_difference = false;
    for (0..16) |yy| {
        for (0..16) |xx| {
            const uv = [2]F{
                @as(F, @floatFromInt(xx)) / 15.0,
                @as(F, @floatFromInt(yy)) / 15.0,
            };
            const sample = evalSpeckle2D(uv, params);
            try testing.expect(sample >= @min(params.foreground, params.background));
            try testing.expect(sample <= @max(params.foreground, params.background));
            if (sample != evalSpeckle2D(uv, seed_changed)) {
                found_seed_difference = true;
            }
        }
    }
    try testing.expect(found_seed_difference);
}

test "procedural speckle SIMD fallback matches scalar evaluation" {
    var speckle_params = Speckle2DParams{};
    speckle_params.cells_per_uv = .{ 11.0, 9.0 };
    speckle_params.seed = 42;
    const params = speckle_params.toFuncShaderParams();

    var coords_0: [S]F = undefined;
    var coords_1: [S]F = undefined;
    for (0..S) |lane| {
        coords_0[lane] = @as(F, @floatFromInt(lane)) / @as(F, @floatFromInt(S));
        coords_1[lane] = 1.0 - coords_0[lane];
    }
    const coord_simd = FuncCoordSIMD{
        .coord_0 = coords_0,
        .coord_1 = coords_1,
        .normal_x = @splat(0.0),
        .normal_y = @splat(0.0),
        .normal_z = @splat(1.0),
    };
    const values_simd: [S]F = simd_impl.evalFuncShaderGreyNormSIMD(
        .speckle,
        coord_simd,
        params,
    );

    for (0..S) |lane| {
        const expected = evalFuncShaderBuiltinGreyNorm(
            .speckle,
            .{
                .coord_0 = coords_0[lane],
                .coord_1 = coords_1[lane],
                .normal_x = 0.0,
                .normal_y = 0.0,
                .normal_z = 1.0,
            },
            params,
        );
        try testing.expectEqual(expected, values_simd[lane]);
    }
}

test "procedural speckle occupancy endpoints behave exactly" {
    if (comptime speckle_shape == .perlin) return;
    var params = Speckle2DParams{};
    params.cells_per_uv = .{ 8.0, 8.0 };
    params.occupancy = 0.0;
    for (0..8) |yy| {
        for (0..8) |xx| {
            const uv = [2]F{
                @as(F, @floatFromInt(xx)) / 7.0,
                @as(F, @floatFromInt(yy)) / 7.0,
            };
            try testing.expectEqual(params.background, evalSpeckle2D(uv, params));
        }
    }

    params.occupancy = 1.0;
    var found_speckle = false;
    for (0..16) |yy| {
        for (0..16) |xx| {
            const uv = [2]F{
                @as(F, @floatFromInt(xx)) / 15.0,
                @as(F, @floatFromInt(yy)) / 15.0,
            };
            if (evalSpeckle2D(uv, params) < params.background) {
                found_speckle = true;
            }
        }
    }
    try testing.expect(found_speckle);
}

test "generated speckle list validates before coordinate conversion" {
    var params = Speckle2DParams{};
    params.uv_offset[0] = std.math.nan(F);
    try testing.expectError(
        error.InvalidSpeckleUVOffset,
        generateSpeckleList2D(testing.allocator, params),
    );
}

test "generated speckle list matches cell hash evaluation" {
    if (comptime speckle_shape == .perlin) return;
    var params = Speckle2DParams{};
    params.cells_per_uv = .{ 4.0, 3.0 };
    params.uv_offset = .{ -0.25, 0.4 };
    params.occupancy = 0.7;
    const speckles = try generateSpeckleList2D(testing.allocator, params);
    defer testing.allocator.free(speckles.disk_by_cell);
    defer testing.allocator.free(speckles.disks);

    for (0..9) |yy| {
        for (0..9) |xx| {
            const uv = [2]F{
                @as(F, @floatFromInt(xx)) / 8.0,
                @as(F, @floatFromInt(yy)) / 8.0,
            };
            const expected = evalSpeckle2D(uv, params);
            const actual = switch (comptime buildconfig.speckle_evaluator) {
                .cell_hash, .list_naive => evalSpeckleList2DNaive(uv, speckles),
                .list_indexed, .mask_1bit, .mask_u8 => evalSpeckleList2DIndexed(uv, speckles),
            };
            try testing.expectEqual(expected, actual);
        }
    }
}

test "procedural speckle exact fast paths preserve endpoint behavior" {
    if (comptime speckle_shape == .disk) {
        try testing.expectEqual(@as(F, 1.0), speckleDiskMask(0.25, 0.5, 0.0));
        try testing.expectEqual(@as(F, 0.0), speckleDiskMask(0.2501, 0.5, 0.0));
    }

    var params = Speckle2DParams{};
    params.foreground = 0.375;
    params.background = params.foreground;
    try testing.expectEqual(params.background, evalSpeckle2D(.{ 0.37, 0.61 }, params));
}

fn speckleMaskTestParams() Speckle2DParams {
    return .{
        .seed = 0x85ebca6b,
        .cells_per_uv = .{ 3.25, 2.5 },
        .uv_offset = .{ 0.37, -0.42 },
        .occupancy = 0.72,
        .radius_mean = 0.38,
        .radius_jitter = 0.07,
        .foreground = 0.15,
        .background = 0.85,
    };
}

test "Perlin mask is deterministic varied bounded and balanced" {
    if (comptime speckle_shape != .perlin) return;

    var params = Speckle2DParams{
        .seed = 0x85ebca6b,
        .cells_per_uv = .{ 7.25, 5.5 },
        .uv_offset = .{ -2.375, -0.625 },
        .foreground = 0.15,
        .background = 0.85,
    };
    params.occupancy = -1.0;
    params.radius_mean = -1.0;
    params.radius_jitter = -1.0;
    params.edge_softness = -1.0;

    const first = try generateSpeckleMask2D(testing.allocator, params);
    defer testing.allocator.free(first.bits);
    const repeated = try generateSpeckleMask2D(testing.allocator, params);
    defer testing.allocator.free(repeated.bits);
    try testing.expectEqualSlices(u8, first.bits, repeated.bits);

    params.seed +%= 1;
    const changed = try generateSpeckleMask2D(testing.allocator, params);
    defer testing.allocator.free(changed.bits);
    try testing.expect(!std.mem.eql(u8, first.bits, changed.bits));

    var min_value: u8 = std.math.maxInt(u8);
    var max_value: u8 = 0;
    var has_intermediate = false;
    var sum: u64 = 0;
    for (first.bits) |value| {
        min_value = @min(min_value, value);
        max_value = @max(max_value, value);
        has_intermediate = has_intermediate or (value > 0 and value < 255);
        sum += value;
    }
    const mean_coverage = @as(F, @floatFromInt(sum)) /
        (@as(F, @floatFromInt(first.bits.len)) * 255.0);
    try testing.expect(min_value < 32 and max_value > 223);
    try testing.expect(has_intermediate);
    try testing.expect(mean_coverage > 0.3 and mean_coverage < 0.7);

    for ([_][2]F{ .{ 0.0, 0.0 }, .{ 0.37, 0.61 }, .{ 1.0, 1.0 } }) |uv| {
        const value = evalSpeckleMask2D(uv, first);
        try testing.expect(value >= @min(params.foreground, params.background));
        try testing.expect(value <= @max(params.foreground, params.background));
    }
    try testing.expectEqual(
        evalSpeckleMask2D(.{ 0.0, 1.0 }, first),
        evalSpeckleMask2D(.{ -2.0, 3.0 }, first),
    );
}

test "direct 1-bit speckle mask preserves list-derived bytes and lattice values" {
    if (comptime buildconfig.speckle_evaluator != .mask_1bit) return;

    const params = speckleMaskTestParams();
    const speckles = try generateSpeckleList2D(testing.allocator, params);
    defer testing.allocator.free(speckles.disk_by_cell);
    defer testing.allocator.free(speckles.disks);

    var mask_storage: [1200]u8 = undefined;
    var fixed = std.heap.FixedBufferAllocator.init(&mask_storage);
    const mask = try generateSpeckleMask2D(fixed.allocator(), params);
    const expected_byte_count = mask.row_stride * mask.dims[1];
    try testing.expectEqual(expected_byte_count, mask.bits.len);
    if (comptime buildconfig.speckle_mask_samples_per_cell == 12) {
        try testing.expectEqual(@as(usize, 155), mask.bits.len);
    }

    const list_bits = try testing.allocator.alloc(u8, mask.bits.len);
    defer testing.allocator.free(list_bits);
    @memset(list_bits, 0);
    for (speckles.disks) |disk| {
        rasterizeSpeckleMaskDisk(
            list_bits,
            mask.row_stride,
            mask.uv_to_texel,
            params,
            disk,
        );
    }
    try testing.expectEqualSlices(u8, list_bits, mask.bits);

    var found_foreground = false;
    var found_background = false;
    for (0..mask.dims[1]) |yy| {
        for (0..mask.dims[0]) |xx| {
            const uv = [2]F{
                @as(F, @floatFromInt(xx)) / mask.uv_to_texel[0],
                @as(F, @floatFromInt(yy)) / mask.uv_to_texel[1],
            };
            const value = evalSpeckleMask2D(uv, mask);
            try testing.expectApproxEqAbs(
                evalSpeckleList2DIndexed(uv, speckles),
                value,
                unit_tol,
            );
            found_foreground = found_foreground or value == mask.params.foreground;
            found_background = found_background or value == mask.params.background;
        }
    }
    try testing.expect(found_foreground and found_background);
}

test "speckle mask handles degenerate and oversized inputs" {
    if (comptime buildconfig.speckle_evaluator != .mask_1bit and
        buildconfig.speckle_evaluator != .mask_u8) return;

    var params = speckleMaskTestParams();
    if (comptime speckle_shape != .perlin) {
        params.occupancy = 0.0;
        const empty = try generateSpeckleMask2D(testing.allocator, params);
        defer testing.allocator.free(empty.bits);
        try testing.expectEqual(@as(usize, 0), empty.bits.len);
        try testing.expectEqual(
            params.background,
            evalSpeckleMask2D(.{ 0.37, 0.61 }, empty),
        );
    }

    params.occupancy = 1.0;
    params.foreground = params.background;
    const equal_colour = try generateSpeckleMask2D(testing.allocator, params);
    defer testing.allocator.free(equal_colour.bits);
    try testing.expectEqual(@as(usize, 0), equal_colour.bits.len);
    try testing.expectEqual(
        params.background,
        evalSpeckleMask2D(.{ 0.37, 0.61 }, equal_colour),
    );

    params = speckleMaskTestParams();
    if (comptime buildconfig.speckle_evaluator == .mask_u8) {
        params.cells_per_uv = .{ 3000.0, 3000.0 };
        try testing.expectError(
            error.SpeckleMaskTooLarge,
            generateSpeckleMask2D(testing.allocator, params),
        );
    }
    params.cells_per_uv = .{ 4000.0, 4000.0 };
    try testing.expectError(
        error.SpeckleMaskTooLarge,
        generateSpeckleMask2D(testing.allocator, params),
    );
}

test "speckle mask clamps UV endpoints" {
    if (comptime buildconfig.speckle_evaluator != .mask_1bit and
        buildconfig.speckle_evaluator != .mask_u8) return;

    const mask = try generateSpeckleMask2D(testing.allocator, speckleMaskTestParams());
    defer testing.allocator.free(mask.bits);

    try testing.expectEqual(
        evalSpeckleMask2D(.{ 0.0, 1.0 }, mask),
        evalSpeckleMask2D(.{ -2.0, 3.0 }, mask),
    );
    try testing.expectEqual(
        evalSpeckleMask2D(.{ 1.0, 0.0 }, mask),
        evalSpeckleMask2D(.{ 3.0, -2.0 }, mask),
    );
    try testing.expectEqual(
        mask.params.background,
        evalSpeckleMask2D(.{ std.math.nan(F), 0.5 }, mask),
    );
    try testing.expectEqual(
        mask.params.background,
        evalSpeckleMask2D(.{ 0.5, std.math.inf(F) }, mask),
    );
}

test "u8 speckle mask agrees with quantized indexed lattice evaluation" {
    if (comptime buildconfig.speckle_evaluator != .mask_u8) return;

    const params = speckleMaskTestParams();
    const speckles = try generateSpeckleList2D(testing.allocator, params);
    defer testing.allocator.free(speckles.disk_by_cell);
    defer testing.allocator.free(speckles.disks);
    const mask = try generateSpeckleMask2D(testing.allocator, params);
    defer testing.allocator.free(mask.bits);

    try testing.expectEqual(@as(u8, 0), quantizeSpeckleCoverage(-1.0));
    try testing.expectEqual(@as(u8, 0), quantizeSpeckleCoverage(0.0));
    try testing.expectEqual(@as(u8, 1), quantizeSpeckleCoverage(1.0 / 255.0));
    try testing.expectEqual(@as(u8, 128), quantizeSpeckleCoverage(0.5));
    try testing.expectEqual(@as(u8, 254), quantizeSpeckleCoverage(254.0 / 255.0));
    try testing.expectEqual(@as(u8, 255), quantizeSpeckleCoverage(1.0));
    try testing.expectEqual(@as(u8, 255), quantizeSpeckleCoverage(2.0));
    try testing.expectEqual(mask.dims[0], mask.row_stride);
    try testing.expectEqual(mask.dims[0] * mask.dims[1], mask.bits.len);
    var found_intermediate = false;
    for (0..mask.dims[1]) |yy| {
        for (0..mask.dims[0]) |xx| {
            const uv = [2]F{
                @as(F, @floatFromInt(xx)) / mask.uv_to_texel[0],
                @as(F, @floatFromInt(yy)) / mask.uv_to_texel[1],
            };
            const analytic = if (comptime speckle_shape == .perlin)
                evalPerlinSpeckle2D(uv, params)
            else
                evalSpeckleList2DIndexed(uv, speckles);
            const analytic_coverage = (analytic - params.background) /
                (params.foreground - params.background);
            const expected_byte = quantizeSpeckleCoverage(analytic_coverage);
            const actual_byte = mask.bits[yy * mask.row_stride + xx];
            const byte_diff = if (actual_byte > expected_byte)
                actual_byte - expected_byte
            else
                expected_byte - actual_byte;
            try testing.expect(byte_diff <= 1);

            const coverage = @as(F, @floatFromInt(actual_byte)) / 255.0;
            const expected = params.background +
                coverage * (params.foreground - params.background);
            try testing.expectEqual(expected, evalSpeckleMask2D(uv, mask));
            found_intermediate = found_intermediate or
                (actual_byte != 0 and actual_byte != 255);
        }
    }
    if (comptime speckle_shape == .gaussian or speckle_boundary_blur) {
        try testing.expect(found_intermediate);
    }
}
