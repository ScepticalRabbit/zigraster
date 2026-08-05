// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");
const root = @import("root");

const build_options = if (@hasDecl(root, "build_options"))
    root.build_options
else
    struct {
        pub const precision = "f64";
        pub const simd = "on";
        pub const newton_solver = "fast";
        pub const simd_vec_width: comptime_int = 0;
        pub const simd_vector_width: comptime_int = 0;
    };

pub const comptime_eval_branch_quota: comptime_int = 50000;

// --------------------------------------------------------------------------------------
// Main Config
// --------------------------------------------------------------------------------------

pub const Config = struct {
    simd: SimdMode = default_simd,
    newton_solver_mode: NewtonSolverMode = default_newton_solver_mode,
    simd_vec_width: comptime_int = defaultSimdVecWidthForPrecision(F),
    max_nodal_fields: comptime_int = 8,
    max_image_channels: comptime_int = 8,
    raster_newton_iter_max: comptime_int = 10,
    distortion_newton_iter_max: comptime_int = 15,
    interp_lut_size: comptime_int = 1024,
    save_frame_buff_count: comptime_int = 3,
    precision: type = F,
    tol: Tol = defaultTolForPrecision(F),
};

pub const default_precision = parsePrecision(build_options.precision);
pub const F = default_precision;
pub const Scal = F;
pub const Scalar = Scal;

pub const default_simd = parseSimd(build_options.simd);
pub const default_newton_solver_mode =
    parseNewtonSolverMode(build_options.newton_solver);

pub const config = configForPrecision(F);

pub const SimdWidth = config.simd_vec_width;
pub const SaveFrameBuffCount = config.save_frame_buff_count;
pub const NewtonMode = config.newton_solver_mode;

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const SimdMode = enum {
    off,
    on,
};

pub const SimdTexInterpMode = enum {
    inner,
    over_pixels,
};

pub const NewtonSolverMode = enum {
    fast,
    robust,
};

pub const TexSIMDPolicy = struct {
    pub fn resolve(
        comptime channels: comptime_int,
        comptime is_linear: bool,
        comptime uses_lut: bool,
    ) SimdTexInterpMode {
        if (is_linear) {
            return .over_pixels;
        }
        if (channels == 1) {
            if (uses_lut) {
                return .inner;
            }
            return .over_pixels;
        }
        return .inner;
    }
};

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn defaultSimdVecWidthForPrecision(comptime precision: type) comptime_int {
    const opt_width = comptime buildOptionsSimdVecWidth();
    if (opt_width > 0) {
        return opt_width;
    }
    return switch (precision) {
        f32 => 16,
        f64 => 8,
        else => @compileError("Only f32 and f64 precision are supped."),
    };
}

pub fn defaultTolForPrecision(comptime precision: type) Tol {
    return switch (precision) {
        f32 => tol_f32,
        f64 => tol_f64,
        else => @compileError("Only f32 and f64 precision are supped."),
    };
}

pub fn configForPrecision(comptime precision: type) Config {
    _ = defaultSimdVecWidthForPrecision(precision);
    return .{
        .precision = precision,
        .newton_solver_mode = default_newton_solver_mode,
        .simd_vec_width = defaultSimdVecWidthForPrecision(precision),
        .tol = defaultTolForPrecision(precision),
    };
}

fn parsePrecision(comptime precision: []const u8) type {
    if (std.mem.eql(u8, precision, "f32")) {
        return f32;
    }
    if (std.mem.eql(u8, precision, "f64")) {
        return f64;
    }
    @compileError("build_options.precision must be \"f32\" or \"f64\".");
}

fn buildOptionsSimdVecWidth() comptime_int {
    if (@hasDecl(build_options, "simd_vec_width")) {
        return build_options.simd_vec_width;
    }
    if (@hasDecl(build_options, "simd_vector_width")) {
        return build_options.simd_vector_width;
    }
    return 0;
}

fn parseSimd(comptime simd: []const u8) SimdMode {
    if (std.mem.eql(u8, simd, "on")) {
        return .on;
    }
    if (std.mem.eql(u8, simd, "off")) {
        return .off;
    }
    @compileError("build_options.simd must be \"on\" or \"off\".");
}

fn parseNewtonSolverMode(comptime newton_solver: []const u8) NewtonSolverMode {
    if (std.mem.eql(u8, newton_solver, "fast")) {
        return .fast;
    }
    if (std.mem.eql(u8, newton_solver, "robust")) {
        return .robust;
    }
    @compileError(
        "build_options.newton_solver must be \"fast\" or \"robust\".",
    );
}

// --------------------------------------------------------------------------------------
// Major Internal Types Shared Across The File
// --------------------------------------------------------------------------------------

pub const EdgeTol = struct {
    tri_weight_inclusion: Scal = 1e-9,
    simd_raster_weight_inclusion: Scal = 1e-9,
};

pub const HullTol = struct {
    scal_inclusion: Scal = 1.0e-6,
    simd_inclusion: Scal = 1.0e-6,
    corner_midside_ang_lower_deg: Scal = 20.0,
    corner_midside_ang_upper_deg: Scal = 180.0,
    // Node-only front-end AABB pad used when RasterConfig.hull_mode == .off.
    // Set from a hull-suite ablation study against the 0.3 baseline:
    // 0.25/0.20/0.15/0.10 passed and 0.05 failed, so we keep 0.15 as a
    // conservative def with margin.
    no_hull_bbox_rel_pad: Scal = 0.15,
};

pub const CullingTol = struct {
    higher_order_backface_nz: Scal = 1e-3,
    tri3_signed_area: Scal = 1e-6,
    projective_z_min: Scal = 1e-12,
};

pub const NormalTol = struct {
    normalise_magnitude: Scal = 1e-12,
};

pub const NewtonTol = struct {
    resid: Scal = 1e-8,
    norm_resid: Scal = 3e-11,
    stagnation_norm_resid: Scal = 1e-10,
    rel_det: Scal = 1e-12,
    para_dom: Scal = 1e-7,
    para_step_abs: Scal = 1e-12,
    para_step_rel: Scal = 1e-12,
    max_para_step: Scal = 0.5,
};

pub const DistortionTol = struct {
    resid: Scal = 1e-10,
    delta: Scal = 1e-10,
    det: Scal = 1e-12,
};

pub const NewtonSeedTol = struct {
    det: Scal = 1e-10,
    para_dom: Scal = 1e-5,
    resid_sq: Scal = 1e-4,
};

pub const GeometryTol = struct {
    bilinear_para_dom: Scal = 1e-8,
    bilinear_denom: Scal = 1e-12,
    quadratic_area: Scal = 1e-12,
    depth_buff_inv_z_cmp: Scal = 1e-12,
};

pub const TexTol = struct {
    lancsoz_centre_snap: Scal = 1e-6,
    samp_coeff_sum: Scal = 1e-9,
};

pub const PSFTol = struct {
    supp_radius_inclusion: Scal = 1e-12,
    pixel_box_identity_supp_radius: Scal = 1e-12,
    anisotropic_axis_align: Scal = 1e-12,
};

pub const ImageTol = struct {
    auto_scale_range: Scal = 1e-9,
};

pub const LegacyTol = struct {
    oldraster_area: Scal = 1e-12,
};

pub const Tol = struct {
    edge: EdgeTol = .{},
    hull: HullTol = .{},
    culling: CullingTol = .{},
    normals: NormalTol = .{},
    newton: NewtonTol = .{},
    distortion: DistortionTol = .{},
    newton_seed: NewtonSeedTol = .{},
    geometry: GeometryTol = .{},
    tex: TexTol = .{},
    psf: PSFTol = .{},
    image: ImageTol = .{},
    legacy: LegacyTol = .{},
};

pub const tol_f64 = Tol{};

pub const tol_f32 = Tol{
    .edge = .{
        .tri_weight_inclusion = 1e-6,
        .simd_raster_weight_inclusion = 1e-6,
    },
    .hull = .{
        .scal_inclusion = 1.0e-5,
        .simd_inclusion = 1.0e-5,
        .corner_midside_ang_lower_deg = 20.0,
        .corner_midside_ang_upper_deg = 180.0,
        .no_hull_bbox_rel_pad = 0.15,
    },
    .culling = .{
        .higher_order_backface_nz = 1e-3,
        .tri3_signed_area = 1e-5,
        .projective_z_min = 1e-9,
    },
    .normals = .{
        .normalise_magnitude = 1e-8,
    },
    .newton = .{
        .resid = 1e-4,
        .norm_resid = 1.5e-5,
        .stagnation_norm_resid = 7.5e-5,
        .rel_det = 1e-6,
        .para_dom = 5e-5,
        .para_step_abs = 2e-6,
        .para_step_rel = 2e-6,
        .max_para_step = 0.5,
    },
    .distortion = .{
        .resid = 1e-5,
        .delta = 1e-5,
        .det = 1e-7,
    },
    .newton_seed = .{
        .det = 1e-7,
        .para_dom = 2e-4,
        .resid_sq = 5e-3,
    },
    .geometry = .{
        .bilinear_para_dom = 5e-5,
        .bilinear_denom = 1e-7,
        .quadratic_area = 1e-8,
        .depth_buff_inv_z_cmp = 1e-8,
    },
    .tex = .{
        .lancsoz_centre_snap = 1e-6,
        .samp_coeff_sum = 1e-6,
    },
    .psf = .{
        .supp_radius_inclusion = 1e-6,
        .pixel_box_identity_supp_radius = 1e-6,
        .anisotropic_axis_align = 1e-6,
    },
    .image = .{
        .auto_scale_range = 1e-8,
    },
    .legacy = .{
        .oldraster_area = 1e-8,
    },
};

// --------------------------------------------------------------------------------------
// Generic Low-Level Helpers
// --------------------------------------------------------------------------------------

pub const VecSF = @Vector(SimdWidth, F);
pub const VecSU = @Vector(SimdWidth, usize);
pub const VecSI = @Vector(SimdWidth, isize);
pub const VecSB = @Vector(SimdWidth, bool);
pub const VecSU8 = @Vector(SimdWidth, u8);

pub const Tri3FixedConfig = switch (F) {
    f32 => struct {
        pub const Coord = i32;
        pub const Setup = i64;
        pub const Edge = i64;
        pub const frac_bits: comptime_int = 12;
    },
    f64 => struct {
        pub const Coord = i64;
        pub const Setup = i128;
        pub const Edge = i64;
        pub const frac_bits: comptime_int = 16;
    },
    else => @compileError("Unsupped Riley floating-point type."),
};

pub const Tri3FixedCoord = Tri3FixedConfig.Coord;
pub const Tri3FixedSetup = Tri3FixedConfig.Setup;
pub const Tri3FixedEdge = Tri3FixedConfig.Edge;
pub const Tri3FixedFracBits = Tri3FixedConfig.frac_bits;
pub const Tri3FixedOne: Tri3FixedEdge = @as(Tri3FixedEdge, 1) << Tri3FixedFracBits;
pub const VecSTri3FixedEdge = @Vector(SimdWidth, Tri3FixedEdge);

comptime {
    if (Tri3FixedFracBits <= 0) {
        @compileError("tri3 fixed-point precision must be positive.");
    }
    if (@bitSizeOf(Tri3FixedEdge) < @bitSizeOf(Tri3FixedCoord)) {
        @compileError(
            "tri3 edge type must not be narrower than coordinate type.",
        );
    }
}

// --------------------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------------------

test "TexSIMDPolicy resolves tex cases" {
    const expectEqual = std.testing.expectEqual;

    try expectEqual(
        SimdTexInterpMode.over_pixels,
        TexSIMDPolicy.resolve(1, true, false),
    );
    try expectEqual(
        SimdTexInterpMode.inner,
        TexSIMDPolicy.resolve(1, false, true),
    );
    try expectEqual(
        SimdTexInterpMode.inner,
        TexSIMDPolicy.resolve(1, false, true),
    );
    try expectEqual(
        SimdTexInterpMode.over_pixels,
        TexSIMDPolicy.resolve(1, false, false),
    );
    try expectEqual(
        SimdTexInterpMode.over_pixels,
        TexSIMDPolicy.resolve(3, true, false),
    );
    try expectEqual(
        SimdTexInterpMode.inner,
        TexSIMDPolicy.resolve(3, false, true),
    );
    try expectEqual(
        SimdTexInterpMode.inner,
        TexSIMDPolicy.resolve(3, false, false),
    );
}
