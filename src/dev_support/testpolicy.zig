// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const buildconfig = @import("../riley/zig/buildconfig.zig");
const gk = @import("../riley/zig/geometrykernels.zig");

const cfg = buildconfig.config;
const F = buildconfig.F;

pub const GoldModePolicy = enum {
    shared_by_precision,
    split_by_precision_and_simd,
};

pub const GoldSuite = enum {
    basic,
    full_shader,
    full_texture,
    full_dist_psf,
    full_ssaa_pxmap,
    full_hull,
    full_tiling,
    full_scene_camera_threads,
    full_image_output,
};

pub const MeshNameContext = enum {
    fixture_case,
    benchmark_data,
    gold_case,
    sphere_gold_case,
};

pub fn goldModePolicy(comptime suite: GoldSuite) GoldModePolicy {
    _ = suite;
    return .shared_by_precision;
}

pub fn suiteDirName(comptime suite: GoldSuite) []const u8 {
    return switch (suite) {
        .basic => "basic",
        .full_shader => "full_shader",
        .full_texture => "full_texture",
        .full_dist_psf => "full_dist_psf",
        .full_ssaa_pxmap => "full_ssaa_pxmap",
        .full_hull => "full_hull",
        .full_tiling => "full_tiling",
        .full_scene_camera_threads => "full_scene_camera_threads",
        .full_image_output => "full_image_output",
    };
}

pub fn goldRoot(comptime suite: GoldSuite) []const u8 {
    return switch (suite) {
        .basic => if (F == f64) "gold/basic" else "gold/basic_f32",
        .full_shader => if (F == f64)
            "gold/full_shader"
        else
            "gold/full_shader_f32",
        .full_texture => if (F == f64)
            "gold/full_texture"
        else
            "gold/full_texture_f32",
        .full_dist_psf => if (F == f64)
            "gold/full_dist_psf"
        else
            "gold/full_dist_psf_f32",
        .full_ssaa_pxmap => if (F == f64)
            "gold/full_ssaa_pxmap"
        else
            "gold/full_ssaa_pxmap_f32",
        .full_hull => if (F == f64)
            "gold/full_hull"
        else
            "gold/full_hull_f32",
        .full_tiling => if (F == f64)
            "gold/full_tiling"
        else
            "gold/full_tiling_f32",
        .full_scene_camera_threads => if (F == f64)
            "gold/full_scene_camera_threads"
        else
            "gold/full_scene_camera_threads_f32",
        .full_image_output => if (F == f64)
            "gold/full_image_output"
        else
            "gold/full_image_output_f32",
    };
}

pub fn canonicalCaseMeshType(mesh_type: gk.MeshType) gk.MeshType {
    return mesh_type;
}

pub fn sphereGoldCaseMeshType(mesh_type: gk.MeshType) gk.MeshType {
    return mesh_type;
}

pub fn meshName(
    comptime context: MeshNameContext,
    mesh_type: gk.MeshType,
) []const u8 {
    return switch (context) {
        .fixture_case => switch (mesh_type) {
            .tri3opt => "tri3",
            else => @tagName(mesh_type),
        },
        .benchmark_data => switch (mesh_type) {
            .tri3opt => "tri3",
            else => @tagName(mesh_type),
        },
        .gold_case => @tagName(canonicalCaseMeshType(mesh_type)),
        .sphere_gold_case => @tagName(sphereGoldCaseMeshType(mesh_type)),
    };
}

pub fn shouldSkipBenchGeomTest(mesh_type: gk.MeshType, case_name: []const u8) bool {
    _ = mesh_type;
    _ = case_name;
    return false;
    // return mesh_type == .tri3opt and
    //     F == f32 and
    //     buildconfig.config.simd == .on and
    //     std.mem.eql(u8, case_name, "geom");
}

pub fn benchGeomSkipWarning() []const u8 {
    return "Skipping bench geom tri3opt gold check for f32 SIMD only: " ++
        "the tri3opt path can still show tiny snapped-grid differences on " ++
        "very small geom elements.";
}
