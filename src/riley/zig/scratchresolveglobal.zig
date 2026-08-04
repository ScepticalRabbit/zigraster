// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const cfg = @import("buildconfig.zig").config;
const std = @import("std");
const cam = @import("camera.zig");
const ndarray = @import("ndarray.zig");
const buildconfig = @import("buildconfig.zig");
const F = buildconfig.F;
const impl = if (cfg.simd == .on)
    @import("scratchresolveglobal_simd.zig")
else
    @import("scratchresolveglobal_scalar.zig");
const subpxframe = @import("subpxframe.zig");

pub fn resolve(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    target: *const subpxframe.SubpxTarget,
    camera: *const cam.CameraPrepared,
    background_value: F,
    image_out_arr: *ndarray.NDArray(F),
    requested_workers: u16,
) !usize {
    return impl.resolve(
        outer_alloc,
        io,
        target,
        camera,
        background_value,
        image_out_arr,
        requested_workers,
    );
}

pub fn resolveRows(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    target: *const subpxframe.SubpxTarget,
    camera: *const cam.CameraPrepared,
    background_value: F,
    image_out_arr: *ndarray.NDArray(F),
    image_y_min: usize,
    image_y_max: usize,
    requested_workers: u16,
) !usize {
    return impl.resolveRows(
        outer_alloc,
        io,
        target,
        camera,
        background_value,
        image_out_arr,
        image_y_min,
        image_y_max,
        requested_workers,
    );
}
