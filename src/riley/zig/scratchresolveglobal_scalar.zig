// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const cam = @import("camera.zig");
const ndarray = @import("ndarray.zig");
const buildconfig = @import("buildconfig.zig");
const F = buildconfig.F;
const common = @import("scratchresolveglobal_common.zig");
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
    return common.resolveParallel(
        false,
        outer_alloc,
        io,
        target,
        camera,
        background_value,
        image_out_arr,
        0,
        target.domain.image_h_subpx / @as(usize, camera.sub_sample),
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
    return common.resolveParallel(
        false,
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
const std = @import("std");
