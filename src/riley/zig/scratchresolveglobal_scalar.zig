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
    target: *const subpxframe.SubpxTarget,
    camera: *const cam.CameraPrepared,
    background_value: F,
    image_out_arr: *ndarray.NDArray(F),
) void {
    common.resolve(target, camera, background_value, image_out_arr);
}

pub fn resolveRows(
    target: *const subpxframe.SubpxTarget,
    camera: *const cam.CameraPrepared,
    background_value: F,
    image_out_arr: *ndarray.NDArray(F),
    image_y_min: usize,
    image_y_max: usize,
) void {
    common.resolveRows(
        target,
        camera,
        background_value,
        image_out_arr,
        image_y_min,
        image_y_max,
    );
}
