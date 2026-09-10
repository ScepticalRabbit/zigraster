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
const camera = @import("../riley/zig/camera.zig");
const cameraops = @import("../riley/zig/cameraops.zig");
const common = @import("../dev_support/tests.zig");
const gengold_twoshapes = @import("../gengold/gen_gold_twoshapes.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const iio = @import("../riley/zig/imageio.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const orch = @import("../dev_support/orchestration.zig");
const policy = @import("../dev_support/testpolicy.zig");
const rastcfg = @import("../riley/zig/rasterconfig.zig");
const riley = @import("../riley/zig/riley.zig");
const Rotation = @import("../riley/zig/rotation.zig").Rotation;
const sceneops = @import("../riley/zig/sceneops.zig");
const tcfg = @import("../dev_support/testconfig.zig");
const texops = @import("../riley/zig/textureops.zig");

const F = buildconfig.F;
const CameraInput = camera.CameraInput;
const MeshInput = mo.MeshInput;
const Timestamp = std.Io.Clock.Timestamp;
const TwoShapesShaderKind = gengold_twoshapes.TwoShapesShaderKind;

const pixel_num_twoshapes = [_]u32{ 160, 100 };
const pixel_size_twoshapes = [2]F{ @floatCast(5.3e-6), @floatCast(5.3e-6) };
const focal_length_twoshapes: F = @floatCast(50.0e-3);

pub fn runTwoShapesCaseTest(
    allocator: std.mem.Allocator,
    io: std.Io,
    mesh_type: gk.MeshType,
    shader_kind: TwoShapesShaderKind,
    texture_u8_grey: texops.Tex(u8, 1),
    texture_u8_rgb: texops.Tex(u8, 3),
    texture_u16_grey: texops.Tex(u16, 1),
    texture_f64_grey: texops.Tex(F, 1),
    gold_dir_root: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const meshes = try gengold_twoshapes.buildTwoShapesScene(
        aa,
        io,
        mesh_type,
        shader_kind,
        texture_u8_grey,
        texture_u8_rgb,
        texture_u16_grey,
        texture_f64_grey,
    );

    const case_dir_name = try std.fmt.allocPrint(
        aa,
        "twoshapes_{s}_{s}",
        .{ @tagName(mesh_type), @tagName(shader_kind) },
    );

    const gold_dir = try std.fmt.allocPrint(
        aa,
        "{s}/{s}",
        .{ gold_dir_root, case_dir_name },
    );

    const target = sceneops.boundsCenterOverMeshes(meshes);
    const rot = Rotation.init(
        0,
        std.math.degreesToRadians(20.0),
        std.math.degreesToRadians(-20.0),
    );
    const pos = cameraops.posFillFrameFromRotOverMeshesAndTarg(
        meshes,
        target,
        pixel_num_twoshapes,
        pixel_size_twoshapes,
        focal_length_twoshapes,
        rot,
        0.9,
    );

    const camera_input = CameraInput{
        .pixels_num = pixel_num_twoshapes,
        .pixels_size = pixel_size_twoshapes,
        .pos_world = pos,
        .rot_world = rot,
        .roi_cent_world = target,
        .focal_length = focal_length_twoshapes,
        .sub_sample = 2,
        .distortion = .none,
    };

    var run_config = config;
    run_config.save_strategy = .memory;
    run_config.background_value = 127.5;

    const start_time = Timestamp.now(io, .awake);
    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), run_config.total_threads) },
    };

    const result = try riley.raster(
        aa,
        &render_groups,
        &[_]CameraInput{camera_input},
        meshes,
        run_config,
        null,
    );

    var render_result = result orelse return error.NoResult;
    defer aa.free(render_result.slice);

    const end_time = Timestamp.now(io, .awake);
    const duration_ms = @as(
        F,
        @floatFromInt(start_time.durationTo(end_time).raw.nanoseconds),
    ) / 1e6;

    const frames_num = if (render_result.dims.len == 5)
        render_result.dims[1]
    else
        render_result.dims[0];
    const is_rgb = shader_kind.isRgb();
    const channels_num: usize = if (is_rgb) 3 else 1;

    for (0..frames_num) |ff| {
        for (0..channels_num) |ch| {
            const gold_path = try common.findGoldPath(
                aa,
                io,
                gold_dir,
                0,
                ff,
                ch,
                false,
            );

            common.compareNDArrayToGold(
                aa,
                io,
                &render_result,
                0,
                ff,
                ch,
                1,
                gold_path,
                tcfg.REL_TOL,
                tcfg.ABS_TOL,
            ) catch |err| {
                if (tcfg.TEST_CASE_VERBOSE) {
                    std.debug.print(
                        "FAIL {s} frame {d} ch {d} ({d:.2} ms)\n",
                        .{ case_dir_name, ff, ch, duration_ms },
                    );
                }
                return err;
            };
        }
    }

    if (tcfg.TEST_CASE_VERBOSE) {
        std.debug.print(
            "PASS {s} ({d:.2} ms, {d} frames)\n",
            .{ case_dir_name, duration_ms, frames_num },
        );
    }
}

pub fn run(allocator: std.mem.Allocator, io: std.Io) !void {
    const config = tcfg.getRasterConfig(.testing);
    const gold_dir_root = policy.goldRoot(.basic);

    const texture_u8_grey = try iio.loadImage(
        u8,
        1,
        allocator,
        io,
        "texture/speck128_mono_u8.bmp",
        .bmp,
    );
    defer texture_u8_grey.deinit(allocator);

    const texture_u8_rgb = try iio.loadImage(
        u8,
        3,
        allocator,
        io,
        "texture/speck128_rgb_u8.bmp",
        .bmp,
    );
    defer texture_u8_rgb.deinit(allocator);

    const texture_u16_grey = try iio.loadImage(
        u16,
        1,
        allocator,
        io,
        "texture/speck128_mono_u16.tiff",
        .tiff,
    );
    defer texture_u16_grey.deinit(allocator);

    var texture_f64_grey = try gengold_twoshapes.convertU16TexToF64(
        allocator,
        texture_u16_grey,
    );
    defer texture_f64_grey.deinit(allocator);

    const mesh_types = [_]gk.MeshType{
        .tri3,
        .tri6,
        .quad4,
        .quad8,
        .quad9,
    };
    const shader_kinds = std.enums.values(TwoShapesShaderKind);

    for (mesh_types) |mesh_type| {
        for (shader_kinds) |shader_kind| {
            try runTwoShapesCaseTest(
                allocator,
                io,
                mesh_type,
                shader_kind,
                texture_u8_grey,
                texture_u8_rgb,
                texture_u16_grey,
                texture_f64_grey,
                gold_dir_root,
                config,
            );
        }
    }
}
