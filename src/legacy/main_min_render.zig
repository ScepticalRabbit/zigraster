// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const common = @import("dev_support/benchcommon.zig");
const gengold = @import("dev_support/gengold.zig");
const minsuite = @import("dev_support/minsuite.zig");
const tcfg = @import("dev_support/testconfig.zig");
const riley = @import("riley/zig/riley.zig");
const mo = @import("riley/zig/meshpipeline.zig");
const gk = @import("riley/zig/geometrykernels.zig");
const iio = @import("riley/zig/imageio.zig");
const texops = @import("riley/zig/textureops.zig");
const Rotation = @import("riley/zig/rotation.zig").Rotation;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const aa = arena.allocator();

    const texture_grey = try iio.loadImage(
        u8,
        1,
        aa,
        io,
        "texture/speckle.bmp",
        .bmp,
    );
    const texture_rgb = try iio.loadImage(
        u8,
        3,
        aa,
        io,
        "texture/speckle_rgb.bmp",
        .bmp,
    );

    const out_dir = "out/min";
    const pixel_num_sphere = [_]u32{ 160, 100 };
    const pixel_num_multi = [_]u32{ 640, 400 };
    const render_defaults_sphere = common.BenchRenderDefaults{
        .pixels_num = pixel_num_sphere,
        .sub_sample = 2,
        .focal_leng = 50.0e-3,
        .pixels_size = .{ 5.3e-6, 5.3e-6 },
        .fov_scale = 1.0,
        .rot = Rotation.init(0, 0, 0),
    };

    const mesh_types = comptime std.enums.values(gk.MeshType);
    const shader_types = [_]common.ShaderType{
        .nodal_grey,
        .nodal_rgb,
        .tex8_grey,
        .tex8_rgb,
    };
    const samp_cfgs = [_]texops.TextureSampleConfig{
        .{ .sample = .nearest, .mode = .direct },
        .{ .sample = .linear, .mode = .direct },
        .{ .sample = .cubic_catmull_rom, .mode = .direct },
        .{ .sample = .cubic_catmull_rom, .mode = .lut_lerp },
        .{ .sample = .cubic_mitchell_netravali, .mode = .lut_lerp },
        .{ .sample = .lanczos3, .mode = .lut_lerp },
        .{ .sample = .cubic_bspline, .mode = .lut_lerp },
        .{ .sample = .quintic_bspline, .mode = .direct },
        .{ .sample = .quintic_bspline, .mode = .lut_lerp },
    };

    var config = tcfg.getRasterConfig(.preview);
    config.save_strategy = .disk;
    config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .fimg, .bits = null, .scaling = .none },
        .{ .format = .bmp, .bits = 8, .scaling = .auto },
    };

    std.debug.print("Rendering MIN Suite (sphere200/base) to {s}...\n", .{out_dir});
    for (mesh_types) |mt| {
        for (shader_types) |st| {
            for (samp_cfgs) |sc| {
                const data_dir = try std.fmt.allocPrint(
                    aa,
                    "data/min/{s}_sphere200",
                    .{@tagName(mt)},
                );

                const is_rgb = (st == .tex8_rgb or st == .nodal_rgb);
                const is_allowed_rgb = (st == .nodal_rgb) or
                    (st == .tex8_rgb and sc.sample == .cubic_catmull_rom and sc.mode == .lut_lerp);

                if (is_rgb and !is_allowed_rgb) continue;

                if (common.shouldRun(
                    .{ .run = .all, .skip_quad4ibi_sphere = true },
                    mt,
                    st,
                    sc,
                    data_dir,
                )) {
                    var r_config = tcfg.getRasterConfig(.bench);
                    r_config.save_strategy = .disk;
                    const case_name = try minsuite.calcMinCaseName(
                        aa,
                        mt,
                        st,
                        sc,
                    );
                    const case_out_dir = try std.fs.path.join(
                        aa,
                        &[_][]const u8{
                            out_dir,
                            "sphere200",
                            "base",
                            case_name,
                        },
                    );
                    _ = try common.runBenchmarkQuietWithImageOut(
                        u8,
                        aa,
                        io,
                        mt,
                        st,
                        sc,
                        null,
                        data_dir,
                        render_defaults_sphere,
                        texture_grey,
                        texture_rgb,
                        r_config,
                        "",
                        case_out_dir,
                    );
                }
            }
        }
    }

    std.debug.print("Rendering MIN Suite (sphere200multicull) to {s}...\n", .{out_dir});
    for (mesh_types) |mt| {
        for (shader_types) |st| {
            for (samp_cfgs) |sc| {
                const data_dir = try std.fmt.allocPrint(
                    aa,
                    "data/min/{s}_sphere200",
                    .{@tagName(mt)},
                );

                const is_rgb = (st == .tex8_rgb or st == .nodal_rgb);
                const is_allowed_rgb = (st == .nodal_rgb) or
                    (st == .tex8_rgb and sc.sample == .cubic_catmull_rom and sc.mode == .lut_lerp);

                if (is_rgb and !is_allowed_rgb) continue;

                if (common.shouldRun(
                    .{ .run = .all, .skip_quad4ibi_sphere = true },
                    mt,
                    st,
                    sc,
                    data_dir,
                )) {
                    var r_config = tcfg.getRasterConfig(.bench);
                    r_config.save_strategy = .disk;
                    _ = try minsuite.runSphere200MultiCullQuiet(
                        aa,
                        io,
                        mt,
                        st,
                        sc,
                        data_dir,
                        pixel_num_sphere,
                        texture_grey,
                        texture_rgb,
                        r_config,
                        out_dir ++ "/sphere200multicull",
                        0.75,
                    );
                }
            }
        }
    }

    std.debug.print("Rendering MIN Suite (multimesh) to {s}...\n", .{out_dir});
    const multi_dir_paths = [_][]const u8{
        "data/min/tri3_twoelems/",
        "data/min/tri6_twoelems/",
        "data/min/quad4_twoelems/",
        "data/min/quad8_twoelems/",
        "data/min/quad9_twoelems/",
    };

    try gengold.runMultimeshGenerationExt(
        aa,
        io,
        config,
        out_dir ++ "/multimesh/base",
        &multi_dir_paths,
        pixel_num_multi,
    );
    try gengold.runMultimeshMixedGenerationExt(
        aa,
        io,
        config,
        out_dir ++ "/multimesh/allelem_allshade",
        &multi_dir_paths,
        pixel_num_multi,
    );
    try gengold.runMultimeshMixedRGBGenerationExt(
        aa,
        io,
        config,
        out_dir ++ "/multimesh/allelem_allshade_rgb",
        &multi_dir_paths,
        pixel_num_multi,
    );

    std.debug.print("Done. MIN Suite rendering complete.\n", .{});
}
