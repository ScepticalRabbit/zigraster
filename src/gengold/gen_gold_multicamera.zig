// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");

const benchcommon = @import("../dev_support/benchcommon.zig");
const policy = @import("../dev_support/testpolicy.zig");
const orch = @import("../dev_support/orchestration.zig");
const tcfg = @import("../dev_support/testconfig.zig");
const buildconfig = @import("../riley/zig/buildconfig.zig");
const cfg = buildconfig.config;
const cam = @import("../riley/zig/camera.zig");
const iio = @import("../riley/zig/imageio.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const meshio = @import("../riley/zig/meshio.zig");
const texops = @import("../riley/zig/textureops.zig");
const uvio = @import("../riley/zig/uvio.zig");
const riley = @import("../riley/zig/riley.zig");

const simd_on = cfg.simd == .on;

const RenderCase = struct {
    case_name: []const u8,
    data_dir: []const u8,
    mesh_type: gk.MeshType,
    channels: usize,
    shader: union(enum) {
        nodal_grey,
        tex8_rgb: texops.TextureSampleConfig,
    },
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const out_root = policy.goldRoot(.sphere200multicam);
    const pixel_num = [_]u32{ 800, 500 };

    const texture_rgb = try iio.loadImage(
        u8,
        3,
        allocator,
        io,
        "texture/speckle_rgb.bmp",
        .bmp,
    );
    defer texture_rgb.deinit(allocator);

    const render_cases = [_]RenderCase{
        .{
            .case_name = "tri3_nodal_grey",
            .data_dir = "data/min/tri3_sphere200",
            .mesh_type = .tri3,
            .channels = 1,
            .shader = .nodal_grey,
        },
        .{
            .case_name = "tri6_tex8_rgb_cubic_catmull_rom_lut_lerp",
            .data_dir = "data/min/tri6_sphere200",
            .mesh_type = .tri6,
            .channels = 3,
            .shader = .{
                .tex8_rgb = .{
                    .sample = .cubic_catmull_rom,
                    .mode = .lut_lerp,
                },
            },
        },
    };

    const cwd = std.Io.Dir.cwd();
    cwd.createDir(io, out_root, .default_dir) catch |err| {
        if (err != error.PathAlreadyExists) return err;
    };

    for (render_cases) |render_case| {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const aa = arena.allocator();

        const coord_path = try std.fs.path.join(
            aa,
            &[_][]const u8{ render_case.data_dir, "coords.csv" },
        );
        const connect_path = try std.fs.path.join(
            aa,
            &[_][]const u8{ render_case.data_dir, "connect.csv" },
        );
        const field_path = try std.fs.path.join(
            aa,
            &[_][]const u8{ render_case.data_dir, "field.csv" },
        );
        const sim_data = try meshio.loadSimData(
            aa,
            io,
            coord_path,
            connect_path,
            null,
            null,
        );
        const field_raw = try benchcommon.loadNDArrayFromCSV(
            aa,
            io,
            field_path,
            if (render_case.channels == 3) 3 else 1,
            true,
        );
        const cameras = try orch.initStereoCamerasForCoords(
            aa,
            &sim_data.coords,
            pixel_num,
            1.0,
            10.0,
        );
        defer for (cameras) |camera| camera.deinit(aa);
        const camera_inputs = [_]cam.CameraInput{
            cam.CameraInput{
                .pixels_num = cameras[0].pixels_num,
                .pixels_size = cameras[0].pixels_size,
                .pos_world = cameras[0].pos_world,
                .rot_world = cameras[0].rot_world,
                .roi_cent_world = cameras[0].roi_cent_world,
                .focal_length = cameras[0].focal_length,
                .sub_sample = cameras[0].sub_sample,
                .distortion = cameras[0].distortion,
            },
            cam.CameraInput{
                .pixels_num = cameras[1].pixels_num,
                .pixels_size = cameras[1].pixels_size,
                .pos_world = cameras[1].pos_world,
                .rot_world = cameras[1].rot_world,
                .roi_cent_world = cameras[1].roi_cent_world,
                .focal_length = cameras[1].focal_length,
                .sub_sample = cameras[1].sub_sample,
                .distortion = cameras[1].distortion,
            },
        };

        const mesh_input = switch (render_case.shader) {
            .nodal_grey => mo.MeshInput{
                .mesh_type = render_case.mesh_type,
                .coords = sim_data.coords,
                .connect = sim_data.connect,
                .disp = null,
                .shader = .{
                    .nodal = .{
                        .field = .{
                            .array = field_raw,
                            .array_mem = field_raw.slice,
                        },
                        .scaling = .auto,
                    },
                },
            },
            .tex8_rgb => |samp_cfg| blk: {
                const uv_path = try std.fmt.allocPrint(
                    aa,
                    "{s}/uvs.csv",
                    .{render_case.data_dir},
                );
                const uv_map = try uvio.loadUVMap(aa, io, uv_path);
                break :blk mo.MeshInput{
                    .mesh_type = render_case.mesh_type,
                    .coords = sim_data.coords,
                    .connect = sim_data.connect,
                    .disp = null,
                    .shader = .{
                        .tex_rgb_u8 = .{
                            .uvs = uv_map.array,
                            .tex = texture_rgb,
                            .samp_cfg = samp_cfg,
                        },
                    },
                };
            },
        };

        var config = tcfg.getRasterConfig(.gold);
        config.save_strategy = .disk;
        config.image_save_opts = &[_]iio.ImageSaveOpts{
            .{
                .format = .fimg,
                .bits = null,
                .scaling = .none,
                .channels = render_case.channels,
            },
        };

        const out_dir_path = try std.fs.path.join(
            aa,
            &[_][]const u8{ out_root, render_case.case_name },
        );
        std.debug.print(
            "Rendering multicamera gold: {s}/{s}\n",
            .{ out_root, render_case.case_name },
        );

        const render_groups = [_]riley.RenderGroupSpec{
            .{ .io = io, .workers = @max(@as(u16, 1), config.total_threads) },
        };
        _ = try riley.raster(
            aa,
            &render_groups,
            &camera_inputs,
            &[_]mo.MeshInput{mesh_input},
            config,
            out_dir_path,
        );
    }

    std.debug.print(
        "Done. Multicamera gold references established for .simd = .{s}.\n",
        .{if (simd_on) "on" else "off"},
    );
}
