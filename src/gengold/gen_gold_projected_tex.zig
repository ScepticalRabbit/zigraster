// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const policy = @import("../dev_support/testpolicy.zig");
const orch = @import("../dev_support/orchestration.zig");
const tcfg = @import("../dev_support/testconfig.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const meshio = @import("../riley/zig/meshio.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const CameraInput = @import("../riley/zig/camera.zig").CameraInput;
const iio = @import("../riley/zig/imageio.zig");
const rastcfg = @import("../riley/zig/rasterconfig.zig");
const rotation = @import("../riley/zig/rotation.zig");
const riley = @import("../riley/zig/riley.zig");
const vec = @import("../riley/zig/vecstack.zig");
const buildconfig = @import("../riley/zig/buildconfig.zig");
const F = buildconfig.F;

const data_root = "data/min";
const test_type = "sphere200";

const SphereCasePrepared = struct {
    coords: meshio.Coords,
    connect: meshio.Connect,
    camera_ref: CameraInput,
    camera_stereo: CameraInput,
};

fn loadSphereCase(
    allocator: std.mem.Allocator,
    io: std.Io,
    mesh_type: gk.MeshType,
) !SphereCasePrepared {
    const data_dir = try std.fmt.allocPrint(
        allocator,
        "{s}/{s}_{s}",
        .{ data_root, @tagName(mesh_type), test_type },
    );
    const coord_path = try std.fs.path.join(
        allocator,
        &[_][]const u8{ data_dir, "coords.csv" },
    );
    const connect_path = try std.fs.path.join(
        allocator,
        &[_][]const u8{ data_dir, "connect.csv" },
    );

    const sim_data = try meshio.loadSimData(
        allocator,
        io,
        coord_path,
        connect_path,
        null,
        null,
    );

    const camera_ref_prep = try orch.initCameraForCoords(
        allocator,
        &sim_data.coords,
        .{ 640, 400 },
        1.0,
    );

    const camera_ref = CameraInput{
        .pixels_num = camera_ref_prep.pixels_num,
        .pixels_size = camera_ref_prep.pixels_size,
        .pos_world = camera_ref_prep.pos_world,
        .rot_world = camera_ref_prep.rot_world,
        .roi_cent_world = camera_ref_prep.roi_cent_world,
        .focal_length = camera_ref_prep.focal_length,
        .sub_sample = camera_ref_prep.sub_sample,
        .distortion = camera_ref_prep.distortion,
    };

    const stereo_angle: F = 15.0 * std.math.pi / 180.0;
    const cam_dist: F = camera_ref_prep.pos_world.get(2);
    const pos_stereo = vec.initVec3(
        F,
        cam_dist * @sin(stereo_angle),
        0.0,
        cam_dist * @cos(stereo_angle),
    );
    const rot_stereo = rotation.Rotation.init(0.0, stereo_angle, 0.0);

    const camera_stereo = CameraInput{
        .pixels_num = camera_ref_prep.pixels_num,
        .pixels_size = camera_ref_prep.pixels_size,
        .pos_world = pos_stereo,
        .rot_world = rot_stereo,
        .roi_cent_world = camera_ref_prep.roi_cent_world,
        .focal_length = camera_ref_prep.focal_length,
        .sub_sample = camera_ref_prep.sub_sample,
        .distortion = camera_ref_prep.distortion,
    };

    return .{
        .coords = sim_data.coords,
        .connect = sim_data.connect,
        .camera_ref = camera_ref,
        .camera_stereo = camera_stereo,
    };
}

fn renderCase(
    allocator: std.mem.Allocator,
    io: std.Io,
    mesh_input: mo.MeshInput,
    camera_target: CameraInput,
    out_dir_path: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    var out_dir = try orch.openDirEnsured(io, out_dir_path);
    out_dir.close(io);

    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), config.total_threads) },
    };
    const images = try riley.raster(
        allocator,
        &render_groups,
        &[_]CameraInput{camera_target},
        &[_]mo.MeshInput{mesh_input},
        config,
        out_dir_path,
    );
    if (images) |img| {
        allocator.free(img.slice);
        img.deinit(allocator);
    }
}

pub fn main(init: std.process.Init) !void {
    try mainWithOutputRoot(init, policy.goldRoot(.projected_tex));
}

pub fn mainWithOutputRoot(
    init: std.process.Init,
    output_root: []const u8,
) !void {
    const io = init.io;
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const allocator = arena.allocator();

    const mesh_types = [_]gk.MeshType{
        .tri3,
        .tri6,
        .quad4ibi,
        .quad4newton,
        .quad8,
        .quad9,
    };

    var config = tcfg.getRasterConfig(.gold_gen);
    config.save_strategy = .disk;
    config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .fimg, .bits = null, .scaling = .none },
        .{ .format = .tiff, .bits = 8, .scaling = .auto },
    };

    const tex_grey = try iio.loadImage(
        u8,
        1,
        allocator,
        io,
        "texture/speckle.bmp",
        .bmp,
    );
    const tex_rgb = try iio.loadImage(
        u8,
        3,
        allocator,
        io,
        "texture/speckle_rgb.bmp",
        .bmp,
    );

    for (mesh_types) |mesh_type| {
        const prepared = try loadSphereCase(
            allocator,
            io,
            mesh_type,
        );

        inline for ([_]bool{ false, true }) |is_stereo| {
            const cam_name = if (is_stereo) "stereo" else "ref";
            const target_cam = if (is_stereo)
                prepared.camera_stereo
            else
                prepared.camera_ref;

            // 1. Greyscale projected texture
            {
                const case_dir_name = try std.fmt.allocPrint(
                    allocator,
                    "{s}_{s}_projtex_grey_{s}",
                    .{ test_type, @tagName(mesh_type), cam_name },
                );
                const out_dir_path = try std.fs.path.join(
                    allocator,
                    &[_][]const u8{ output_root, case_dir_name },
                );

                const mesh_input = mo.MeshInput{
                    .mesh_type = mesh_type,
                    .coords = prepared.coords,
                    .connect = prepared.connect,
                    .disp = null,
                    .shader = .{
                        .projected_tex_u8 = .{
                            .ref_camera = prepared.camera_ref,
                            .ref_coords = null,
                            .tex = tex_grey,
                        },
                    },
                };

                try renderCase(
                    allocator,
                    io,
                    mesh_input,
                    target_cam,
                    out_dir_path,
                    config,
                );
            }

            // 2. RGB projected texture
            {
                const case_dir_name = try std.fmt.allocPrint(
                    allocator,
                    "{s}_{s}_projtex_rgb_{s}",
                    .{ test_type, @tagName(mesh_type), cam_name },
                );
                const out_dir_path = try std.fs.path.join(
                    allocator,
                    &[_][]const u8{ output_root, case_dir_name },
                );

                const mesh_input = mo.MeshInput{
                    .mesh_type = mesh_type,
                    .coords = prepared.coords,
                    .connect = prepared.connect,
                    .disp = null,
                    .shader = .{
                        .projected_tex_rgb_u8 = .{
                            .ref_camera = prepared.camera_ref,
                            .ref_coords = null,
                            .tex = tex_rgb,
                        },
                    },
                };

                try renderCase(
                    allocator,
                    io,
                    mesh_input,
                    target_cam,
                    out_dir_path,
                    config,
                );
            }
        }
    }
}
