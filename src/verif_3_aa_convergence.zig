// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const buildconfig = @import("riley/zig/buildconfig.zig");
const F = buildconfig.F;

const orch = @import("dev_support/orchestration.zig");
const riley = @import("riley/zig/riley.zig");
const rastcfg = @import("riley/zig/rasterconfig.zig");
const meshio = @import("riley/zig/meshio.zig");
const uvio = @import("riley/zig/uvio.zig");
const mo = @import("riley/zig/meshpipeline.zig");
const gk = @import("riley/zig/geometrykernels.zig");
const iio = @import("riley/zig/imageio.zig");
const imageops = @import("riley/zig/imageops.zig");
const matslice = @import("riley/zig/matslice.zig");
const ndarray = @import("riley/zig/ndarray.zig");
const camera_mod = @import("riley/zig/camera.zig");
const cameraops = @import("riley/zig/cameraops.zig");
const Rotation = @import("riley/zig/rotation.zig").Rotation;
const sceneops = @import("riley/zig/sceneops.zig");

const MeshInput = mo.MeshInput;
const CameraPrepared = camera_mod.CameraPrepared;
const CameraInput = camera_mod.CameraInput;
const NDArrayOps = ndarray.NDArrayOps(F);
const MatSlice = matslice.MatSlice(F);

const bunny_mesh_types = [_]gk.MeshType{
    .tri3,
    .tri6,
    .quad4ibi,
    .quad8,
    .quad9,
};

const bunny_names = [_][]const u8{
    "tri3",
    "tri6",
    "quad4",
    "quad8",
    "quad9",
};

const bunny_dir_paths = [_][]const u8{
    "data/rabbits/riley_tri3/",
    "data/rabbits/riley_tri6/",
    "data/rabbits/riley_quad4/",
    "data/rabbits/riley_quad8/",
    "data/rabbits/riley_quad9/",
};

const ssaa_levels_lowres = [_]u8{ 32, 16, 8, 4, 2, 1 };
const ssaa_levels_highres = [_]u8{ 32, 16, 8, 4, 2, 1 };

const shader_names = [_][]const u8{ "funcconst", "tex" };

const resolution_cases = [_]struct {
    tag: []const u8,
    pixel_num: [2]u32,
    ssaa_levels: []const u8,
}{
    .{
        .tag = "lowres",
        .pixel_num = .{ 160, 100 },
        .ssaa_levels = &ssaa_levels_lowres,
    },
    .{
        .tag = "highres",
        .pixel_num = .{ 640, 400 },
        .ssaa_levels = &ssaa_levels_highres,
    },
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const aa = arena.allocator();

    const out_verif_root = "verif/verif_3";
    const fov_scale: F = 1.02;

    std.debug.print("Loading speckle texture...\n", .{});
    const texture = try iio.loadImage(
        u8,
        1,
        aa,
        io,
        "texture/speckle-simple.tiff",
        .tiff,
    );

    for (bunny_mesh_types, 0..) |mesh_type, ii| {
        const bunny_name = bunny_names[ii];
        const dir_path = bunny_dir_paths[ii];

        std.debug.print("Processing Riley mesh: {s}\n", .{bunny_name});

        const sim_datas = try meshio.loadMultiSimData(
            aa,
            io,
            &[_][]const u8{dir_path},
            .{
                .field_files = null,
                .disp_files = null,
            },
        );
        const sim_data = sim_datas[0];

        const uv_path = try std.fmt.allocPrint(aa, "{s}uvs.csv", .{dir_path});
        const uv_map = try uvio.loadUVMap(aa, io, uv_path);

        for (shader_names) |shader_name| {
            for (resolution_cases) |resolution_case| {
                const pixel_num = resolution_case.pixel_num;
                const out_dir_path = try std.fmt.allocPrint(
                    aa,
                    "{s}/c_{s}_{s}_{s}",
                    .{
                        out_verif_root,
                        bunny_name,
                        shader_name,
                        resolution_case.tag,
                    },
                );
                const out_dir = try orch.openDirEnsured(io, out_dir_path);

                var ref_mat = try MatSlice.initAlloc(
                    aa,
                    pixel_num[1],
                    pixel_num[0],
                );
                var ssaa_mat = try MatSlice.initAlloc(
                    aa,
                    pixel_num[1],
                    pixel_num[0],
                );
                var diff_mat = try MatSlice.initAlloc(
                    aa,
                    pixel_num[1],
                    pixel_num[0],
                );

                for (resolution_case.ssaa_levels, 0..) |ssaa, jj| {
                    var render_arena = std.heap.ArenaAllocator.init(aa);
                    defer render_arena.deinit();
                    const ra = render_arena.allocator();

                    const rot = Rotation.init(0.0, std.math.pi, 0.0);

                    var mesh_input = MeshInput{
                        .mesh_type = mesh_type,
                        .coords = try orch.copyCoords(ra, sim_data.coords),
                        .connect = sim_data.connect,
                        .disp = null,
                        .shader = undefined,
                    };

                    if (std.mem.eql(u8, shader_name, "funcconst")) {
                        mesh_input.shader = .{ .func = .{
                            .uvs = uv_map.array,
                            .coord_mode = .uv,
                            .builtin = .constant,
                            .params = .{},
                            .bits = 8,
                            .scaling = .auto,
                            .normal_type = .none,
                        } };
                    } else {
                        mesh_input.shader = .{ .tex_u8 = .{
                            .uvs = uv_map.array,
                            .tex = texture,
                            .samp_cfg = .{
                                .sample = .cubic_catmull_rom,
                                .mode = .lut_lerp,
                            },
                            .bits = 8,
                            .scaling = .none,
                            .normal_type = .none,
                        } };
                    }

                    const mesh_inputs = &[_]MeshInput{mesh_input};
                    const roi_pos = sceneops.boundsCenterOverMeshes(mesh_inputs);
                    const cam_pos = cameraops.posFillFrameFromRotOverMeshes(
                        mesh_inputs,
                        pixel_num,
                        orch.default_pixel_size,
                        orch.default_focal_length,
                        rot,
                        fov_scale,
                    );

                    std.debug.print(
                        "  Shader: {s}, Tag: {s}, SSAA: {d}\n",
                        .{ shader_name, resolution_case.tag, ssaa },
                    );

                    const camera = try CameraPrepared.init(
                        ra,
                        .{
                            .pixels_num = pixel_num,
                            .pixels_size = orch.default_pixel_size,
                            .pos_world = cam_pos,
                            .rot_world = rot,
                            .roi_cent_world = roi_pos,
                            .focal_length = orch.default_focal_length,
                            .sub_sample = ssaa,
                        },
                    );

                    const camera_input = CameraInput{
                        .pixels_num = camera.pixels_num,
                        .pixels_size = camera.pixels_size,
                        .pos_world = camera.pos_world,
                        .rot_world = camera.rot_world,
                        .roi_cent_world = camera.roi_cent_world,
                        .focal_length = camera.focal_length,
                        .sub_sample = camera.sub_sample,
                        .distortion = camera.distortion,
                    };
                    const config = rastcfg.RasterConfig{
                        .save_strategy = .memory,
                        .report = .off,
                        .subpixel_center_map = .per_tile,
                    };

                    const render_groups = [_]riley.RenderGroupSpec{
                        .{ .io = io, .workers = @max(@as(u16, 1), config.total_threads) },
                    };
                    const images = try riley.raster(
                        ra,
                        &render_groups,
                        &[_]@TypeOf(camera_input){camera_input},
                        mesh_inputs,
                        config,
                        null,
                    );

                    if (images) |img| {
                        const fixed_idxs = [_]usize{ 0, 0, 0, 0, 0 };
                        try NDArrayOps.extractMat(
                            ra,
                            &img,
                            &fixed_idxs,
                            3,
                            4,
                            &ssaa_mat,
                        );

                        if (jj == 0) {
                            @memcpy(ref_mat.slice, ssaa_mat.slice);
                        }

                        for (0..pixel_num[1]) |rr| {
                            for (0..pixel_num[0]) |cc| {
                                const val_ref = ref_mat.get(rr, cc);
                                const val_ssaa = ssaa_mat.get(rr, cc);
                                diff_mat.set(rr, cc, val_ref - val_ssaa);
                            }
                        }

                        var name_buf: [64]u8 = undefined;
                        const ssaa_base = try std.fmt.bufPrint(
                            name_buf[0..],
                            "ssaa{d}",
                            .{ssaa},
                        );
                        try iio.saveMatAsImage(
                            io,
                            out_dir,
                            ssaa_base,
                            &ssaa_mat,
                            .{
                                .format = .bmp,
                                .bits = 8,
                                .scaling = .auto,
                            },
                        );
                        const ssaa_csv_name = try std.fmt.allocPrint(
                            ra,
                            "ssaa{d}.csv",
                            .{ssaa},
                        );
                        try ssaa_mat.saveCSV(io, out_dir, ssaa_csv_name);

                        const diff_base = try std.fmt.bufPrint(
                            name_buf[0..],
                            "diff_ssaa{d}",
                            .{ssaa},
                        );
                        try iio.saveMatAsImage(
                            io,
                            out_dir,
                            diff_base,
                            &diff_mat,
                            .{
                                .format = .bmp,
                                .bits = 8,
                                .scaling = .auto,
                            },
                        );
                        const diff_csv_name = try std.fmt.allocPrint(
                            ra,
                            "diff_ssaa{d}.csv",
                            .{ssaa},
                        );
                        try diff_mat.saveCSV(io, out_dir, diff_csv_name);
                    }
                }
            }
        }
    }

    std.debug.print("Convergence study complete.\n", .{});
}
