// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");

const riley = @import("riley/zig/riley.zig");
const rastcfg = @import("riley/zig/rasterconfig.zig");
const meshio = @import("riley/zig/meshio.zig");
const uvio = @import("riley/zig/uvio.zig");
const mo = @import("riley/zig/meshpipeline.zig");
const gk = @import("riley/zig/geometrykernels.zig");
const iio = @import("riley/zig/imageio.zig");
const shaderops = @import("riley/zig/shaderops.zig");
const camera_mod = @import("riley/zig/camera.zig");
const cameraops = @import("riley/zig/cameraops.zig");
const sceneops = @import("riley/zig/sceneops.zig");
const Rotation = @import("riley/zig/rotation.zig").Rotation;

const MeshInput = mo.MeshInput;
const MeshType = gk.MeshType;
const CameraInput = camera_mod.CameraInput;
const DistortionModel = camera_mod.DistortionModel;
const BrownConrady = camera_mod.BrownConrady;
const PointSpreadFunc = camera_mod.PointSpreadFunc;
const FuncShaderBuiltin = shaderops.FuncShaderBuiltin;
const FuncShaderParams = shaderops.FuncShaderParams;

const OUT_DIR_ROOT = "./out/distortion";
const PIXELS_NUM = [2]u32{ 1600, 1000 };
const PIXELS_SIZE = [2]f64{ 5.3e-6, 5.3e-6 };
const FOCAL_LENGTH: f64 = 50.0e-3;
const FOV_SCALE: f64 = 1.0;
const SUB_SAMPLE: u32 = 2;
const RENDER_THREADS: u16 = 4;
const CAMERA_ROT = Rotation.init(0.0, 0.0, 0.0);
const SHADER_BUILTIN = FuncShaderBuiltin.checker;
const SHADER_PARAMS = FuncShaderParams{
    .coord_scale = .{ 36.0, 36.0 },
    .coord_offset = .{ 0.0, 0.0 },
};
const PSF_CASES = [_]struct {
    name: []const u8,
    psf: PointSpreadFunc,
}{
    .{
        .name = "pixel_box",
        .psf = .{ .pixel_box = .{} },
    },
    .{
        .name = "gaussian_heavy",
        .psf = .{ .gaussian = .{
            .sigma_px = 0.8,
            .supp_rad_px = 2.5,
            .separable = .yes,
        } },
    },
};

const DISTORTION_CASES = [_]struct {
    name: []const u8,
    model: DistortionModel,
}{
    .{
        .name = "none",
        .model = .none,
    },
    .{
        .name = "heavy_pincushion",
        .model = .{ .brown_conrady = BrownConrady{
            .k1 = 12.0,
            .k2 = 40.0,
            .k3 = 0.0,
            .p1 = 0.0,
            .p2 = 0.0,
        } },
    },
    .{
        .name = "heavy_barrel",
        .model = .{ .brown_conrady = BrownConrady{
            .k1 = -12.0,
            .k2 = 40.0,
            .k3 = 0.0,
            .p1 = 0.0,
            .p2 = 0.0,
        } },
    },
};

fn ensureDir(io: std.Io, dir_path: []const u8) !void {
    const cwd = std.Io.Dir.cwd();
    var parts = std.mem.splitScalar(u8, dir_path, '/');
    var path_buf: [512]u8 = undefined;
    var path_len: usize = 0;

    while (parts.next()) |part| {
        if (part.len == 0 or std.mem.eql(u8, part, ".")) continue;
        if (path_len > 0) {
            path_buf[path_len] = '/';
            path_len += 1;
        }
        @memcpy(path_buf[path_len .. path_len + part.len], part);
        path_len += part.len;
        cwd.createDir(io, path_buf[0..path_len], .default_dir) catch |err| {
            if (err != error.PathAlreadyExists) return err;
        };
    }
}

fn makeCameraInput(
    coords: *const meshio.Coords,
    distortion: DistortionModel,
    psf: PointSpreadFunc,
) CameraInput {
    const roi_pos = sceneops.boundsCenter(coords);
    const cam_pos = cameraops.posFillFrameFromRot(
        coords,
        PIXELS_NUM,
        PIXELS_SIZE,
        FOCAL_LENGTH,
        CAMERA_ROT,
        FOV_SCALE,
    );
    return .{
        .pixels_num = PIXELS_NUM,
        .pixels_size = PIXELS_SIZE,
        .pos_world = cam_pos,
        .rot_world = CAMERA_ROT,
        .roi_cent_world = roi_pos,
        .focal_length = FOCAL_LENGTH,
        .sub_sample = SUB_SAMPLE,
        .distortion = distortion,
        .psf = psf,
    };
}

fn renderCase(
    allocator: std.mem.Allocator,
    io: std.Io,
    threaded_io_io: std.Io,
    mesh_type: MeshType,
    mesh_name: []const u8,
    distortion_case: @TypeOf(DISTORTION_CASES[0]),
    psf_case: @TypeOf(PSF_CASES[0]),
    sim_data: meshio.SimData,
    uvs: uvio.UVMap,
    config: rastcfg.RasterConfig,
) !void {
    var out_dir_buf: [256]u8 = undefined;
    const out_dir = try std.fmt.bufPrint(
        &out_dir_buf,
        "{s}/{s}_{s}_{s}",
        .{ OUT_DIR_ROOT, mesh_name, distortion_case.name, psf_case.name },
    );
    try ensureDir(io, out_dir);

    const mesh_input = MeshInput{
        .mesh_type = mesh_type,
        .coords = sim_data.coords,
        .connect = sim_data.connect,
        .disp = null,
        .shader = .{ .func = .{
            .uvs = uvs.array,
            .coord_mode = .uv,
            .builtin = SHADER_BUILTIN,
            .params = SHADER_PARAMS,
            .bits = 8,
            .scaling = .none,
            .normal_type = .none,
        } },
    };
    const camera_input = makeCameraInput(
        &sim_data.coords,
        distortion_case.model,
        psf_case.psf,
    );
    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = threaded_io_io, .workers = RENDER_THREADS },
    };

    std.debug.print(
        "Rendering {s} {s} {s} -> {s}\n",
        .{ mesh_name, distortion_case.name, psf_case.name, out_dir },
    );
    const images = try riley.raster(
        allocator,
        &render_groups,
        &[_]CameraInput{camera_input},
        &[_]MeshInput{mesh_input},
        config,
        out_dir,
    );
    if (images) |img| {
        allocator.free(img.slice);
        img.deinit(allocator);
    }
}

pub fn main(init: std.process.Init) !void {
    const outer_alloc = init.gpa;
    const io = init.io;

    var arena = std.heap.ArenaAllocator.init(outer_alloc);
    defer arena.deinit();
    const aa = arena.allocator();

    const config = rastcfg.RasterConfig{
        .save_strategy = .disk,
        .image_save_mode = .grey,
        .image_save_opts = &[_]iio.ImageSaveOpts{
            .{ .format = .bmp, .bits = 8, .scaling = .auto, .channels = 1 },
        },
        .report = .off,
        .total_threads = RENDER_THREADS,
        .frame_batch_size_per_group = 1,
        .max_geom_jobs_in_flight_per_group = 1,
        .max_geom_workers_per_job = RENDER_THREADS,
        .max_raster_workers_per_job = RENDER_THREADS,
    };

    try ensureDir(io, OUT_DIR_ROOT);

    var threaded_io = riley.getThreadedIo(
        aa,
        init.minimal,
        config.total_threads,
    );
    defer threaded_io.deinit();

    const mesh_types = comptime std.enums.values(MeshType);
    for (mesh_types) |mesh_type| {
        const mesh_name = @tagName(mesh_type);
        const data_dir = try std.fmt.allocPrint(
            aa,
            "data/bench/{s}_fullraster",
            .{mesh_name},
        );
        const coord_path = try std.fmt.allocPrint(aa, "{s}/coords.csv", .{data_dir});
        const connect_path = try std.fmt.allocPrint(aa, "{s}/connect.csv", .{data_dir});
        const uv_path = try std.fmt.allocPrint(aa, "{s}/uvs.csv", .{data_dir});

        var sim_data = try meshio.loadSimData(
            aa,
            threaded_io.io(),
            coord_path,
            connect_path,
            null,
            null,
        );
        defer sim_data.deinit(aa);

        var uvs = try uvio.loadUVMap(aa, threaded_io.io(), uv_path);
        defer uvs.deinit(aa);

        for (DISTORTION_CASES) |dist_case| {
            for (PSF_CASES) |psf_case| {
                try renderCase(
                    aa,
                    io,
                    threaded_io.io(),
                    mesh_type,
                    mesh_name,
                    dist_case,
                    psf_case,
                    sim_data,
                    uvs,
                    config,
                );
            }
        }
    }

    std.debug.print("Done. Output written under {s}/\n", .{OUT_DIR_ROOT});
}
