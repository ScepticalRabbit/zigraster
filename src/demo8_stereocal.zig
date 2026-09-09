// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");

const demoframes = @import("dev_support/demoframes.zig");
const buildconfig = @import("riley/zig/buildconfig.zig");
const riley = @import("riley/zig/riley.zig");
const RasterConfig = riley.RasterConfig;
const meshio = @import("riley/zig/meshio.zig");
const uvio = @import("riley/zig/uvio.zig");
const iio = @import("riley/zig/imageio.zig");
const mo = @import("riley/zig/meshpipeline.zig");
const MeshInput = mo.MeshInput;
const camera_mod = @import("riley/zig/camera.zig");
const CameraInput = camera_mod.CameraInput;
const cameraio = @import("riley/zig/cameraio.zig");
const cameraops = @import("riley/zig/cameraops.zig");
const sceneops = @import("riley/zig/sceneops.zig");
const vecstack = @import("riley/zig/vecstack.zig");
const Rotation = @import("riley/zig/rotation.zig").Rotation;
const DistortionModel = camera_mod.DistortionModel;
const BrownConrady = camera_mod.BrownConrady;
const BrownConradyExt = camera_mod.BrownConradyExt;
const CameraInput = camera_mod.CameraInput;
const StereoPairInput = camera_mod.StereoPairInput;
const F = buildconfig.F;

const DATA_DIR = "data/calplate/tri3_calplate3d/";
const TEXTURE_PATH = "texture/cal_target.tiff";
const OUT_DIR_ROOT = "./out/demo8_stereocal";
const PIXELS_NUM = [2]u32{ 2464, 2056 };
const PIXELS_SIZE = [2]F{
    @floatCast(3.45e-6),
    @floatCast(3.45e-6),
};
const FOCAL_LENGTH: F = @floatCast(50.0e-3);
const FOV_SCALE_FACTOR: F = 1.0;
const STEREO_ANGLE_DEG: F = 20.0;
const SUB_SAMPLE: u32 = 2;
const FRAMES_MAX: usize = 8;

const MATCHED_ROI = [3]F{
    @floatCast(0.0125),
    @floatCast(0.0175),
    @floatCast(0.0005),
};
const MATCHED_CAM0_POS = [3]F{
    @floatCast(0.0125),
    @floatCast(0.0175),
    @floatCast(0.160864856482),
};
const MATCHED_CAM1_POS = [3]F{
    @floatCast(0.067348011198),
    @floatCast(0.0175),
    @floatCast(0.151193672270),
};

const TOTAL_THREADS: u16 = 8;
const RENDER_GROUP_COUNT: u16 = 8;
const WORKERS_PER_GROUP: u16 = 1;

const DistortionCase = enum {
    none,
    brown_conrady,
    brown_conrady_ext,
};

const DISTORTION_CASE: DistortionCase = .brown_conrady;

const STEREO_FILE_NAME = "stereo_data_opengl.csv";

fn buildDistortion() DistortionModel {
    return switch (DISTORTION_CASE) {
        .none => .none,
        .brown_conrady => .{ .brown_conrady = BrownConrady{
            .k1 = -0.2,
            .k2 = 0.1,
            .k3 = 0.0,
            .p1 = 0.0001,
            .p2 = -0.0001,
        } },
        .brown_conrady_ext => .{ .brown_conrady_ext = BrownConradyExt{
            .k1 = -0.2,
            .k2 = 0.1,
            .k3 = -0.01,
            .k4 = -0.04,
            .k5 = 0.18,
            .k6 = -0.02,
            .p1 = 0.0001,
            .p2 = -0.0001,
        } },
    };
}

pub fn main(init: std.process.Init) !void {
    const outer_alloc = init.gpa;

    var arena = std.heap.ArenaAllocator.init(outer_alloc);
    defer arena.deinit();
    const aa = arena.allocator();

    // -------------------------------------------------------------------------
    // 1. Setup paths and parameters
    // -------------------------------------------------------------------------
    var coord_sys = camera_mod.CameraCoordSys.opengl;
    if (init.minimal.args.vector.len > 1) {
        const arg = std.mem.span(init.minimal.args.vector[1]);
        if (std.mem.eql(u8, arg, "opencv")) {
            coord_sys = .opencv;
        }
    }
    const stereo_file_name = if (coord_sys == .opencv)
        "stereo_data_opencv.csv"
    else
        "stereo_data_opengl.csv";

    const coord_path = DATA_DIR ++ "coords.csv";
    const conn_path = DATA_DIR ++ "connect.csv";
    const uv_path = DATA_DIR ++ "uvs.csv";
    const disp_paths = &[_][]const u8{
        DATA_DIR ++ "field_disp_x.csv",
        DATA_DIR ++ "field_disp_y.csv",
        DATA_DIR ++ "field_disp_z.csv",
    };

    const managed_ios = try outer_alloc.alloc(std.Io.Threaded, RENDER_GROUP_COUNT);
    defer {
        for (managed_ios) |*managed_io| {
            managed_io.deinit();
        }
        outer_alloc.free(managed_ios);
    }

    const render_groups = try outer_alloc.alloc(
        riley.RenderGroupSpec,
        RENDER_GROUP_COUNT,
    );
    defer outer_alloc.free(render_groups);

    for (0..RENDER_GROUP_COUNT) |gg| {
        managed_ios[gg] = riley.getThreadedIo(
            outer_alloc,
            init.minimal,
            WORKERS_PER_GROUP,
        );
        render_groups[gg] = .{
            .io = managed_ios[gg].io(),
            .workers = WORKERS_PER_GROUP,
        };
    }
    const io = render_groups[0].io;

    const cwd = std.Io.Dir.cwd();
    cwd.createDir(io, "out", .default_dir) catch |err| {
        if (err != error.PathAlreadyExists) return err;
    };
    cwd.deleteTree(io, OUT_DIR_ROOT) catch |err| {
        if (err != error.FileNotFound) return err;
    };
    cwd.createDir(io, OUT_DIR_ROOT, .default_dir) catch |err| {
        if (err != error.PathAlreadyExists) return err;
    };
    var out_dir = try cwd.openDir(io, OUT_DIR_ROOT, .{});
    defer out_dir.close(io);
    out_dir.deleteFile(io, "cameradata.csv") catch |err| {
        if (err != error.FileNotFound) return err;
    };

    // -------------------------------------------------------------------------
    // 2. Load calibration plate mesh, sample frames, and apply texture
    // -------------------------------------------------------------------------
    var sim_data = try meshio.loadSimData(
        aa,
        io,
        coord_path,
        conn_path,
        null,
        disp_paths,
    );
    defer sim_data.deinit(aa);
    const disp_source = sim_data.disp orelse return error.MissingDisplacement;
    const frame_indices = try demoframes.evenlySpacedIndices(
        aa,
        disp_source.getTimeN(),
        FRAMES_MAX,
    );
    var selected_disp = try demoframes.selectFieldFrames(
        aa,
        &disp_source,
        frame_indices,
    );
    defer selected_disp.deinit(aa);

    var uvs = try uvio.loadUVMap(aa, io, uv_path);
    defer uvs.deinit(aa);

    var texture = try iio.loadImage(
        u8,
        1,
        aa,
        io,
        TEXTURE_PATH,
        .tiff,
    );
    defer texture.deinit(aa);

    const distortion = buildDistortion();

    const target_roi = vecstack.initVec3(
        F,
        MATCHED_ROI[0],
        MATCHED_ROI[1],
        MATCHED_ROI[2],
    );
    const roi_pos_orig = sceneops.boundsCenter(&sim_data.coords);
    const roi_shift = target_roi.sub(roi_pos_orig);
    for (0..sim_data.coords.mat.rows_num) |nn| {
        sim_data.coords.mat.set(
            nn,
            0,
            sim_data.coords.mat.get(nn, 0) + roi_shift.get(0),
        );
        sim_data.coords.mat.set(
            nn,
            1,
            sim_data.coords.mat.get(nn, 1) + roi_shift.get(1),
        );
        sim_data.coords.mat.set(
            nn,
            2,
            sim_data.coords.mat.get(nn, 2) + roi_shift.get(2),
        );
    }
    const roi_pos = sceneops.boundsCenter(&sim_data.coords);

    // -------------------------------------------------------------------------
    // 3. Create, save, and reload stereo camera pair
    // -------------------------------------------------------------------------
    const cam0_rot = Rotation.init(
        std.math.degreesToRadians(0.0),
        std.math.degreesToRadians(0.0),
        std.math.degreesToRadians(0.0),
    );
    const cam1_rot = Rotation.init(
        std.math.degreesToRadians(0.0),
        std.math.degreesToRadians(STEREO_ANGLE_DEG),
        std.math.degreesToRadians(0.0),
    );

    const cam0_pos = vecstack.initVec3(
        F,
        MATCHED_CAM0_POS[0],
        MATCHED_CAM0_POS[1],
        MATCHED_CAM0_POS[2],
    );
    const cam1_pos = vecstack.initVec3(
        F,
        MATCHED_CAM1_POS[0],
        MATCHED_CAM1_POS[1],
        MATCHED_CAM1_POS[2],
    );

    const cam0_in = CameraInput{
        .pixels_num = PIXELS_NUM,
        .pixels_size = PIXELS_SIZE,
        .pos_world = cam0_pos,
        .rot_world = cam0_rot,
        .roi_cent_world = roi_pos,
        .focal_length = FOCAL_LENGTH,
        .sub_sample = SUB_SAMPLE,
        .distortion = distortion,
    };
    var cam1_in = cam0_in;
    cam1_in.rot_world = cam1_rot;
    cam1_in.pos_world = cam1_pos;

    var stereo_pair = StereoPairInput{
        .cameras = .{ cam0_in, cam1_in },
    };

    // Save stereo pair to output directory
    try cameraio.saveStereoPair(io, out_dir, stereo_file_name, stereo_pair);

    // Load stereo pair back from output directory (standalone test)
    stereo_pair = try cameraio.loadStereoPair(aa, io, out_dir, stereo_file_name);

    // -------------------------------------------------------------------------
    // 4. Build mesh and raster configuration
    // -------------------------------------------------------------------------
    const mesh_input = MeshInput{
        .mesh_type = .tri3,
        .coords = sim_data.coords,
        .connect = sim_data.connect,
        .disp = selected_disp,
        .shader = .{ .tex_u8 = .{
            .uvs = uvs.array,
            .tex = texture,
            .samp_cfg = .{
                .sample = .cubic_catmull_rom,
                .mode = .lut_lerp,
            },
            .bits = 8,
            .scaling = .none,
        } },
    };

    const config = RasterConfig{
        .render_mode = .offline,
        .total_threads = TOTAL_THREADS,
        .frame_batch_size_per_group = FRAMES_MAX,
        .max_geom_jobs_in_flight_per_group = FRAMES_MAX,
        .max_geom_workers_per_job = 1,
        .geom_scheduling_mode = .spread,
        .max_raster_workers_per_job = 1,
        .save_strategy = .disk,
        .tile_size_min = 8,
        .tile_size_max = 128,
        .background_value = 128.0,
        .image_save_opts = &[_]iio.ImageSaveOpts{
            .{ .format = .bmp, .bits = 8, .scaling = .auto },
        },
        .report = .bench,
    };

    // -------------------------------------------------------------------------
    // 5. Render stereocal poses
    // -------------------------------------------------------------------------
    const meshes = [_]MeshInput{mesh_input};
    const images = try riley.raster(
        aa,
        render_groups,
        &stereo_pair.cameras,
        &meshes,
        config,
        OUT_DIR_ROOT,
    );

    if (images) |img| {
        aa.free(img.slice);
        img.deinit(aa);
    }
}
