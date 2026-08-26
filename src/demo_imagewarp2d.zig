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
const riley = @import("riley/zig/riley.zig");
const RasterConfig = riley.RasterConfig;
const meshio = @import("riley/zig/meshio.zig");
const iio = @import("riley/zig/imageio.zig");
const mo = @import("riley/zig/meshpipeline.zig");
const MeshInput = mo.MeshInput;
const gk = @import("riley/zig/geometrykernels.zig");
const camera_mod = @import("riley/zig/camera.zig");
const cameraio = @import("riley/zig/cameraio.zig");
const cameraops = @import("riley/zig/cameraops.zig");
const sceneops = @import("riley/zig/sceneops.zig");
const Rotation = @import("riley/zig/rotation.zig").Rotation;
const CameraInput = camera_mod.CameraInput;
const DistortionModel = camera_mod.DistortionModel;
const BrownConrady = camera_mod.BrownConrady;
const F = buildconfig.F;

const DATA_DIR = "data/FE/platehole3d_2mr_63f/";
const TEXTURE_PATH = "texture/speckle.bmp";
const OUT_DIR_ROOT = "./out/demo-imagewarp2d";

const PIXELS_NUM = [2]u32{ 2464, 2056 };
const PIXELS_SIZE = [2]F{
    @floatCast(3.45e-6),
    @floatCast(3.45e-6),
};
const FOCAL_LENGTH: F = @floatCast(50.0e-3);
const FOV_SCALE_FACTOR: F = @floatCast(0.65);
const SUB_SAMPLE: u32 = 2;
const STEREO_ANGLE_DEG: F = 20.0;
const VERTICAL_TRANSLATION_Y: F = 0.005; // 5mm vertical shift (half hole diameter)

const TOTAL_THREADS: u16 = 8;
const FRAME_BATCH_SIZE_PER_GROUP: u16 = 1;
const MAX_GEOM_JOBS_IN_FLIGHT_PER_GROUP: u16 = 1;
const RENDER_GROUP_COUNT: usize = 8;
const WORKERS_PER_GROUP: u16 = 1;

fn buildDistortion() DistortionModel {
    return .{ .brown_conrady = BrownConrady{
        .k1 = -0.2,
        .k2 = 0.1,
        .k3 = 0.0,
        .p1 = 0.0001,
        .p2 = -0.0001,
    } };
}

pub fn main(init: std.process.Init) !void {
    const outer_alloc = init.gpa;

    var arena = std.heap.ArenaAllocator.init(outer_alloc);
    defer arena.deinit();
    const aa = arena.allocator();

    // 1. Setup Rasteriser Configuration
    const config = RasterConfig{
        .render_mode = .offline,
        .total_threads = TOTAL_THREADS,
        .frame_batch_size_per_group = FRAME_BATCH_SIZE_PER_GROUP,
        .max_geom_jobs_in_flight_per_group = MAX_GEOM_JOBS_IN_FLIGHT_PER_GROUP,
        .max_geom_workers_per_job = 1,
        .geom_scheduling_mode = .spread,
        .max_raster_workers_per_job = 1,
        .save_strategy = .disk,
        .tile_size_min = 8,
        .tile_size_max = 128,
        .background_value = 128.0,
        .image_save_opts = &[_]iio.ImageSaveOpts{
            .{ .format = .bmp, .bits = 8, .scaling = .none },
        },
        .report = .bench,
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
            aa,
            init.minimal,
            WORKERS_PER_GROUP,
        );
        render_groups[gg] = .{
            .io = managed_ios[gg].io(),
            .workers = WORKERS_PER_GROUP,
        };
    }
    const io = render_groups[0].io;

    // 2. Load Mesh Simulation Geometry
    std.debug.print(
        "Loading plate with hole mesh from {s}...\n",
        .{DATA_DIR},
    );
    const coord_path = DATA_DIR ++ "coords.csv";
    const conn_path = DATA_DIR ++ "connect.csv";

    const sim_data = try meshio.loadSimData(
        aa,
        io,
        coord_path,
        conn_path,
        null,
        null,
    );

    // 3. Create Custom Displacement Field (2 frames: 0 disp, then vertical shift)
    const node_count = sim_data.coords.mat.rows_num;
    var disp_field = try meshio.Field.initAlloc(aa, 2, node_count, 3);
    for (0..node_count) |nn| {
        // Frame 0: undeformed reference
        disp_field.array.set(&[_]usize{ 0, nn, 0 }, 0.0);
        disp_field.array.set(&[_]usize{ 0, nn, 1 }, 0.0);
        disp_field.array.set(&[_]usize{ 0, nn, 2 }, 0.0);

        // Frame 1: translated vertically along the long axis (Y)
        disp_field.array.set(&[_]usize{ 1, nn, 0 }, 0.0);
        disp_field.array.set(&[_]usize{ 1, nn, 1 }, VERTICAL_TRANSLATION_Y);
        disp_field.array.set(&[_]usize{ 1, nn, 2 }, 0.0);
    }

    // 4. Load Texture for Projected Warping
    std.debug.print("Loading speckle texture from {s}...\n", .{TEXTURE_PATH});
    const texture = try iio.loadImage(
        u8,
        1,
        aa,
        io,
        TEXTURE_PATH,
        .bmp,
    );

    // 5. Setup Cameras (Stereo Rig)
    std.debug.print("Setting up stereo camera pair...\n", .{});
    const roi_pos = sceneops.boundsCenter(&sim_data.coords);
    const distortion = buildDistortion();

    // Camera 0: face on (reference camera)
    const cam0_rot = Rotation.init(
        std.math.degreesToRadians(0.0),
        std.math.degreesToRadians(0.0),
        std.math.degreesToRadians(0.0),
    );
    const cam0_pos = cameraops.posFillFrameFromRot(
        &sim_data.coords,
        PIXELS_NUM,
        PIXELS_SIZE,
        FOCAL_LENGTH,
        cam0_rot,
        FOV_SCALE_FACTOR,
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

    // Camera 1: stereo angle
    const cam1_rot = Rotation.init(
        std.math.degreesToRadians(0.0),
        std.math.degreesToRadians(STEREO_ANGLE_DEG),
        std.math.degreesToRadians(0.0),
    );
    const cam1_pos = cameraops.posFillFrameFromRot(
        &sim_data.coords,
        PIXELS_NUM,
        PIXELS_SIZE,
        FOCAL_LENGTH,
        cam1_rot,
        FOV_SCALE_FACTOR,
    );
    const cam1_in = CameraInput{
        .pixels_num = PIXELS_NUM,
        .pixels_size = PIXELS_SIZE,
        .pos_world = cam1_pos,
        .rot_world = cam1_rot,
        .roi_cent_world = roi_pos,
        .focal_length = FOCAL_LENGTH,
        .sub_sample = SUB_SAMPLE,
        .distortion = distortion,
    };

    // 6. Prepare Mesh Input with ProjectedTex Shader
    std.debug.print("Configuring ProjectedTex 2D image warper...\n", .{});
    const mesh_input = MeshInput{
        .mesh_type = .quad8,
        .coords = sim_data.coords,
        .connect = sim_data.connect,
        .disp = disp_field,
        .shader = .{
            .projected_tex_u8 = .{
                .ref_camera = cam0_in,
                .ref_coords = null,
                .tex = texture,
            },
        },
    };

    // 7. Run the Rasteriser
    std.debug.print(
        "Rendering projected 2D image warping to {s}/...\n",
        .{OUT_DIR_ROOT},
    );
    const meshes = [_]MeshInput{mesh_input};
    const cams_in = [_]CameraInput{ cam0_in, cam1_in };

    std.Io.Dir.cwd().deleteTree(io, OUT_DIR_ROOT) catch |err| {
        if (err != error.FileNotFound) return err;
    };

    const images = try riley.raster(
        aa,
        render_groups,
        &cams_in,
        &meshes,
        config,
        OUT_DIR_ROOT,
    );

    if (images) |img| {
        aa.free(img.slice);
        img.deinit(aa);
    }

    var out_dir = try std.Io.Dir.cwd().openDir(io, OUT_DIR_ROOT, .{});
    defer out_dir.close(io);

    var cam0_opengl = cam0_in;
    cam0_opengl.coord_sys = .opengl;
    var cam1_opengl = cam1_in;
    cam1_opengl.coord_sys = .opengl;
    try cameraio.saveStereoPair(
        io,
        out_dir,
        "stereo_data_opengl.csv",
        .{ .cameras = .{ cam0_opengl, cam1_opengl } },
    );

    var cam0_opencv = cam0_in;
    cam0_opencv.coord_sys = .opencv;
    var cam1_opencv = cam1_in;
    cam1_opencv.coord_sys = .opencv;
    try cameraio.saveStereoPair(
        io,
        out_dir,
        "stereo_data_opencv.csv",
        .{ .cameras = .{ cam0_opencv, cam1_opencv } },
    );

    std.debug.print(
        "Demo complete. Warped images saved to {s}/\n",
        .{OUT_DIR_ROOT},
    );
}
