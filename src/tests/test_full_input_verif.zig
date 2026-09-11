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
const common_full = @import("../gengold/gen_gold_full_common.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const iio = @import("../riley/zig/imageio.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const ndarray = @import("../riley/zig/ndarray.zig");
const rastcfg = @import("../riley/zig/rasterconfig.zig");
const report = @import("../riley/zig/report.zig");
const riley = @import("../riley/zig/riley.zig");
const tcfg = @import("../dev_support/testconfig.zig");
const valinp = @import("../riley/zig/validateinput.zig");

const F = buildconfig.F;
const CameraInput = camera.CameraInput;
const MeshInput = mo.MeshInput;
const RenderGroupSpec = riley.RenderGroupSpec;
const RasterConfig = rastcfg.RasterConfig;
const NDArray = ndarray.NDArray(F);

const BaselineFixture = struct {
    prep: common_full.Scene3Prepared,
    textures: common_full.FullTextures,
    config: RasterConfig,

    pub fn init(allocator: std.mem.Allocator, io: std.Io) !BaselineFixture {
        var prep = try common_full.prepareScene3(allocator, io);
        errdefer prep.deinit(allocator);

        var textures = try common_full.FullTextures.init(allocator, io);
        errdefer textures.deinit(allocator);

        var config = tcfg.getRasterConfig(.testing);
        config.save_strategy = .memory;
        config.image_save_mode = .grey;
        config.background_value = 0.5;

        return .{
            .prep = prep,
            .textures = textures,
            .config = config,
        };
    }

    pub fn deinit(self: *BaselineFixture, allocator: std.mem.Allocator) void {
        self.prep.deinit(allocator);
        self.textures.deinit(allocator);
    }
};

fn runRender(
    render_groups: []const RenderGroupSpec,
    cam_inps: []const CameraInput,
    mesh_inps: []const MeshInput,
    config: RasterConfig,
) !?NDArray {
    return riley.rasterReport(
        std.testing.allocator,
        render_groups,
        cam_inps,
        mesh_inps,
        config,
        null,
        null,
    );
}

// --------------------------------------------------------------------------
// Top-Level Input Cardinality Tests
// --------------------------------------------------------------------------

fn testNoRenderGroups(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    _ = io;
    const cam_inps = common_full.createScene3Cameras();
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);

    const empty_groups = [_]RenderGroupSpec{};
    try std.testing.expectError(
        error.NoRenderGroups,
        runRender(&empty_groups, &cam_inps, &mesh_inps, fixture.config),
    );
}

fn testNoCameras(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    const empty_cams = [_]CameraInput{};
    try std.testing.expectError(
        error.NoCameras,
        runRender(&render_groups, &empty_cams, &mesh_inps, fixture.config),
    );
}

fn testNoMeshes(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    const cam_inps = common_full.createScene3Cameras();
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    const empty_meshes = [_]MeshInput{};
    try std.testing.expectError(
        error.NoMeshes,
        runRender(&render_groups, &cam_inps, &empty_meshes, fixture.config),
    );
}

// --------------------------------------------------------------------------
// Mesh & Solver Compatibility Tests
// --------------------------------------------------------------------------

fn testDistortionNotSuppedWithTri3Opt(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var cam_inps = common_full.createScene3Cameras();
    cam_inps[0].distortion = .{
        .brown_conrady = .{
            .k1 = 0.01,
            .k2 = 0.0,
            .k3 = 0.0,
            .p1 = 0.0,
            .p2 = 0.0,
        },
    };
    var mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    mesh_inps[0].mesh_type = .tri3opt;

    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};
    try std.testing.expectError(
        error.DistortionNotSuppedWithTri3Opt,
        runRender(&render_groups, &cam_inps, &mesh_inps, fixture.config),
    );
}

// --------------------------------------------------------------------------
// Render Group & Threading Config Tests
// --------------------------------------------------------------------------

fn testInvalidRenderGroupWorkers(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    const cam_inps = common_full.createScene3Cameras();
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 0 }};

    try std.testing.expectError(
        error.InvalidRenderGroupWorkers,
        runRender(&render_groups, &cam_inps, &mesh_inps, fixture.config),
    );
}

fn testFullStatsRequiresSingleRasterWorker(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    const cam_inps = common_full.createScene3Cameras();
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 2 }};

    var config = fixture.config;
    config.report = .full_stats;
    config.max_raster_workers_per_job = 2;

    try std.testing.expectError(
        error.FullStatsRequiresSingleRasterWorker,
        runRender(&render_groups, &cam_inps, &mesh_inps, config),
    );
}

fn testInvalidTotalThreads(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    const cam_inps = common_full.createScene3Cameras();
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    var config = fixture.config;
    config.total_threads = 0;

    try std.testing.expectError(
        error.InvalidTotalThreads,
        runRender(&render_groups, &cam_inps, &mesh_inps, config),
    );
}

fn testInvalidFrameBatchSize(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    const cam_inps = common_full.createScene3Cameras();
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    var config = fixture.config;
    config.frame_batch_size_per_group = 0;

    try std.testing.expectError(
        error.InvalidFrameBatchSize,
        runRender(&render_groups, &cam_inps, &mesh_inps, config),
    );
}

fn testInvalidGeomJobsInFlight(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    const cam_inps = common_full.createScene3Cameras();
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    var config = fixture.config;
    config.max_geom_jobs_in_flight_per_group = 0;

    try std.testing.expectError(
        error.InvalidGeomJobsInFlight,
        runRender(&render_groups, &cam_inps, &mesh_inps, config),
    );
}

fn testInvalidGeomWorkersPerJob(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    const cam_inps = common_full.createScene3Cameras();
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    var config = fixture.config;
    config.max_geom_workers_per_job = 0;

    try std.testing.expectError(
        error.InvalidGeomWorkersPerJob,
        runRender(&render_groups, &cam_inps, &mesh_inps, config),
    );
}

fn testInvalidRasterWorkersPerJob(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    const cam_inps = common_full.createScene3Cameras();
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    var config = fixture.config;
    config.max_raster_workers_per_job = 0;

    try std.testing.expectError(
        error.InvalidRasterWorkersPerJob,
        runRender(&render_groups, &cam_inps, &mesh_inps, config),
    );
}

// --------------------------------------------------------------------------
// Tile Size and Subpixel Tiling Config Tests
// --------------------------------------------------------------------------

fn testInvalidTileSizeMin(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    const cam_inps = common_full.createScene3Cameras();
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    var config = fixture.config;
    config.tile_size_min = 0;

    try std.testing.expectError(
        error.InvalidTileSizeMin,
        runRender(&render_groups, &cam_inps, &mesh_inps, config),
    );
}

fn testInvalidTileSizeMax(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    const cam_inps = common_full.createScene3Cameras();
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    var config = fixture.config;
    config.tile_size_max = 0;

    try std.testing.expectError(
        error.InvalidTileSizeMax,
        runRender(&render_groups, &cam_inps, &mesh_inps, config),
    );
}

fn testInvalidTileSizeRange(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    const cam_inps = common_full.createScene3Cameras();
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    var config = fixture.config;
    config.tile_size_min = 64;
    config.tile_size_max = 32;

    try std.testing.expectError(
        error.InvalidTileSizeRange,
        runRender(&render_groups, &cam_inps, &mesh_inps, config),
    );
}

fn testInvalidTileSizeOverride(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    const cam_inps = common_full.createScene3Cameras();
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    var config = fixture.config;
    config.tile_size_min = 16;
    config.tile_size_max = 64;
    config.tile_size_override = 128;

    try std.testing.expectError(
        error.InvalidTileSizeOverride,
        runRender(&render_groups, &cam_inps, &mesh_inps, config),
    );
}

fn testInvalidGlobalSubpxTileSizeMin(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    const cam_inps = common_full.createScene3Cameras();
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    var config = fixture.config;
    config.global_subpx_tile_size_min = 0;

    try std.testing.expectError(
        error.InvalidGlobalSubpxTileSizeMin,
        runRender(&render_groups, &cam_inps, &mesh_inps, config),
    );
}

fn testInvalidGlobalSubpxTileSizeMax(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    const cam_inps = common_full.createScene3Cameras();
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    var config = fixture.config;
    config.global_subpx_tile_size_max = 0;

    try std.testing.expectError(
        error.InvalidGlobalSubpxTileSizeMax,
        runRender(&render_groups, &cam_inps, &mesh_inps, config),
    );
}

fn testInvalidGlobalSubpxTileSizeRange(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    const cam_inps = common_full.createScene3Cameras();
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    var config = fixture.config;
    config.global_subpx_tile_size_min = 128;
    config.global_subpx_tile_size_max = 64;

    try std.testing.expectError(
        error.InvalidGlobalSubpxTileSizeRange,
        runRender(&render_groups, &cam_inps, &mesh_inps, config),
    );
}

fn testInvalidGlobalSubpxTileSizeOverride(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    const cam_inps = common_full.createScene3Cameras();
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    var config = fixture.config;
    config.global_subpx_tile_size_min = 16;
    config.global_subpx_tile_size_max = 64;
    config.global_subpx_tile_size_override = 128;

    try std.testing.expectError(
        error.InvalidGlobalSubpxTileSizeOverride,
        runRender(&render_groups, &cam_inps, &mesh_inps, config),
    );
}

fn testInvalidGlobalSubpxStripeSizeMin(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    const cam_inps = common_full.createScene3Cameras();
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    var config = fixture.config;
    config.global_subpx_stripe_size_min = 0;

    try std.testing.expectError(
        error.InvalidGlobalSubpxStripeSizeMin,
        runRender(&render_groups, &cam_inps, &mesh_inps, config),
    );
}

fn testInvalidGlobalSubpxStripeSizeMax(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    const cam_inps = common_full.createScene3Cameras();
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    var config = fixture.config;
    config.global_subpx_stripe_size_max = 0;

    try std.testing.expectError(
        error.InvalidGlobalSubpxStripeSizeMax,
        runRender(&render_groups, &cam_inps, &mesh_inps, config),
    );
}

fn testInvalidGlobalSubpxStripeSizeRange(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    const cam_inps = common_full.createScene3Cameras();
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    var config = fixture.config;
    config.global_subpx_stripe_size_min = 128;
    config.global_subpx_stripe_size_max = 64;

    try std.testing.expectError(
        error.InvalidGlobalSubpxStripeSizeRange,
        runRender(&render_groups, &cam_inps, &mesh_inps, config),
    );
}

fn testInvalidGlobalSubpxStripeSizeOverride(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    const cam_inps = common_full.createScene3Cameras();
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    var config = fixture.config;
    config.global_subpx_stripe_size_min = 16;
    config.global_subpx_stripe_size_max = 64;
    config.global_subpx_stripe_size_override = 128;

    try std.testing.expectError(
        error.InvalidGlobalSubpxStripeSizeOverride,
        runRender(&render_groups, &cam_inps, &mesh_inps, config),
    );
}

// --------------------------------------------------------------------------
// Background Value, Save Strategy & Full Stats Config Tests
// --------------------------------------------------------------------------

fn testInvalidBackgroundValue(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    const cam_inps = common_full.createScene3Cameras();
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    var config = fixture.config;
    config.background_value = std.math.nan(F);

    try std.testing.expectError(
        error.InvalidBackgroundValue,
        runRender(&render_groups, &cam_inps, &mesh_inps, config),
    );
}

fn testInvalidImageSaveOpts(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    const cam_inps = common_full.createScene3Cameras();
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    var config = fixture.config;
    config.save_strategy = .disk;
    config.image_save_opts = &[_]iio.ImageSaveOpts{};

    try std.testing.expectError(
        error.InvalidImageSaveOpts,
        runRender(&render_groups, &cam_inps, &mesh_inps, config),
    );
}

fn testInvalidFullStatsFormats(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    const cam_inps = common_full.createScene3Cameras();
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    var config = fixture.config;
    config.report = .full_stats;
    config.max_raster_workers_per_job = 1;
    config.full_stats_opts.formats = &[_]iio.ImageSaveOpts{};

    try std.testing.expectError(
        error.InvalidFullStatsFormats,
        runRender(&render_groups, &cam_inps, &mesh_inps, config),
    );
}

// --------------------------------------------------------------------------
// Camera Parameter Verification Tests
// --------------------------------------------------------------------------

fn testInvalidCameraPixels(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var cam_inps = common_full.createScene3Cameras();
    cam_inps[0].pixels_num = [2]u32{ 0, 100 };
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    try std.testing.expectError(
        error.InvalidCameraPixels,
        runRender(&render_groups, &cam_inps, &mesh_inps, fixture.config),
    );
}

fn testInvalidCameraPixelSize(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var cam_inps = common_full.createScene3Cameras();
    cam_inps[0].pixels_size = [2]F{ -1.0, 5.3e-6 };
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    try std.testing.expectError(
        error.InvalidCameraPixelSize,
        runRender(&render_groups, &cam_inps, &mesh_inps, fixture.config),
    );
}

fn testInvalidCameraFocalLength(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var cam_inps = common_full.createScene3Cameras();
    cam_inps[0].focal_length = 0.0;
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    try std.testing.expectError(
        error.InvalidCameraFocalLength,
        runRender(&render_groups, &cam_inps, &mesh_inps, fixture.config),
    );
}

fn testInvalidCameraSubSample(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var cam_inps = common_full.createScene3Cameras();
    cam_inps[0].sub_sample = 0;
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    try std.testing.expectError(
        error.InvalidCameraSubSample,
        runRender(&render_groups, &cam_inps, &mesh_inps, fixture.config),
    );
}

fn testNonFiniteCameraInput(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var cam_inps = common_full.createScene3Cameras();
    cam_inps[0].pos_world.slice[0] = std.math.nan(F);
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    try std.testing.expectError(
        error.NonFiniteCameraInput,
        runRender(&render_groups, &cam_inps, &mesh_inps, fixture.config),
    );
}

fn testInvalidCameraDistortion(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var cam_inps = common_full.createScene3Cameras();
    cam_inps[0].distortion = .{
        .brown_conrady = .{
            .k1 = std.math.nan(F),
            .k2 = 0.0,
            .k3 = 0.0,
            .p1 = 0.0,
            .p2 = 0.0,
        },
    };
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    try std.testing.expectError(
        error.InvalidCameraDistortion,
        runRender(&render_groups, &cam_inps, &mesh_inps, fixture.config),
    );
}

fn testInvalidCameraPsf(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var cam_inps = common_full.createScene3Cameras();
    cam_inps[0].psf = .{
        .gaussian = .{
            .sigma_px = -1.0,
            .supp_rad_px = 2.0,
        },
    };
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    try std.testing.expectError(
        error.InvalidCameraPsf,
        runRender(&render_groups, &cam_inps, &mesh_inps, fixture.config),
    );
}

// --------------------------------------------------------------------------
// Subpixel Alignment Tests
// --------------------------------------------------------------------------

fn testGlobalSubpxTileSizeNotAligned(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var cam_inps = common_full.createScene3Cameras();
    cam_inps[0].sub_sample = 2;
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    var config = fixture.config;
    config.buffer_mode = .global_subpx_full;
    config.global_subpx_tile_size_min = 16;
    config.global_subpx_tile_size_max = 64;
    config.global_subpx_tile_size_override = 25;

    try std.testing.expectError(
        error.GlobalSubpxTileSizeNotAligned,
        runRender(&render_groups, &cam_inps, &mesh_inps, config),
    );
}

fn testGlobalSubpxStripeSizeNotAligned(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var cam_inps = common_full.createScene3Cameras();
    cam_inps[0].sub_sample = 2;
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    var config = fixture.config;
    config.buffer_mode = .global_subpx_stripe;
    config.global_subpx_stripe_size_min = 16;
    config.global_subpx_stripe_size_max = 64;
    config.global_subpx_stripe_size_override = 25;

    try std.testing.expectError(
        error.GlobalSubpxStripeSizeNotAligned,
        runRender(&render_groups, &cam_inps, &mesh_inps, config),
    );
}

// --------------------------------------------------------------------------
// Benchmark Capture & Output Buffer Tests
// --------------------------------------------------------------------------

fn testInvalidBenchCaptureBuff(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    const cam_inps = common_full.createScene3Cameras();
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    var invalid_capt = [_]report.FrameBenchCapture{std.mem.zeroes(report.FrameBenchCapture)};

    try std.testing.expectError(
        error.InvalidBenchCaptureBuff,
        riley.rasterReport(
            std.testing.allocator,
            &render_groups,
            &cam_inps,
            &mesh_inps,
            fixture.config,
            null,
            invalid_capt[0..],
        ),
    );
}

fn testUnsuppedImageModeFieldCount(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    _ = io;
    _ = fixture;

    try std.testing.expectError(
        error.UnsuppedImageModeFieldCount,
        valinp.calcOutFieldsForImgSaveMode(.grey, 2),
    );
    try std.testing.expectError(
        error.UnsuppedImageModeFieldCount,
        valinp.calcOutFieldsForImgSaveMode(.rgb, 2),
    );
    try std.testing.expectError(
        error.UnsuppedImageModeFieldCount,
        valinp.calcOutFieldsForImgSaveMode(.grey, 4),
    );
}

fn testInvalidOutputBuff(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    const cam_inps = common_full.createScene3Cameras();
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    // Case A: save_strategy == .memory but output buffer is null when required
    try std.testing.expectError(
        error.InvalidOutputBuff,
        riley.rasterReportInto(
            allocator,
            &render_groups,
            &cam_inps,
            &mesh_inps,
            fixture.config,
            null,
            null,
            null,
        ),
    );

    // Case B: save_strategy == .disk but output buffer is provided
    var dummy_dims = [_]usize{ 1, 1, 1, 100, 160 };
    var dummy_arr = try NDArray.initFlat(allocator, dummy_dims[0..]);
    defer {
        allocator.free(dummy_arr.slice);
        dummy_arr.deinit(allocator);
    }

    var disk_config = fixture.config;
    disk_config.save_strategy = .disk;
    const disk_opts = [_]iio.ImageSaveOpts{.{ .format = .bmp }};
    disk_config.image_save_opts = &disk_opts;

    try std.testing.expectError(
        error.InvalidOutputBuff,
        valinp.checkRenderInps(
            &render_groups,
            &cam_inps,
            &mesh_inps,
            disk_config,
            &dummy_arr,
            false,
            null,
        ),
    );

    // Case C: save_strategy == .memory but buffer dimension mismatch
    var bad_dims = [_]usize{ 1, 1, 1, 50, 50 };
    var bad_arr = try NDArray.initFlat(allocator, bad_dims[0..]);
    defer {
        allocator.free(bad_arr.slice);
        bad_arr.deinit(allocator);
    }

    try std.testing.expectError(
        error.InvalidOutputBuff,
        riley.rasterReportInto(
            allocator,
            &render_groups,
            &cam_inps,
            &mesh_inps,
            fixture.config,
            null,
            &bad_arr,
            null,
        ),
    );
}

// --------------------------------------------------------------------------
// Mixed / Multiple Error Precedence Tests
// --------------------------------------------------------------------------

fn testMixedCaseMultipleConfigErrors(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    const cam_inps = common_full.createScene3Cameras();
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    var config = fixture.config;
    config.total_threads = 0;
    config.tile_size_min = 0;
    config.background_value = std.math.nan(F);

    // Should return the first validation error encountered
    try std.testing.expectError(
        error.InvalidTotalThreads,
        runRender(&render_groups, &cam_inps, &mesh_inps, config),
    );
}

fn testMixedCaseCameraAndConfigErrors(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var cam_inps = common_full.createScene3Cameras();
    cam_inps[0].pixels_num = [2]u32{ 0, 0 };
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    var config = fixture.config;
    config.tile_size_min = 64;
    config.tile_size_max = 32;

    // Config validation precedes camera validation
    try std.testing.expectError(
        error.InvalidTileSizeRange,
        runRender(&render_groups, &cam_inps, &mesh_inps, config),
    );
}

fn testMixedCaseEmptyInputs(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    _ = io;
    const empty_groups = [_]RenderGroupSpec{};
    const empty_cams = [_]CameraInput{};
    const empty_meshes = [_]MeshInput{};

    // Group check precedes camera and mesh checks
    try std.testing.expectError(
        error.NoRenderGroups,
        runRender(&empty_groups, &empty_cams, &empty_meshes, fixture.config),
    );
}

// --------------------------------------------------------------------------
// Public Sub-Suite Runner
// --------------------------------------------------------------------------

pub fn run(allocator: std.mem.Allocator, io: std.Io) !void {
    var fixture = try BaselineFixture.init(allocator, io);
    defer fixture.deinit(allocator);

    // Cardinality
    try testNoRenderGroups(allocator, io, &fixture);
    try testNoCameras(allocator, io, &fixture);
    try testNoMeshes(allocator, io, &fixture);

    // Compatibility
    try testDistortionNotSuppedWithTri3Opt(allocator, io, &fixture);

    // Render Groups & Threading
    try testInvalidRenderGroupWorkers(allocator, io, &fixture);
    try testFullStatsRequiresSingleRasterWorker(allocator, io, &fixture);
    try testInvalidTotalThreads(allocator, io, &fixture);
    try testInvalidFrameBatchSize(allocator, io, &fixture);
    try testInvalidGeomJobsInFlight(allocator, io, &fixture);
    try testInvalidGeomWorkersPerJob(allocator, io, &fixture);
    try testInvalidRasterWorkersPerJob(allocator, io, &fixture);

    // Tile Size & Subpixel Tiling
    try testInvalidTileSizeMin(allocator, io, &fixture);
    try testInvalidTileSizeMax(allocator, io, &fixture);
    try testInvalidTileSizeRange(allocator, io, &fixture);
    try testInvalidTileSizeOverride(allocator, io, &fixture);
    try testInvalidGlobalSubpxTileSizeMin(allocator, io, &fixture);
    try testInvalidGlobalSubpxTileSizeMax(allocator, io, &fixture);
    try testInvalidGlobalSubpxTileSizeRange(allocator, io, &fixture);
    try testInvalidGlobalSubpxTileSizeOverride(allocator, io, &fixture);
    try testInvalidGlobalSubpxStripeSizeMin(allocator, io, &fixture);
    try testInvalidGlobalSubpxStripeSizeMax(allocator, io, &fixture);
    try testInvalidGlobalSubpxStripeSizeRange(allocator, io, &fixture);
    try testInvalidGlobalSubpxStripeSizeOverride(allocator, io, &fixture);

    // Config options
    try testInvalidBackgroundValue(allocator, io, &fixture);
    try testInvalidImageSaveOpts(allocator, io, &fixture);
    try testInvalidFullStatsFormats(allocator, io, &fixture);

    // Camera parameters
    try testInvalidCameraPixels(allocator, io, &fixture);
    try testInvalidCameraPixelSize(allocator, io, &fixture);
    try testInvalidCameraFocalLength(allocator, io, &fixture);
    try testInvalidCameraSubSample(allocator, io, &fixture);
    try testNonFiniteCameraInput(allocator, io, &fixture);
    try testInvalidCameraDistortion(allocator, io, &fixture);
    try testInvalidCameraPsf(allocator, io, &fixture);

    // Alignment
    try testGlobalSubpxTileSizeNotAligned(allocator, io, &fixture);
    try testGlobalSubpxStripeSizeNotAligned(allocator, io, &fixture);

    // Benchmark buffer & output buffer
    try testInvalidBenchCaptureBuff(allocator, io, &fixture);
    try testUnsuppedImageModeFieldCount(allocator, io, &fixture);
    try testInvalidOutputBuff(allocator, io, &fixture);

    // Mixed error cases
    try testMixedCaseMultipleConfigErrors(allocator, io, &fixture);
    try testMixedCaseCameraAndConfigErrors(allocator, io, &fixture);
    try testMixedCaseEmptyInputs(allocator, io, &fixture);
}
