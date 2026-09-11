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
const common_full = @import("../dev_support/fullfixtures.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const iio = @import("../riley/zig/imageio.zig");
const matslice = @import("../riley/zig/matslice.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const ndarray = @import("../riley/zig/ndarray.zig");
const rastcfg = @import("../riley/zig/rasterconfig.zig");
const report = @import("../riley/zig/report.zig");
const riley = @import("../riley/zig/riley.zig");
const shaderops = @import("../riley/zig/shaderops_common.zig");
const tcfg = @import("../dev_support/testconfig.zig");
const valarr = @import("../riley/zig/validatearrays.zig");
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

fn expectConfigError(
    io: std.Io,
    fixture: *const BaselineFixture,
    config: RasterConfig,
    expected_error: anyerror,
) !void {
    const cam_inps = common_full.createScene3Cameras();
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};
    try std.testing.expectError(
        expected_error,
        runRender(&render_groups, &cam_inps, &mesh_inps, config),
    );
}

fn expectCameraError(
    io: std.Io,
    fixture: *const BaselineFixture,
    cam_inps: []const CameraInput,
    expected_error: anyerror,
) !void {
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};
    try std.testing.expectError(
        expected_error,
        runRender(&render_groups, cam_inps, &mesh_inps, fixture.config),
    );
}

fn expectRenderInputError(
    io: std.Io,
    fixture: *const BaselineFixture,
    mesh_inps: []const MeshInput,
    config: ?RasterConfig,
    expected_error: anyerror,
) !void {
    const cam_inps = common_full.createScene3Cameras();
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};
    const active_config = config orelse fixture.config;
    try std.testing.expectError(
        expected_error,
        runRender(&render_groups, &cam_inps, mesh_inps, active_config),
    );
}

// --------------------------------------------------------------------------
// Validation Mode Tests
// --------------------------------------------------------------------------

fn testValidateInputModes(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    const cam_inps = common_full.createScene3Cameras();
    const mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    // Default configuration has validate_input = .fast
    const default_config = fixture.config;
    try std.testing.expectEqual(rastcfg.ValidateInput.fast, default_config.validate_input);

    // .off mode succeeds on valid input
    var off_config = fixture.config;
    off_config.validate_input = .off;
    const off_result = try runRender(&render_groups, &cam_inps, &mesh_inps, off_config);
    if (off_result) |*arr| {
        allocator.free(arr.slice);
        arr.deinit(allocator);
    }

    // .fast mode succeeds on valid input
    var fast_config = fixture.config;
    fast_config.validate_input = .fast;
    const fast_result = try runRender(&render_groups, &cam_inps, &mesh_inps, fast_config);
    if (fast_result) |*arr| {
        allocator.free(arr.slice);
        arr.deinit(allocator);
    }

    // .full mode succeeds on valid input
    var full_config = fixture.config;
    full_config.validate_input = .full;
    const full_result = try runRender(&render_groups, &cam_inps, &mesh_inps, full_config);
    if (full_result) |*arr| {
        allocator.free(arr.slice);
        arr.deinit(allocator);
    }
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
    var config = fixture.config;
    config.total_threads = 0;
    try expectConfigError(io, fixture, config, error.InvalidTotalThreads);
}

fn testInvalidFrameBatchSize(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var config = fixture.config;
    config.frame_batch_size_per_group = 0;
    try expectConfigError(io, fixture, config, error.InvalidFrameBatchSize);
}

fn testInvalidGeomJobsInFlight(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var config = fixture.config;
    config.max_geom_jobs_in_flight_per_group = 0;
    try expectConfigError(io, fixture, config, error.InvalidGeomJobsInFlight);
}

fn testInvalidGeomWorkersPerJob(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var config = fixture.config;
    config.max_geom_workers_per_job = 0;
    try expectConfigError(io, fixture, config, error.InvalidGeomWorkersPerJob);
}

fn testInvalidRasterWorkersPerJob(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var config = fixture.config;
    config.max_raster_workers_per_job = 0;
    try expectConfigError(io, fixture, config, error.InvalidRasterWorkersPerJob);
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
    var config = fixture.config;
    config.tile_size_min = 0;
    try expectConfigError(io, fixture, config, error.InvalidTileSizeMin);
}

fn testInvalidTileSizeMax(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var config = fixture.config;
    config.tile_size_max = 0;
    try expectConfigError(io, fixture, config, error.InvalidTileSizeMax);
}

fn testInvalidTileSizeRange(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var config = fixture.config;
    config.tile_size_min = 64;
    config.tile_size_max = 32;
    try expectConfigError(io, fixture, config, error.InvalidTileSizeRange);
}

fn testInvalidTileSizeOverride(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var config = fixture.config;
    config.tile_size_min = 16;
    config.tile_size_max = 64;
    config.tile_size_override = 128;
    try expectConfigError(io, fixture, config, error.InvalidTileSizeOverride);
}

fn testInvalidGlobalSubpxTileSizeMin(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var config = fixture.config;
    config.global_subpx_tile_size_min = 0;
    try expectConfigError(io, fixture, config, error.InvalidGlobalSubpxTileSizeMin);
}

fn testInvalidGlobalSubpxTileSizeMax(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var config = fixture.config;
    config.global_subpx_tile_size_max = 0;
    try expectConfigError(io, fixture, config, error.InvalidGlobalSubpxTileSizeMax);
}

fn testInvalidGlobalSubpxTileSizeRange(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var config = fixture.config;
    config.global_subpx_tile_size_min = 128;
    config.global_subpx_tile_size_max = 64;
    try expectConfigError(io, fixture, config, error.InvalidGlobalSubpxTileSizeRange);
}

fn testInvalidGlobalSubpxTileSizeOverride(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var config = fixture.config;
    config.global_subpx_tile_size_min = 16;
    config.global_subpx_tile_size_max = 64;
    config.global_subpx_tile_size_override = 128;
    try expectConfigError(io, fixture, config, error.InvalidGlobalSubpxTileSizeOverride);
}

fn testInvalidGlobalSubpxStripeSizeMin(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var config = fixture.config;
    config.global_subpx_stripe_size_min = 0;
    try expectConfigError(io, fixture, config, error.InvalidGlobalSubpxStripeSizeMin);
}

fn testInvalidGlobalSubpxStripeSizeMax(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var config = fixture.config;
    config.global_subpx_stripe_size_max = 0;
    try expectConfigError(io, fixture, config, error.InvalidGlobalSubpxStripeSizeMax);
}

fn testInvalidGlobalSubpxStripeSizeRange(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var config = fixture.config;
    config.global_subpx_stripe_size_min = 128;
    config.global_subpx_stripe_size_max = 64;
    try expectConfigError(io, fixture, config, error.InvalidGlobalSubpxStripeSizeRange);
}

fn testInvalidGlobalSubpxStripeSizeOverride(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var config = fixture.config;
    config.global_subpx_stripe_size_min = 16;
    config.global_subpx_stripe_size_max = 64;
    config.global_subpx_stripe_size_override = 128;
    try expectConfigError(io, fixture, config, error.InvalidGlobalSubpxStripeSizeOverride);
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
    var config = fixture.config;
    config.background_value = std.math.nan(F);
    try expectConfigError(io, fixture, config, error.InvalidBackgroundValue);
}

fn testInvalidSaveFrameBuffCount(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var config = fixture.config;
    config.save_frame_buff_count = 0;
    try expectConfigError(io, fixture, config, error.InvalidSaveFrameBuffCount);
}

fn testInvalidImageSaveOpts(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var config = fixture.config;
    config.save_strategy = .disk;
    config.image_save_opts = &[_]iio.ImageSaveOpts{};
    try expectConfigError(io, fixture, config, error.InvalidImageSaveOpts);
}

fn testInvalidFullStatsFormats(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var config = fixture.config;
    config.report = .full_stats;
    config.max_raster_workers_per_job = 1;
    config.full_stats_opts.formats = &[_]iio.ImageSaveOpts{};
    try expectConfigError(io, fixture, config, error.InvalidFullStatsFormats);
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
    try expectCameraError(io, fixture, &cam_inps, error.InvalidCameraPixels);
}

fn testInvalidCameraPixelSize(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var cam_inps = common_full.createScene3Cameras();
    cam_inps[0].pixels_size = [2]F{ -1.0, 5.3e-6 };
    try expectCameraError(io, fixture, &cam_inps, error.InvalidCameraPixelSize);
}

fn testInvalidCameraFocalLength(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var cam_inps = common_full.createScene3Cameras();
    cam_inps[0].focal_length = 0.0;
    try expectCameraError(io, fixture, &cam_inps, error.InvalidCameraFocalLength);
}

fn testInvalidCameraSubSample(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var cam_inps = common_full.createScene3Cameras();
    cam_inps[0].sub_sample = 0;
    try expectCameraError(io, fixture, &cam_inps, error.InvalidCameraSubSample);
}

fn testInvalidCameraRoi(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var cam_inps = common_full.createScene3Cameras();
    cam_inps[0].pos_world.slice[0] = std.math.nan(F);
    try expectCameraError(io, fixture, &cam_inps, error.InvalidCameraRoi);
}

fn testInvalidCameraRotation(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var cam_inps = common_full.createScene3Cameras();
    cam_inps[0].rot_world.matrix.set(0, 0, 2.0);
    try expectCameraError(io, fixture, &cam_inps, error.InvalidCameraRotation);
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
    try expectCameraError(io, fixture, &cam_inps, error.InvalidCameraDistortion);
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
    try expectCameraError(io, fixture, &cam_inps, error.InvalidCameraPsf);
}

// --------------------------------------------------------------------------
// Mesh & Shader Structural Verification Tests
// --------------------------------------------------------------------------

fn testZeroCoordinateCount(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    mesh_inps[0].coords.mat.rows_num = 0;
    try expectRenderInputError(io, fixture, &mesh_inps, null, error.ZeroCoordinateCount);
}

fn testInvalidCoordinateDimensions(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    mesh_inps[0].coords.mat.cols_num = 2;
    try expectRenderInputError(
        io,
        fixture,
        &mesh_inps,
        null,
        error.InvalidCoordinateDimensions,
    );
}

fn testZeroElementCount(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    mesh_inps[0].connect.table.rows_num = 0;
    try expectRenderInputError(io, fixture, &mesh_inps, null, error.ZeroElementCount);
}

fn testInvalidConnectivityDimensions(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    mesh_inps[0].connect.table.cols_num = 2;
    try expectRenderInputError(
        io,
        fixture,
        &mesh_inps,
        null,
        error.InvalidConnectivityDimensions,
    );
}

fn testInvalidDisplacementDimensions(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    if (mesh_inps[0].disp) |*disp_field| {
        const orig_dim = disp_field.array.dims[2];
        disp_field.array.dims[2] = 2;
        defer disp_field.array.dims[2] = orig_dim;

        try expectRenderInputError(
            io,
            fixture,
            &mesh_inps,
            null,
            error.InvalidDisplacementDimensions,
        );
    }
}

fn testInvalidNodalFieldDimensions(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    switch (mesh_inps[0].shader) {
        .nodal => |*nodal_shader| {
            const orig_dim = nodal_shader.field.array.dims[1];
            nodal_shader.field.array.dims[1] = 1;
            defer nodal_shader.field.array.dims[1] = orig_dim;

            try expectRenderInputError(
                io,
                fixture,
                &mesh_inps,
                null,
                error.InvalidNodalFieldDimensions,
            );
        },
        else => {},
    }
}

fn testInvalidTexSampleConfig(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    switch (mesh_inps[2].shader) {
        .tex_u8 => |*tex_shader| {
            tex_shader.samp_cfg = .{
                .sample = .nearest,
                .mode = .lut,
            };
        },
        else => {},
    }
    try expectRenderInputError(
        io,
        fixture,
        &mesh_inps,
        null,
        error.InvalidTexSampleConfig,
    );
}

fn testInvalidShaderBitDepth(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    switch (mesh_inps[0].shader) {
        .nodal => |*nodal_shader| {
            nodal_shader.bits = 7;
        },
        else => {},
    }
    try expectRenderInputError(
        io,
        fixture,
        &mesh_inps,
        null,
        error.InvalidShaderBitDepth,
    );
}

fn testInvalidScalingBounds(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    switch (mesh_inps[0].shader) {
        .nodal => |*nodal_shader| {
            nodal_shader.scaling = .{ .fixed = .{ 10.0, 5.0 } };
        },
        else => {},
    }
    try expectRenderInputError(
        io,
        fixture,
        &mesh_inps,
        null,
        error.InvalidScalingBounds,
    );
}

fn testInvalidFuncShaderParams(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    switch (mesh_inps[1].shader) {
        .func => |*func_shader| {
            func_shader.params.coord_scale[0] = 0.0;
        },
        else => {},
    }
    try expectRenderInputError(
        io,
        fixture,
        &mesh_inps,
        null,
        error.InvalidFuncShaderParams,
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
// Full Payload Scan Verification Tests (validate_input = .full)
// --------------------------------------------------------------------------

fn testInvalidConnectivityIndex(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);

    var full_config = fixture.config;
    full_config.validate_input = .full;

    const orig_idx = mesh_inps[0].connect.table_mem[0];
    mesh_inps[0].connect.table_mem[0] = mesh_inps[0].coords.mat.rows_num + 100;
    defer mesh_inps[0].connect.table_mem[0] = orig_idx;

    try expectRenderInputError(
        io,
        fixture,
        &mesh_inps,
        full_config,
        error.InvalidConnectivityIndex,
    );
}

fn testDegenerateElementIndices(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);

    var full_config = fixture.config;
    full_config.validate_input = .full;

    const orig_idx1 = mesh_inps[0].connect.table_mem[1];
    mesh_inps[0].connect.table_mem[1] = mesh_inps[0].connect.table_mem[0];
    defer mesh_inps[0].connect.table_mem[1] = orig_idx1;

    try expectRenderInputError(
        io,
        fixture,
        &mesh_inps,
        full_config,
        error.DegenerateElementIndices,
    );
}

fn testNonFiniteCoordinates(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);

    var full_config = fixture.config;
    full_config.validate_input = .full;

    const orig_coord = mesh_inps[0].coords.mem[0];
    mesh_inps[0].coords.mem[0] = std.math.nan(F);
    defer mesh_inps[0].coords.mem[0] = orig_coord;

    try expectRenderInputError(
        io,
        fixture,
        &mesh_inps,
        full_config,
        error.NonFiniteCoordinates,
    );
}

fn testNonFiniteDisplacements(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);

    var full_config = fixture.config;
    full_config.validate_input = .full;

    if (mesh_inps[0].disp) |*disp_field| {
        const orig_disp = disp_field.array_mem[0];
        disp_field.array_mem[0] = std.math.nan(F);
        defer disp_field.array_mem[0] = orig_disp;

        try expectRenderInputError(
            io,
            fixture,
            &mesh_inps,
            full_config,
            error.NonFiniteDisplacements,
        );
    }
}

fn testNonFiniteNodalFields(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);

    var full_config = fixture.config;
    full_config.validate_input = .full;

    switch (mesh_inps[0].shader) {
        .nodal => |*nodal_shader| {
            const orig_val = nodal_shader.field.array_mem[0];
            nodal_shader.field.array_mem[0] = std.math.nan(F);
            defer nodal_shader.field.array_mem[0] = orig_val;

            try expectRenderInputError(
                io,
                fixture,
                &mesh_inps,
                full_config,
                error.NonFiniteNodalFields,
            );
        },
        else => {},
    }
}

fn testNonFiniteUvs(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    var mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);

    var full_config = fixture.config;
    full_config.validate_input = .full;

    switch (mesh_inps[1].shader) {
        .func => |*func_shader| {
            if (func_shader.uvs) |uvs| {
                const orig_uv = uvs.slice[0];
                uvs.slice[0] = std.math.nan(F);
                defer uvs.slice[0] = orig_uv;

                try expectRenderInputError(
                    io,
                    fixture,
                    &mesh_inps,
                    full_config,
                    error.NonFiniteUvs,
                );
            }
        },
        else => {},
    }
}

fn testNonFiniteTexels(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    var mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);

    var full_config = fixture.config;
    full_config.validate_input = .full;

    var float_tex_data = [_]F{ 0.1, 0.2, 0.3, 0.4 };
    const float_tex_dims = [_]usize{ 1, 2, 2 };
    var float_tex_arr = try NDArray.initFlat(allocator, float_tex_dims[0..]);
    defer {
        allocator.free(float_tex_arr.slice);
        float_tex_arr.deinit(allocator);
    }
    @memcpy(float_tex_arr.slice, float_tex_data[0..]);

    mesh_inps[2].shader = .{
        .tex_f = .{
            .uvs = fixture.prep.meshes[2].uvs.array,
            .tex = .{
                .array = float_tex_arr,
                .rows_num = 2,
                .cols_num = 2,
            },
            .samp_cfg = .{
                .sample = .linear,
                .mode = .direct,
            },
            .bits = 16,
            .scaling = .auto,
            .normal_type = .none,
        },
    };

    float_tex_arr.slice[0] = std.math.nan(F);

    try expectRenderInputError(
        io,
        fixture,
        &mesh_inps,
        full_config,
        error.NonFiniteTexels,
    );
}

// --------------------------------------------------------------------------
// Cross-Mode Validation Behavior Tests
// --------------------------------------------------------------------------

fn testCrossModeValidation(
    allocator: std.mem.Allocator,
    io: std.Io,
    fixture: *const BaselineFixture,
) !void {
    _ = allocator;
    const cam_inps = common_full.createScene3Cameras();
    var mesh_inps = common_full.buildScene3Meshes(&fixture.prep, &fixture.textures);
    const render_groups = [_]RenderGroupSpec{.{ .io = io, .workers = 1 }};

    const orig_idx = mesh_inps[0].connect.table_mem[0];
    mesh_inps[0].connect.table_mem[0] = mesh_inps[0].coords.mat.rows_num + 50;
    defer mesh_inps[0].connect.table_mem[0] = orig_idx;

    var fast_config = fixture.config;
    fast_config.validate_input = .fast;
    _ = try valinp.checkRenderInps(
        &render_groups,
        &cam_inps,
        &mesh_inps,
        fast_config,
        null,
        false,
        null,
    );

    var full_config = fixture.config;
    full_config.validate_input = .full;
    try std.testing.expectError(
        error.InvalidConnectivityIndex,
        valarr.checkRenderArrs(&mesh_inps),
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
    var config = fixture.config;
    config.total_threads = 0;
    config.tile_size_min = 0;
    config.background_value = std.math.nan(F);

    // Should return the first validation error encountered
    try expectConfigError(io, fixture, config, error.InvalidTotalThreads);
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

    // Validation modes
    try testValidateInputModes(allocator, io, &fixture);

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
    try testInvalidSaveFrameBuffCount(allocator, io, &fixture);
    try testInvalidImageSaveOpts(allocator, io, &fixture);
    try testInvalidFullStatsFormats(allocator, io, &fixture);

    // Camera parameters
    try testInvalidCameraPixels(allocator, io, &fixture);
    try testInvalidCameraPixelSize(allocator, io, &fixture);
    try testInvalidCameraFocalLength(allocator, io, &fixture);
    try testInvalidCameraSubSample(allocator, io, &fixture);
    try testInvalidCameraRoi(allocator, io, &fixture);
    try testInvalidCameraRotation(allocator, io, &fixture);
    try testInvalidCameraDistortion(allocator, io, &fixture);
    try testInvalidCameraPsf(allocator, io, &fixture);

    // Mesh and shader structural parameters
    try testZeroCoordinateCount(allocator, io, &fixture);
    try testInvalidCoordinateDimensions(allocator, io, &fixture);
    try testZeroElementCount(allocator, io, &fixture);
    try testInvalidConnectivityDimensions(allocator, io, &fixture);
    try testInvalidDisplacementDimensions(allocator, io, &fixture);
    try testInvalidNodalFieldDimensions(allocator, io, &fixture);
    try testInvalidTexSampleConfig(allocator, io, &fixture);
    try testInvalidShaderBitDepth(allocator, io, &fixture);
    try testInvalidScalingBounds(allocator, io, &fixture);
    try testInvalidFuncShaderParams(allocator, io, &fixture);

    // Alignment
    try testGlobalSubpxTileSizeNotAligned(allocator, io, &fixture);
    try testGlobalSubpxStripeSizeNotAligned(allocator, io, &fixture);

    // Benchmark buffer & output buffer
    try testInvalidBenchCaptureBuff(allocator, io, &fixture);
    try testUnsuppedImageModeFieldCount(allocator, io, &fixture);
    try testInvalidOutputBuff(allocator, io, &fixture);

    // Full payload scan tests
    try testInvalidConnectivityIndex(allocator, io, &fixture);
    try testDegenerateElementIndices(allocator, io, &fixture);
    try testNonFiniteCoordinates(allocator, io, &fixture);
    try testNonFiniteDisplacements(allocator, io, &fixture);
    try testNonFiniteNodalFields(allocator, io, &fixture);
    try testNonFiniteUvs(allocator, io, &fixture);
    try testNonFiniteTexels(allocator, io, &fixture);

    // Cross-mode validation tests
    try testCrossModeValidation(allocator, io, &fixture);

    // Mixed error cases
    try testMixedCaseMultipleConfigErrors(allocator, io, &fixture);
    try testMixedCaseCameraAndConfigErrors(allocator, io, &fixture);
    try testMixedCaseEmptyInputs(allocator, io, &fixture);
}
