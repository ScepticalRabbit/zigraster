const std = @import("std");

const buildconfig = @import("riley/zig/buildconfig.zig");
const camera = @import("riley/zig/camera.zig");
const cameraops = @import("riley/zig/cameraops.zig");
const iio = @import("riley/zig/imageio.zig");
const meshio = @import("riley/zig/meshio.zig");
const mo = @import("riley/zig/meshpipeline.zig");
const riley = @import("riley/zig/riley.zig");
const Rotation = @import("riley/zig/rotation.zig").Rotation;
const sceneops = @import("riley/zig/sceneops.zig");

const F = buildconfig.F;

pub fn main(init: std.process.Init) !void {
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const allocator = arena.allocator();

    var coord_values = [_]F{
        -1.0, -1.0, 0.0,
        1.0,  -1.0, 0.0,
        0.0,  1.0,  0.0,
    };
    var connect_values = [_]usize{ 0, 1, 2 };
    const coords = meshio.Coords.init(&coord_values, 3);
    const connect = meshio.Connect.init(&connect_values, 1, 3);
    const mesh = mo.MeshInput{
        .mesh_type = .tri3opt,
        .coords = coords,
        .connect = connect,
        .disp = null,
        .shader = .{ .func = .{
            .coord_mode = .world_reference,
            .builtin = .checker,
            .params = .{
                .coord_scale = .{ 4.0, 4.0 },
                .settings = .{ .checker = .{} },
            },
            .scaling = .auto,
        } },
    };
    const pixels_num = [2]u32{ 512, 512 };
    const pixels_size = [2]F{ 0.02, 0.02 };
    const focal_length: F = 1.0;
    const rotation = Rotation.init(0.0, 0.0, 0.0);
    const target = sceneops.boundsCenter(&coords);
    const position = cameraops.posFillFrameFromRotAndTarg(
        &coords,
        target,
        pixels_num,
        pixels_size,
        focal_length,
        rotation,
        1.0,
    );
    const camera_input = camera.CameraInput{
        .pixels_num = pixels_num,
        .pixels_size = pixels_size,
        .pos_world = position,
        .rot_world = rotation,
        .roi_cent_world = target,
        .focal_length = focal_length,
        .sub_sample = 1,
    };
    const config = riley.RasterConfig{
        .save_strategy = .disk,
        .image_save_mode = .grey,
        .image_save_opts = &.{
            .{ .format = iio.ImageFormat.bmp, .bits = 8, .scaling = .none },
        },
    };
    const groups = [_]riley.RenderGroupSpec{.{ .io = init.io, .workers = 1 }};
    if (try riley.raster(
        allocator,
        &groups,
        &.{camera_input},
        &.{mesh},
        config,
        "./out/demo0_quickstart",
    )) |images| {
        allocator.free(images.slice);
        var images_mut = images;
        images_mut.deinit(allocator);
    }
}
