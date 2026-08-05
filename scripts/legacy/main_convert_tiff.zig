const std = @import("std");
const iio = @import("imageio");

const INPUT_PATH = "texture/cal_target.tiff";
const OUTPUT_BASE_NAME = "cal_target-simple";
const OUTPUT_DIR_PATH = "texture";

pub fn main(init: std.process.Init) !void {
    const io = init.io;

    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const allocator = arena.allocator();

    var texture = try iio.CLoadTIFF(
        u8,
        1,
        allocator,
        io,
        INPUT_PATH,
    );
    defer texture.deinit(allocator);

    const cwd = std.Io.Dir.cwd();
    var out_dir = try cwd.openDir(io, OUTPUT_DIR_PATH, .{});
    defer out_dir.close(io);

    const tiff_opts = iio.ImageSaveOpts{
        .format = .tiff,
        .bits = 8,
        .scaling = .none,
        .channels = 1,
    };
    try iio.saveImage(
        io,
        out_dir,
        OUTPUT_BASE_NAME,
        &texture.array,
        0,
        tiff_opts,
    );

    const bmp_opts = iio.ImageSaveOpts{
        .format = .bmp,
        .bits = 8,
        .scaling = .none,
        .channels = 1,
    };
    try iio.saveImage(
        io,
        out_dir,
        OUTPUT_BASE_NAME,
        &texture.array,
        0,
        bmp_opts,
    );
}
