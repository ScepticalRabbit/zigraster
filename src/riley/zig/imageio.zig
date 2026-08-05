// --------------------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------------------
const std = @import("std");

const buildconfig = @import("buildconfig.zig");
const F = buildconfig.F;

const matslice = @import("matslice.zig");
const ndarray = @import("ndarray.zig");

const texops = @import("textureops.zig");
const clibtiff = @import("clibtiff.zig");

const imageops = @import("imageops.zig");
const csvio = @import("csvio.zig");

const temp_test_root_dir = "temp-tests";
const temp_test_dir = "temp-tests/imageio";

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const ImageFormat = enum {
    csv,
    fimg,
    ppm,
    bmp,
    tiff,
};

pub const ImageSaveOpts = struct {
    format: ImageFormat,
    bits: ?u8 = 8,
    scaling: imageops.ScaleStrategy = .none,
    channels: usize = 1,

    pub fn init(
        format: ImageFormat,
        bits: ?u8,
        scaling: imageops.ScaleStrategy,
        channels: usize,
    ) ImageSaveOpts {
        return .{
            .format = format,
            .bits = bits,
            .scaling = scaling,
            .channels = channels,
        };
    }
};

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn loadImage(
    comptime T: type,
    comptime channels: usize,
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    format: ImageFormat,
) !texops.Tex(T, channels) {
    return switch (format) {
        .csv => try loadCSV(T, channels, allocator, io, path),
        .fimg => {
            return try loadFIMGTex(T, channels, allocator, io, path);
        },
        .ppm => try loadPPM(T, channels, allocator, io, path),
        .bmp => try loadBMP(T, channels, allocator, io, path),
        .tiff => try loadTIFF(T, channels, allocator, io, path),
    };
}

pub fn saveImage(
    io: std.Io,
    out_dir: std.Io.Dir,
    file_name_no_ext: []const u8,
    image_arr: *const ndarray.NDArray(F),
    start_field: usize,
    opts: ImageSaveOpts,
) !void {
    var name_buff: [1024]u8 = undefined;
    const ext = switch (opts.format) {
        .csv => ".csv",
        .fimg => ".fimg",
        .ppm => ".ppm",
        .bmp => ".bmp",
        .tiff => ".tiff",
    };
    const file_name = try std.fmt.bufPrint(
        name_buff[0..],
        "{s}{s}",
        .{ file_name_no_ext, ext },
    );

    switch (opts.format) {
        .csv => try saveCSV(io, out_dir, file_name, image_arr, start_field, opts),
        .fimg => try saveFIMG(io, out_dir, file_name, image_arr, start_field, opts),
        .ppm => try savePPM(io, out_dir, file_name, image_arr, start_field, opts),
        .bmp => try saveBMP(io, out_dir, file_name, image_arr, start_field, opts),
        .tiff => try saveTIFF(io, out_dir, file_name, image_arr, start_field, opts),
    }
}

pub fn saveMatAsImage(
    io: std.Io,
    out_dir: std.Io.Dir,
    file_name_no_ext: []const u8,
    image: *const matslice.MatSlice(F),
    opts: ImageSaveOpts,
) !void {
    var dims = [_]usize{ 1, image.rows_num, image.cols_num };
    var strides = [_]usize{ image.rows_num * image.cols_num, image.cols_num, 1 };
    const arr = ndarray.NDArray(F){
        .slice = image.slice,
        .dims = &dims,
        .strides = &strides,
    };
    try saveImage(io, out_dir, file_name_no_ext, &arr, 0, opts);
}

pub fn saveImages(
    io: std.Io,
    out_dir: ?std.Io.Dir,
    camera_idx: usize,
    frame_idx: usize,
    num_fields: u8,
    pixels_num: [2]u32,
    frame_arr: *const ndarray.NDArray(F),
    channels_override: ?usize,
    opts_slice: []const ImageSaveOpts,
) !void {
    const save_dir = out_dir orelse return;
    var name_buff: [1024]u8 = undefined;
    _ = pixels_num;

    for (opts_slice) |opts| {
        const channels = channels_override orelse opts.channels;
        var save_opts = opts;
        save_opts.channels = channels;
        var ff: usize = 0;
        while (ff + channels <= @as(usize, num_fields)) {
            const file_name = try formatFrameFieldBaseName(
                name_buff[0..],
                camera_idx,
                frame_idx,
                ff,
                channels,
            );

            try saveImage(io, save_dir, file_name, frame_arr, ff, save_opts);
            ff += channels;
        }
    }
}

pub fn formatFrameFieldBaseName(
    buff: []u8,
    camera_idx: usize,
    frame_idx: usize,
    field_idx: usize,
    channels: usize,
) ![]const u8 {
    return if (channels == 1)
        std.fmt.bufPrint(
            buff,
            "cam{d}_frame{d}_field{d}",
            .{ camera_idx, frame_idx, field_idx },
        )
    else if (channels == 3)
        std.fmt.bufPrint(
            buff,
            "cam{d}_frame{d}_field{d}_rgb",
            .{ camera_idx, frame_idx, field_idx },
        )
    else
        std.fmt.bufPrint(
            buff,
            "cam{d}_frame{d}_field{d}_{d}",
            .{ camera_idx, frame_idx, field_idx, field_idx + channels - 1 },
        );
}

pub fn savePPM(
    io: std.Io,
    out_dir: std.Io.Dir,
    file_name: []const u8,
    image_arr: *const ndarray.NDArray(F),
    start_field: usize,
    opts: ImageSaveOpts,
) !void {
    const ppm_file: std.Io.File = try out_dir.createFile(io, file_name, .{});
    defer ppm_file.close(io);

    var write_buf: [4096]u8 = undefined;
    var file_writer = ppm_file.writer(io, &write_buf);
    const writer = &file_writer.interface;

    const bits = opts.bits orelse 8;
    const max_val_f = imageops.getScaleMax(bits);
    const max_val = @as(u32, @intFromFloat(max_val_f));

    const rows = image_arr.dims[1];
    const cols = image_arr.dims[2];

    try writer.print("P3\n{d} {d}\n{d}\n", .{ cols, rows, max_val });

    // We need scaling params per field if we are doing auto scaling
    var params_array: [buildconfig.config.max_image_channels]imageops.ScalingParams = undefined;
    const channels = @min(opts.channels, buildconfig.config.max_image_channels);
    for (0..channels) |ch| {
        const field_idx = start_field + ch;
        const slice = image_arr.getSlice(&[_]usize{ field_idx, 0, 0 }, 0);
        const mat = matslice.MatSlice(F).init(slice, rows, cols);
        params_array[ch] = imageops.getScalingParams(&mat, opts.scaling);
    }

    for (0..rows) |rr| {
        for (0..cols) |cc| {
            for (0..3) |ch| {
                const field_idx = if (ch < channels) start_field + ch else start_field;
                const raw_val = image_arr.get(&[_]usize{ field_idx, rr, cc });
                const params = if (ch < channels) params_array[ch] else params_array[0];

                var val = imageops.applyScaling(raw_val, opts.scaling, opts.bits, params);
                if (opts.bits == null) {
                    val *= imageops.getScaleMax(bits);
                }
                const px = @as(u32, @intFromFloat(imageops.applyClamping(val, bits)));
                try writer.print("{d}", .{px});
                if (ch < 2) try writer.writeByte(' ');
            }
            try writer.writeByte(' ');
        }
        try writer.writeByte('\n');
    }

    try writer.flush();
}

pub fn saveCSV(
    io: std.Io,
    out_dir: std.Io.Dir,
    file_name: []const u8,
    image_arr: *const ndarray.NDArray(F),
    start_field: usize,
    opts: ImageSaveOpts,
) !void {
    const rows = image_arr.dims[1];
    const cols = image_arr.dims[2];

    var params_array: [buildconfig.config.max_image_channels]imageops.ScalingParams = undefined;
    const channels = @min(opts.channels, buildconfig.config.max_image_channels);
    for (0..channels) |ch| {
        const field_idx = start_field + ch;
        const slice = image_arr.getSlice(&[_]usize{ field_idx, 0, 0 }, 0);
        const mat = matslice.MatSlice(F).init(slice, rows, cols);
        params_array[ch] = imageops.getScalingParams(&mat, opts.scaling);
    }

    const SaveCtx = struct {
        image_arr: *const ndarray.NDArray(F),
        start_field: usize,
        opts: ImageSaveOpts,
        params_array: [buildconfig.config.max_image_channels]imageops.ScalingParams,

        fn getVal(ctx: @This(), row: usize, col: usize, ch: usize) F {
            const field_idx = ctx.start_field + ch;
            const raw_val = ctx.image_arr.get(&[_]usize{ field_idx, row, col });
            const params = ctx.params_array[ch];

            var val = imageops.applyScaling(
                raw_val,
                ctx.opts.scaling,
                ctx.opts.bits,
                params,
            );
            if (ctx.opts.scaling == .none and ctx.opts.bits != null) {
                val = imageops.applyClamping(val, ctx.opts.bits);
            }
            return val;
        }
    };

    try csvio.savePackedGridCSV(
        io,
        out_dir,
        file_name,
        rows,
        cols,
        channels,
        SaveCtx{
            .image_arr = image_arr,
            .start_field = start_field,
            .opts = opts,
            .params_array = params_array,
        },
        SaveCtx.getVal,
    );
}

pub fn saveFIMG(
    io: std.Io,
    out_dir: std.Io.Dir,
    file_name: []const u8,
    image_arr: *const ndarray.NDArray(F),
    start_field: usize,
    opts: ImageSaveOpts,
) !void {
    const rows = image_arr.dims[1];
    const cols = image_arr.dims[2];
    const channels = @min(opts.channels, buildconfig.config.max_image_channels);

    const file = try out_dir.createFile(io, file_name, .{});
    defer file.close(io);

    var write_buf: [4096]u8 = undefined;
    var file_writer = file.writer(io, &write_buf);
    const writer = &file_writer.interface;

    // Header: ASCII
    try writer.print("FIMG\n{d} {d} {d}\n", .{ cols, rows, channels });

    // Payload: Binary F Little-Endian
    // Format is Planar: Channel by Channel (matching our ndarray.NDArray layout for fields)
    for (0..channels) |ch| {
        const field_idx = start_field + ch;
        for (0..rows) |rr| {
            for (0..cols) |cc| {
                const val = image_arr.get(&[_]usize{ field_idx, rr, cc });
                const le_val = std.mem.nativeToLittle(F, val);
                try writer.writeAll(std.mem.asBytes(&le_val));
            }
        }
    }

    try writer.flush();
}

pub fn saveBMP(
    io: std.Io,
    out_dir: std.Io.Dir,
    file_name: []const u8,
    image_arr: *const ndarray.NDArray(F),
    start_field: usize,
    opts: ImageSaveOpts,
) !void {
    const file = try out_dir.createFile(io, file_name, .{});
    defer file.close(io);

    var write_buf: [4096]u8 = undefined;
    var file_writer = file.writer(io, &write_buf);
    const writer = &file_writer.interface;

    const bits = opts.bits orelse 8;
    const is_16bit = (bits == 16);
    const bpp: u16 = if (is_16bit) 48 else 24;

    const rows = image_arr.dims[1];
    const cols = image_arr.dims[2];

    const width = @as(u32, @intCast(cols));
    const height = @as(u32, @intCast(rows));
    const bytes_per_px: u32 = if (is_16bit) 6 else 3;
    const row_padding = (4 - (width * bytes_per_px) % 4) % 4;
    const row_size = width * bytes_per_px + row_padding;
    const data_size = row_size * height;
    const header_size: u32 = 14 + 40;
    const file_size = header_size + data_size;

    // Bitmap File Header (14 bytes)
    try writer.writeAll("BM");
    try writer.writeInt(u32, file_size, .little);
    try writer.writeInt(u16, 0, .little);
    try writer.writeInt(u16, 0, .little);
    try writer.writeInt(u32, header_size, .little);

    // DIB Header (BITMAPINFOHEADER - 40 bytes)
    try writer.writeInt(u32, 40, .little);
    try writer.writeInt(i32, @intCast(width), .little);
    try writer.writeInt(i32, @intCast(height), .little);
    try writer.writeInt(u16, 1, .little);
    try writer.writeInt(u16, bpp, .little);
    try writer.writeInt(u32, 0, .little); // BI_RGB
    try writer.writeInt(u32, data_size, .little);
    try writer.writeInt(i32, 2835, .little);
    try writer.writeInt(i32, 2835, .little);
    try writer.writeInt(u32, 0, .little);
    try writer.writeInt(u32, 0, .little);

    var params_array: [buildconfig.config.max_image_channels]imageops.ScalingParams = undefined;
    const channels = @min(opts.channels, buildconfig.config.max_image_channels);
    for (0..@min(channels, 3)) |ch| {
        const field_idx = start_field + ch;
        const slice = image_arr.getSlice(&[_]usize{ field_idx, 0, 0 }, 0);
        const mat = matslice.MatSlice(F).init(slice, rows, cols);
        params_array[ch] = imageops.getScalingParams(&mat, opts.scaling);
    }

    // BMP data is bottom-up
    var rr: usize = rows;
    while (rr > 0) {
        rr -= 1;
        for (0..cols) |cc| {
            // Write BGR
            const bgr_inds = [_]usize{ 2, 1, 0 };
            for (bgr_inds) |ch| {
                const field_idx = if (ch < channels) start_field + ch else start_field;
                const raw_val = image_arr.get(&[_]usize{ field_idx, rr, cc });
                const params = if (ch < channels) params_array[ch] else params_array[0];

                var val = imageops.applyScaling(raw_val, opts.scaling, opts.bits, params);
                if (opts.bits == null and opts.scaling != .none) {
                    val *= if (is_16bit) 65535.0 else 255.0;
                }
                const px_f = imageops.applyClamping(val, bits);
                if (is_16bit) {
                    try writer.writeInt(u16, @as(u16, @intFromFloat(px_f)), .little);
                } else {
                    try writer.writeByte(@as(u8, @intFromFloat(px_f)));
                }
            }
        }
        for (0..row_padding) |_| try writer.writeByte(0);
    }
    try writer.flush();
}
pub fn saveTIFF(
    io: std.Io,
    out_dir: std.Io.Dir,
    file_name: []const u8,
    image_arr: *const ndarray.NDArray(F),
    start_field: usize,
    opts: ImageSaveOpts,
) !void {
    const rows = image_arr.dims[1];
    const cols = image_arr.dims[2];
    const slice = image_arr.getSlice(&[_]usize{ start_field, 0, 0 }, 0);
    const image = matslice.MatSlice(F).init(slice, rows, cols);

    const file = try out_dir.createFile(io, file_name, .{});
    defer file.close(io);

    var write_buf: [4096]u8 = undefined;
    var file_writer = file.writer(io, &write_buf);
    const writer = &file_writer.interface;

    const bits = opts.bits orelse 8;
    const width = @as(u32, @intCast(image.cols_num));
    const height = @as(u32, @intCast(image.rows_num));
    const pixel_data_offset: u32 = 8;
    const bytes_per_pixel: u32 = if (bits == 16) 2 else 1;
    const pixel_data_size = width * height * bytes_per_pixel;
    const ifd_offset = pixel_data_offset + pixel_data_size;

    // 1. Header (8 bytes)
    try writer.writeAll("II"); // Little Endian
    try writer.writeInt(u16, 42, .little);
    try writer.writeInt(u32, ifd_offset, .little);

    const params = imageops.getScalingParams(&image, opts.scaling);
    const max_v = imageops.getScaleMax(bits);

    // 2. Pixel Data
    for (0..image.rows_num) |r| {
        for (0..image.cols_num) |c| {
            const raw_val = image.get(r, c);
            var val = imageops.applyScaling(raw_val, opts.scaling, opts.bits, params);
            if (opts.bits == null and opts.scaling != .none) {
                val *= max_v;
            }

            const px_clamped = imageops.applyClamping(val, bits);
            if (bits == 16) {
                try writer.writeInt(u16, @as(u16, @intFromFloat(px_clamped)), .little);
            } else {
                try writer.writeByte(@as(u8, @intFromFloat(px_clamped)));
            }
        }
    }

    // 3. IFD
    const num_tags: u16 = 10;
    try writer.writeInt(u16, num_tags, .little);

    const Tag = struct {
        id: u16,
        type: u16,
        count: u32,
        value: u32,
        fn write(self: @This(), w: anytype) !void {
            try w.writeInt(u16, self.id, .little);
            try w.writeInt(u16, self.type, .little);
            try w.writeInt(u32, self.count, .little);
            try w.writeInt(u32, self.value, .little);
        }
    };

    try (Tag{ .id = 256, .type = 3, .count = 1, .value = width }).write(writer);
    try (Tag{ .id = 257, .type = 3, .count = 1, .value = height }).write(writer);
    try (Tag{ .id = 258, .type = 3, .count = 1, .value = bits }).write(writer);
    try (Tag{ .id = 259, .type = 3, .count = 1, .value = 1 }).write(writer);
    try (Tag{ .id = 262, .type = 3, .count = 1, .value = 1 }).write(writer);
    try (Tag{
        .id = 273,
        .type = 4,
        .count = 1,
        .value = pixel_data_offset,
    }).write(writer);
    try (Tag{ .id = 277, .type = 3, .count = 1, .value = 1 }).write(writer);
    try (Tag{ .id = 278, .type = 3, .count = 1, .value = height }).write(writer);
    try (Tag{ .id = 279, .type = 4, .count = 1, .value = pixel_data_size }).write(writer);
    try (Tag{ .id = 296, .type = 3, .count = 1, .value = 2 }).write(writer);

    try writer.writeInt(u32, 0, .little);
    try writer.flush();
}

pub fn loadPPM(
    comptime T: type,
    comptime channels: usize,
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) !texops.Tex(T, channels) {
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.openFile(io, path, .{ .mode = .read_only });
    defer file.close(io);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    var read_buf: [4096]u8 = undefined;
    var file_reader = file.reader(io, &read_buf);
    const reader = &file_reader.interface;

    var header: [2]u8 = undefined;
    try reader.readSliceAll(&header);
    if (!std.mem.eql(u8, &header, "P3")) return error.NotAPPM;

    const readToken = struct {
        fn readToken(r: anytype, a: std.mem.Allocator) ![]const u8 {
            while (true) {
                const char = try r.takeByte();
                if (char == '#') {
                    _ = try r.takeDelimiter('\n');
                    continue;
                }
                if (std.ascii.isWhitespace(char)) continue;

                var list: std.ArrayList(u8) = .empty;
                try list.append(a, char);
                while (true) {
                    const next_char = r.takeByte() catch |err| switch (err) {
                        error.EndOfStream => break,
                        else => return err,
                    };
                    if (std.ascii.isWhitespace(next_char)) break;
                    if (next_char == '#') break;
                    try list.append(a, next_char);
                }
                return list.toOwnedSlice(a);
            }
        }
    }.readToken;

    const width_str = try readToken(reader, aa);
    const height_str = try readToken(reader, aa);
    const max_val_str = try readToken(reader, aa);

    const width = try std.fmt.parseInt(usize, width_str, 10);
    const height = try std.fmt.parseInt(usize, height_str, 10);
    const max_val = try std.fmt.parseInt(u32, max_val_str, 10);

    var tex = try texops.Tex(T, channels).init(allocator, height, width);
    errdefer tex.deinit(allocator);

    const max_val_f = @as(F, @floatFromInt(max_val));

    for (0..height) |rr| {
        for (0..width) |cc| {
            var rgb: [3]u32 = undefined;
            for (0..3) |i| {
                const val_str = try readToken(reader, aa);
                rgb[i] = try std.fmt.parseInt(u32, val_str, 10);
            }

            if (channels == 3) {
                const s0 = @as(F, @floatFromInt(rgb[0])) / max_val_f;
                const s1 = @as(F, @floatFromInt(rgb[1])) / max_val_f;
                const s2 = @as(F, @floatFromInt(rgb[2])) / max_val_f;
                tex.setVal(0, rr, cc, convertToTarg(T, s0));
                tex.setVal(1, rr, cc, convertToTarg(T, s1));
                tex.setVal(2, rr, cc, convertToTarg(T, s2));
            } else if (channels == 1) {
                const val = toGreyScale(rgb[0], rgb[1], rgb[2]);
                tex.setVal(
                    0,
                    rr,
                    cc,
                    convertToTarg(T, val / max_val_f),
                );
            }
        }
    }

    return tex;
}

pub fn loadCSV(
    comptime T: type,
    comptime channels: usize,
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) !texops.Tex(T, channels) {
    const array = if (channels == 1)
        try csvio.loadScalarCsv2D(allocator, io, path)
    else
        try csvio.loadPackedCsv2D(allocator, io, path, channels);
    defer {
        allocator.free(array.slice);
        var tmp = array;
        tmp.deinit(allocator);
    }

    const rows = array.dims[0];
    const cols = array.dims[1];
    var tex = try texops.Tex(T, channels).init(allocator, rows, cols);
    errdefer tex.deinit(allocator);

    for (0..rows) |rr| {
        for (0..cols) |cc| {
            for (0..channels) |ch| {
                const val = if (channels == 1)
                    array.get(&[_]usize{ rr, cc })
                else
                    array.get(&[_]usize{ rr, cc, ch });
                tex.setVal(ch, rr, cc, convertValue(T, val));
            }
        }
    }

    return tex;
}

pub fn loadFIMG(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) !ndarray.NDArray(F) {
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.openFile(io, path, .{ .mode = .read_only });
    defer file.close(io);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    var read_buf: [4096]u8 = undefined;
    var file_reader = file.reader(io, &read_buf);
    const reader = &file_reader.interface;

    // 1. Validate Signature
    var sig: [4]u8 = undefined;
    try reader.readSliceAll(&sig);
    if (!std.mem.eql(u8, &sig, "FIMG")) return error.InvalidFormat;

    // Skip any leading whitespace/newlines before tokens
    const readToken = struct {
        fn readToken(r: anytype, a: std.mem.Allocator) ![]const u8 {
            while (true) {
                const char = try r.takeByte();
                if (char == '#') {
                    _ = try r.takeDelimiter('\n');
                    continue;
                }
                if (std.ascii.isWhitespace(char)) continue;

                var list: std.ArrayList(u8) = .empty;
                try list.append(a, char);
                while (true) {
                    const next_char = r.takeByte() catch |err| switch (err) {
                        error.EndOfStream => break,
                        else => return err,
                    };
                    if (std.ascii.isWhitespace(next_char)) break;
                    if (next_char == '#') break;
                    try list.append(a, next_char);
                }
                return list.toOwnedSlice(a);
            }
        }
    }.readToken;

    const width_str = try readToken(reader, aa);
    const height_str = try readToken(reader, aa);
    const chan_str = try readToken(reader, aa);

    const width = try std.fmt.parseInt(usize, width_str, 10);
    const height = try std.fmt.parseInt(usize, height_str, 10);
    const chans = try std.fmt.parseInt(usize, chan_str, 10);
    const sample_count = chans * height * width;
    const file_stat = try file.stat(io);
    const payload_bytes = file_stat.size - file_reader.logicalPos();
    if (sample_count == 0) return error.InvalidFormat;
    if (payload_bytes % sample_count != 0) return error.InvalidFormat;
    const bytes_per_sample = payload_bytes / sample_count;

    var array = try ndarray.NDArray(F).initFlat(allocator, &[_]usize{ chans, height, width });
    errdefer array.deinit(allocator);

    // 2. Read Binary Payload (f32/f64 LE)
    // The file is stored Planar: [chans, height, width]
    for (0..chans) |ch| {
        for (0..height) |rr| {
            for (0..width) |cc| {
                const val = switch (bytes_per_sample) {
                    4 => blk: {
                        var bytes: [4]u8 = undefined;
                        try reader.readSliceAll(&bytes);
                        const le_val = std.mem.bytesAsValue(f32, &bytes).*;
                        break :blk @as(
                            F,
                            @floatCast(std.mem.littleToNative(f32, le_val)),
                        );
                    },
                    8 => blk: {
                        var bytes: [8]u8 = undefined;
                        try reader.readSliceAll(&bytes);
                        const le_val = std.mem.bytesAsValue(f64, &bytes).*;
                        break :blk @as(
                            F,
                            @floatCast(std.mem.littleToNative(f64, le_val)),
                        );
                    },
                    else => return error.InvalidFormat,
                };
                array.set(&[_]usize{ ch, rr, cc }, val);
            }
        }
    }

    return array;
}

fn loadFIMGTex(
    comptime T: type,
    comptime channels: usize,
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) !texops.Tex(T, channels) {
    const array = try loadFIMG(allocator, io, path);
    defer {
        allocator.free(array.slice);
        var array_tmp = array;
        array_tmp.deinit(allocator);
    }

    if (array.dims[0] != channels) {
        return error.ChannelMismatch;
    }

    var tex = try texops.Tex(T, channels).init(
        allocator,
        array.dims[1],
        array.dims[2],
    );
    errdefer tex.deinit(allocator);

    for (0..channels) |ch| {
        for (0..array.dims[1]) |rr| {
            for (0..array.dims[2]) |cc| {
                const val = array.get(&[_]usize{ ch, rr, cc });
                tex.setVal(ch, rr, cc, convertValue(T, val));
            }
        }
    }

    return tex;
}

pub fn loadBMP(
    comptime T: type,
    comptime channels: usize,
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) !texops.Tex(T, channels) {
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.openFile(io, path, .{ .mode = .read_only });
    defer file.close(io);

    var read_buf: [4096]u8 = undefined;
    var file_reader = file.reader(io, &read_buf);
    const reader = &file_reader.interface;

    var header: [14]u8 = undefined;
    try reader.readSliceAll(&header);
    if (!std.mem.eql(u8, header[0..2], "BM")) return error.NotABMP;

    const offset = std.mem.readInt(u32, header[10..14], .little);
    const dib_size = try reader.takeInt(u32, .little);

    var width: i32 = 0;
    var height: i32 = 0;
    var bit_count: u16 = 0;

    if (dib_size >= 40) {
        width = try reader.takeInt(i32, .little);
        height = try reader.takeInt(i32, .little);
        _ = try reader.takeInt(u16, .little);
        bit_count = try reader.takeInt(u16, .little);
        const compression = try reader.takeInt(u32, .little);
        if (compression != 0) return error.CompressionNotSupped;
        try file_reader.seekBy(@as(i64, @intCast(dib_size)) - 20);
    } else return error.UnsuppedDIBHeader;

    const abs_height = @as(usize, @intCast(@abs(height)));
    const abs_width = @as(usize, @intCast(@abs(width)));

    var tex = try texops.Tex(T, channels).init(
        allocator,
        abs_height,
        abs_width,
    );
    errdefer tex.deinit(allocator);

    if (bit_count == 24) {
        try file_reader.seekTo(offset);
        const row_padding = (4 - (abs_width * 3) % 4) % 4;

        for (0..abs_height) |y| {
            const r = if (height > 0) abs_height - 1 - y else y;

            for (0..abs_width) |x| {
                var bgr: [3]u8 = undefined;
                try reader.readSliceAll(&bgr);

                if (channels == 3) {
                    tex.setVal(0, r, x, convertValue(T, bgr[2]));
                    tex.setVal(1, r, x, convertValue(T, bgr[1]));
                    tex.setVal(2, r, x, convertValue(T, bgr[0]));
                } else if (channels == 1) {
                    const val = toGreyScale(bgr[2], bgr[1], bgr[0]);
                    tex.setVal(0, r, x, convertValue(T, val));
                }
            }

            try file_reader.seekBy(@intCast(row_padding));
        }
    } else if (bit_count == 48) {
        try file_reader.seekTo(offset);
        const row_padding = (4 - (abs_width * 6) % 4) % 4;

        for (0..abs_height) |y| {
            const r = if (height > 0) abs_height - 1 - y else y;

            for (0..abs_width) |x| {
                var bgr: [3]u16 = undefined;

                bgr[0] = try reader.takeInt(u16, .little);
                bgr[1] = try reader.takeInt(u16, .little);
                bgr[2] = try reader.takeInt(u16, .little);
                if (channels == 3) {
                    tex.setVal(0, r, x, convertValue(T, bgr[2]));
                    tex.setVal(1, r, x, convertValue(T, bgr[1]));
                    tex.setVal(2, r, x, convertValue(T, bgr[0]));
                } else if (channels == 1) {
                    const val = toGreyScale(bgr[2], bgr[1], bgr[0]);
                    tex.setVal(0, r, x, convertValue(T, val));
                }
            }

            try file_reader.seekBy(@intCast(row_padding));
        }
    } else if (bit_count == 8) {
        try file_reader.seekTo(14 + dib_size);
        const palette_size = (offset - (14 + dib_size)) / 4;

        const palette = try allocator.alloc([4]u8, palette_size);
        defer allocator.free(palette);

        for (0..palette_size) |i| try reader.readSliceAll(&palette[i]);

        try file_reader.seekTo(offset);
        const row_padding = (4 - abs_width % 4) % 4;

        for (0..abs_height) |y| {
            const r = if (height > 0) abs_height - 1 - y else y;

            for (0..abs_width) |x| {
                const index = try reader.takeByte();
                const color = palette[index];

                if (channels == 3) {
                    tex.setVal(0, r, x, convertValue(T, color[2]));
                    tex.setVal(1, r, x, convertValue(T, color[1]));
                    tex.setVal(2, r, x, convertValue(T, color[0]));
                } else if (channels == 1) {
                    const val = toGreyScale(color[2], color[1], color[0]);
                    tex.setVal(0, r, x, convertValue(T, val));
                }
            }

            try file_reader.seekBy(@intCast(row_padding));
        }
    } else return error.UnsuppedBitCount;

    return tex;
}

pub fn loadTIFF(
    comptime T: type,
    comptime channels: usize,
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) !texops.Tex(T, channels) {
    const cwd = std.Io.Dir.cwd();
    const file = try cwd.openFile(io, path, .{ .mode = .read_only });
    defer file.close(io);

    var read_buf: [4096]u8 = undefined;
    var file_reader = file.reader(io, &read_buf);
    const reader = &file_reader.interface;

    var header: [4]u8 = undefined;
    try reader.readSliceAll(&header);

    const is_little = if (std.mem.eql(u8, header[0..2], "II"))
        true
    else if (std.mem.eql(u8, header[0..2], "MM"))
        false
    else
        return error.NotATIFF;

    const endian: std.builtin.Endian = if (is_little) .little else .big;
    const magic = std.mem.readInt(u16, header[2..4], endian);
    if (magic != 42) return error.InvalidTIFFMagic;

    const ifd_offset = try reader.takeInt(u32, endian);
    try file_reader.seekTo(ifd_offset);

    const num_tags = try reader.takeInt(u16, endian);
    var width: u32 = 0;
    var height: u32 = 0;
    var bits_per_sample: u16 = 8;
    var strip_offsets: u32 = 0;
    var samples_per_pixel: u16 = 1;

    for (0..num_tags) |_| {
        const tag_id = try reader.takeInt(u16, endian);
        const tag_type = try reader.takeInt(u16, endian);
        const tag_count = try reader.takeInt(u32, endian);
        const tag_value = try reader.takeInt(u32, endian);

        switch (tag_id) {
            256 => width = if (tag_type == 3)
                @as(u32, @intCast(tag_value & 0xFFFF))
            else
                tag_value,
            257 => height = if (tag_type == 3)
                @as(u32, @intCast(tag_value & 0xFFFF))
            else
                tag_value,
            258 => bits_per_sample = @intCast(tag_value & 0xFFFF),
            273 => strip_offsets = tag_value,
            277 => samples_per_pixel = @intCast(tag_value & 0xFFFF),
            else => {},
        }
        _ = tag_count;
    }

    if (samples_per_pixel != 1) return error.UnsuppedTIFFColorSpace;

    var tex = try texops.Tex(T, channels).init(
        allocator,
        height,
        width,
    );
    errdefer tex.deinit(allocator);

    try file_reader.seekTo(strip_offsets);

    const max_val_f: F = if (bits_per_sample == 16) 65535.0 else 255.0;

    for (0..height) |rr| {
        for (0..width) |cc| {
            const val_raw: F = if (bits_per_sample == 16)
                @as(F, @floatFromInt(try reader.takeInt(u16, endian)))
            else
                @as(F, @floatFromInt(try reader.takeByte()));

            const norm = val_raw / max_val_f;

            if (channels == 3) {
                const out_val = convertToTarg(T, norm);
                tex.setVal(0, rr, cc, out_val);
                tex.setVal(1, rr, cc, out_val);
                tex.setVal(2, rr, cc, out_val);
            } else if (channels == 1) {
                tex.setVal(
                    0,
                    rr,
                    cc,
                    convertToTarg(T, norm),
                );
            }
        }
    }

    return tex;
}

pub fn CLoadTIFF(
    comptime T: type,
    comptime channels: usize,
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) !texops.Tex(T, channels) {
    _ = io;
    var libtiff = try clibtiff.LibTiff.init();
    defer libtiff.deinit();

    const path_c = try allocator.dupeZ(u8, path);
    defer allocator.free(path_c);

    const tif = libtiff.open(path_c, "r") orelse return error.OpenFailed;
    defer libtiff.close(tif);

    var w: u32 = 0;
    var h: u32 = 0;
    _ = libtiff.getField(tif, 256, &w);
    _ = libtiff.getField(tif, 257, &h);

    const pixel_count = w * h;
    const raster = try allocator.alloc(u32, pixel_count);
    defer allocator.free(raster);

    if (libtiff.readRGBAImage(tif, w, h, raster.ptr, 0) == 0) return error.ReadFailed;

    var tex = try texops.Tex(T, channels).init(allocator, h, w);
    errdefer tex.deinit(allocator);

    for (0..h) |row| {
        const src_row = (h - 1 - row) * w;
        for (0..w) |col| {
            const pixel = raster[src_row + col];
            const r = @as(u8, @intCast(pixel & 0xFF));
            const g = @as(u8, @intCast((pixel >> 8) & 0xFF));
            const b = @as(u8, @intCast((pixel >> 16) & 0xFF));

            if (channels == 3) {
                tex.setVal(0, row, col, convertValue(T, r));
                tex.setVal(1, row, col, convertValue(T, g));
                tex.setVal(2, row, col, convertValue(T, b));
            } else if (channels == 1) {
                const val = toGreyScale(r, g, b);
                tex.setVal(0, row, col, convertValue(T, val));
            }
        }
    }

    return tex;
}

inline fn toGreyScale(r: anytype, g: anytype, b: anytype) F {
    return 0.299 * @as(F, @floatFromInt(r)) +
        0.587 * @as(F, @floatFromInt(g)) +
        0.114 * @as(F, @floatFromInt(b));
}

fn convertToTarg(comptime T: type, norm: F) T {
    const scale = switch (@typeInfo(T)) {
        .int => |info| (@as(F, 1.0) * @as(
            F,
            @floatFromInt((@as(u64, 1) << info.bits) - 1),
        )),
        .float => 1.0,
        else => @compileError("Unsupped type"),
    };
    const val = norm * scale;
    switch (@typeInfo(T)) {
        .int => return @as(T, @intFromFloat(@round(@max(0.0, @min(scale, val))))),
        .float => return @as(T, @floatCast(val)),
        else => @compileError("Unsupped type"),
    }
}

fn convertValue(comptime T: type, val: anytype) T {
    const val_f64 = switch (@typeInfo(@TypeOf(val))) {
        .int => @as(F, @floatFromInt(val)),
        .float => @as(F, val),
        else => @compileError("Unsupped type"),
    };

    switch (@typeInfo(T)) {
        .int => return @as(T, @intFromFloat(@round(val_f64))),
        .float => return @as(T, @floatCast(val_f64)),
        else => @compileError("Unsupped type"),
    }
}

// --------------------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------------------

const testing = std.testing;

const frac_tol: F = if (F == f32) 1e-2 else 1e-6;

fn ensureTempTestDir(io: std.Io) !void {
    const cwd = std.Io.Dir.cwd();
    cwd.createDir(io, temp_test_root_dir, .default_dir) catch |err| {
        if (err != error.PathAlreadyExists) return err;
    };
    cwd.createDir(io, temp_test_dir, .default_dir) catch |err| {
        if (err != error.PathAlreadyExists) return err;
    };
}

fn deleteTempTestDir(io: std.Io) void {
    const cwd = std.Io.Dir.cwd();
    cwd.deleteTree(io, temp_test_root_dir) catch {};
}

test "Verify hand-written TIFF loader" {
    const allocator = testing.allocator;
    const io = std.testing.io;
    const cwd = std.Io.Dir.cwd();
    try ensureTempTestDir(io);
    defer deleteTempTestDir(io);
    const out_dir = cwd;
    const rows_num = 19;
    const cols_num = 23;
    const mat_size = rows_num * cols_num;
    const mat_mem = try allocator.alloc(F, mat_size);
    defer allocator.free(mat_mem);

    for (0..rows_num) |rr| {
        for (0..cols_num) |cc| {
            // Exercise a spread of 8-bit grayscale vals without requiring libtiff.
            mat_mem[rr * cols_num + cc] = @floatFromInt((rr * 17 + cc * 29) % 256);
        }
    }
    const mat = matslice.MatSlice(F).init(mat_mem, rows_num, cols_num);

    try saveMatAsImage(
        io,
        out_dir,
        temp_test_dir ++ "/speckle-simple",
        &mat,
        .{ .format = .tiff, .bits = 8, .scaling = .none },
    );

    var tex_zig = try loadImage(
        u8,
        1,
        allocator,
        io,
        temp_test_dir ++ "/speckle-simple.tiff",
        .tiff,
    );
    defer tex_zig.deinit(allocator);

    try testing.expectEqual(rows_num, tex_zig.rows_num);
    try testing.expectEqual(cols_num, tex_zig.cols_num);

    for (0..rows_num) |rr| {
        for (0..cols_num) |cc| {
            const expected = mat_mem[rr * cols_num + cc];
            const actual = tex_zig.getVal(0, rr, cc);
            try testing.expectEqual(expected, actual);
        }
    }
}

test "Save and Load All Formats 8-bit and 16-bit" {
    const allocator = testing.allocator;
    const io = std.testing.io;
    const cwd = std.Io.Dir.cwd();
    try ensureTempTestDir(io);
    defer deleteTempTestDir(io);
    const out_dir = cwd;

    const rows = 4;
    const cols = 4;
    const mat_mem = try allocator.alloc(F, rows * cols);
    defer allocator.free(mat_mem);
    // Data in [0, 100] range
    for (0..rows) |rr| {
        for (0..cols) |cc| {
            mat_mem[rr * cols + cc] = @as(F, @floatFromInt(rr * cols + cc)) * 5.0;
        }
    }
    const mat = matslice.MatSlice(F).init(mat_mem, rows, cols);

    const formats = [_]ImageFormat{ .csv, .fimg, .ppm, .tiff, .bmp };
    const bit_depths = [_]u8{ 8, 16 };

    for (formats) |fmt| {
        for (bit_depths) |bits| {
            const base_name = try std.fmt.allocPrint(
                allocator,
                temp_test_dir ++ "/test_io_{s}_{d}bit",
                .{ @tagName(fmt), bits },
            );
            defer allocator.free(base_name);

            // 1. Save with auto-scaling
            try saveMatAsImage(
                io,
                out_dir,
                base_name,
                &mat,
                .{ .format = fmt, .bits = bits, .scaling = .auto },
            );

            var ext_buff: [1024]u8 = undefined;
            const ext = switch (fmt) {
                .csv => ".csv",
                .fimg => ".fimg",
                .ppm => ".ppm",
                .bmp => ".bmp",
                .tiff => ".tiff",
            };
            const full_path = try std.fmt.bufPrint(
                ext_buff[0..],
                "{s}{s}",
                .{ base_name, ext },
            );

            // 2. Load back into u8 or u16
            if (bits == 8) {
                var loaded = try loadImage(
                    u8,
                    1,
                    allocator,
                    io,
                    full_path,
                    fmt,
                );
                defer loaded.deinit(allocator);
                try testing.expectEqual(rows, loaded.rows_num);
                try testing.expectEqual(cols, loaded.cols_num);
            } else {
                var loaded = try loadImage(
                    u16,
                    1,
                    allocator,
                    io,
                    full_path,
                    fmt,
                );
                defer loaded.deinit(allocator);
                try testing.expectEqual(rows, loaded.rows_num);
                try testing.expectEqual(cols, loaded.cols_num);
            }
        }
    }
}

test "Scaling Strategy: Fractional" {
    const allocator = testing.allocator;
    const io = std.testing.io;
    const cwd = std.Io.Dir.cwd();
    try ensureTempTestDir(io);
    defer deleteTempTestDir(io);

    const rows = 2;
    const cols = 2;
    const mat_mem = try allocator.alloc(F, rows * cols);
    defer allocator.free(mat_mem);
    // Data [0, 10]
    mat_mem[0] = 0.0;
    mat_mem[1] = 10.0;
    mat_mem[2] = 5.0;
    mat_mem[3] = 2.5;
    const mat = matslice.MatSlice(F).init(mat_mem, rows, cols);

    // Test 1: Frac [0.4, 0.6], bits = null (CSV) -> should map [0, 10] to [0.4, 0.6]
    const opts1 = ImageSaveOpts{
        .format = .csv,
        .bits = null,
        .scaling = .{ .frac = .{ 0.4, 0.6 } },
    };
    try saveMatAsImage(io, cwd, temp_test_dir ++ "/test_frac_float", &mat, opts1);

    var loaded1 = try loadImage(
        F,
        1,
        allocator,
        io,
        temp_test_dir ++ "/test_frac_float.csv",
        .csv,
    );
    defer loaded1.deinit(allocator);
    try testing.expectApproxEqAbs(loaded1.getVal(0, 0, 0), 0.4, frac_tol);
    try testing.expectApproxEqAbs(loaded1.getVal(0, 0, 1), 0.6, frac_tol);
    try testing.expectApproxEqAbs(loaded1.getVal(0, 1, 0), 0.5, frac_tol);
    try testing.expectApproxEqAbs(loaded1.getVal(0, 1, 1), 0.45, frac_tol);

    // Test 2: Frac [0.4, 0.6], bits = 8 (CSV) -> should map [0, 10] to [0.4*255, 0.6*255]
    const opts2 = ImageSaveOpts{
        .format = .csv,
        .bits = 8,
        .scaling = .{ .frac = .{ 0.4, 0.6 } },
    };
    try saveMatAsImage(io, cwd, temp_test_dir ++ "/test_frac_bits", &mat, opts2);

    var loaded2 = try loadImage(
        F,
        1,
        allocator,
        io,
        temp_test_dir ++ "/test_frac_bits.csv",
        .csv,
    );
    defer loaded2.deinit(allocator);
    try testing.expectApproxEqAbs(
        loaded2.getVal(0, 0, 0),
        0.4 * 255.0,
        frac_tol,
    );
    try testing.expectApproxEqAbs(
        loaded2.getVal(0, 0, 1),
        0.6 * 255.0,
        frac_tol,
    );
    try testing.expectApproxEqAbs(
        loaded2.getVal(0, 1, 0),
        0.5 * 255.0,
        frac_tol,
    );
    try testing.expectApproxEqAbs(
        loaded2.getVal(0, 1, 1),
        0.45 * 255.0,
        frac_tol,
    );
}

test "FIMG Save and Load Roundtrip" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const cwd = std.Io.Dir.cwd();

    const width: usize = 4;
    const height: usize = 3;
    const channels: usize = 2;

    var tex = try texops.Tex(F, channels).init(
        allocator,
        height,
        width,
    );
    defer tex.deinit(allocator);

    for (0..channels) |ch| {
        for (0..height) |rr| {
            for (0..width) |cc| {
                const val = @as(F, @floatFromInt(ch * 100 + rr * 10 + cc)) * 1.123456789;
                tex.setVal(ch, rr, cc, val);
            }
        }
    }

    try ensureTempTestDir(io);
    defer deleteTempTestDir(io);
    const file_base = temp_test_dir ++ "/test_roundtrip";
    const file_full = file_base ++ ".fimg";

    var dims = [_]usize{ channels, height, width };
    var strides = [_]usize{ height * width, width, 1 };
    const arr = ndarray.NDArray(F){
        .slice = tex.array.slice,
        .dims = &dims,
        .strides = &strides,
    };

    const opts = ImageSaveOpts{
        .format = .fimg,
        .channels = channels,
    };

    try saveImage(io, cwd, file_base, &arr, 0, opts);

    var loaded = try loadImage(F, channels, allocator, io, file_full, .fimg);
    defer loaded.deinit(allocator);

    try std.testing.expectEqual(loaded.rows_num, height);
    try std.testing.expectEqual(loaded.cols_num, width);

    for (0..channels) |ch| {
        for (0..height) |rr| {
            for (0..width) |cc| {
                const expected = tex.getVal(ch, rr, cc);
                const actual = loaded.getVal(ch, rr, cc);
                try std.testing.expectEqual(expected, actual);
            }
        }
    }
}
