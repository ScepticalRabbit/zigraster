const std = @import("std");
const print = std.debug.print;
const time = std.time;
const Instant = time.Instant;

const Vec3f = @import("vecstack.zig").Vec3f;
const Vec3SliceOps = @import("vecstack.zig").Vec3SliceOps;

const Mat44Ops = @import("matstack.zig").Mat44Ops;

const VecSlice = @import("vecslice.zig").VecSlice;
const MatSlice = @import("matslice.zig").MatSlice;
const NDArray = @import("ndarray.zig").NDArray;
const sliceops = @import("sliceops.zig");

const Camera = @import("camera.zig").Camera;

// TODO: comptime these to allow switching from f32 to f64 to compare precision

pub fn worldToRasterCoords(coord_world: Vec3f, camera: *const Camera) Vec3f {
    // TODO: simplify this to a matrix mult
    var coord_raster: Vec3f = Mat44Ops.mulVec3(f64, 
    										   camera.world_to_cam_mat, 
    										   coord_world);

    coord_raster.elems[0] = camera.image_dist 
                            * coord_raster.elems[0] 
                            / (-coord_raster.elems[2]);
    coord_raster.elems[1] = camera.image_dist 
                            * coord_raster.elems[1] 
                            / (-coord_raster.elems[2]);

    coord_raster.elems[0] = 2.0 * coord_raster.elems[0] 
                            / camera.image_dims[0];
    coord_raster.elems[1] = 2.0 * coord_raster.elems[1] 
                            / camera.image_dims[1];

    coord_raster.elems[0] = (coord_raster.elems[0] + 1.0) 
    	/ 2.0 * @as(f64, @floatFromInt(camera.pixels_num[0]));
    coord_raster.elems[1] = (1.0 - coord_raster.elems[1]) 
    	/ 2.0 * @as(f64, @floatFromInt(camera.pixels_num[1]));
    coord_raster.elems[2] = -1.0 * coord_raster.elems[2];

    return coord_raster;
}

pub fn edgeFun3(vert_0: Vec3f, vert_1: Vec3f, vert_2: Vec3f) f64 {
    return ((vert_2.get(0) - vert_0.get(0)) 
          * (vert_1.get(1) - vert_0.get(1)) 
          - (vert_2.get(1) - vert_0.get(1)) 
          * (vert_1.get(0) - vert_0.get(0)));
}

pub fn boundIndexMin(min_val: f64) usize {
    var min_ind: usize = @as(usize, @intFromFloat(@floor(min_val)));
    if (min_ind < 0) {
        min_ind = 0;
    }
    return min_ind;
}

pub fn boundIndexMax(max_val: f64, pixels_num: usize) usize {
    var max_ind: usize = @as(usize, @intFromFloat(@ceil(max_val)));
    if (max_ind > (pixels_num - 1)) {
        max_ind = (pixels_num - 1);
    }
    return max_ind;
}

pub fn averageImage(image_subpx: *const MatSlice(f64), 
                    sub_samp: u8, 
                    image_avg: *MatSlice(f64)) void {
                    
    const num_px_x: usize = (image_subpx.cols_n) / @as(usize, sub_samp);
    const num_px_y: usize = (image_subpx.rows_n) / @as(usize, sub_samp);
    const sub_samp_us: usize = @as(usize, sub_samp);
    const sub_samp_f: f64 = @as(f64, @floatFromInt(sub_samp));
    const subpx_per_px: f64 = sub_samp_f * sub_samp_f;

    // TODO: do some error checking on the Matrices here to check dims agree
    // with the variables above

    var px_sum: f64 = 0.0;

    for (0..num_px_y) |iy| {
        for (0..num_px_x) |ix| {
            px_sum = 0.0;
            for (0..sub_samp_us) |sy| {
                for (0..sub_samp_us) |sx| {
                    px_sum += image_subpx.get(sub_samp_us * iy + sy, 
                                              sub_samp_us * ix + sx);
                }
            }
            image_avg.set(iy, ix, px_sum / subpx_per_px);
        }
    }
}

// TODO: this could be a tagged union of nested structs with the methods. But wait until there
// is a third case 
const ImageFormat = enum {
    csv,
    ppm,    
};

pub fn saveImage(io: std.Io,
                 out_dir: std.Io.Dir, 
                 file_name_no_ext: []const u8,
                 image: *const MatSlice(f64),
                 format: ImageFormat,
                 ) !void {
                    
    var name_buff: [1024]u8 = undefined;                
       
    switch (format) {
        .csv => {
            const ext = ".csv";
            const file_name_ext = try std.fmt.bufPrint(name_buff[0..], 
                                                       "{s}{s}", 
                                                       .{ file_name_no_ext, ext });     
            try saveCSV(io,out_dir,file_name_ext,image);
        },
        .ppm => {
            const ext = ".ppm";
            const file_name_ext = try std.fmt.bufPrint(name_buff[0..], 
                                                       "{s}{s}", 
                                                      .{ file_name_no_ext, ext });
            try saveScaledPPM(io,out_dir,file_name_ext,image);
        },
    }
}

// NOTE: just for debugging due to how floats are scaled
pub fn saveScaledPPM(io: std.Io,
                     out_dir: std.Io.Dir, 
                     file_name: []const u8,
                     image: *const MatSlice(f64),
                     ) !void {

    const ppm_file: std.Io.File = try out_dir.createFile(io, file_name, .{});
    defer ppm_file.close(io);

    var write_buf: [4096]u8 = undefined;
    var file_writer = ppm_file.writer(io, &write_buf);
    const writer = &file_writer.interface;

    try writer.print("P3\n{d} {d}\n255\n", .{ image.cols_n, image.rows_n});

    const px_min: f64 = std.mem.min(f64,image.elems);
    const px_max: f64 = std.mem.max(f64,image.elems);
    const px_rng: f64 = px_max - px_min;

    for (0..image.rows_n) |rr| {
        for (0..image.cols_n) |cc| {
            const px_scaled = @as(u8,
                @intFromFloat((image.get(rr,cc) - px_min)/px_rng * 255.0)
            );  
            try writer.print("{d} {d} {d}\n", .{px_scaled,px_scaled,px_scaled});
        }
    }

    try writer.flush();   
}

pub fn saveCSV(io: std.Io,
               out_dir: std.Io.Dir, 
               file_name: []const u8,
               image: *const MatSlice(f64),
               ) !void {

    const csv_file = try out_dir.createFile(io, file_name, .{});
    defer csv_file.close(io);

    var write_buf: [4096]u8 = undefined;
    var file_writer = csv_file.writer(io,&write_buf);
    const writer = &file_writer.interface;

    for (0..image.rows_n) |rr| {
        for (0..image.cols_n) |cc| {
            try writer.print("{d},", .{image.get(rr, cc)});
        }
        try writer.print("\n",.{});
    }

    try writer.flush();
}
