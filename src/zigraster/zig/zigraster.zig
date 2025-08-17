const std = @import("std");
const print = std.debug.print;
const testing = std.testing;

pub const CVec2U32 = extern struct {
    x: u32,
    y: u32,
};

pub const CVec2F = extern struct {
    x: f64,
    y: f64,
};

pub const CVec3F = extern struct {
    x: f64,
    y: f64,
    z: f64,
};

pub const CMat44F = extern struct {
    elems: [*c]f64,
    elems_num: usize,
};


pub const CNDArrayF = extern struct {
    elems: [*c]f64,
    dims: [*c]usize,
    elems_num: usize,
    dims_num: usize,
};

pub const CCamera = extern struct {
    pixels_num: CVec2U32,
    pixels_size: CVec2F,
    pos_world: CVec3F,
    rot_world: CVec3F,
    roi_cent_world: CVec3F,
    subsample: u8,
    sensor_size: CVec2F,
    image_dims: CVec2F,
    image_dist: f64,
    cam_to_world: CMat44F,
    world_to_cam: CMat44F,
};

// Function for testing sending a complex struct to Zig
pub export fn printCamera(cam: *const CCamera) void {
    print("\nZig Camera:\n", .{});
    print("--------------------\n", .{});
    print("pixels_num[y,x]=[{},{}]\n", .{ cam.pixels_num.y, cam.pixels_num.x });
    print("pixels_size[y,x]=[{},{}]\n", .{ cam.pixels_size.y, cam.pixels_size.x });
    print("pos_world[x,y,z]=[{},{},{}]\n",.{cam.pos_world.x,cam.pos_world.y,cam.pos_world.z});
    print("rot_world[x,y,z]=[{},{},{}]\n",.{cam.rot_world.x,cam.rot_world.y,cam.rot_world.z});
    print("roi_cent[x,y,z]=[{},{},{}]\n",.{cam.roi_cent_world.x,cam.roi_cent_world.y,cam.roi_cent_world.z});
    print("subsample={}\n",.{cam.subsample});
    print("sensor_size[y,x]=[{},{}]\n", .{ cam.sensor_size.y, cam.sensor_size.x });
    print("image_dims[y,x]=[{},{}]\n", .{ cam.image_dims.y, cam.image_dims.x });
    print("image_dist={}\n",.{cam.image_dist});
    print("\n", .{});
}

