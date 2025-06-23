const std = @import("std");
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
    mat: [*c]f64,
    numel: usize,
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