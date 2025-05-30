const std = @import("std");
const print = std.debug.print;
const testing = std.testing;
const expectEqual = testing.expectEqual;
const expectApproxEqAbs = testing.expectApproxEqAbs;

const Coords = @import("meshio.zig").Coords;
const vector = @import("vector.zig");
const Vec2f = vector.Vec2f;
const Vec3f = vector.Vec3f;
const matrix = @import("matrix.zig");
const Mat33f = matrix.Mat33f;
const Mat33Ops = matrix.Mat33Ops;
const Mat44f = matrix.Mat44f;
const Mat44Ops = matrix.Mat44Ops;
const Rotation = @import("rotation.zig").Rotation;

pub const Camera = struct {
    // TODO: which of these actually need to be stored?
    pixels_num: [2]u32,
    pixels_size: [2]f64,
    pos_world: Vec3f,
    rot_world: Rotation,
    roi_cent_world: Vec3f,
    focal_length: f64,
    sub_sample: u8,
    sensor_size: [2]f64,
    image_dims: [2]f64,
    image_dist: f64,
    cam_to_world_mat: Mat44f,
    world_to_cam_mat: Mat44f,

    pub fn init(pixels_num: [2]u32, pixels_size: [2]f64, pos_world: Vec3f, rot_world: Rotation, roi_cent_world: Vec3f, focal_length: f64, sub_sample: u8) Camera {
        const sensor_size = CameraOps.calc_sensor_size(pixels_num, pixels_size);
        const image_dist: f64 = (pos_world.sub(roi_cent_world)).vecLen();

        var image_dims: [2]f64 = undefined;
        image_dims[0] = (image_dist / focal_length) * sensor_size[0];
        image_dims[1] = (image_dist / focal_length) * sensor_size[1];

        var cam_to_world_mat: Mat44f = Mat44f.initIdentity();
        cam_to_world_mat.insertColVec(3, 0, 3, pos_world);
        cam_to_world_mat.insertSubMat(0, 0, 3, 3, rot_world.matrix);

        const world_to_cam_mat = Mat44Ops.inv(f64, cam_to_world_mat);

        return .{
            .pixels_num = pixels_num,
            .pixels_size = pixels_size,
            .pos_world = pos_world,
            .rot_world = rot_world,
            .roi_cent_world = roi_cent_world,
            .focal_length = focal_length,
            .sub_sample = sub_sample,
            .sensor_size = sensor_size,
            .image_dims = image_dims,
            .image_dist = image_dist,
            .cam_to_world_mat = cam_to_world_mat,
            .world_to_cam_mat = world_to_cam_mat,
        };
    }
};

pub const CameraOps = struct {
    // TODO: maybe this should return a Vec2f?
    pub fn fov_from_cam_rot(cam_rot: Rotation, coords_world: *const Coords) [2]f64 {
        const world_to_cam_mat = Mat33Ops.inv(f64, cam_rot.matrix);

        const bb_min_x = std.mem.min(f64, coords_world.x[0..]);
        const bb_min_y = std.mem.min(f64, coords_world.y[0..]);
        const bb_min_z = std.mem.min(f64, coords_world.z[0..]);
        const bb_max_x = std.mem.max(f64, coords_world.x[0..]);
        const bb_max_y = std.mem.max(f64, coords_world.y[0..]);
        const bb_max_z = std.mem.max(f64, coords_world.z[0..]);

        print("\n",.{});
        print("bb_min=[{d},{d},{d}]\n",.{bb_min_x,bb_min_y,bb_min_z});
        print("bb_max=[{d},{d},{d}]",.{bb_max_x,bb_max_y,bb_max_z});

        print("\nCam to world mat:\n",.{});
        cam_rot.matrix.matPrint();
        print("\nWorld to cam mat:\n",.{});
        world_to_cam_mat.matPrint();
        print("\n",.{});

        var bb_world_vecs: [8]Vec3f = undefined;
        bb_world_vecs[0] = vector.initVec3(f64, bb_min_x, bb_min_y, bb_max_z);
        bb_world_vecs[1] = vector.initVec3(f64, bb_max_x, bb_min_y, bb_max_z);
        bb_world_vecs[2] = vector.initVec3(f64, bb_max_x, bb_max_y, bb_max_z);
        bb_world_vecs[3] = vector.initVec3(f64, bb_min_x, bb_max_y, bb_max_z);
        bb_world_vecs[4] = vector.initVec3(f64, bb_min_x, bb_min_y, bb_min_z);
        bb_world_vecs[5] = vector.initVec3(f64, bb_max_x, bb_min_y, bb_min_z);
        bb_world_vecs[6] = vector.initVec3(f64, bb_max_x, bb_max_y, bb_min_z);
        bb_world_vecs[7] = vector.initVec3(f64, bb_min_x, bb_max_y, bb_min_z);

        var bb_cam_vec: Vec3f = undefined;
        bb_cam_vec = world_to_cam_mat.mulVec(bb_world_vecs[0]);
        var bb_cam_max = [_]f64{ bb_cam_vec.get(0), bb_cam_vec.get(1) };
        var bb_cam_min = [_]f64{ bb_cam_vec.get(0), bb_cam_vec.get(1) };

        for (bb_world_vecs[1..]) |vec| {
            bb_cam_vec = world_to_cam_mat.mulVec(vec);

            if (bb_cam_vec.get(0) > bb_cam_max[0]) {
                bb_cam_max[0] = bb_cam_vec.get(0);
            } else if (bb_cam_vec.get(0) < bb_cam_min[0]) {
                bb_cam_min[0] = bb_cam_vec.get(0);
            }

            if (bb_cam_vec.get(1) > bb_cam_max[1]) {
                bb_cam_max[1] = bb_cam_vec.get(1);
            } else if (bb_cam_vec.get(1) < bb_cam_min[1]) {
                bb_cam_min[1] = bb_cam_vec.get(1);
            }
        }

        const fov_x = bb_cam_max[0] - bb_cam_min[0];
        const fov_y = bb_cam_max[1] - bb_cam_min[1];
        const fov_leng = [2]f64{ fov_x, fov_y };
        return fov_leng;
    }

    pub fn calc_sensor_size(pixels_num: [2]u32, pixels_size: [2]f64) [2]f64 {
        var sensor_size: [2]f64 = undefined;
        sensor_size[0] = @as(f64, @floatFromInt(pixels_num[0])) * pixels_size[0];
        sensor_size[1] = @as(f64, @floatFromInt(pixels_num[1])) * pixels_size[1];
        return sensor_size;
    }

    pub fn image_dist_from_fov(pixels_num: [2]u32, pixels_size: [2]f64, focal_leng: f64, fov_leng: [2]f64) [2]f64 {
        const sensor_size = calc_sensor_size(pixels_num, pixels_size);

        var fov_angle: [2]f64 = undefined;
        fov_angle[0] = 2 * std.math.atan(sensor_size[0] / (2 * focal_leng));
        fov_angle[1] = 2 * std.math.atan(sensor_size[1] / (2 * focal_leng));

        var image_dist: [2]f64 = undefined;
        image_dist[0] = fov_leng[0] / (2 * std.math.tan(fov_angle[0] / 2));
        image_dist[1] = fov_leng[1] / (2 * std.math.tan(fov_angle[1] / 2));

        return image_dist;
    }

    pub fn calc_cam_pos(roi_pos_world: Vec3f, cam_rot: Rotation, image_dist: f64) Vec3f {
        var cam_z_axis_vec = cam_rot.matrix.getColVec(2);
        cam_z_axis_vec = cam_z_axis_vec.mulScalar(image_dist);
        const cam_pos = roi_pos_world.add(cam_z_axis_vec);
        return cam_pos;
    }

    pub fn roi_cent_from_coords(coords_world: *const Coords) Vec3f {
        var max_vec: Vec3f = undefined;
        max_vec.elems[0] = std.mem.max(f64,coords_world.x[0..]);
        max_vec.elems[1] = std.mem.max(f64,coords_world.y[0..]);
        max_vec.elems[2] = std.mem.max(f64,coords_world.z[0..]);

        var min_vec: Vec3f = undefined;
        min_vec.elems[0] = std.mem.min(f64,coords_world.x[0..]);
        min_vec.elems[1] = std.mem.min(f64,coords_world.y[0..]);
        min_vec.elems[2] = std.mem.min(f64,coords_world.z[0..]);

        var roi_cent: Vec3f = max_vec.sub(min_vec);
        roi_cent = roi_cent.mulScalar(0.5);
        return roi_cent;
    }

    pub fn pos_fill_frame_from_rot(coords_world: *const Coords, pixels_num: [2]u32, pixels_size: [2]f64, focal_leng: f64, cam_rot: Rotation, frame_fill: f64) Vec3f {
        var fov_leng: [2]f64 = fov_from_cam_rot(cam_rot,coords_world);
        fov_leng[0] = frame_fill*fov_leng[0];
        fov_leng[1] = frame_fill*fov_leng[1];

        const image_dists: [2]f64 = image_dist_from_fov(pixels_num, pixels_size, focal_leng, fov_leng);
        const image_dist = @max(image_dists[0],image_dists[1]);

        print("\nfov_leng=[{any},{any}]\n",.{fov_leng[0],fov_leng[1]});
        print("image_dists=[{any},{any}]\n",.{image_dists[0],image_dists[1]});
        print("image_dist={any}\n",.{image_dist});

        const roi_pos: Vec3f = roi_cent_from_coords(coords_world);

        const cam_pos: Vec3f = calc_cam_pos(roi_pos, cam_rot, image_dist);

        return cam_pos;
    }
};

//------------------------------------------------------------------------------
const test_tol: f64 = 1e-4;
const pix_num = [_]u32{ 500, 500 };
const pix_size = [_]f64{ 5e-3, 5e-3 };
const foc_leng: f64 = 50.0;
const rotat_world = Rotation.init(0, 0, std.math.degreesToRadians(-45));
const bb: f64 = 20.0;
const coord_n: usize = 8;
const coord_x = [_]f64{ -bb, bb, bb, -bb, -bb, bb, bb, -bb };
const coord_y = [_]f64{ bb, bb, -bb, -bb, bb, bb, -bb, -bb };
const coord_z = [_]f64{ bb, bb, bb, bb, -bb, -bb, -bb, -bb };
const roi_world_arr = [_]f64{ 0, 0, 0 };
const roi_world = Vec3f.initSlice(&roi_world_arr);
const sub_samp: u8 = 2;

const fov_exp = [2]f64{ 40.0, 56.56854249 };
const image_dist_exp = [2]f64{ 800.0, 1131.3708499 };
const sensor_size_exp = [2]f64{ 2.5, 2.5 };
const cam_pos_arr = [_]f64{ 0.0, 800.0, 800.0 };
const cam_pos_exp = Vec3f.initSlice(&cam_pos_arr);

//TODO
test "CameraOps.pos_fill_frame_from_rot" {

}

test "CameraOps.calc_cam_pos" {
    const coords = try Coords.init(testing.allocator, coord_n);
    defer coords.deinit();

    @memcpy(coords.x, coord_x[0..]);
    @memcpy(coords.y, coord_y[0..]);
    @memcpy(coords.z, coord_z[0..]);

    const fov_leng = CameraOps.fov_from_cam_rot(rotat_world, &coords);
    const image_dist = CameraOps.image_dist_from_fov(pix_num, pix_size, foc_leng, fov_leng);
    const image_dist_max = @max(image_dist[0],image_dist[1]);
    const cam_pos = CameraOps.calc_cam_pos(roi_world, rotat_world, image_dist_max);

    try expectApproxEqAbs(cam_pos_exp.get(0), cam_pos.get(0),test_tol);
    try expectApproxEqAbs(cam_pos_exp.get(1), cam_pos.get(1),test_tol);
    try expectApproxEqAbs(cam_pos_exp.get(2), cam_pos.get(2),test_tol);
}

test "CameraOps.image_dist_from_fov" {
    const coords = try Coords.init(testing.allocator, coord_n);
    defer coords.deinit();

    @memcpy(coords.x, coord_x[0..]);
    @memcpy(coords.y, coord_y[0..]);
    @memcpy(coords.z, coord_z[0..]);

    const fov_leng = CameraOps.fov_from_cam_rot(rotat_world, &coords);
    const image_dist = CameraOps.image_dist_from_fov(pix_num, pix_size, foc_leng, fov_leng);

    try expectApproxEqAbs(image_dist_exp[0], image_dist[0], test_tol);
    try expectApproxEqAbs(image_dist_exp[1], image_dist[1], test_tol);
}

test "CameraOps.fov_from_cam_rot" {
    const coords = try Coords.init(testing.allocator, coord_n);
    defer coords.deinit();

    @memcpy(coords.x, coord_x[0..]);
    @memcpy(coords.y, coord_y[0..]);
    @memcpy(coords.z, coord_z[0..]);

    const fov_leng = CameraOps.fov_from_cam_rot(rotat_world, &coords);

    try expectApproxEqAbs(fov_exp[0], fov_leng[0], test_tol);
    try expectApproxEqAbs(fov_exp[1], fov_leng[1], test_tol);
}

test "CameraOps.calc_sensor_size" {
    const sensor_size = CameraOps.calc_sensor_size(pix_num, pix_size);

    try expectEqual(sensor_size_exp, sensor_size);
}

