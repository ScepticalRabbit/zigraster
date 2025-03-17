const Vec3f = @import("vector.zig").Vec3f;
const Mat44f = @import("matrix.zig").Mat44f;
const Mat44Ops = @import("matrix.zig").Mat44Ops;
const Rotation = @import("rotation.zig").Rotation;

const Camera = struct {
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
        var sensor_size: [2]f64 = undefined;
        sensor_size[0] = pixels_num[0] * pixels_size[0];
        sensor_size[1] = pixels_num[1] * pixels_size[1];

        const image_dist: f64 = (pos_world.subtract(roi_cent_world)).length();

        var image_dims: [2]f64 = undefined;
        image_dims[0] = (image_dist/focal_length) * sensor_size[0];
        image_dims[1] = (image_dist/focal_length) * sensor_size[1];

        var cam_to_world_mat: Mat44f = Mat44f.initIdentity();
        cam_to_world_mat.insertColVec(3, 0, 3, pos_world);
        cam_to_world_mat.insertSubMat(0, 0, 3, 3, rot_world.matrix);

        const world_to_cam_mat = Mat44Ops.inv(f64,cam_to_world_mat);

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
