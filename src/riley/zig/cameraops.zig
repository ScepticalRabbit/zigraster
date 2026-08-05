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

const cam = @import("camera.zig");
const meshio = @import("meshio.zig");
const mo = @import("meshpipeline.zig");
const sceneops = @import("sceneops.zig");
const vec = @import("vecstack.zig");
const matrix = @import("matstack.zig");
const rotation = @import("rotation.zig");
const rastcfg = @import("rasterconfig.zig");
const F = buildconfig.F;

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn prepareCameraSlice(
    allocator: std.mem.Allocator,
    camera_inputs: []const cam.CameraInput,
) ![]cam.CameraPrepared {
    const cameras = try allocator.alloc(cam.CameraPrepared, camera_inputs.len);
    for (camera_inputs, 0..) |camera_input, cc| {
        cameras[cc] = cam.CameraPrepared.init(
            allocator,
            camera_input,
        ) catch |err| {
            for (0..cc) |pp| cameras[pp].deinit(allocator);
            allocator.free(cameras);
            return err;
        };
    }

    return cameras;
}

pub fn toOpenGLInput(input: cam.CameraInput) cam.CameraInput {
    if (input.coord_sys == .opengl) return input;
    var opengl_input = input;
    const r_opencv = input.rot_world.matrix;
    const r_opencv_t = r_opencv.transpose();
    const neg_t = vec.initVec3(
        F,
        -input.pos_world.get(0),
        -input.pos_world.get(1),
        -input.pos_world.get(2),
    );
    opengl_input.pos_world = r_opencv_t.mulVec(neg_t);
    var r_riley = r_opencv_t;
    r_riley.slice[1] = -r_riley.slice[1];
    r_riley.slice[2] = -r_riley.slice[2];
    r_riley.slice[4] = -r_riley.slice[4];
    r_riley.slice[5] = -r_riley.slice[5];
    r_riley.slice[7] = -r_riley.slice[7];
    r_riley.slice[8] = -r_riley.slice[8];
    opengl_input.rot_world = rotation.Rotation.fromMat33(r_riley);
    opengl_input.coord_sys = .opengl;
    return opengl_input;
}

const CameraPlaneMetrics = struct {
    sensor_size: [2]F,
    focal_px: [2]F,
    principal_point_px: [2]F,
    roi_plane_dist: F,
    roi_plane_size: [2]F,
    avg_leng_per_pixel: F,
    avg_pixel_per_leng: F,
};

pub fn calcPlaneMetrics(camera_input: cam.CameraInput) CameraPlaneMetrics {
    const opengl_input = toOpenGLInput(camera_input);
    const scaling = calcFOVScaling(
        opengl_input,
        opengl_input.roi_cent_world,
    );
    const sensor_size = calcSensorSize(
        opengl_input.pixels_num,
        opengl_input.pixels_size,
    );
    const focal_px_x = opengl_input.focal_length / opengl_input.pixels_size[0];
    const focal_px_y = opengl_input.focal_length / opengl_input.pixels_size[1];
    const principal_x = 0.5 * @as(F, @floatFromInt(opengl_input.pixels_num[0]));
    const principal_y = 0.5 * @as(F, @floatFromInt(opengl_input.pixels_num[1]));
    return .{
        .sensor_size = sensor_size,
        .focal_px = .{ focal_px_x, focal_px_y },
        .principal_point_px = .{ principal_x, principal_y },
        .roi_plane_dist = scaling.plane_dist,
        .roi_plane_size = scaling.plane_size,
        .avg_leng_per_pixel = 0.5 * (scaling.leng_per_pixel[0] + scaling.leng_per_pixel[1]),
        .avg_pixel_per_leng = 0.5 * (scaling.pixel_per_leng[0] + scaling.pixel_per_leng[1]),
    };
}

pub fn fovFromCamRot(
    cam_rot: rotation.Rotation,
    coords_world: *const meshio.Coords,
) [2]F {
    return sceneops.extentInRotatedFrame(cam_rot, coords_world);
}

pub fn fovFromCamRotOverMeshes(
    cam_rot: rotation.Rotation,
    meshes: []const mo.MeshInput,
) [2]F {
    return sceneops.extentInRotatedFrameOverMeshes(cam_rot, meshes);
}

pub fn calcSensorSize(pixels_num: [2]u32, pixels_size: [2]F) [2]F {
    return .{
        @as(F, @floatFromInt(pixels_num[0])) * pixels_size[0],
        @as(F, @floatFromInt(pixels_num[1])) * pixels_size[1],
    };
}

pub fn imageDistFromFov(
    pixels_num: [2]u32,
    pixels_size: [2]F,
    focal_leng: F,
    fov_leng: [2]F,
) [2]F {
    const sensor_size = calcSensorSize(pixels_num, pixels_size);

    const fov_angle = [2]F{
        2 * std.math.atan(sensor_size[0] / (2 * focal_leng)),
        2 * std.math.atan(sensor_size[1] / (2 * focal_leng)),
    };

    return .{
        fov_leng[0] / (2 * std.math.tan(fov_angle[0] / 2)),
        fov_leng[1] / (2 * std.math.tan(fov_angle[1] / 2)),
    };
}

pub fn calcFOVScaling(
    camera_input: cam.CameraInput,
    plane_cent_world: vec.Vec3f,
) cam.FOVScaling {
    const cam_z_axis = camera_input.rot_world.matrix.getColVec(2);
    const plane_vec = (&camera_input.pos_world).sub(plane_cent_world);
    const plane_dist = @abs(plane_vec.dot(cam_z_axis));
    const sensor_size = calcSensorSize(
        camera_input.pixels_num,
        camera_input.pixels_size,
    );

    const plane_size = [2]F{
        (plane_dist / camera_input.focal_length) * sensor_size[0],
        (plane_dist / camera_input.focal_length) * sensor_size[1],
    };
    const leng_per_pixel = [2]F{
        plane_size[0] / @as(F, @floatFromInt(camera_input.pixels_num[0])),
        plane_size[1] / @as(F, @floatFromInt(camera_input.pixels_num[1])),
    };

    return .{
        .plane_dist = plane_dist,
        .plane_size = plane_size,
        .leng_per_pixel = leng_per_pixel,
        .pixel_per_leng = .{
            1.0 / leng_per_pixel[0],
            1.0 / leng_per_pixel[1],
        },
    };
}

pub fn calcCamPos(
    roi_pos_world: vec.Vec3f,
    cam_rot: rotation.Rotation,
    image_dist: F,
) vec.Vec3f {
    var cam_z_axis_vec = cam_rot.matrix.getColVec(2);
    cam_z_axis_vec = cam_z_axis_vec.mulScal(image_dist);
    return (&roi_pos_world).add(cam_z_axis_vec);
}

pub fn lookAtPoint(
    pos_world: vec.Vec3f,
    targ_world: vec.Vec3f,
) rotation.Rotation {
    const cam_z_axis = (&pos_world).sub(targ_world);
    const cam_z_leng = cam_z_axis.vecLen();

    if (cam_z_leng == 0.0) {
        return rotation.Rotation.init(0, 0, 0);
    }

    const cam_z_unit = cam_z_axis.mulScal(1.0 / cam_z_leng);
    const alpha_z = std.math.atan2(cam_z_unit.get(1), cam_z_unit.get(0));
    const beta_y = std.math.atan2(
        @sqrt(
            cam_z_unit.get(0) * cam_z_unit.get(0) +
                cam_z_unit.get(1) * cam_z_unit.get(1),
        ),
        cam_z_unit.get(2),
    );

    return rotation.Rotation.init(alpha_z, beta_y, 0.0);
}

pub fn imageDistFillFrameFromRot(
    coords_world: *const meshio.Coords,
    pixels_num: [2]u32,
    pixels_size: [2]F,
    focal_leng: F,
    cam_rot: rotation.Rotation,
    frame_fill: F,
) F {
    var fov_leng = fovFromCamRot(cam_rot, coords_world);
    fov_leng[0] = frame_fill * fov_leng[0];
    fov_leng[1] = frame_fill * fov_leng[1];

    const image_dists = imageDistFromFov(
        pixels_num,
        pixels_size,
        focal_leng,
        fov_leng,
    );
    return @max(image_dists[0], image_dists[1]);
}

pub fn posFillFrameFromRot(
    coords_world: *const meshio.Coords,
    pixels_num: [2]u32,
    pixels_size: [2]F,
    focal_leng: F,
    cam_rot: rotation.Rotation,
    frame_fill: F,
) vec.Vec3f {
    const image_dist = imageDistFillFrameFromRot(
        coords_world,
        pixels_num,
        pixels_size,
        focal_leng,
        cam_rot,
        frame_fill,
    );
    return calcCamPos(
        sceneops.boundsCenter(coords_world),
        cam_rot,
        image_dist,
    );
}

pub fn posFillFrameFromRotAndTarg(
    coords_world: *const meshio.Coords,
    targ_world: vec.Vec3f,
    pixels_num: [2]u32,
    pixels_size: [2]F,
    focal_leng: F,
    cam_rot: rotation.Rotation,
    frame_fill: F,
) vec.Vec3f {
    const image_dist = imageDistFillFrameFromRotAndTarg(
        coords_world,
        targ_world,
        pixels_num,
        pixels_size,
        focal_leng,
        cam_rot,
        frame_fill,
    );
    return calcCamPos(targ_world, cam_rot, image_dist);
}

pub fn posFillFrameFromRotOverMeshes(
    meshes: []const mo.MeshInput,
    pixels_num: [2]u32,
    pixels_size: [2]F,
    focal_leng: F,
    cam_rot: rotation.Rotation,
    frame_fill: F,
) vec.Vec3f {
    var fov_leng = fovFromCamRotOverMeshes(cam_rot, meshes);
    fov_leng[0] = frame_fill * fov_leng[0];
    fov_leng[1] = frame_fill * fov_leng[1];

    const image_dists = imageDistFromFov(
        pixels_num,
        pixels_size,
        focal_leng,
        fov_leng,
    );
    const image_dist = @max(image_dists[0], image_dists[1]);
    return calcCamPos(
        sceneops.boundsCenterOverMeshes(meshes),
        cam_rot,
        image_dist,
    );
}

pub fn posFillFrameFromRotOverMeshesAndTarg(
    meshes: []const mo.MeshInput,
    targ_world: vec.Vec3f,
    pixels_num: [2]u32,
    pixels_size: [2]F,
    focal_leng: F,
    cam_rot: rotation.Rotation,
    frame_fill: F,
) vec.Vec3f {
    const image_dist = imageDistFillFrameFromRotOverMeshesAndTarg(
        meshes,
        targ_world,
        pixels_num,
        pixels_size,
        focal_leng,
        cam_rot,
        frame_fill,
    );
    return calcCamPos(targ_world, cam_rot, image_dist);
}

// --------------------------------------------------------------------------------------
// Generic Low-Level Helpers
// --------------------------------------------------------------------------------------

fn imageDistFillFrameFromRotAndTarg(
    coords_world: *const meshio.Coords,
    targ_world: vec.Vec3f,
    pixels_num: [2]u32,
    pixels_size: [2]F,
    focal_leng: F,
    cam_rot: rotation.Rotation,
    frame_fill: F,
) F {
    const world_to_cam_mat = matrix.Mat33Ops.inv(F, cam_rot.matrix);
    var coord_cam = world_to_cam_mat.mulVec(coords_world.getVec3(0).sub(targ_world));
    var max_abs_x = @abs(coord_cam.get(0));
    var max_abs_y = @abs(coord_cam.get(1));

    for (1..coords_world.mat.rows_num) |nn| {
        coord_cam = world_to_cam_mat.mulVec(coords_world.getVec3(nn).sub(targ_world));
        max_abs_x = @max(max_abs_x, @abs(coord_cam.get(0)));
        max_abs_y = @max(max_abs_y, @abs(coord_cam.get(1)));
    }

    const fov_leng = [2]F{
        2.0 * frame_fill * max_abs_x,
        2.0 * frame_fill * max_abs_y,
    };
    const image_dists = imageDistFromFov(
        pixels_num,
        pixels_size,
        focal_leng,
        fov_leng,
    );
    return @max(image_dists[0], image_dists[1]);
}

fn imageDistFillFrameFromRotOverMeshesAndTarg(
    meshes: []const mo.MeshInput,
    targ_world: vec.Vec3f,
    pixels_num: [2]u32,
    pixels_size: [2]F,
    focal_leng: F,
    cam_rot: rotation.Rotation,
    frame_fill: F,
) F {
    const world_to_cam_mat = matrix.Mat33Ops.inv(F, cam_rot.matrix);
    var max_abs_x: F = 0.0;
    var max_abs_y: F = 0.0;
    var is_first = true;

    for (meshes) |mesh| {
        for (0..mesh.coords.mat.rows_num) |nn| {
            const coord_cam = world_to_cam_mat.mulVec(
                mesh.coords.getVec3(nn).sub(targ_world),
            );
            if (is_first) {
                max_abs_x = @abs(coord_cam.get(0));
                max_abs_y = @abs(coord_cam.get(1));
                is_first = false;
            } else {
                max_abs_x = @max(max_abs_x, @abs(coord_cam.get(0)));
                max_abs_y = @max(max_abs_y, @abs(coord_cam.get(1)));
            }
        }
    }

    const fov_leng = [2]F{
        2.0 * frame_fill * max_abs_x,
        2.0 * frame_fill * max_abs_y,
    };
    const image_dists = imageDistFromFov(
        pixels_num,
        pixels_size,
        focal_leng,
        fov_leng,
    );
    return @max(image_dists[0], image_dists[1]);
}
