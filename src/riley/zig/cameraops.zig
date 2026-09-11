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
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const FrameFitMode = enum(u32) {
    contain = 0,
    cover = 1,
    horizontal = 2,
    vertical = 3,
};

pub const OrbitCam = struct {
    pos: vec.Vec3f,
    rot: rotation.Rotation,
};

pub const StereoPairPosRot = struct {
    cam0_pos: vec.Vec3f,
    cam0_rot: rotation.Rotation,
    cam1_pos: vec.Vec3f,
    cam1_rot: rotation.Rotation,
};

pub const CameraPlaneMetrics = struct {
    sensor_size: [2]F,
    focal_px: [2]F,
    principal_point_px: [2]F,
    roi_plane_dist: F,
    roi_plane_size: [2]F,
    avg_leng_per_pixel: F,
    avg_pixel_per_leng: F,
};

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

pub fn coverageToFovScale(coverage: F) F {
    std.debug.assert(coverage > 0.0);
    return 1.0 / coverage;
}

pub fn fovScaleToCoverage(fov_scale: F) F {
    std.debug.assert(fov_scale > 0.0);
    return 1.0 / fov_scale;
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

pub fn imageDistFrameCoords(
    coords_world: *const meshio.Coords,
    pixels_num: [2]u32,
    pixels_size: [2]F,
    focal_leng: F,
    cam_rot: rotation.Rotation,
    fov_scale: F,
    fit_mode: FrameFitMode,
) F {
    var fov_leng = fovFromCamRot(cam_rot, coords_world);
    fov_leng[0] = fov_scale * fov_leng[0];
    fov_leng[1] = fov_scale * fov_leng[1];

    const image_dists = imageDistFromFov(
        pixels_num,
        pixels_size,
        focal_leng,
        fov_leng,
    );
    return selectFitDist(image_dists, fit_mode);
}

pub fn imageDistFrameCoordsTarg(
    coords_world: *const meshio.Coords,
    targ_world: vec.Vec3f,
    pixels_num: [2]u32,
    pixels_size: [2]F,
    focal_leng: F,
    cam_rot: rotation.Rotation,
    fov_scale: F,
    fit_mode: FrameFitMode,
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
        2.0 * fov_scale * max_abs_x,
        2.0 * fov_scale * max_abs_y,
    };
    const image_dists = imageDistFromFov(
        pixels_num,
        pixels_size,
        focal_leng,
        fov_leng,
    );
    return selectFitDist(image_dists, fit_mode);
}

pub fn imageDistFrameMeshes(
    meshes: []const mo.MeshInput,
    pixels_num: [2]u32,
    pixels_size: [2]F,
    focal_leng: F,
    cam_rot: rotation.Rotation,
    fov_scale: F,
    fit_mode: FrameFitMode,
) F {
    var fov_leng = fovFromCamRotOverMeshes(cam_rot, meshes);
    fov_leng[0] = fov_scale * fov_leng[0];
    fov_leng[1] = fov_scale * fov_leng[1];

    const image_dists = imageDistFromFov(
        pixels_num,
        pixels_size,
        focal_leng,
        fov_leng,
    );
    return selectFitDist(image_dists, fit_mode);
}

pub fn imageDistFrameMeshesTarg(
    meshes: []const mo.MeshInput,
    targ_world: vec.Vec3f,
    pixels_num: [2]u32,
    pixels_size: [2]F,
    focal_leng: F,
    cam_rot: rotation.Rotation,
    fov_scale: F,
    fit_mode: FrameFitMode,
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
        2.0 * fov_scale * max_abs_x,
        2.0 * fov_scale * max_abs_y,
    };
    const image_dists = imageDistFromFov(
        pixels_num,
        pixels_size,
        focal_leng,
        fov_leng,
    );
    return selectFitDist(image_dists, fit_mode);
}

pub fn posFrameCoords(
    coords_world: *const meshio.Coords,
    pixels_num: [2]u32,
    pixels_size: [2]F,
    focal_leng: F,
    cam_rot: rotation.Rotation,
    fov_scale: F,
    fit_mode: FrameFitMode,
) vec.Vec3f {
    const image_dist = imageDistFrameCoords(
        coords_world,
        pixels_num,
        pixels_size,
        focal_leng,
        cam_rot,
        fov_scale,
        fit_mode,
    );
    return calcCamPos(
        sceneops.boundsCenter(coords_world),
        cam_rot,
        image_dist,
    );
}

pub fn posFrameCoordsTarg(
    coords_world: *const meshio.Coords,
    targ_world: vec.Vec3f,
    pixels_num: [2]u32,
    pixels_size: [2]F,
    focal_leng: F,
    cam_rot: rotation.Rotation,
    fov_scale: F,
    fit_mode: FrameFitMode,
) vec.Vec3f {
    const image_dist = imageDistFrameCoordsTarg(
        coords_world,
        targ_world,
        pixels_num,
        pixels_size,
        focal_leng,
        cam_rot,
        fov_scale,
        fit_mode,
    );
    return calcCamPos(targ_world, cam_rot, image_dist);
}

pub fn posFrameMesh(
    mesh: mo.MeshInput,
    pixels_num: [2]u32,
    pixels_size: [2]F,
    focal_leng: F,
    cam_rot: rotation.Rotation,
    fov_scale: F,
    fit_mode: FrameFitMode,
) vec.Vec3f {
    return posFrameCoords(
        &mesh.coords,
        pixels_num,
        pixels_size,
        focal_leng,
        cam_rot,
        fov_scale,
        fit_mode,
    );
}

pub fn posFrameMeshes(
    meshes: []const mo.MeshInput,
    pixels_num: [2]u32,
    pixels_size: [2]F,
    focal_leng: F,
    cam_rot: rotation.Rotation,
    fov_scale: F,
    fit_mode: FrameFitMode,
) vec.Vec3f {
    const image_dist = imageDistFrameMeshes(
        meshes,
        pixels_num,
        pixels_size,
        focal_leng,
        cam_rot,
        fov_scale,
        fit_mode,
    );
    return calcCamPos(
        sceneops.boundsCenterOverMeshes(meshes),
        cam_rot,
        image_dist,
    );
}

pub fn posFrameMeshesTarg(
    meshes: []const mo.MeshInput,
    targ_world: vec.Vec3f,
    pixels_num: [2]u32,
    pixels_size: [2]F,
    focal_leng: F,
    cam_rot: rotation.Rotation,
    fov_scale: F,
    fit_mode: FrameFitMode,
) vec.Vec3f {
    const image_dist = imageDistFrameMeshesTarg(
        meshes,
        targ_world,
        pixels_num,
        pixels_size,
        focal_leng,
        cam_rot,
        fov_scale,
        fit_mode,
    );
    return calcCamPos(targ_world, cam_rot, image_dist);
}

pub fn posOrbitCam(
    targ_world: vec.Vec3f,
    azimuth_rad: F,
    elevation_rad: F,
    dist: F,
) OrbitCam {
    const cos_elev = @cos(elevation_rad);
    const offset = vec.initVec3(
        F,
        dist * cos_elev * @cos(azimuth_rad),
        dist * cos_elev * @sin(azimuth_rad),
        dist * @sin(elevation_rad),
    );
    const pos = (&targ_world).add(offset);
    const rot = lookAtPoint(pos, targ_world);
    return .{
        .pos = pos,
        .rot = rot,
    };
}

pub fn posStereoPair(
    targ_world: vec.Vec3f,
    dist: F,
    stereo_angle_rad: F,
    baseline_angle_rad: F,
) StereoPairPosRot {
    const half_angle = 0.5 * stereo_angle_rad;
    const cam0_orbit = posOrbitCam(
        targ_world,
        baseline_angle_rad - half_angle,
        0.0,
        dist,
    );
    const cam1_orbit = posOrbitCam(
        targ_world,
        baseline_angle_rad + half_angle,
        0.0,
        dist,
    );
    return .{
        .cam0_pos = cam0_orbit.pos,
        .cam0_rot = cam0_orbit.rot,
        .cam1_pos = cam1_orbit.pos,
        .cam1_rot = cam1_orbit.rot,
    };
}

pub fn calcPixelResolution(
    camera_input: cam.CameraInput,
    targ_world: vec.Vec3f,
) F {
    const opengl_input = toOpenGLInput(camera_input);
    const scaling = calcFOVScaling(opengl_input, targ_world);
    return 0.5 * (scaling.leng_per_pixel[0] + scaling.leng_per_pixel[1]);
}

// --------------------------------------------------------------------------------------
// Deprecated Backwards-Compatible Aliases
// --------------------------------------------------------------------------------------

pub fn imageDistFillFrameFromRot(
    coords_world: *const meshio.Coords,
    pixels_num: [2]u32,
    pixels_size: [2]F,
    focal_leng: F,
    cam_rot: rotation.Rotation,
    fov_scale: F,
) F {
    return imageDistFrameCoords(
        coords_world,
        pixels_num,
        pixels_size,
        focal_leng,
        cam_rot,
        fov_scale,
        .contain,
    );
}

pub fn posFillFrameFromRot(
    coords_world: *const meshio.Coords,
    pixels_num: [2]u32,
    pixels_size: [2]F,
    focal_leng: F,
    cam_rot: rotation.Rotation,
    fov_scale: F,
) vec.Vec3f {
    return posFrameCoords(
        coords_world,
        pixels_num,
        pixels_size,
        focal_leng,
        cam_rot,
        fov_scale,
        .contain,
    );
}

pub fn posFillFrameFromRotAndTarg(
    coords_world: *const meshio.Coords,
    targ_world: vec.Vec3f,
    pixels_num: [2]u32,
    pixels_size: [2]F,
    focal_leng: F,
    cam_rot: rotation.Rotation,
    fov_scale: F,
) vec.Vec3f {
    return posFrameCoordsTarg(
        coords_world,
        targ_world,
        pixels_num,
        pixels_size,
        focal_leng,
        cam_rot,
        fov_scale,
        .contain,
    );
}

pub fn posFillFrameFromRotOverMeshes(
    meshes: []const mo.MeshInput,
    pixels_num: [2]u32,
    pixels_size: [2]F,
    focal_leng: F,
    cam_rot: rotation.Rotation,
    fov_scale: F,
) vec.Vec3f {
    return posFrameMeshes(
        meshes,
        pixels_num,
        pixels_size,
        focal_leng,
        cam_rot,
        fov_scale,
        .contain,
    );
}

pub fn posFillFrameFromRotOverMeshesAndTarg(
    meshes: []const mo.MeshInput,
    targ_world: vec.Vec3f,
    pixels_num: [2]u32,
    pixels_size: [2]F,
    focal_leng: F,
    cam_rot: rotation.Rotation,
    fov_scale: F,
) vec.Vec3f {
    return posFrameMeshesTarg(
        meshes,
        targ_world,
        pixels_num,
        pixels_size,
        focal_leng,
        cam_rot,
        fov_scale,
        .contain,
    );
}

// --------------------------------------------------------------------------------------
// Generic Low-Level Helpers
// --------------------------------------------------------------------------------------

fn selectFitDist(image_dists: [2]F, fit_mode: FrameFitMode) F {
    return switch (fit_mode) {
        .contain => @max(image_dists[0], image_dists[1]),
        .cover => @min(image_dists[0], image_dists[1]),
        .horizontal => image_dists[0],
        .vertical => image_dists[1],
    };
}

// --------------------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------------------

test "coverageToFovScale and fovScaleToCoverage roundtrip" {
    const coverage: F = 0.8;
    const fov_scale = coverageToFovScale(coverage);
    try std.testing.expectApproxEqRel(@as(F, 1.25), fov_scale, 1e-6);
    const roundtrip = fovScaleToCoverage(fov_scale);
    try std.testing.expectApproxEqRel(coverage, roundtrip, 1e-6);
}

test "FrameFitMode distance selection" {
    const dists = [2]F{ 100.0, 50.0 };
    try std.testing.expectEqual(@as(F, 100.0), selectFitDist(dists, .contain));
    try std.testing.expectEqual(@as(F, 50.0), selectFitDist(dists, .cover));
    try std.testing.expectEqual(@as(F, 100.0), selectFitDist(dists, .horizontal));
    try std.testing.expectEqual(@as(F, 50.0), selectFitDist(dists, .vertical));
}

test "posOrbitCam placement and rotation" {
    const targ = vec.initVec3(F, 0.0, 0.0, 0.0);
    const dist: F = 100.0;
    const orbit = posOrbitCam(targ, 0.0, 0.0, dist);
    try std.testing.expectApproxEqRel(@as(F, 100.0), orbit.pos.get(0), 1e-6);
    try std.testing.expectApproxEqRel(@as(F, 0.0), orbit.pos.get(1), 1e-6);
    try std.testing.expectApproxEqRel(@as(F, 0.0), orbit.pos.get(2), 1e-6);
}

test "posStereoPair symmetric separation" {
    const targ = vec.initVec3(F, 0.0, 0.0, 0.0);
    const dist: F = 100.0;
    const stereo_angle: F = std.math.pi / 6.0;
    const stereo = posStereoPair(targ, dist, stereo_angle, 0.0);
    const diff = (&stereo.cam0_pos).sub(stereo.cam1_pos);
    const baseline = diff.vecLen();
    const expected_baseline = 2.0 * dist * @sin(0.5 * stereo_angle);
    try std.testing.expectApproxEqRel(expected_baseline, baseline, 1e-5);
}

test "posFrameCoords framing modes" {
    var raw_coords = [_]F{
        -10.0, -5.0, 0.0,
        10.0,  -5.0, 0.0,
        10.0,  5.0,  0.0,
        -10.0, 5.0,  0.0,
    };
    const coords = meshio.Coords.init(&raw_coords, 4);
    const pixels_num = [2]u32{ 100, 100 };
    const pixels_size = [2]F{ 0.1, 0.1 };
    const focal_leng: F = 10.0;
    const cam_rot = rotation.Rotation.init(0.0, 0.0, 0.0);

    const pos_contain = posFrameCoords(
        &coords,
        pixels_num,
        pixels_size,
        focal_leng,
        cam_rot,
        1.0,
        .contain,
    );
    try std.testing.expectApproxEqRel(@as(F, 0.0), pos_contain.get(0), 1e-5);
    try std.testing.expectApproxEqRel(@as(F, 0.0), pos_contain.get(1), 1e-5);
    try std.testing.expectApproxEqRel(@as(F, 20.0), pos_contain.get(2), 1e-5);

    const pos_cover = posFrameCoords(
        &coords,
        pixels_num,
        pixels_size,
        focal_leng,
        cam_rot,
        1.0,
        .cover,
    );
    try std.testing.expectApproxEqRel(@as(F, 10.0), pos_cover.get(2), 1e-5);

    const pos_horiz = posFrameCoords(
        &coords,
        pixels_num,
        pixels_size,
        focal_leng,
        cam_rot,
        1.0,
        .horizontal,
    );
    try std.testing.expectApproxEqRel(@as(F, 20.0), pos_horiz.get(2), 1e-5);

    const pos_vert = posFrameCoords(
        &coords,
        pixels_num,
        pixels_size,
        focal_leng,
        cam_rot,
        1.0,
        .vertical,
    );
    try std.testing.expectApproxEqRel(@as(F, 10.0), pos_vert.get(2), 1e-5);
}

test "posFrameCoordsTarg framing with offset target" {
    var raw_coords = [_]F{
        0.0,  0.0, 0.0,
        20.0, 0.0, 0.0,
    };
    const coords = meshio.Coords.init(&raw_coords, 2);
    const targ = vec.initVec3(F, 10.0, 0.0, 0.0);
    const pixels_num = [2]u32{ 100, 100 };
    const pixels_size = [2]F{ 0.1, 0.1 };
    const focal_leng: F = 10.0;
    const cam_rot = rotation.Rotation.init(0.0, 0.0, 0.0);

    const pos = posFrameCoordsTarg(
        &coords,
        targ,
        pixels_num,
        pixels_size,
        focal_leng,
        cam_rot,
        1.0,
        .contain,
    );
    try std.testing.expectApproxEqRel(@as(F, 10.0), pos.get(0), 1e-5);
    try std.testing.expectApproxEqRel(@as(F, 0.0), pos.get(1), 1e-5);
    try std.testing.expectApproxEqRel(@as(F, 20.0), pos.get(2), 1e-5);
}
