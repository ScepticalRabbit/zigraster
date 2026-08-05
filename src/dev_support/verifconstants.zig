// --------------------------------------------------------------------------
const buildconfig = @import("../riley/zig/buildconfig.zig");
const F = buildconfig.F;
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const cam = @import("../riley/zig/camera.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const rotation = @import("../riley/zig/rotation.zig");

pub const output_dir_name = "verif";

pub const CameraDistortionCase = struct {
    case_name: []const u8,
    distortion: cam.DistortionModel,
};

pub const DistortCase = struct {
    case_name: []const u8,
    mesh_type: gk.MeshType,
    data_dir: []const u8,
    camera_input: cam.CameraInput,
};

fn edgeCameraInput(
    pos_world: [3]F,
    roi_cent_world: [3]F,
) cam.CameraInput {
    return .{
        .pixels_num = .{ 800, 500 },
        .pixels_size = .{ 5.3e-6, 5.3e-6 },
        .pos_world = .{ .slice = pos_world },
        .rot_world = rotation.Rotation.init(0.0, 0.0, 0.0),
        .roi_cent_world = .{ .slice = roi_cent_world },
        .focal_length = 5.0e-2,
        .sub_sample = 1,
        .distortion = .none,
    };
}

fn brownConradyDistortion(
    k1: F,
    k2: F,
    k3: F,
    p1: F,
    p2: F,
) cam.DistortionModel {
    return .{
        .brown_conrady = .{
            .k1 = k1,
            .k2 = k2,
            .k3 = k3,
            .p1 = p1,
            .p2 = p2,
        },
    };
}

pub const camera_distortion_cases = [_]CameraDistortionCase{
    .{
        .case_name = "none",
        .distortion = .none,
    },
    .{
        .case_name = "mild_barrel",
        .distortion = brownConradyDistortion(
            -5.0e-2,
            1.0e-2,
            0.0,
            0.0,
            0.0,
        ),
    },
    .{
        .case_name = "mild_pincushion",
        .distortion = brownConradyDistortion(
            5.0e-2,
            -1.0e-2,
            0.0,
            0.0,
            0.0,
        ),
    },
    .{
        .case_name = "strong_barrel",
        .distortion = brownConradyDistortion(
            -1.5e-1,
            3.0e-2,
            0.0,
            0.0,
            0.0,
        ),
    },
    .{
        .case_name = "mixed_asymmetric",
        .distortion = brownConradyDistortion(
            -8.0e-2,
            1.5e-2,
            0.0,
            1.0e-3,
            -1.0e-3,
        ),
    },
};

pub fn cameraInputWithDistortion(
    camera_input: cam.CameraInput,
    distortion_case: CameraDistortionCase,
) cam.CameraInput {
    var distorted_camera_input = camera_input;
    distorted_camera_input.distortion = distortion_case.distortion;
    return distorted_camera_input;
}

pub const distort_cases = [_]DistortCase{
    .{
        .case_name = "bulge",
        .mesh_type = .tri6,
        .data_dir = "data/edge/tri6_distort_bulge",
        .camera_input = edgeCameraInput(
            .{ 5.0, 2.8301270189221928, 242.00527248356275 },
            .{ 5.0, 2.8301270189221928, 0.0 },
        ),
    },
    .{
        .case_name = "bulge",
        .mesh_type = .quad8,
        .data_dir = "data/edge/quad8_distort_bulge",
        .camera_input = edgeCameraInput(
            .{ 5.0, 5.0, 332.07547169811323 },
            .{ 5.0, 5.0, 0.0 },
        ),
    },
    .{
        .case_name = "bulge",
        .mesh_type = .quad9,
        .data_dir = "data/edge/quad9_distort_bulge",
        .camera_input = edgeCameraInput(
            .{ 5.0, 5.0, 332.07547169811323 },
            .{ 5.0, 5.0, 0.0 },
        ),
    },
    .{
        .case_name = "tan",
        .mesh_type = .tri6,
        .data_dir = "data/edge/tri6_distort_tan",
        .camera_input = edgeCameraInput(
            .{ 5.0, 4.3301270189221930, 179.74112154016652 },
            .{ 5.0, 4.3301270189221930, 0.0 },
        ),
    },
    .{
        .case_name = "tan",
        .mesh_type = .quad8,
        .data_dir = "data/edge/quad8_distort_tan",
        .camera_input = edgeCameraInput(
            .{ 5.0, 5.0, 207.54716981132077 },
            .{ 5.0, 5.0, 0.0 },
        ),
    },
    .{
        .case_name = "tan",
        .mesh_type = .quad9,
        .data_dir = "data/edge/quad9_distort_tan",
        .camera_input = edgeCameraInput(
            .{ 5.0, 5.0, 207.54716981132077 },
            .{ 5.0, 5.0, 0.0 },
        ),
    },
    .{
        .case_name = "stretch",
        .mesh_type = .tri3,
        .data_dir = "data/edge/tri3_distort_stretch",
        .camera_input = edgeCameraInput(
            .{ 59.33012701892219, 0.0, 1539.2249934154347 },
            .{ 59.33012701892219, 0.0, 0.0 },
        ),
    },
    .{
        .case_name = "stretch",
        .mesh_type = .tri6,
        .data_dir = "data/edge/tri6_distort_stretch",
        .camera_input = edgeCameraInput(
            .{ 59.33012701892219, 0.0, 1539.2249934154347 },
            .{ 59.33012701892219, 0.0, 0.0 },
        ),
    },
    .{
        .case_name = "stretch",
        .mesh_type = .quad4newton,
        .data_dir = "data/edge/quad4_distort_stretch",
        .camera_input = edgeCameraInput(
            .{ 60.0, 5.0, 1556.6037735849059 },
            .{ 60.0, 5.0, 0.0 },
        ),
    },
    .{
        .case_name = "stretch",
        .mesh_type = .quad8,
        .data_dir = "data/edge/quad8_distort_stretch",
        .camera_input = edgeCameraInput(
            .{ 60.0, 5.0, 1556.6037735849059 },
            .{ 60.0, 5.0, 0.0 },
        ),
    },
    .{
        .case_name = "stretch",
        .mesh_type = .quad9,
        .data_dir = "data/edge/quad9_distort_stretch",
        .camera_input = edgeCameraInput(
            .{ 60.0, 5.0, 1556.6037735849059 },
            .{ 60.0, 5.0, 0.0 },
        ),
    },
    .{
        .case_name = "shear",
        .mesh_type = .tri3,
        .data_dir = "data/edge/tri3_distort_shear",
        .camera_input = edgeCameraInput(
            .{ 57.5, 4.330127018922193, 1491.7452830188683 },
            .{ 57.5, 4.330127018922193, 0.0 },
        ),
    },
    .{
        .case_name = "shear",
        .mesh_type = .tri6,
        .data_dir = "data/edge/tri6_distort_shear",
        .camera_input = edgeCameraInput(
            .{ 57.5, 4.330127018922193, 1491.7452830188683 },
            .{ 57.5, 4.330127018922193, 0.0 },
        ),
    },
    .{
        .case_name = "shear",
        .mesh_type = .quad4newton,
        .data_dir = "data/edge/quad4_distort_shear",
        .camera_input = edgeCameraInput(
            .{ 60.0, 5.0, 1556.6037735849059 },
            .{ 60.0, 5.0, 0.0 },
        ),
    },
    .{
        .case_name = "shear",
        .mesh_type = .quad8,
        .data_dir = "data/edge/quad8_distort_shear",
        .camera_input = edgeCameraInput(
            .{ 60.0, 5.0, 1556.6037735849059 },
            .{ 60.0, 5.0, 0.0 },
        ),
    },
    .{
        .case_name = "shear",
        .mesh_type = .quad9,
        .data_dir = "data/edge/quad9_distort_shear",
        .camera_input = edgeCameraInput(
            .{ 60.0, 5.0, 1556.6037735849059 },
            .{ 60.0, 5.0, 0.0 },
        ),
    },
    .{
        .case_name = "rot",
        .mesh_type = .tri3,
        .data_dir = "data/edge/tri3_distort_rot",
        .camera_input = edgeCameraInput(
            .{ 5.0, 2.886751345948129, 144.92861657761585 },
            .{ 5.0, 2.886751345948129, 0.0 },
        ),
    },
    .{
        .case_name = "rot",
        .mesh_type = .tri6,
        .data_dir = "data/edge/tri6_distort_rot",
        .camera_input = edgeCameraInput(
            .{ 5.0, 2.886751345948129, 144.92861657761585 },
            .{ 5.0, 2.886751345948129, 0.0 },
        ),
    },
    .{
        .case_name = "rot",
        .mesh_type = .quad4newton,
        .data_dir = "data/edge/quad4_distort_rot",
        .camera_input = edgeCameraInput(
            .{ 5.0, 5.0, 167.27358490566039 },
            .{ 5.0, 5.0, 0.0 },
        ),
    },
    .{
        .case_name = "rot",
        .mesh_type = .quad8,
        .data_dir = "data/edge/quad8_distort_rot",
        .camera_input = edgeCameraInput(
            .{ 5.0, 5.0, 167.27358490566039 },
            .{ 5.0, 5.0, 0.0 },
        ),
    },
    .{
        .case_name = "rot",
        .mesh_type = .quad9,
        .data_dir = "data/edge/quad9_distort_rot",
        .camera_input = edgeCameraInput(
            .{ 5.0, 5.0, 167.27358490566039 },
            .{ 5.0, 5.0, 0.0 },
        ),
    },
};
