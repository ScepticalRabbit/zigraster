// --------------------------------------------------------------------------
// Riley: A High Performance Rasteriser for DIC UQ
//
// Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
// Licensed under the MIT License (see LICENSE file for details)
//
// Authors: scepticalrabbit (Lloyd Fletcher)
// --------------------------------------------------------------------------
const std = @import("std");
const buildconfig = @import("../riley/zig/buildconfig.zig");
const camera = @import("../riley/zig/camera.zig");
const cameraops = @import("../riley/zig/cameraops.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const iio = @import("../riley/zig/imageio.zig");
const meshio = @import("../riley/zig/meshio.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const orch = @import("../dev_support/orchestration.zig");
const rastcfg = @import("../riley/zig/rasterconfig.zig");
const riley = @import("../riley/zig/riley.zig");
const Rotation = @import("../riley/zig/rotation.zig").Rotation;
const sceneops = @import("../riley/zig/sceneops.zig");
const shaderops = @import("../riley/zig/shaderops_common.zig");
const tcfg = @import("../dev_support/testconfig.zig");
const texops = @import("../riley/zig/textureops.zig");
const uvio = @import("../riley/zig/uvio.zig");
const vec = @import("../riley/zig/vecstack.zig");

const F = buildconfig.F;
const CameraInput = camera.CameraInput;
const MeshInput = mo.MeshInput;

pub const pixel_num_scene0 = [_]u32{ 160, 100 };
pub const pixel_size_scene0 = [2]F{ @floatCast(5.3e-6), @floatCast(5.3e-6) };
pub const focal_length_scene0: F = @floatCast(50.0e-3);

// 12mm Sphere on left in front (Z = 0.0), 10mm Cylinder on right behind (Z = -0.010).
// Horizontal overlap is 20% of 12mm sphere diameter (2.4mm).
// Sphere span in X: [-10.3, +1.7] mm; Cylinder span in X: [-0.7, +9.3] mm.
pub const sphere_center_scene0 = [3]F{ -0.0043, 0.0, 0.0 };
pub const cylinder_center_scene0 = [3]F{ 0.0043, 0.0, -0.010 };

pub const FullTextures = struct {
    tex_u8_mono: texops.Tex(u8, 1),
    tex_u8_rgb: texops.Tex(u8, 3),
    tex_u16_mono: texops.Tex(u16, 1),
    tex_u16_rgb: texops.Tex(u16, 3),
    tex_f64_mono: texops.Tex(F, 1),
    tex_f64_rgb: texops.Tex(F, 3),

    pub fn init(allocator: std.mem.Allocator, io: std.Io) !FullTextures {
        const tex_u8_mono = try iio.loadImage(
            u8,
            1,
            allocator,
            io,
            "texture/speck128_mono_u8.bmp",
            .bmp,
        );
        errdefer tex_u8_mono.deinit(allocator);

        const tex_u8_rgb = try iio.loadImage(
            u8,
            3,
            allocator,
            io,
            "texture/speck128_rgb_u8.bmp",
            .bmp,
        );
        errdefer tex_u8_rgb.deinit(allocator);

        const tex_u16_mono = try iio.loadImage(
            u16,
            1,
            allocator,
            io,
            "texture/speck128_mono_u16.tiff",
            .tiff,
        );
        errdefer tex_u16_mono.deinit(allocator);

        var tex_u16_rgb = try texops.Tex(u16, 3).init(
            allocator,
            tex_u8_rgb.rows_num,
            tex_u8_rgb.cols_num,
        );
        errdefer tex_u16_rgb.deinit(allocator);
        for (tex_u8_rgb.array.slice, 0..) |pixel_val, ii| {
            tex_u16_rgb.array.slice[ii] = @as(u16, pixel_val) * 257;
        }

        var tex_f64_mono = try texops.Tex(F, 1).init(
            allocator,
            tex_u16_mono.rows_num,
            tex_u16_mono.cols_num,
        );
        errdefer tex_f64_mono.deinit(allocator);
        for (tex_u16_mono.array.slice, 0..) |pixel_val, ii| {
            tex_f64_mono.array.slice[ii] = @as(F, @floatFromInt(pixel_val)) / 65535.0;
        }

        var tex_f64_rgb = try texops.Tex(F, 3).init(
            allocator,
            tex_u8_rgb.rows_num,
            tex_u8_rgb.cols_num,
        );
        errdefer tex_f64_rgb.deinit(allocator);
        for (tex_u8_rgb.array.slice, 0..) |pixel_val, ii| {
            tex_f64_rgb.array.slice[ii] = @as(F, @floatFromInt(pixel_val)) / 255.0;
        }

        return .{
            .tex_u8_mono = tex_u8_mono,
            .tex_u8_rgb = tex_u8_rgb,
            .tex_u16_mono = tex_u16_mono,
            .tex_u16_rgb = tex_u16_rgb,
            .tex_f64_mono = tex_f64_mono,
            .tex_f64_rgb = tex_f64_rgb,
        };
    }

    pub fn deinit(self: *FullTextures, allocator: std.mem.Allocator) void {
        self.tex_u8_mono.deinit(allocator);
        self.tex_u8_rgb.deinit(allocator);
        self.tex_u16_mono.deinit(allocator);
        self.tex_u16_rgb.deinit(allocator);
        self.tex_f64_mono.deinit(allocator);
        self.tex_f64_rgb.deinit(allocator);
    }
};

pub fn sliceFieldToTwoFrames(
    allocator: std.mem.Allocator,
    field: meshio.Field,
) !meshio.Field {
    const time_n = field.getTimeN();
    const node_n = field.getCoordN();
    const chan_n = field.getFieldsN();
    var result = try meshio.Field.initAlloc(allocator, 2, node_n, chan_n);
    const max_time_idx = if (time_n > 0) time_n - 1 else 0;

    for (0..node_n) |nn| {
        for (0..chan_n) |cc| {
            result.array.set(
                &.{ 0, nn, cc },
                field.array.get(&.{ 0, nn, cc }),
            );
            result.array.set(
                &.{ 1, nn, cc },
                field.array.get(&.{ max_time_idx, nn, cc }),
            );
        }
    }
    return result;
}

pub fn buildRgbFieldTwoFrames(
    allocator: std.mem.Allocator,
    temp_2f: meshio.Field,
    disp_2f: meshio.Field,
) !meshio.Field {
    const node_n = temp_2f.getCoordN();
    var field = try meshio.Field.initAlloc(allocator, 2, node_n, 3);
    for (0..2) |tt| {
        for (0..node_n) |nn| {
            field.array.set(&.{ tt, nn, 0 }, temp_2f.array.get(&.{ tt, nn, 0 }));
            field.array.set(&.{ tt, nn, 1 }, disp_2f.array.get(&.{ tt, nn, 0 }));
            field.array.set(&.{ tt, nn, 2 }, disp_2f.array.get(&.{ tt, nn, 1 }));
        }
    }
    return field;
}

pub fn loadShapeMeshScene0(
    allocator: std.mem.Allocator,
    io: std.Io,
    shape_dir_name: []const u8,
    mesh_type: gk.MeshType,
    shader_input: shaderops.ShaderInput,
) !MeshInput {
    const elem_str = switch (mesh_type) {
        .tri3opt => "tri3",
        else => @tagName(mesh_type),
    };
    const dir = try std.fmt.allocPrint(
        allocator,
        "data/shapes/{s}/{s}/",
        .{ shape_dir_name, elem_str },
    );
    const temp_files = &[_][]const u8{
        try std.fmt.allocPrint(allocator, "{s}temperature.csv", .{dir}),
    };
    const disp_files = &[_][]const u8{
        try std.fmt.allocPrint(allocator, "{s}disp_x.csv", .{dir}),
        try std.fmt.allocPrint(allocator, "{s}disp_y.csv", .{dir}),
        try std.fmt.allocPrint(allocator, "{s}disp_z.csv", .{dir}),
    };
    const sim = try meshio.loadSimData(
        allocator,
        io,
        try std.fmt.allocPrint(allocator, "{s}coords.csv", .{dir}),
        try std.fmt.allocPrint(allocator, "{s}connect.csv", .{dir}),
        temp_files,
        disp_files,
    );
    const raw_disp = sim.disp orelse return error.MissingDisplacement;
    const disp_2f = try sliceFieldToTwoFrames(allocator, raw_disp);

    return .{
        .mesh_type = mesh_type,
        .coords = sim.coords,
        .connect = sim.connect,
        .disp = disp_2f,
        .shader = shader_input,
    };
}

pub fn loadShapeUVs(
    allocator: std.mem.Allocator,
    io: std.Io,
    shape_dir_name: []const u8,
    mesh_type: gk.MeshType,
) !uvio.UVMap {
    const elem_str = switch (mesh_type) {
        .tri3opt => "tri3",
        else => @tagName(mesh_type),
    };
    const path = try std.fmt.allocPrint(
        allocator,
        "data/shapes/{s}/{s}/uvs.csv",
        .{ shape_dir_name, elem_str },
    );
    defer allocator.free(path);
    return uvio.loadUVMap(allocator, io, path);
}

pub fn loadShapeTempField(
    allocator: std.mem.Allocator,
    io: std.Io,
    shape_dir_name: []const u8,
    mesh_type: gk.MeshType,
) !meshio.Field {
    const elem_str = switch (mesh_type) {
        .tri3opt => "tri3",
        else => @tagName(mesh_type),
    };
    const path = try std.fmt.allocPrint(
        allocator,
        "data/shapes/{s}/{s}/temperature.csv",
        .{ shape_dir_name, elem_str },
    );
    defer allocator.free(path);
    var raw_temp = try meshio.loadFieldCsv(allocator, io, path);
    defer raw_temp.deinit(allocator);
    return sliceFieldToTwoFrames(allocator, raw_temp);
}

pub fn loadShapeDispField(
    allocator: std.mem.Allocator,
    io: std.Io,
    shape_dir_name: []const u8,
    mesh_type: gk.MeshType,
) !meshio.Field {
    const elem_str = switch (mesh_type) {
        .tri3opt => "tri3",
        else => @tagName(mesh_type),
    };
    const dir = try std.fmt.allocPrint(
        allocator,
        "data/shapes/{s}/{s}/",
        .{ shape_dir_name, elem_str },
    );
    defer allocator.free(dir);
    const dispx = try std.fmt.allocPrint(allocator, "{s}disp_x.csv", .{dir});
    defer allocator.free(dispx);
    const dispy = try std.fmt.allocPrint(allocator, "{s}disp_y.csv", .{dir});
    defer allocator.free(dispy);
    const dispz = try std.fmt.allocPrint(allocator, "{s}disp_z.csv", .{dir});
    defer allocator.free(dispz);
    const disp_files = &[_][]const u8{
        dispx,
        dispy,
        dispz,
    };
    var raw_disp = try meshio.loadDispCsv(allocator, io, disp_files);
    defer raw_disp.deinit(allocator);
    return sliceFieldToTwoFrames(allocator, raw_disp);
}

pub fn buildScene0Meshes(
    allocator: std.mem.Allocator,
    sphere_mesh: MeshInput,
    cylinder_mesh: MeshInput,
) ![]MeshInput {
    var meshes = try allocator.alloc(MeshInput, 2);
    meshes[0] = sphere_mesh;
    meshes[1] = cylinder_mesh;

    sceneops.centerMeshGroupAt(
        meshes,
        sceneops.meshGroupSingle(0),
        sphere_center_scene0,
    );
    sceneops.centerMeshGroupAt(
        meshes,
        sceneops.meshGroupSingle(1),
        cylinder_center_scene0,
    );

    return meshes;
}

pub fn createScene0Camera(meshes: []const MeshInput) CameraInput {
    const target = sceneops.boundsCenterOverMeshes(meshes);
    const rot = Rotation.init(
        0,
        std.math.degreesToRadians(10.0),
        std.math.degreesToRadians(-10.0),
    );
    const pos = cameraops.posFillFrameFromRotOverMeshesAndTarg(
        meshes,
        target,
        pixel_num_scene0,
        pixel_size_scene0,
        focal_length_scene0,
        rot,
        0.98,
    );

    return .{
        .pixels_num = pixel_num_scene0,
        .pixels_size = pixel_size_scene0,
        .pos_world = pos,
        .rot_world = rot,
        .roi_cent_world = target,
        .focal_length = focal_length_scene0,
        .sub_sample = 2,
        .distortion = .none,
    };
}

pub const Scene0Prepared = struct {
    mesh_type: gk.MeshType,
    sphere_coords: meshio.Coords,
    sphere_connect: meshio.Connect,
    sphere_disp: meshio.Field,
    sphere_uvs: uvio.UVMap,
    sphere_temp: meshio.Field,
    sphere_rgb: meshio.Field,

    cylinder_coords: meshio.Coords,
    cylinder_connect: meshio.Connect,
    cylinder_disp: meshio.Field,
    cylinder_uvs: uvio.UVMap,
    cylinder_temp: meshio.Field,
    cylinder_rgb: meshio.Field,

    camera_input: CameraInput,

    pub fn deinit(self: *Scene0Prepared, allocator: std.mem.Allocator) void {
        allocator.free(self.sphere_coords.mem);
        self.sphere_connect.deinit(allocator);
        self.sphere_disp.deinit(allocator);
        self.sphere_uvs.deinit(allocator);
        self.sphere_temp.deinit(allocator);
        self.sphere_rgb.deinit(allocator);

        allocator.free(self.cylinder_coords.mem);
        self.cylinder_connect.deinit(allocator);
        self.cylinder_disp.deinit(allocator);
        self.cylinder_uvs.deinit(allocator);
        self.cylinder_temp.deinit(allocator);
        self.cylinder_rgb.deinit(allocator);
    }
};

pub fn prepareScene0(
    allocator: std.mem.Allocator,
    io: std.Io,
    mesh_type: gk.MeshType,
) !Scene0Prepared {
    const elem_str = switch (mesh_type) {
        .tri3opt => "tri3",
        else => @tagName(mesh_type),
    };

    const sphere_dir = try std.fmt.allocPrint(
        allocator,
        "data/shapes/sphere_surf/{s}/",
        .{elem_str},
    );
    defer allocator.free(sphere_dir);

    const sphere_temp_path = try std.fmt.allocPrint(
        allocator,
        "{s}temperature.csv",
        .{sphere_dir},
    );
    defer allocator.free(sphere_temp_path);
    const sphere_temp_files = &[_][]const u8{sphere_temp_path};

    const sphere_dispx = try std.fmt.allocPrint(
        allocator,
        "{s}disp_x.csv",
        .{sphere_dir},
    );
    defer allocator.free(sphere_dispx);
    const sphere_dispy = try std.fmt.allocPrint(
        allocator,
        "{s}disp_y.csv",
        .{sphere_dir},
    );
    defer allocator.free(sphere_dispy);
    const sphere_dispz = try std.fmt.allocPrint(
        allocator,
        "{s}disp_z.csv",
        .{sphere_dir},
    );
    defer allocator.free(sphere_dispz);
    const sphere_disp_files = &[_][]const u8{
        sphere_dispx,
        sphere_dispy,
        sphere_dispz,
    };

    const sphere_coords_path = try std.fmt.allocPrint(
        allocator,
        "{s}coords.csv",
        .{sphere_dir},
    );
    defer allocator.free(sphere_coords_path);
    const sphere_connect_path = try std.fmt.allocPrint(
        allocator,
        "{s}connect.csv",
        .{sphere_dir},
    );
    defer allocator.free(sphere_connect_path);

    const sphere_sim = try meshio.loadSimData(
        allocator,
        io,
        sphere_coords_path,
        sphere_connect_path,
        sphere_temp_files,
        sphere_disp_files,
    );
    var raw_sphere_disp = sphere_sim.disp orelse return error.MissingDisplacement;
    var raw_sphere_temp = sphere_sim.field orelse return error.MissingField;

    var sphere_coords = sphere_sim.coords;
    const sphere_connect = sphere_sim.connect;
    const sphere_disp = try sliceFieldToTwoFrames(allocator, raw_sphere_disp);
    raw_sphere_disp.deinit(allocator);
    const sphere_temp = try sliceFieldToTwoFrames(allocator, raw_sphere_temp);
    raw_sphere_temp.deinit(allocator);

    const sphere_uvs = try loadShapeUVs(allocator, io, "sphere_surf", mesh_type);
    const sphere_rgb = try buildRgbFieldTwoFrames(
        allocator,
        sphere_temp,
        sphere_disp,
    );

    const cyl_dir = try std.fmt.allocPrint(
        allocator,
        "data/shapes/cylinder_surf/{s}/",
        .{elem_str},
    );
    defer allocator.free(cyl_dir);

    const cyl_temp_path = try std.fmt.allocPrint(
        allocator,
        "{s}temperature.csv",
        .{cyl_dir},
    );
    defer allocator.free(cyl_temp_path);
    const cyl_temp_files = &[_][]const u8{cyl_temp_path};

    const cyl_dispx = try std.fmt.allocPrint(
        allocator,
        "{s}disp_x.csv",
        .{cyl_dir},
    );
    defer allocator.free(cyl_dispx);
    const cyl_dispy = try std.fmt.allocPrint(
        allocator,
        "{s}disp_y.csv",
        .{cyl_dir},
    );
    defer allocator.free(cyl_dispy);
    const cyl_dispz = try std.fmt.allocPrint(
        allocator,
        "{s}disp_z.csv",
        .{cyl_dir},
    );
    defer allocator.free(cyl_dispz);
    const cyl_disp_files = &[_][]const u8{
        cyl_dispx,
        cyl_dispy,
        cyl_dispz,
    };

    const cyl_coords_path = try std.fmt.allocPrint(
        allocator,
        "{s}coords.csv",
        .{cyl_dir},
    );
    defer allocator.free(cyl_coords_path);
    const cyl_connect_path = try std.fmt.allocPrint(
        allocator,
        "{s}connect.csv",
        .{cyl_dir},
    );
    defer allocator.free(cyl_connect_path);

    const cyl_sim = try meshio.loadSimData(
        allocator,
        io,
        cyl_coords_path,
        cyl_connect_path,
        cyl_temp_files,
        cyl_disp_files,
    );
    var raw_cyl_disp = cyl_sim.disp orelse return error.MissingDisplacement;
    var raw_cyl_temp = cyl_sim.field orelse return error.MissingField;

    var cylinder_coords = cyl_sim.coords;
    const cylinder_connect = cyl_sim.connect;
    const cylinder_disp = try sliceFieldToTwoFrames(allocator, raw_cyl_disp);
    raw_cyl_disp.deinit(allocator);
    const cylinder_temp = try sliceFieldToTwoFrames(allocator, raw_cyl_temp);
    raw_cyl_temp.deinit(allocator);

    const cylinder_uvs = try loadShapeUVs(allocator, io, "cylinder_surf", mesh_type);
    const cylinder_rgb = try buildRgbFieldTwoFrames(
        allocator,
        cylinder_temp,
        cylinder_disp,
    );

    sceneops.centerCoordsAt(&sphere_coords, sphere_center_scene0);
    sceneops.centerCoordsAt(&cylinder_coords, cylinder_center_scene0);

    const dummy_sphere = MeshInput{
        .mesh_type = mesh_type,
        .coords = sphere_coords,
        .connect = sphere_connect,
        .disp = sphere_disp,
        .shader = .{ .nodal = .{ .field = sphere_temp } },
    };
    const dummy_cylinder = MeshInput{
        .mesh_type = mesh_type,
        .coords = cylinder_coords,
        .connect = cylinder_connect,
        .disp = cylinder_disp,
        .shader = .{ .nodal = .{ .field = cylinder_temp } },
    };
    const dummy_meshes = [_]MeshInput{ dummy_sphere, dummy_cylinder };
    const camera_input = createScene0Camera(&dummy_meshes);

    return .{
        .mesh_type = mesh_type,
        .sphere_coords = sphere_coords,
        .sphere_connect = sphere_connect,
        .sphere_disp = sphere_disp,
        .sphere_uvs = sphere_uvs,
        .sphere_temp = sphere_temp,
        .sphere_rgb = sphere_rgb,
        .cylinder_coords = cylinder_coords,
        .cylinder_connect = cylinder_connect,
        .cylinder_disp = cylinder_disp,
        .cylinder_uvs = cylinder_uvs,
        .cylinder_temp = cylinder_temp,
        .cylinder_rgb = cylinder_rgb,
        .camera_input = camera_input,
    };
}

// --------------------------------------------------------------------------
// Scene 1: Single Cube Surface (tri3, sharp checker, grey background)
// --------------------------------------------------------------------------

pub const pixel_num_scene1 = [_]u32{ 128, 128 };
pub const pixel_size_scene1 = [2]F{ @floatCast(5.3e-6), @floatCast(5.3e-6) };
pub const focal_length_scene1: F = @floatCast(50.0e-3);
pub const grey_background_scene1: F = 0.5;

pub const Scene1Prepared = struct {
    coords: meshio.Coords,
    connect: meshio.Connect,
    uvs: uvio.UVMap,
    camera_input: CameraInput,

    pub fn deinit(self: *Scene1Prepared, allocator: std.mem.Allocator) void {
        allocator.free(self.coords.mem);
        self.connect.deinit(allocator);
        self.uvs.deinit(allocator);
    }
};

pub fn getRepresentativePolynomialMap() camera.PolynomialMap {
    var poly_map = camera.PolynomialMap{
        .order = .quadratic,
    };
    poly_map.coeffs_u[1] = 0.03;
    poly_map.coeffs_u[4] = 5.0;
    poly_map.coeffs_v[2] = -0.03;
    poly_map.coeffs_v[3] = 5.0;
    return poly_map;
}

pub fn createScene1Camera(meshes: []const MeshInput) CameraInput {
    const target = sceneops.boundsCenterOverMeshes(meshes);
    const rot = Rotation.init(
        0,
        std.math.degreesToRadians(5.0),
        std.math.degreesToRadians(-5.0),
    );
    const pos = cameraops.posFillFrameFromRotOverMeshesAndTarg(
        meshes,
        target,
        pixel_num_scene1,
        pixel_size_scene1,
        focal_length_scene1,
        rot,
        0.95,
    );

    return .{
        .pixels_num = pixel_num_scene1,
        .pixels_size = pixel_size_scene1,
        .pos_world = pos,
        .rot_world = rot,
        .roi_cent_world = target,
        .focal_length = focal_length_scene1,
        .sub_sample = 1,
        .distortion = .none,
    };
}

pub fn prepareScene1(
    allocator: std.mem.Allocator,
    io: std.Io,
) !Scene1Prepared {
    const dir = "data/shapes/cube_surf/tri3/";
    const coords_path = try std.fmt.allocPrint(allocator, "{s}coords.csv", .{dir});
    defer allocator.free(coords_path);
    const connect_path = try std.fmt.allocPrint(allocator, "{s}connect.csv", .{dir});
    defer allocator.free(connect_path);
    const uvs_path = try std.fmt.allocPrint(allocator, "{s}uvs.csv", .{dir});
    defer allocator.free(uvs_path);

    const sim = try meshio.loadSimData(
        allocator,
        io,
        coords_path,
        connect_path,
        null,
        null,
    );
    var coords = sim.coords;
    errdefer allocator.free(coords.mem);
    var connect = sim.connect;
    errdefer connect.deinit(allocator);
    var uvs = try uvio.loadUVMap(allocator, io, uvs_path);
    errdefer uvs.deinit(allocator);

    sceneops.centerCoordsAt(&coords, .{ 0.0, 0.0, 0.0 });

    const dummy_mesh = MeshInput{
        .mesh_type = .tri3,
        .coords = coords,
        .connect = connect,
        .disp = null,
        .shader = .{
            .func = .{
                .uvs = uvs.array,
                .builtin = .checker,
                .params = .{
                    .coord_scale = .{ 24.0, 24.0 },
                    .coord_offset = .{ 0.0, 0.0 },
                },
                .coord_mode = .uv,
            },
        },
    };
    const dummy_meshes = [_]MeshInput{dummy_mesh};
    const camera_input = createScene1Camera(&dummy_meshes);

    return .{
        .coords = coords,
        .connect = connect,
        .uvs = uvs,
        .camera_input = camera_input,
    };
}

// --------------------------------------------------------------------------
// Scene 2: Two Spheres (Front Left Cropped, Back Right Occluded)
// --------------------------------------------------------------------------

pub const sphere1_center_scene2 = [3]F{ -0.0025, 0.0, 0.0 };
pub const sphere2_center_scene2 = [3]F{ 0.0038, 0.0, -1.02 };

pub const Scene2Prepared = struct {
    mesh_type: gk.MeshType,
    sphere1_coords: meshio.Coords,
    sphere1_connect: meshio.Connect,
    sphere1_uvs: uvio.UVMap,

    sphere2_coords: meshio.Coords,
    sphere2_connect: meshio.Connect,
    sphere2_uvs: uvio.UVMap,

    pub fn deinit(self: *Scene2Prepared, allocator: std.mem.Allocator) void {
        allocator.free(self.sphere1_coords.mem);
        self.sphere1_connect.deinit(allocator);
        self.sphere1_uvs.deinit(allocator);

        allocator.free(self.sphere2_coords.mem);
        self.sphere2_connect.deinit(allocator);
        self.sphere2_uvs.deinit(allocator);
    }
};

pub fn createScene2Camera(
    pixel_num: [2]u32,
    sub_sample: u8,
) CameraInput {
    const fov_h: F = 0.01194;
    const sensor_h = @as(F, @floatFromInt(pixel_num[1])) * pixel_size_scene0[1];
    const z_cam = focal_length_scene0 * fov_h / sensor_h;

    return .{
        .pixels_num = pixel_num,
        .pixels_size = pixel_size_scene0,
        .pos_world = vec.initVec3(F, 0.0, 0.0, z_cam),
        .rot_world = Rotation.init(0, 0, 0),
        .roi_cent_world = vec.initVec3(F, 0.0, 0.0, 0.0),
        .focal_length = focal_length_scene0,
        .sub_sample = sub_sample,
        .distortion = .none,
    };
}

pub fn buildScene2Meshes(
    prep: *const Scene2Prepared,
    textures: *const FullTextures,
) [2]MeshInput {
    const sphere1_shader = shaderops.ShaderInput{
        .func = .{
            .uvs = prep.sphere1_uvs.array,
            .builtin = .checker,
            .params = .{
                .coord_scale = .{ 24.0, 24.0 },
                .coord_offset = .{ 0.0, 0.0 },
            },
            .coord_mode = .uv,
            .normal_type = .none,
        },
    };

    const sphere2_shader = shaderops.ShaderInput{
        .tex_u8 = .{
            .uvs = prep.sphere2_uvs.array,
            .tex = textures.tex_u8_mono,
            .samp_cfg = .{
                .sample = .cubic_catmull_rom,
                .mode = .direct,
            },
            .normal_type = .none,
        },
    };

    return [_]MeshInput{
        .{
            .mesh_type = prep.mesh_type,
            .coords = prep.sphere1_coords,
            .connect = prep.sphere1_connect,
            .disp = null,
            .shader = sphere1_shader,
        },
        .{
            .mesh_type = prep.mesh_type,
            .coords = prep.sphere2_coords,
            .connect = prep.sphere2_connect,
            .disp = null,
            .shader = sphere2_shader,
        },
    };
}

pub fn prepareScene2(
    allocator: std.mem.Allocator,
    io: std.Io,
    mesh_type: gk.MeshType,
) !Scene2Prepared {
    const elem_str = switch (mesh_type) {
        .tri3opt => "tri3",
        else => @tagName(mesh_type),
    };

    const sphere_dir = try std.fmt.allocPrint(
        allocator,
        "data/shapes/sphere_surf/{s}/",
        .{elem_str},
    );
    defer allocator.free(sphere_dir);

    const coords1_path = try std.fmt.allocPrint(
        allocator,
        "{s}coords.csv",
        .{sphere_dir},
    );
    defer allocator.free(coords1_path);
    const connect1_path = try std.fmt.allocPrint(
        allocator,
        "{s}connect.csv",
        .{sphere_dir},
    );
    defer allocator.free(connect1_path);
    const uvs1_path = try std.fmt.allocPrint(
        allocator,
        "{s}uvs.csv",
        .{sphere_dir},
    );
    defer allocator.free(uvs1_path);

    const sim1 = try meshio.loadSimData(
        allocator,
        io,
        coords1_path,
        connect1_path,
        null,
        null,
    );
    var sphere1_coords = sim1.coords;
    errdefer allocator.free(sphere1_coords.mem);
    var sphere1_connect = sim1.connect;
    errdefer sphere1_connect.deinit(allocator);
    var sphere1_uvs = try uvio.loadUVMap(allocator, io, uvs1_path);
    errdefer sphere1_uvs.deinit(allocator);

    const sim2 = try meshio.loadSimData(
        allocator,
        io,
        coords1_path,
        connect1_path,
        null,
        null,
    );
    var sphere2_coords = sim2.coords;
    errdefer allocator.free(sphere2_coords.mem);
    var sphere2_connect = sim2.connect;
    errdefer sphere2_connect.deinit(allocator);
    var sphere2_uvs = try uvio.loadUVMap(allocator, io, uvs1_path);
    errdefer sphere2_uvs.deinit(allocator);

    sceneops.centerCoordsAt(&sphere1_coords, sphere1_center_scene2);
    sceneops.centerCoordsAt(&sphere2_coords, sphere2_center_scene2);

    return .{
        .mesh_type = mesh_type,
        .sphere1_coords = sphere1_coords,
        .sphere1_connect = sphere1_connect,
        .sphere1_uvs = sphere1_uvs,
        .sphere2_coords = sphere2_coords,
        .sphere2_connect = sphere2_connect,
        .sphere2_uvs = sphere2_uvs,
    };
}


