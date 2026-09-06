const std = @import("std");

const buildconfig = @import("riley/zig/buildconfig.zig");
const camera = @import("riley/zig/camera.zig");
const cameraops = @import("riley/zig/cameraops.zig");
const gk = @import("riley/zig/geometrykernels.zig");
const iio = @import("riley/zig/imageio.zig");
const meshio = @import("riley/zig/meshio.zig");
const mo = @import("riley/zig/meshpipeline.zig");
const riley = @import("riley/zig/riley.zig");
const Rotation = @import("riley/zig/rotation.zig").Rotation;
const sceneops = @import("riley/zig/sceneops.zig");
const shaderops = @import("riley/zig/shaderops_common.zig");
const texops = @import("riley/zig/textureops.zig");
const uvio = @import("riley/zig/uvio.zig");

const F = buildconfig.F;
const MeshInput = mo.MeshInput;
const out_dir_root = "./out/demo9_feature_zoo";
const pixel_size = [2]F{ @floatCast(5.3e-6), @floatCast(5.3e-6) };
const focal_length: F = @floatCast(50.0e-3);

const Case = struct {
    name: []const u8,
    mesh_type: gk.MeshType,
};

const cases = [_]Case{
    .{ .name = "cube_quad9", .mesh_type = .quad9 },
    .{ .name = "cube_tri6", .mesh_type = .tri6 },
    .{ .name = "cylinder_quad8", .mesh_type = .quad8 },
    .{ .name = "cylinder_tri6", .mesh_type = .tri6 },
    .{ .name = "plate_quad4ibi", .mesh_type = .quad4ibi },
    .{ .name = "plate_quad4newton", .mesh_type = .quad4newton },
    .{ .name = "plate_tri3", .mesh_type = .tri3 },
};

const mesh_centers = [cases.len][3]F{
    .{ 0.016, -0.0065, 0.0 },
    .{ 0.027, -0.0065, 0.0 },
    .{ 0.016, -0.0195, 0.0 },
    .{ 0.027, -0.0195, 0.0 },
    .{ -0.013, 0.018, 0.0 },
    .{ 0.013, 0.018, 0.0 },
    .{ -0.008, -0.013, 0.0 },
};

fn buildRgbField(
    allocator: std.mem.Allocator,
    temp: meshio.Field,
    disp: meshio.Field,
) !meshio.Field {
    const time_n = temp.getTimeN();
    const node_n = temp.getCoordN();
    var field = try meshio.Field.initAlloc(allocator, time_n, node_n, 3);
    for (0..time_n) |tt| {
        for (0..node_n) |nn| {
            field.array.set(&.{ tt, nn, 0 }, temp.array.get(&.{ tt, nn, 0 }));
            field.array.set(&.{ tt, nn, 1 }, disp.array.get(&.{ tt, nn, 0 }));
            field.array.set(&.{ tt, nn, 2 }, disp.array.get(&.{ tt, nn, 1 }));
        }
    }
    return field;
}

fn textureShader(
    comptime T: type,
    comptime C: usize,
    texture: texops.Tex(T, C),
    uvs: uvio.UVMap,
    comptime bits: u8,
    linear: bool,
    normal_type: shaderops.NormalType,
) shaderops.ShaderInput {
    const samp_cfg = texops.TexSampConfig{
        .sample = if (linear) .linear else .cubic_catmull_rom,
        .mode = if (linear) .direct else .lut_lerp,
    };
    const input = shaderops.TexInput(T, C){
        .uvs = uvs.array,
        .tex = texture,
        .samp_cfg = samp_cfg,
        .bits = bits,
        .scaling = .auto,
        .normal_type = normal_type,
    };
    if (C == 1 and T == u8) return .{ .tex_u8 = input };
    if (C == 1 and T == u16) return .{ .tex_u16 = input };
    if (C == 3 and T == u8) return .{ .tex_rgb_u8 = input };
    if (C == 3 and T == u16) return .{ .tex_rgb_u16 = input };
    unreachable;
}

fn loadMesh(
    comptime T: type,
    comptime C: usize,
    comptime bits: u8,
    allocator: std.mem.Allocator,
    io: std.Io,
    case: Case,
    case_index: usize,
    texture: texops.Tex(T, C),
) !MeshInput {
    const dir = try std.fmt.allocPrint(
        allocator,
        "data/feature_zoo/{s}/",
        .{case.name},
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
    const uvs = try uvio.loadUVMap(
        allocator,
        io,
        try std.fmt.allocPrint(allocator, "{s}uvs.csv", .{dir}),
    );
    const temp = sim.field orelse return error.MissingTemperature;
    const disp = sim.disp orelse return error.MissingDisplacement;
    const normal_type: shaderops.NormalType = switch (case_index % 3) {
        0 => .none,
        1 => .exact,
        else => .avg,
    };

    const shader: shaderops.ShaderInput = switch (case_index) {
        0, 2, 6 => textureShader(
            T,
            C,
            texture,
            uvs,
            bits,
            case_index != 0,
            normal_type,
        ),
        1, 5 => blk: {
            const field = if (C == 1)
                temp
            else
                try buildRgbField(allocator, temp, disp);
            break :blk .{ .nodal = .{
                .field = field,
                .bits = bits,
                .scaling = .auto,
                .scale_over = if (case_index == 1)
                    .over_frames
                else
                    .within_frames,
                .normal_type = normal_type,
            } };
        },
        3, 4 => blk: {
            const params = if (case_index == 3)
                shaderops.FuncShaderParams{
                    .coord_scale = .{ 1000.0, 1000.0 },
                    .settings = .{ .checker = .{} },
                }
            else
                shaderops.FuncShaderParams{
                    .coord_scale = .{ 1.0, 1.0 },
                    .settings = .{ .eggbox = .{
                        .pitch = .{ 0.005, 0.005 },
                    } },
                };
            const input = shaderops.FuncInput{
                .coord_mode = if (case_index == 3)
                    .world_reference
                else
                    .world_deformed,
                .builtin = if (case_index == 3) .checker else .eggbox,
                .params = params,
                .bits = bits,
                .scaling = .auto,
                .normal_type = normal_type,
            };
            break :blk if (C == 1)
                .{ .func = input }
            else
                .{ .func_rgb = input };
        },
        else => unreachable,
    };
    return .{
        .mesh_type = case.mesh_type,
        .coords = sim.coords,
        .connect = sim.connect,
        .disp = if (case_index % 2 == 1) disp else null,
        .shader = shader,
    };
}

fn buildScene(
    comptime T: type,
    comptime C: usize,
    comptime bits: u8,
    allocator: std.mem.Allocator,
    io: std.Io,
    texture: texops.Tex(T, C),
) ![]MeshInput {
    var meshes = std.ArrayList(MeshInput).empty;
    var groups = std.ArrayList(sceneops.MeshGroup).empty;
    for (cases, 0..) |case, index| {
        try meshes.append(
            allocator,
            try loadMesh(T, C, bits, allocator, io, case, index, texture),
        );
        try groups.append(allocator, sceneops.meshGroupSingle(index));
    }
    const plate = &meshes.items[6];
    for (0..plate.coords.mat.rows_num) |node| {
        const x = plate.coords.mat.get(node, 0);
        const y = plate.coords.mat.get(node, 1);
        plate.coords.mat.set(node, 0, -y);
        plate.coords.mat.set(node, 1, x);
    }
    for (groups.items, mesh_centers) |group, center| {
        sceneops.centerMeshGroupAt(meshes.items, group, center);
    }
    return try meshes.toOwnedSlice(allocator);
}

fn makeCamera(
    meshes: []MeshInput,
    pixels_num: [2]u32,
    rot: Rotation,
    sub_sample: u32,
    distortion: camera.DistortionModel,
    psf: camera.PointSpreadFunc,
) camera.CameraInput {
    const target = sceneops.boundsCenterOverMeshes(meshes);
    return .{
        .pixels_num = pixels_num,
        .pixels_size = pixel_size,
        .pos_world = cameraops.posFillFrameFromRotOverMeshesAndTarg(
            meshes,
            target,
            pixels_num,
            pixel_size,
            focal_length,
            rot,
            1.1,
        ),
        .rot_world = rot,
        .roi_cent_world = target,
        .focal_length = focal_length,
        .sub_sample = sub_sample,
        .distortion = distortion,
        .psf = psf,
    };
}

fn buildCameras(meshes: []MeshInput) [6]camera.CameraInput {
    const deg = std.math.degreesToRadians;
    const brown: camera.DistortionModel = .{ .brown_conrady = .{
        .k1 = -0.12,
        .k2 = 0.035,
        .p1 = 0.0002,
        .p2 = -0.0001,
    } };
    const gaussian: camera.PointSpreadFunc = .{ .gaussian = .{
        .sigma_px = 0.65,
        .supp_rad_px = 2.0,
    } };
    return .{
        makeCamera(meshes, .{ 1024, 1024 }, Rotation.init(0, 0, 0), 1, .none, .{ .pixel_box = .{} }),
        makeCamera(meshes, .{ 1024, 1024 }, Rotation.init(0, deg(25.0), 0), 4, brown, .{ .pixel_box = .{} }),
        makeCamera(meshes, .{ 1024, 1229 }, Rotation.init(0, deg(-28.0), 0), 4, .none, gaussian),
        makeCamera(meshes, .{ 1229, 1024 }, Rotation.init(deg(90.0), deg(25.0), 0), 4, brown, gaussian),
        makeCamera(meshes, .{ 1024, 1024 }, Rotation.init(deg(18.0), deg(38.0), deg(26.0)), 4, .none, .{ .anisotropic_gaussian = .{
            .sigma_x_px = 0.55,
            .sigma_y_px = 0.9,
            .theta_rad = deg(25.0),
            .supp_rad_px = 2.5,
        } }),
        makeCamera(meshes, .{ 1229, 1024 }, Rotation.init(deg(-90.0), deg(-20.0), deg(5.0)), 4, brown, .{ .pixel_box = .{} }),
    };
}

fn renderCase(
    comptime T: type,
    comptime C: usize,
    comptime bits: u8,
    allocator: std.mem.Allocator,
    io: std.Io,
    texture_path: []const u8,
    case_name: []const u8,
) !void {
    const texture = try iio.loadImage(
        T,
        C,
        allocator,
        io,
        texture_path,
        if (T == u8) .bmp else if (C == 1) .tiff else .fimg,
    );
    const meshes = try buildScene(T, C, bits, allocator, io, texture);
    const cameras = buildCameras(meshes);
    const out_dir = try std.fs.path.join(
        allocator,
        &.{ out_dir_root, case_name },
    );
    const config = riley.RasterConfig{
        .render_mode = .offline,
        .total_threads = 4,
        .max_raster_workers_per_job = 1,
        .save_strategy = .disk,
        .image_save_mode = if (C == 1) .grey else .rgb,
        .background_value = 0.5 * (@as(F, @floatFromInt((@as(u32, 1) << bits) - 1))),
        .image_save_opts = &.{
            .{
                .format = if (bits == 8) .bmp else .tiff,
                .bits = bits,
                .scaling = .none,
            },
        },
    };
    const groups = [_]riley.RenderGroupSpec{.{ .io = io, .workers = 4 }};
    if (try riley.raster(
        allocator,
        &groups,
        &cameras,
        meshes,
        config,
        out_dir,
    )) |images| {
        allocator.free(images.slice);
        var images_mut = images;
        images_mut.deinit(allocator);
    }
}

pub fn main(init: std.process.Init) !void {
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const allocator = arena.allocator();
    const io = init.io;
    std.Io.Dir.cwd().deleteTree(io, out_dir_root) catch |err| {
        if (err != error.FileNotFound) return err;
    };
    try renderCase(u8, 1, 8, allocator, io, "texture/speck128_mono_u8.bmp", "mono-u8");
    try renderCase(u16, 1, 16, allocator, io, "texture/speck128_mono_u16.tiff", "mono-u16");
    try renderCase(u8, 3, 8, allocator, io, "texture/speck128_rgb_u8.bmp", "rgb-u8");
    try renderCase(u16, 3, 16, allocator, io, "data/feature_zoo/texture_rgb_u16.fimg", "rgb-u16");
}
