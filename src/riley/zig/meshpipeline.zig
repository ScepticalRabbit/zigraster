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
const F = buildconfig.F;

const ndarray = @import("ndarray.zig");
const matslice = @import("matslice.zig");

const meshio = @import("meshio.zig");

const uvio = @import("uvio.zig");

const imageio = @import("imageio.zig");

const rastcfg = @import("rasterconfig.zig");
const imageops = @import("imageops.zig");
const cam = @import("camera.zig");
const pce = @import("parachunkexec.zig");
const scalingpolicy = @import("scalingpolicy.zig");
const rops = @import("rasterops.zig");
const report = @import("report.zig");
const sceneops = @import("sceneops.zig");
const texops = @import("textureops.zig");
const Timestamp = std.Io.Clock.Timestamp;

const shaderops = @import("shaderops.zig");
const normals = @import("normals.zig");
const geomkerns = @import("geometrykernels.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

// Input: Raw user data for all frames.
// meshio.Coords/Fields: Node-order [total_nodes, ...]
// meshio.Connect: Connectivity table links nodes to elems.
pub const MeshInput = struct {
    mesh_type: geomkerns.MeshType,
    coords: meshio.Coords,
    connect: meshio.Connect,
    disp: ?meshio.Field,
    shader: shaderops.ShaderInput,
};

// Static: Persistent multi-frame resources in the engine's memory.
// meshio.Coords/Fields: Node-order [total_nodes, ...]
// Shader UVs: Elem-order (gathered during static init as they are usually static).
pub const MeshStatic = struct {
    mesh_type: geomkerns.MeshType,
    coords_orig: meshio.Coords,
    connect: meshio.Connect,
    disp: ?meshio.Field,
    shader: shaderops.ShaderStatic,
};

// Workspace: Temporary node-order working area for the geometry pipeline.
// meshio.Coords: Node-order [total_nodes, 3]. Holds coords for a single frame after
// displacement.
pub const MeshFrameWorkspace = struct {
    coords_nodes: meshio.Coords,
    coords_nodes_def_world: ?meshio.Coords,
    vis_orig_elem_inds: []usize,
    elem_bboxes: []rops.ElemBBox,
    elems_in_image: usize,
    raster_hull: ?ndarray.NDArray(F),
    vis_counts_by_chunk: []usize,
    vis_offsets_by_chunk: []usize,
};

// Frame: Wraps the Prep payload with per-frame spatial metadata.
// Prep means culled elem-order ndarray.NDArray data ready for the raster loop.
pub const MeshFrame = struct {
    mesh: MeshPrepared,
    elem_bboxes: []rops.ElemBBox,
    elems_in_image: usize,
    total_elems_num: usize,
    raster_hull: ?ndarray.NDArray(F),
    frame_workspace: MeshFrameWorkspace,
};

// Prep: Data culled and gathered for the raster loop for a SINGLE frame.
// Prep means culled elem-order ndarray.NDArray data ready for the raster loop.
// Elem-order [vis_elems, field, nodes_per_elem]
pub const MeshPrepared = struct {
    mesh_type: geomkerns.MeshType,
    coords: ndarray.NDArray(F),
    shader: shaderops.ShaderPrepared,
};

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn calcNodesPerElem(meshes: []const MeshPrepared) F {
    var nodes_sum: usize = 0;
    for (meshes) |mesh| {
        nodes_sum += mesh.mesh_type.getNodesNum();
    }
    return @as(F, @floatFromInt(nodes_sum)) / @as(F, @floatFromInt(meshes.len));
}

pub fn countFrames(meshes: []const MeshInput) usize {
    const dim_time_pre: usize = 0;
    var num_time: usize = 1;

    for (meshes) |mesh| {
        if (mesh.disp) |field| {
            num_time = @max(num_time, field.array.dims[dim_time_pre]);
        } else switch (mesh.shader) {
            .nodal => |shader| {
                num_time = @max(
                    num_time,
                    shader.field.array.dims[dim_time_pre],
                );
            },
            else => {},
        }
    }
    return num_time;
}

pub fn countOutputFields(meshes: []const MeshInput) u8 {
    var num_fields: u8 = 0;
    for (meshes) |mesh| {
        const mesh_fields: u8 = switch (mesh.shader) {
            .nodal => |shader| shader.field.getFieldsN(),
            .tex_u8, .tex_u16, .tex_f => 1,
            .tex_rgb_u8, .tex_rgb_u16, .tex_rgb_f => 3,
            .func => 1,
            .func_rgb => 3,
        };
        num_fields = @max(num_fields, mesh_fields);
    }
    return num_fields;
}

pub fn countStaticMeshElems(mesh_static: []const MeshStatic) usize {
    var total: usize = 0;
    for (mesh_static) |mesh| {
        total += mesh.connect.table.rows_num;
    }
    return total;
}

pub fn countStaticMeshNodes(mesh_static: []const MeshStatic) usize {
    var total: usize = 0;
    for (mesh_static) |mesh| {
        total += mesh.coords_orig.mat.rows_num;
    }
    return total;
}

// External helper to turn a SimData struct into a MeshInput with associted shader data for
// rendering
pub fn meshInputFromSimDataSlice(
    outer_alloc: std.mem.Allocator,
    io: std.Io,
    sim_datas: []const meshio.SimData,
    mesh_types: []const geomkerns.MeshType,
    shader_mode: enum { nodal, tex },
    uv_paths: ?[]const []const u8,
    tex_path: ?[]const u8,
    uv_file: ?[]const u8,
) ![]MeshInput {
    var mesh_inputs = try outer_alloc.alloc(MeshInput, sim_datas.len);
    var initialized_count: usize = 0;
    errdefer {
        for (0..initialized_count) |ii| {
            switch (mesh_inputs[ii].shader) {
                .tex_u8 => |tex| {
                    outer_alloc.free(tex.uvs.slice);
                },
                .tex_u16 => |tex| {
                    outer_alloc.free(tex.uvs.slice);
                },
                .tex_f => |tex| {
                    outer_alloc.free(tex.uvs.slice);
                },
                .tex_rgb_u8 => |tex| {
                    outer_alloc.free(tex.uvs.slice);
                },
                .tex_rgb_u16 => |tex| {
                    outer_alloc.free(tex.uvs.slice);
                },
                .tex_rgb_f => |tex| {
                    outer_alloc.free(tex.uvs.slice);
                },
                .func => |tex_func| {
                    if (tex_func.uvs) |uvs| outer_alloc.free(uvs.slice);
                },
                .func_rgb => |tex_func| {
                    if (tex_func.uvs) |uvs| outer_alloc.free(uvs.slice);
                },
                else => {},
            }
        }
        outer_alloc.free(mesh_inputs);
    }

    const uv_file_name = uv_file orelse "uvs.csv";

    for (sim_datas, 0..) |sim_data, ii| {
        mesh_inputs[ii] = MeshInput{
            .mesh_type = mesh_types[ii],
            .coords = sim_data.coords,
            .connect = sim_data.connect,
            .disp = sim_data.disp,
            .shader = undefined,
        };

        if (shader_mode == .nodal) {
            if (sim_data.field) |field| {
                mesh_inputs[ii].shader = .{ .nodal = .{
                    .field = field,
                    .bits = 8,
                    .normal_type = .none,
                } };
            } else {
                return error.MissingFieldData;
            }
        } else {
            const paths = uv_paths orelse return error.MissingUVPaths;
            const path_uvs = try std.fmt.allocPrint(
                outer_alloc,
                "{s}{s}",
                .{ paths[ii], uv_file_name },
            );
            defer outer_alloc.free(path_uvs);

            const uvmap = try uvio.loadUVMap(outer_alloc, io, path_uvs);

            const format: imageio.ImageFormat = if (std.mem.endsWith(u8, tex_path.?, ".bmp"))
                .bmp
            else
                .tiff;

            const tex = try imageio.loadImage(
                u8,
                1,
                outer_alloc,
                io,
                tex_path.?,
                format,
            );

            mesh_inputs[ii].shader = .{ .tex_u8 = .{
                .uvs = uvmap.array,
                .tex = tex,
                .samp_cfg = .{ .sample = .cubic_catmull_rom, .mode = .lut_lerp },
                .normal_type = .none,
            } };
        }
        initialized_count += 1;
    }
    return mesh_inputs;
}

// -----------------------------------------------------------------------------------------
// Mesh Initialisation: Initial mesh and coordinate preparation
// -----------------------------------------------------------------------------------------

pub fn initMeshStatic(
    allocator: std.mem.Allocator,
    mesh_input: *const MeshInput,
) !MeshStatic {
    const coords_orig = try sceneops.duplicateCoords(
        allocator,
        mesh_input.coords,
    );

    var shader_static: shaderops.ShaderStatic = undefined;
    switch (mesh_input.shader) {
        .nodal => |nodal_in| {
            shader_static = .{ .nodal = .{
                .field = nodal_in.field,
                .bits = nodal_in.bits,
                .scaling = nodal_in.scaling,
                .scale_over = nodal_in.scale_over,
                .normal_type = nodal_in.normal_type,
            } };
        },
        .tex_u8 => |tex_in| {
            const elem_uvs = try prepUVs(
                allocator,
                &tex_in.uvs,
                &mesh_input.connect,
            );
            shader_static = .{ .tex_u8 = .{
                .elem_uvs = elem_uvs,
                .tex = tex_in.tex,
                .samp_cfg = tex_in.samp_cfg,
                .bits = tex_in.bits,
                .scaling = tex_in.scaling,
                .normal_type = tex_in.normal_type,
            } };
        },
        .tex_u16 => |tex_in| {
            const elem_uvs = try prepUVs(
                allocator,
                &tex_in.uvs,
                &mesh_input.connect,
            );
            shader_static = .{ .tex_u16 = .{
                .elem_uvs = elem_uvs,
                .tex = tex_in.tex,
                .samp_cfg = tex_in.samp_cfg,
                .bits = tex_in.bits,
                .scaling = tex_in.scaling,
                .normal_type = tex_in.normal_type,
            } };
        },
        .tex_f => |tex_in| {
            const elem_uvs = try prepUVs(
                allocator,
                &tex_in.uvs,
                &mesh_input.connect,
            );
            shader_static = .{ .tex_f = .{
                .elem_uvs = elem_uvs,
                .tex = tex_in.tex,
                .samp_cfg = tex_in.samp_cfg,
                .bits = tex_in.bits,
                .scaling = tex_in.scaling,
                .normal_type = tex_in.normal_type,
            } };
        },
        .tex_rgb_u8 => |tex_in| {
            const elem_uvs = try prepUVs(
                allocator,
                &tex_in.uvs,
                &mesh_input.connect,
            );
            shader_static = .{ .tex_rgb_u8 = .{
                .elem_uvs = elem_uvs,
                .tex = tex_in.tex,
                .samp_cfg = tex_in.samp_cfg,
                .bits = tex_in.bits,
                .scaling = tex_in.scaling,
                .normal_type = tex_in.normal_type,
            } };
        },
        .tex_rgb_u16 => |tex_in| {
            const elem_uvs = try prepUVs(
                allocator,
                &tex_in.uvs,
                &mesh_input.connect,
            );
            shader_static = .{ .tex_rgb_u16 = .{
                .elem_uvs = elem_uvs,
                .tex = tex_in.tex,
                .samp_cfg = tex_in.samp_cfg,
                .bits = tex_in.bits,
                .scaling = tex_in.scaling,
                .normal_type = tex_in.normal_type,
            } };
        },
        .tex_rgb_f => |tex_in| {
            const elem_uvs = try prepUVs(
                allocator,
                &tex_in.uvs,
                &mesh_input.connect,
            );
            shader_static = .{ .tex_rgb_f = .{
                .elem_uvs = elem_uvs,
                .tex = tex_in.tex,
                .samp_cfg = tex_in.samp_cfg,
                .bits = tex_in.bits,
                .scaling = tex_in.scaling,
                .normal_type = tex_in.normal_type,
            } };
        },
        .func => |tex_func_in| {
            const elem_uvs = if (tex_func_in.uvs) |uvs|
                try prepUVs(
                    allocator,
                    &uvs,
                    &mesh_input.connect,
                )
            else
                null;
            shader_static = .{ .func = .{
                .elem_uvs = elem_uvs,
                .coord_mode = tex_func_in.coord_mode,
                .builtin = tex_func_in.builtin,
                .params = shaderops.normFuncShaderParams(
                    tex_func_in.builtin,
                    tex_func_in.params,
                ),
                .bits = tex_func_in.bits,
                .scaling = tex_func_in.scaling,
                .normal_type = tex_func_in.normal_type,
            } };
        },
        .func_rgb => |tex_func_in| {
            const elem_uvs = if (tex_func_in.uvs) |uvs|
                try prepUVs(
                    allocator,
                    &uvs,
                    &mesh_input.connect,
                )
            else
                null;
            shader_static = .{ .func_rgb = .{
                .elem_uvs = elem_uvs,
                .coord_mode = tex_func_in.coord_mode,
                .builtin = tex_func_in.builtin,
                .params = shaderops.normFuncShaderParams(
                    tex_func_in.builtin,
                    tex_func_in.params,
                ),
                .bits = tex_func_in.bits,
                .scaling = tex_func_in.scaling,
                .normal_type = tex_func_in.normal_type,
            } };
        },
    }

    return .{
        .mesh_type = mesh_input.mesh_type,
        .coords_orig = coords_orig,
        .connect = mesh_input.connect,
        .disp = mesh_input.disp,
        .shader = shader_static,
    };
}

pub const GeomTimes = struct {
    coord_ops: u64 = 0,
    cull_ops: u64 = 0,
    prep_hulls_shaders: u64 = 0,
    remap_inds: u64 = 0,
};

// -----------------------------------------------------------------------------------------
// Mesh Pipeline: Main entry point into the mesh frame pipeline
// -----------------------------------------------------------------------------------------

pub fn prepMeshFrames(
    arena_alloc: std.mem.Allocator,
    chunk_exec: *pce.ParaChunkExecutor,
    workers_num: usize,
    camera: *const cam.CameraPrepared,
    raster_halo_px: u16,
    config: rastcfg.RasterConfig,
    frame_idx: usize,
    static_meshes: []const MeshStatic,
    nodal_global_scaling: []const ?imageops.ScalingParams,
    frame_meshes: []MeshFrame,
    timing: *GeomTimes,
) !FrameGeomResult {
    var res = FrameGeomResult{
        .total_elems_num = 0,
        .total_elems_in_image = 0,
    };

    for (static_meshes, 0..) |*mesh_static, ii| {
        // Only needed for nodal interpolation shading and only if not .none. If .none we
        // directly render float fields unscaled.
        var nodal_frame_scaling: ?imageops.ScalingParams = null;
        switch (mesh_static.shader) {
            .nodal => |s| {
                if (s.scale_over == .over_frames) {
                    nodal_frame_scaling = nodal_global_scaling[ii];
                } else { // .within_frames
                    nodal_frame_scaling = imageops.getScalingParamsNDArray(
                        &s.field.array,
                        frame_idx,
                        s.scaling,
                    );
                }
            },
            else => {},
        }

        // Prepares meshes for each frame including coord transforms to camera space and
        // data reshaping to elem order for a given frame.
        frame_meshes[ii] = try prepMeshFrameWithHalo(
            arena_alloc,
            chunk_exec,
            workers_num,
            camera,
            raster_halo_px,
            config,
            mesh_static,
            frame_idx,
            nodal_frame_scaling,
            timing,
        );
        res.total_elems_num += frame_meshes[ii].total_elems_num;
        res.total_elems_in_image += frame_meshes[ii].elems_in_image;
    }

    return res;
}

pub fn prepMeshFrame(
    allocator: std.mem.Allocator,
    chunk_exec: *pce.ParaChunkExecutor,
    workers_num: usize,
    camera: *const cam.CameraPrepared,
    config: rastcfg.RasterConfig,
    mesh_static: *const MeshStatic,
    frame_idx: usize,
    scaling_params: ?imageops.ScalingParams,
    timing: *GeomTimes,
) !MeshFrame {
    return prepMeshFrameWithHalo(
        allocator,
        chunk_exec,
        workers_num,
        camera,
        config.raster_halo_px_override orelse camera.prep_psf.halo_px,
        config,
        mesh_static,
        frame_idx,
        scaling_params,
        timing,
    );
}

pub fn prepMeshFrameWithHalo(
    allocator: std.mem.Allocator,
    chunk_exec: *pce.ParaChunkExecutor,
    workers_num: usize,
    camera: *const cam.CameraPrepared,
    raster_halo_px: u16,
    config: rastcfg.RasterConfig,
    mesh_static: *const MeshStatic,
    frame_idx: usize,
    scaling_params: ?imageops.ScalingParams,
    timing: *GeomTimes,
) !MeshFrame {
    return switch (mesh_static.mesh_type) {
        inline else => |MT| {
            var pipeline = try FrameMeshPipeline(MT).init(
                allocator,
                camera,
                raster_halo_px,
                mesh_static,
                frame_idx,
                config.hull_mode,
                scaling_params,
                chunk_exec,
                workers_num,
            );
            return try pipeline.run(timing);
        },
    };
}

pub const FrameGeomResult = struct {
    total_elems_num: usize,
    total_elems_in_image: usize,
};

// Outside of pipeline because these are static - we gather these into an NDarray once
// for all frames and cameras
fn prepUVs(
    outer_alloc: std.mem.Allocator,
    uvs: *const ndarray.NDArray(F),
    connect: *const meshio.Connect,
) !ndarray.NDArray(F) {
    const elems_num = connect.getElemsNum();
    const nodes_per_elem = connect.getNodesPerElem();
    var elem_uv_arr = try ndarray.NDArray(F).initFlat(
        outer_alloc,
        &[_]usize{ elems_num, 2, nodes_per_elem },
    );
    @memset(elem_uv_arr.slice, 0.0);

    for (0..elems_num) |ee| {
        const coord_inds = connect.getElem(ee);
        for (0..nodes_per_elem) |nn| {
            for (0..2) |uu| {
                const val = uvs.get(&[_]usize{ coord_inds[nn], uu });
                elem_uv_arr.set(&[_]usize{ ee, uu, nn }, val);
            }
        }
    }

    return elem_uv_arr;
}

fn initMeshFrameWorkspace(
    allocator: std.mem.Allocator,
    mesh_static: *const MeshStatic,
) !MeshFrameWorkspace {
    return .{
        .coords_nodes = try meshio.Coords.initAlloc(
            allocator,
            mesh_static.coords_orig.mat.rows_num,
        ),
        .coords_nodes_def_world = if (switch (mesh_static.shader) {
            .func => |func_static| func_static.coord_mode == .world_deformed,
            .func_rgb => |func_static| func_static.coord_mode == .world_deformed,
            else => false,
        })
            try meshio.Coords.initAlloc(
                allocator,
                mesh_static.coords_orig.mat.rows_num,
            )
        else
            null,
        .vis_orig_elem_inds = &.{},
        .elem_bboxes = &.{},
        .elems_in_image = 0,
        .raster_hull = null,
        .vis_counts_by_chunk = &.{},
        .vis_offsets_by_chunk = &.{},
    };
}

//------------------------------------------------------------------------------------------
// Geometry Pipeline Stages: meshio.Coords, Transform, Culling and Gathering
//------------------------------------------------------------------------------------------

fn prefixVisCounts(mesh_workspace: *MeshFrameWorkspace) void {
    var running_total: usize = 0;
    for (mesh_workspace.vis_counts_by_chunk, 0..) |vis_count, cc| {
        mesh_workspace.vis_offsets_by_chunk[cc] = running_total;
        running_total += vis_count;
    }
    mesh_workspace.elems_in_image = running_total;
}

fn getConnectNodesNum(connect: *const meshio.Connect) usize {
    var max_node_idx: usize = 0;
    for (connect.table_mem) |node_idx| {
        max_node_idx = @max(max_node_idx, node_idx);
    }
    return max_node_idx + 1;
}

//------------------------------------------------------------------------------------------
// Frame Mesh Pipeline: Implementation of the frame-by-frame geometry pipeline
//------------------------------------------------------------------------------------------

fn FrameMeshPipeline(comptime MT: geomkerns.MeshType) type {
    return struct {
        const FrameMeshPipelineType = @This();

        allocator: std.mem.Allocator,
        camera: *const cam.CameraPrepared,
        raster_halo_px: u16,
        mesh_static: *const MeshStatic,
        frame_idx: usize,
        hull_mode: rastcfg.HullMode,
        scaling_params: ?imageops.ScalingParams,
        chunk_exec: *pce.ParaChunkExecutor,
        workers_num: usize,
        nodes_num: usize,
        elems_num: usize,
        node_chunk_size: usize,
        elem_chunk_size: usize,
        elem_chunks_num: usize,
        vis_chunk_size: usize,
        mesh_workspace: MeshFrameWorkspace,

        fn init(
            allocator: std.mem.Allocator,
            camera: *const cam.CameraPrepared,
            raster_halo_px: u16,
            mesh_static: *const MeshStatic,
            frame_idx: usize,
            hull_mode: rastcfg.HullMode,
            scaling_params: ?imageops.ScalingParams,
            chunk_exec: *pce.ParaChunkExecutor,
            workers_num: usize,
        ) !FrameMeshPipelineType {
            const nodes_num = mesh_static.coords_orig.mat.rows_num;
            const elems_num = mesh_static.connect.getElemsNum();
            const node_chunk_size = scalingpolicy.geometryNodeChunkSize(
                nodes_num,
                workers_num,
            );
            const elem_chunk_size = scalingpolicy.geometryElemChunkSize(
                elems_num,
                workers_num,
            );

            return .{
                .allocator = allocator,
                .camera = camera,
                .raster_halo_px = raster_halo_px,
                .mesh_static = mesh_static,
                .frame_idx = frame_idx,
                .hull_mode = hull_mode,
                .scaling_params = scaling_params,
                .chunk_exec = chunk_exec,
                .workers_num = workers_num,
                .nodes_num = nodes_num,
                .elems_num = elems_num,
                .node_chunk_size = node_chunk_size,
                .elem_chunk_size = elem_chunk_size,
                .elem_chunks_num = pce.getStaticPartitionsNum(
                    chunk_exec,
                    elems_num,
                    elem_chunk_size,
                ),
                .vis_chunk_size = 1,
                .mesh_workspace = try initMeshFrameWorkspace(
                    allocator,
                    mesh_static,
                ),
            };
        }

        fn run(
            self: *FrameMeshPipelineType,
            timing: *GeomTimes,
        ) !MeshFrame {
            const time_start_coords = Timestamp.now(self.chunk_exec.io, .awake);
            self.displaceCoords();
            self.transformCoords();
            const time_end_coords = Timestamp.now(self.chunk_exec.io, .awake);
            timing.coord_ops += @intCast(time_start_coords.durationTo(
                time_end_coords,
            ).raw.nanoseconds);

            const time_start_cull = Timestamp.now(self.chunk_exec.io, .awake);
            try self.cullVis();
            var mesh_prep = try self.gatherVisCoords();
            const time_end_cull = Timestamp.now(self.chunk_exec.io, .awake);
            timing.cull_ops += @intCast(time_start_cull.durationTo(
                time_end_cull,
            ).raw.nanoseconds);

            const time_start_prep = Timestamp.now(self.chunk_exec.io, .awake);
            try self.prepareRasterHulls(&mesh_prep.coords);
            try self.prepareShader(&mesh_prep);
            const time_end_prep = Timestamp.now(self.chunk_exec.io, .awake);
            timing.prep_hulls_shaders += @intCast(time_start_prep.durationTo(
                time_end_prep,
            ).raw.nanoseconds);

            const time_start_remap = Timestamp.now(self.chunk_exec.io, .awake);
            self.remapVisElemInds();
            const time_end_remap = Timestamp.now(self.chunk_exec.io, .awake);
            timing.remap_inds += @intCast(time_start_remap.durationTo(
                time_end_remap,
            ).raw.nanoseconds);

            return .{
                .mesh = mesh_prep,
                .elem_bboxes = self.mesh_workspace.elem_bboxes,
                .elems_in_image = self.mesh_workspace.elems_in_image,
                .total_elems_num = self.mesh_static.connect.getElemsNum(),
                .raster_hull = self.mesh_workspace.raster_hull,
                .frame_workspace = self.mesh_workspace,
            };
        }

        const DisplaceCoordsStage = struct {
            frame_workspace: *MeshFrameWorkspace,
            mesh_static: *const MeshStatic,
            frame_idx: usize,
        };

        fn runDisplaceCoords(
            ctx_ptr: *anyopaque,
            chunk_idx: usize,
            range_start: usize,
            range_end: usize,
        ) void {
            _ = chunk_idx;
            const stage: *DisplaceCoordsStage = @ptrCast(@alignCast(ctx_ptr));

            const actual_frame_idx = if (stage.mesh_static.disp) |disp|
                @min(stage.frame_idx, disp.array.dims[0] - 1)
            else
                0;

            for (range_start..range_end) |nn| {
                const coord_off = nn * 3;
                stage.frame_workspace.coords_nodes.mem[coord_off + 0] =
                    stage.mesh_static.coords_orig.mem[coord_off + 0];
                stage.frame_workspace.coords_nodes.mem[coord_off + 1] =
                    stage.mesh_static.coords_orig.mem[coord_off + 1];
                stage.frame_workspace.coords_nodes.mem[coord_off + 2] =
                    stage.mesh_static.coords_orig.mem[coord_off + 2];

                if (stage.mesh_static.disp) |disp| {
                    for (0..3) |cc| {
                        stage.frame_workspace.coords_nodes.mem[coord_off + cc] +=
                            disp.array.get(&[_]usize{ actual_frame_idx, nn, cc });
                    }
                }
            }
        }

        fn displaceCoords(self: *FrameMeshPipelineType) void {
            var displace_stage = DisplaceCoordsStage{
                .frame_workspace = &self.mesh_workspace,
                .mesh_static = self.mesh_static,
                .frame_idx = self.frame_idx,
            };
            pce.runStaticRange(
                self.chunk_exec,
                &displace_stage,
                runDisplaceCoords,
                self.nodes_num,
                self.node_chunk_size,
            );
            if (self.mesh_workspace.coords_nodes_def_world) |*coords_def_world| {
                @memcpy(coords_def_world.mem, self.mesh_workspace.coords_nodes.mem);
            }
        }

        const TransformCoordsStage = struct {
            camera: *const cam.CameraPrepared,
            frame_workspace: *MeshFrameWorkspace,
        };

        fn runTransformCoords(
            ctx_ptr: *anyopaque,
            chunk_idx: usize,
            range_start: usize,
            range_end: usize,
        ) void {
            _ = chunk_idx;
            const stage: *TransformCoordsStage = @ptrCast(@alignCast(ctx_ptr));

            if (MT == .tri3 or MT == .tri3opt) {
                rops.nodesToRasterRangeInPlace(
                    stage.camera,
                    &stage.frame_workspace.coords_nodes,
                    range_start,
                    range_end,
                );
            } else {
                rops.nodesToClipPxLengRangeInPlace(
                    stage.camera,
                    &stage.frame_workspace.coords_nodes,
                    range_start,
                    range_end,
                );
            }
        }

        fn transformCoords(self: *FrameMeshPipelineType) void {
            var transform_stage = TransformCoordsStage{
                .camera = self.camera,
                .frame_workspace = &self.mesh_workspace,
            };

            pce.runStaticRange(
                self.chunk_exec,
                &transform_stage,
                runTransformCoords,
                self.nodes_num,
                self.node_chunk_size,
            );
        }

        const CullVisibleCountStage = struct {
            camera: *const cam.CameraPrepared,
            raster_halo_px: u16,
            connect: *const meshio.Connect,
            coords_nodes: *const meshio.Coords,
            vis_counts_by_chunk: []usize,
            hull_mode: rastcfg.HullMode,
        };

        fn runCullVisibleCount(
            ctx_ptr: *anyopaque,
            chunk_idx: usize,
            range_start: usize,
            range_end: usize,
        ) void {
            const stage: *CullVisibleCountStage = @ptrCast(@alignCast(ctx_ptr));
            const hull_convex_fallback_on = stage.hull_mode == .on_convex_fallback;
            var vis_count: usize = 0;

            for (range_start..range_end) |ee| {
                const bbox = if (MT == .tri3 or MT == .tri3opt)
                    rops.calcVisibleNodeBBoxTri3WithHalo(
                        MT,
                        stage.camera,
                        stage.coords_nodes,
                        stage.connect,
                        ee,
                        stage.raster_halo_px,
                    )
                else if (stage.hull_mode == .off)
                    rops.calcVisibleNodeBBoxHighOrdNoHullWithHalo(
                        MT,
                        stage.camera,
                        stage.coords_nodes,
                        stage.connect,
                        ee,
                        stage.raster_halo_px,
                    )
                else
                    rops.calcVisibleNodeBBoxHighOrdWithHalo(
                        MT,
                        stage.camera,
                        stage.coords_nodes,
                        stage.connect,
                        ee,
                        hull_convex_fallback_on,
                        stage.raster_halo_px,
                    );

                if (bbox != null) {
                    vis_count += 1;
                }
            }

            stage.vis_counts_by_chunk[chunk_idx] = vis_count;
        }

        fn cullVis(self: *FrameMeshPipelineType) !void {
            self.mesh_workspace.vis_counts_by_chunk = try self.allocator.alloc(
                usize,
                self.elem_chunks_num,
            );

            self.mesh_workspace.vis_offsets_by_chunk = try self.allocator.alloc(
                usize,
                self.elem_chunks_num,
            );

            @memset(self.mesh_workspace.vis_counts_by_chunk, 0);
            @memset(self.mesh_workspace.vis_offsets_by_chunk, 0);

            var cull_count_stage = CullVisibleCountStage{
                .camera = self.camera,
                .raster_halo_px = self.raster_halo_px,
                .connect = &self.mesh_static.connect,
                .coords_nodes = &self.mesh_workspace.coords_nodes,
                .vis_counts_by_chunk = self.mesh_workspace.vis_counts_by_chunk,
                .hull_mode = self.hull_mode,
            };

            pce.runStaticRange(
                self.chunk_exec,
                &cull_count_stage,
                runCullVisibleCount,
                self.elems_num,
                self.elem_chunk_size,
            );

            prefixVisCounts(&self.mesh_workspace);

            self.mesh_workspace.vis_orig_elem_inds =
                try self.allocator.alloc(usize, self.mesh_workspace.elems_in_image);
            self.mesh_workspace.elem_bboxes = try self.allocator.alloc(
                rops.ElemBBox,
                self.mesh_workspace.elems_in_image,
            );

            var cull_fill_stage = CullVisibleFillStage{
                .camera = self.camera,
                .raster_halo_px = self.raster_halo_px,
                .connect = &self.mesh_static.connect,
                .coords_nodes = &self.mesh_workspace.coords_nodes,
                .vis_orig_elem_inds = self.mesh_workspace.vis_orig_elem_inds,
                .elem_bboxes = self.mesh_workspace.elem_bboxes,
                .vis_offsets_by_chunk = self.mesh_workspace.vis_offsets_by_chunk,
                .hull_mode = self.hull_mode,
            };
            pce.runStaticRange(
                self.chunk_exec,
                &cull_fill_stage,
                runCullVisibleFill,
                self.elems_num,
                self.elem_chunk_size,
            );

            self.vis_chunk_size = scalingpolicy.geometryVisibleChunkSize(
                self.mesh_workspace.elems_in_image,
                self.workers_num,
            );
        }

        const CullVisibleFillStage = struct {
            camera: *const cam.CameraPrepared,
            raster_halo_px: u16,
            connect: *const meshio.Connect,
            coords_nodes: *const meshio.Coords,
            vis_orig_elem_inds: []usize,
            elem_bboxes: []rops.ElemBBox,
            vis_offsets_by_chunk: []const usize,
            hull_mode: rastcfg.HullMode,
        };

        fn runCullVisibleFill(
            ctx_ptr: *anyopaque,
            chunk_idx: usize,
            range_start: usize,
            range_end: usize,
        ) void {
            const stage: *CullVisibleFillStage = @ptrCast(@alignCast(ctx_ptr));
            const hull_convex_fallback_on = stage.hull_mode == .on_convex_fallback;
            var write_idx = stage.vis_offsets_by_chunk[chunk_idx];

            for (range_start..range_end) |ee| {
                const bbox = if (MT == .tri3 or MT == .tri3opt)
                    rops.calcVisibleNodeBBoxTri3WithHalo(
                        MT,
                        stage.camera,
                        stage.coords_nodes,
                        stage.connect,
                        ee,
                        stage.raster_halo_px,
                    )
                else if (stage.hull_mode == .off)
                    rops.calcVisibleNodeBBoxHighOrdNoHullWithHalo(
                        MT,
                        stage.camera,
                        stage.coords_nodes,
                        stage.connect,
                        ee,
                        stage.raster_halo_px,
                    )
                else
                    rops.calcVisibleNodeBBoxHighOrdWithHalo(
                        MT,
                        stage.camera,
                        stage.coords_nodes,
                        stage.connect,
                        ee,
                        hull_convex_fallback_on,
                        stage.raster_halo_px,
                    );

                if (bbox) |b| {
                    stage.vis_orig_elem_inds[write_idx] = ee;
                    stage.elem_bboxes[write_idx] = b;
                    write_idx += 1;
                }
            }
        }

        const GatherVisibleCoordsStage = struct {
            connect: *const meshio.Connect,
            coords_nodes: *const meshio.Coords,
            vis_orig_elem_inds: []const usize,
            elem_coords: *ndarray.NDArray(F),
        };

        fn runGatherVisibleCoords(
            ctx_ptr: *anyopaque,
            chunk_idx: usize,
            range_start: usize,
            range_end: usize,
        ) void {
            _ = chunk_idx;
            const stage: *GatherVisibleCoordsStage = @ptrCast(@alignCast(ctx_ptr));
            const N = comptime MT.getNodesNum();

            for (range_start..range_end) |pp| {
                const orig_ee = stage.vis_orig_elem_inds[pp];
                const coord_inds = stage.connect.getElem(orig_ee);
                for (0..N) |nn| {
                    const node_idx = coord_inds[nn];
                    stage.elem_coords.set(
                        &[_]usize{ pp, 0, nn },
                        stage.coords_nodes.x(node_idx),
                    );
                    stage.elem_coords.set(
                        &[_]usize{ pp, 1, nn },
                        stage.coords_nodes.y(node_idx),
                    );
                    stage.elem_coords.set(
                        &[_]usize{ pp, 2, nn },
                        stage.coords_nodes.z(node_idx),
                    );
                }
            }
        }

        fn gatherVisElemCoordsFromNodes(
            self: *FrameMeshPipelineType,
            coords_nodes: *const meshio.Coords,
        ) !ndarray.NDArray(F) {
            const N = comptime MT.getNodesNum();
            var elem_coords = try ndarray.NDArray(F).initFlat(
                self.allocator,
                &[_]usize{ self.mesh_workspace.elems_in_image, 3, N },
            );

            var gather_coords_stage = GatherVisibleCoordsStage{
                .connect = &self.mesh_static.connect,
                .coords_nodes = coords_nodes,
                .vis_orig_elem_inds = self.mesh_workspace.vis_orig_elem_inds,
                .elem_coords = &elem_coords,
            };

            pce.runStaticRange(
                self.chunk_exec,
                &gather_coords_stage,
                runGatherVisibleCoords,
                self.mesh_workspace.elems_in_image,
                self.vis_chunk_size,
            );

            return elem_coords;
        }

        fn gatherVisCoords(self: *FrameMeshPipelineType) !MeshPrepared {
            const elem_coords = try self.gatherVisElemCoordsFromNodes(
                &self.mesh_workspace.coords_nodes,
            );

            return .{
                .mesh_type = MT,
                .coords = elem_coords,
                .shader = undefined,
            };
        }

        const PrepareRasterHullsStage = struct {
            camera: *const cam.CameraPrepared,
            elem_coords: *const ndarray.NDArray(F),
            raster_hull: *ndarray.NDArray(F),
            hull_mode: rastcfg.HullMode,
        };

        fn runPrepareRasterHulls(
            ctx_ptr: *anyopaque,
            chunk_idx: usize,
            range_start: usize,
            range_end: usize,
        ) void {
            _ = chunk_idx;
            const stage: *PrepareRasterHullsStage = @ptrCast(@alignCast(ctx_ptr));
            const hull_convex_fallback_on = stage.hull_mode == .on_convex_fallback;
            rops.prepareVisibleRasterHullsRange(
                MT,
                stage.camera,
                stage.elem_coords,
                stage.raster_hull,
                hull_convex_fallback_on,
                range_start,
                range_end,
            );
        }

        fn prepareRasterHulls(
            self: *FrameMeshPipelineType,
            elem_coords: *const ndarray.NDArray(F),
        ) !void {
            if (MT == .tri3 or MT == .tri3opt) {
                self.mesh_workspace.raster_hull = null;
                return;
            }

            if (self.hull_mode == .off) {
                self.mesh_workspace.raster_hull = null;
                return;
            }

            const NH = comptime MT.getNumHullPoints();
            self.mesh_workspace.raster_hull = try ndarray.NDArray(F).initFlat(
                self.allocator,
                &[_]usize{ self.mesh_workspace.elems_in_image, 2, NH },
            );

            var hulls_stage = PrepareRasterHullsStage{
                .camera = self.camera,
                .elem_coords = elem_coords,
                .raster_hull = &self.mesh_workspace.raster_hull.?,
                .hull_mode = self.hull_mode,
            };
            pce.runStaticRange(
                self.chunk_exec,
                &hulls_stage,
                runPrepareRasterHulls,
                self.mesh_workspace.elems_in_image,
                self.vis_chunk_size,
            );
        }

        fn prepareShader(self: *FrameMeshPipelineType, mesh_prep: *MeshPrepared) !void {
            switch (self.mesh_static.shader) {
                .nodal => |nodal_static| {
                    mesh_prep.shader = try self.prepareNodalShader(nodal_static);
                },
                .tex_u8 => |tex_static| {
                    mesh_prep.shader = try prepareTexShader(
                        u8,
                        1,
                        self,
                        tex_static,
                    );
                },
                .tex_u16 => |tex_static| {
                    mesh_prep.shader = try prepareTexShader(
                        u16,
                        1,
                        self,
                        tex_static,
                    );
                },
                .tex_f => |tex_static| {
                    mesh_prep.shader = try prepareTexShader(F, 1, self, tex_static);
                },
                .tex_rgb_u8 => |tex_static| {
                    mesh_prep.shader = try prepareTexShader(
                        u8,
                        3,
                        self,
                        tex_static,
                    );
                },
                .tex_rgb_u16 => |tex_static| {
                    mesh_prep.shader = try prepareTexShader(
                        u16,
                        3,
                        self,
                        tex_static,
                    );
                },
                .tex_rgb_f => |tex_static| {
                    mesh_prep.shader = try prepareTexShader(F, 3, self, tex_static);
                },
                .func => |func_static| {
                    mesh_prep.shader = try prepareFuncShader(1, self, func_static);
                },
                .func_rgb => |func_static| {
                    mesh_prep.shader = try prepareFuncShader(3, self, func_static);
                },
            }
        }

        const GatherVisibleFieldStage = struct {
            connect: *const meshio.Connect,
            field: *const meshio.Field,
            frame_idx: usize,
            vis_orig_elem_inds: []const usize,
            elem_field: *ndarray.NDArray(F),
        };

        fn runGatherVisibleField(
            ctx_ptr: *anyopaque,
            chunk_idx: usize,
            range_start: usize,
            range_end: usize,
        ) void {
            _ = chunk_idx;
            const stage: *GatherVisibleFieldStage = @ptrCast(@alignCast(ctx_ptr));
            const actual_frame_idx = @min(stage.frame_idx, stage.field.array.dims[0] - 1);
            const fields_num = stage.field.getFieldsN();
            const N = comptime MT.getNodesNum();

            for (range_start..range_end) |pp| {
                const orig_ee = stage.vis_orig_elem_inds[pp];
                const coord_inds = stage.connect.getElem(orig_ee);
                for (0..N) |nn| {
                    for (0..@as(usize, fields_num)) |ff| {
                        const field_val = stage.field.array.get(
                            &[_]usize{ actual_frame_idx, coord_inds[nn], ff },
                        );
                        stage.elem_field.set(&[_]usize{ pp, ff, nn }, field_val);
                    }
                }
            }
        }

        fn prepareNodalShader(
            self: *FrameMeshPipelineType,
            nodal_static: shaderops.NodalStatic,
        ) !shaderops.ShaderPrepared {
            const N = comptime MT.getNodesNum();
            var elem_field = try ndarray.NDArray(F).initFlat(
                self.allocator,
                &[_]usize{
                    self.mesh_workspace.elems_in_image,
                    @as(usize, nodal_static.field.getFieldsN()),
                    N,
                },
            );

            var field_stage = GatherVisibleFieldStage{
                .connect = &self.mesh_static.connect,
                .field = &nodal_static.field,
                .frame_idx = self.frame_idx,
                .vis_orig_elem_inds = self.mesh_workspace.vis_orig_elem_inds,
                .elem_field = &elem_field,
            };

            pce.runStaticRange(
                self.chunk_exec,
                &field_stage,
                runGatherVisibleField,
                self.mesh_workspace.elems_in_image,
                self.vis_chunk_size,
            );

            const factors = if (self.scaling_params) |sp|
                imageops.getScaleFactors(nodal_static.scaling, nodal_static.bits, sp)
            else
                imageops.ScaleFactors{ .mul = 1.0, .add = 0.0 };

            return .{ .nodal = .{
                .elem_field = elem_field,
                .bits = nodal_static.bits,
                .scaling = nodal_static.scaling,
                .scale_over = nodal_static.scale_over,
                .scale_mul = factors.mul,
                .scale_add = factors.add,
                .normal_type = nodal_static.normal_type,
                .elem_normals = try self.prepVisNormals(nodal_static.normal_type),
            } };
        }

        const GatherVisibleUVStage = struct {
            elem_uvs_full: ndarray.NDArray(F),
            vis_orig_elem_inds: []const usize,
            elem_uvs: *ndarray.NDArray(F),
        };

        fn runGatherVisibleUV(
            ctx_ptr: *anyopaque,
            chunk_idx: usize,
            range_start: usize,
            range_end: usize,
        ) void {
            _ = chunk_idx;
            const stage: *GatherVisibleUVStage = @ptrCast(@alignCast(ctx_ptr));
            const nodes_num = stage.elem_uvs_full.dims[2];

            for (range_start..range_end) |pp| {
                const orig_ee = stage.vis_orig_elem_inds[pp];
                const src_start = stage.elem_uvs_full.getFlatIdx(&[_]usize{ orig_ee, 0, 0 });
                const dst_start = stage.elem_uvs.getFlatIdx(&[_]usize{ pp, 0, 0 });
                @memcpy(
                    stage.elem_uvs.slice[dst_start .. dst_start + 2 * nodes_num],
                    stage.elem_uvs_full.slice[src_start .. src_start + 2 * nodes_num],
                );
            }
        }

        fn prepareTexShader(
            comptime T: type,
            comptime C: usize,
            self: *FrameMeshPipelineType,
            tex_static: shaderops.TexStatic(T, C),
        ) !shaderops.ShaderPrepared {
            const params = imageops.getScalingParamsTex(
                T,
                C,
                &tex_static.tex,
                tex_static.scaling,
            );
            const factors = imageops.getScaleFactors(
                tex_static.scaling,
                tex_static.bits,
                params,
            );

            var elem_uvs = try ndarray.NDArray(F).initFlat(
                self.allocator,
                &[_]usize{
                    self.mesh_workspace.elems_in_image,
                    2,
                    tex_static.elem_uvs.dims[2],
                },
            );

            var uv_stage = GatherVisibleUVStage{
                .elem_uvs_full = tex_static.elem_uvs,
                .vis_orig_elem_inds = self.mesh_workspace.vis_orig_elem_inds,
                .elem_uvs = &elem_uvs,
            };

            pce.runStaticRange(
                self.chunk_exec,
                &uv_stage,
                runGatherVisibleUV,
                self.mesh_workspace.elems_in_image,
                self.vis_chunk_size,
            );

            const elem_normals = try self.prepVisNormals(tex_static.normal_type);
            if (comptime T == u8 and C == 1) {
                return .{ .tex_u8 = .{
                    .elem_uvs = elem_uvs,
                    .tex = tex_static.tex,
                    .samp_cfg = tex_static.samp_cfg,
                    .bits = tex_static.bits,
                    .scaling = tex_static.scaling,
                    .scale_mul = factors.mul,
                    .scale_add = factors.add,
                    .normal_type = tex_static.normal_type,
                    .elem_normals = elem_normals,
                } };
            } else if (comptime T == u16 and C == 1) {
                return .{ .tex_u16 = .{
                    .elem_uvs = elem_uvs,
                    .tex = tex_static.tex,
                    .samp_cfg = tex_static.samp_cfg,
                    .bits = tex_static.bits,
                    .scaling = tex_static.scaling,
                    .scale_mul = factors.mul,
                    .scale_add = factors.add,
                    .normal_type = tex_static.normal_type,
                    .elem_normals = elem_normals,
                } };
            } else if (comptime T == F and C == 1) {
                return .{ .tex_f = .{
                    .elem_uvs = elem_uvs,
                    .tex = tex_static.tex,
                    .samp_cfg = tex_static.samp_cfg,
                    .bits = tex_static.bits,
                    .scaling = tex_static.scaling,
                    .scale_mul = factors.mul,
                    .scale_add = factors.add,
                    .normal_type = tex_static.normal_type,
                    .elem_normals = elem_normals,
                } };
            } else if (comptime T == u8 and C == 3) {
                return .{ .tex_rgb_u8 = .{
                    .elem_uvs = elem_uvs,
                    .tex = tex_static.tex,
                    .samp_cfg = tex_static.samp_cfg,
                    .bits = tex_static.bits,
                    .scaling = tex_static.scaling,
                    .scale_mul = factors.mul,
                    .scale_add = factors.add,
                    .normal_type = tex_static.normal_type,
                    .elem_normals = elem_normals,
                } };
            } else if (comptime T == u16 and C == 3) {
                return .{ .tex_rgb_u16 = .{
                    .elem_uvs = elem_uvs,
                    .tex = tex_static.tex,
                    .samp_cfg = tex_static.samp_cfg,
                    .bits = tex_static.bits,
                    .scaling = tex_static.scaling,
                    .scale_mul = factors.mul,
                    .scale_add = factors.add,
                    .normal_type = tex_static.normal_type,
                    .elem_normals = elem_normals,
                } };
            } else {
                return .{ .tex_rgb_f = .{
                    .elem_uvs = elem_uvs,
                    .tex = tex_static.tex,
                    .samp_cfg = tex_static.samp_cfg,
                    .bits = tex_static.bits,
                    .scaling = tex_static.scaling,
                    .scale_mul = factors.mul,
                    .scale_add = factors.add,
                    .normal_type = tex_static.normal_type,
                    .elem_normals = elem_normals,
                } };
            }
        }

        fn prepareFuncShader(
            comptime C: usize,
            self: *FrameMeshPipelineType,
            func_static: shaderops.FuncStatic,
        ) !shaderops.ShaderPrepared {
            const factors = imageops.getScaleFactors(
                func_static.scaling,
                func_static.bits,
                .{ .min = 0.0, .range = 1.0 },
            );

            const elem_uvs = if (func_static.coord_mode == .uv) blk: {
                const elem_uvs_full = func_static.elem_uvs orelse
                    return error.MissingUVsForFuncShader;
                var elem_uvs = try ndarray.NDArray(F).initFlat(
                    self.allocator,
                    &[_]usize{
                        self.mesh_workspace.elems_in_image,
                        2,
                        elem_uvs_full.dims[2],
                    },
                );

                var uv_stage = GatherVisibleUVStage{
                    .elem_uvs_full = elem_uvs_full,
                    .vis_orig_elem_inds = self.mesh_workspace.vis_orig_elem_inds,
                    .elem_uvs = &elem_uvs,
                };

                pce.runStaticRange(
                    self.chunk_exec,
                    &uv_stage,
                    runGatherVisibleUV,
                    self.mesh_workspace.elems_in_image,
                    self.vis_chunk_size,
                );
                break :blk elem_uvs;
            } else null;

            const elem_world_ref = if (func_static.coord_mode == .world_reference)
                try self.gatherVisElemCoordsFromNodes(&self.mesh_static.coords_orig)
            else
                null;
            const elem_world_def = if (func_static.coord_mode == .world_deformed)
                try self.gatherVisElemCoordsFromNodes(
                    &(self.mesh_workspace.coords_nodes_def_world orelse
                        return error.MissingWorldDeformedCoords),
                )
            else
                null;

            const elem_normals = try self.prepVisNormals(func_static.normal_type);
            if (comptime C == 1) {
                return .{ .func = .{
                    .elem_uvs = elem_uvs,
                    .elem_world_ref = elem_world_ref,
                    .elem_world_def = elem_world_def,
                    .coord_mode = func_static.coord_mode,
                    .builtin = func_static.builtin,
                    .params = shaderops.normFuncShaderParams(
                        func_static.builtin,
                        func_static.params,
                    ),
                    .bits = func_static.bits,
                    .scaling = func_static.scaling,
                    .scale_mul = factors.mul,
                    .scale_add = factors.add,
                    .normal_type = func_static.normal_type,
                    .elem_normals = elem_normals,
                } };
            } else {
                return .{ .func_rgb = .{
                    .elem_uvs = elem_uvs,
                    .elem_world_ref = elem_world_ref,
                    .elem_world_def = elem_world_def,
                    .coord_mode = func_static.coord_mode,
                    .builtin = func_static.builtin,
                    .params = shaderops.normFuncShaderParams(
                        func_static.builtin,
                        func_static.params,
                    ),
                    .bits = func_static.bits,
                    .scaling = func_static.scaling,
                    .scale_mul = factors.mul,
                    .scale_add = factors.add,
                    .normal_type = func_static.normal_type,
                    .elem_normals = elem_normals,
                } };
            }
        }

        fn prepVisNormals(
            self: *FrameMeshPipelineType,
            normal_type: shaderops.NormalType,
        ) !?ndarray.MappedNDArray(F) {
            if (normal_type == .none) {
                return null;
            }

            return try normals.prepVisNormalsThreaded(
                MT,
                self.allocator,
                &self.mesh_workspace.coords_nodes,
                &self.mesh_static.connect,
                self.mesh_workspace.vis_orig_elem_inds,
                normal_type,
                self.chunk_exec,
                self.elem_chunk_size,
                self.vis_chunk_size,
            );
        }

        fn remapVisElemInds(self: *FrameMeshPipelineType) void {
            for (self.mesh_workspace.elem_bboxes, 0..) |*elem_bbox, pp| {
                elem_bbox.elem_idx = pp;
            }
        }
    };
}
