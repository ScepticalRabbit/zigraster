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
const tol = buildconfig.config.tol;
const ndarray = @import("ndarray.zig");
const meshio = @import("meshio.zig");
const shapefun = @import("shapefun.zig");
const shaderops = @import("shaderops.zig");
const pce = @import("parachunkexec.zig");
const geomkerns = @import("geometrykernels.zig");
const MeshType = geomkerns.MeshType;
const rops = @import("rasterops.zig");

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn calcElemNodeNormal(
    comptime N: usize,
    nodal_derivs: shapefun.NodalDerivs,
    sx: []const F,
    sy: []const F,
    sz: []const F,
    node_idx: usize,
) [3]F {
    var dx_dxi: F = 0;
    var dx_deta: F = 0;
    var dy_dxi: F = 0;
    var dy_deta: F = 0;
    var dz_dxi: F = 0;
    var dz_deta: F = 0;

    for (0..N) |nn| {
        const du = nodal_derivs.dNu[node_idx][nn];
        const dv = nodal_derivs.dNv[node_idx][nn];
        dx_dxi += du * sx[nn];
        dx_deta += dv * sx[nn];
        dy_dxi += du * sy[nn];
        dy_deta += dv * sy[nn];
        dz_dxi += du * sz[nn];
        dz_deta += dv * sz[nn];
    }

    return .{
        dy_dxi * dz_deta - dz_dxi * dy_deta,
        dz_dxi * dx_deta - dx_dxi * dz_deta,
        dx_dxi * dy_deta - dy_dxi * dx_deta,
    };
}

pub fn normaliseNormal(normal_vec: *[3]F) void {
    const nx = normal_vec[0];
    const ny = normal_vec[1];
    const nz = normal_vec[2];
    const magnitude = @sqrt(nx * nx + ny * ny + nz * nz);

    if (magnitude > tol.normals.normalise_magnitude) {
        normal_vec[0] = nx / magnitude;
        normal_vec[1] = ny / magnitude;
        normal_vec[2] = nz / magnitude;
    }
}

fn writePrepNormal(
    prep_normals: *ndarray.NDArray(F),
    pp: usize,
    nn: usize,
    normal_vec: [3]F,
) void {
    const elem_base = prep_normals.planeBase(pp);
    const field_stride = prep_normals.strides[1];
    prep_normals.slice[elem_base + 0 * field_stride + nn] = normal_vec[0];
    prep_normals.slice[elem_base + 1 * field_stride + nn] = normal_vec[1];
    prep_normals.slice[elem_base + 2 * field_stride + nn] = normal_vec[2];
}

//------------------------------------------------------------------------------------------
// Serial Normal Calculation Implementations
//------------------------------------------------------------------------------------------

pub fn calcVisExactNormals(
    coords_nodes: *const meshio.Coords,
    connect: *const meshio.Connect,
    vis_orig_elem_inds: []const usize,
    prep_normals: *ndarray.NDArray(F),
    comptime N: usize,
) void {
    const nodal_derivs = comptime shapefun.getNodalDerivs(N);

    for (vis_orig_elem_inds, 0..) |orig_ee, pp| {
        const coords_elem = rops.gatherElemNodeCoords(N, coords_nodes, connect, orig_ee);
        for (0..N) |nn| {
            var normal_vec = calcElemNodeNormal(
                N,
                nodal_derivs,
                &coords_elem.x,
                &coords_elem.y,
                &coords_elem.z,
                nn,
            );
            normaliseNormal(&normal_vec);
            writePrepNormal(prep_normals, pp, nn, normal_vec);
        }
    }
}

pub fn calcVisAvgNormals(
    allocator: std.mem.Allocator,
    coords_nodes: *const meshio.Coords,
    connect: *const meshio.Connect,
    vis_orig_elem_inds: []const usize,
    prep_normals: *ndarray.NDArray(F),
    comptime N: usize,
) !void {
    const nodal_derivs = comptime shapefun.getNodalDerivs(N);
    var max_node_idx: usize = 0;
    for (connect.table_mem) |node_idx| {
        max_node_idx = @max(max_node_idx, node_idx);
    }

    const nodes_num = max_node_idx + 1;
    const node_normals = try allocator.alloc(F, nodes_num * 3);
    defer allocator.free(node_normals);
    @memset(node_normals, 0.0);

    for (0..connect.getElemsNum()) |ee| {
        const coords_elem = rops.gatherElemNodeCoords(N, coords_nodes, connect, ee);
        const coord_inds = connect.getElem(ee);

        for (0..N) |nn| {
            const normal_vec = calcElemNodeNormal(
                N,
                nodal_derivs,
                &coords_elem.x,
                &coords_elem.y,
                &coords_elem.z,
                nn,
            );
            const node_idx = coord_inds[nn];
            node_normals[node_idx * 3 + 0] += normal_vec[0];
            node_normals[node_idx * 3 + 1] += normal_vec[1];
            node_normals[node_idx * 3 + 2] += normal_vec[2];
        }
    }

    for (vis_orig_elem_inds, 0..) |orig_ee, pp| {
        const coord_inds = connect.getElem(orig_ee);
        for (0..N) |nn| {
            const node_idx = coord_inds[nn];
            var normal_vec = [3]F{
                node_normals[node_idx * 3 + 0],
                node_normals[node_idx * 3 + 1],
                node_normals[node_idx * 3 + 2],
            };
            normaliseNormal(&normal_vec);
            writePrepNormal(prep_normals, pp, nn, normal_vec);
        }
    }
}

pub fn prepVisNormals(
    comptime MT: MeshType,
    allocator: std.mem.Allocator,
    coords_nodes: *const meshio.Coords,
    connect: *const meshio.Connect,
    vis_orig_elem_inds: []const usize,
    normal_type: shaderops.NormalType,
) !ndarray.MappedNDArray(F) {
    const N = comptime MT.getNodesNum();
    var prep_normals = try initIdentityMappedNormals(
        N,
        allocator,
        vis_orig_elem_inds.len,
    );
    switch (normal_type) {
        .none => unreachable,
        .exact => calcVisExactNormals(
            coords_nodes,
            connect,
            vis_orig_elem_inds,
            &prep_normals.array,
            N,
        ),
        .avg => try calcVisAvgNormals(
            allocator,
            coords_nodes,
            connect,
            vis_orig_elem_inds,
            &prep_normals.array,
            N,
        ),
    }
    return prep_normals;
}

//------------------------------------------------------------------------------------------
// Threaded Normal Calculation Stages
//------------------------------------------------------------------------------------------

pub fn prepVisExactNormalsRange(
    comptime MT: MeshType,
    coords_nodes: *const meshio.Coords,
    connect: *const meshio.Connect,
    vis_orig_elem_inds: []const usize,
    prep_normals: *ndarray.NDArray(F),
    vis_start: usize,
    vis_end: usize,
) void {
    const N = comptime MT.getNodesNum();
    const nodal_derivs = comptime shapefun.getNodalDerivs(N);

    for (vis_start..vis_end) |pp| {
        const orig_ee = vis_orig_elem_inds[pp];
        const coords_elem = rops.gatherElemNodeCoords(N, coords_nodes, connect, orig_ee);
        for (0..N) |nn| {
            var normal_vec = calcElemNodeNormal(
                N,
                nodal_derivs,
                &coords_elem.x,
                &coords_elem.y,
                &coords_elem.z,
                nn,
            );
            normaliseNormal(&normal_vec);
            writePrepNormal(prep_normals, pp, nn, normal_vec);
        }
    }
}

pub fn accumAvgNodeNormalsRange(
    comptime MT: MeshType,
    coords_nodes: *const meshio.Coords,
    connect: *const meshio.Connect,
    node_normals: []F,
    elem_start: usize,
    elem_end: usize,
) void {
    const N = comptime MT.getNodesNum();
    const nodal_derivs = comptime shapefun.getNodalDerivs(N);

    for (elem_start..elem_end) |ee| {
        const coords_elem = rops.gatherElemNodeCoords(N, coords_nodes, connect, ee);
        const coord_inds = connect.getElem(ee);

        for (0..N) |nn| {
            const normal_vec = calcElemNodeNormal(
                N,
                nodal_derivs,
                &coords_elem.x,
                &coords_elem.y,
                &coords_elem.z,
                nn,
            );
            const node_idx = coord_inds[nn];
            node_normals[node_idx * 3 + 0] += normal_vec[0];
            node_normals[node_idx * 3 + 1] += normal_vec[1];
            node_normals[node_idx * 3 + 2] += normal_vec[2];
        }
    }
}

pub fn writeVisAvgNormalsRange(
    comptime MT: MeshType,
    connect: *const meshio.Connect,
    vis_orig_elem_inds: []const usize,
    node_normals: []const F,
    prep_normals: *ndarray.NDArray(F),
    vis_start: usize,
    vis_end: usize,
) void {
    const N = comptime MT.getNodesNum();
    for (vis_start..vis_end) |pp| {
        const coord_inds = connect.getElem(vis_orig_elem_inds[pp]);
        for (0..N) |nn| {
            const node_idx = coord_inds[nn];
            var normal_vec = [3]F{
                node_normals[node_idx * 3 + 0],
                node_normals[node_idx * 3 + 1],
                node_normals[node_idx * 3 + 2],
            };
            normaliseNormal(&normal_vec);
            writePrepNormal(prep_normals, pp, nn, normal_vec);
        }
    }
}

pub fn NormalStages(comptime MT: MeshType) type {
    return struct {
        const ExactNormalsStage = struct {
            coords_nodes: *const meshio.Coords,
            connect: *const meshio.Connect,
            vis_orig_elem_inds: []const usize,
            prep_normals: *ndarray.NDArray(F),
        };

        pub fn runPrepVisExactNormals(
            ctx_ptr: *anyopaque,
            chunk_idx: usize,
            range_start: usize,
            range_end: usize,
        ) void {
            _ = chunk_idx;
            const stage: *ExactNormalsStage = @ptrCast(@alignCast(ctx_ptr));
            prepVisExactNormalsRange(
                MT,
                stage.coords_nodes,
                stage.connect,
                stage.vis_orig_elem_inds,
                stage.prep_normals,
                range_start,
                range_end,
            );
        }

        const AccumAvgNormalsStage = struct {
            coords_nodes: *const meshio.Coords,
            connect: *const meshio.Connect,
            chunk_node_normals: []F,
            node_normals_stride: usize,
        };

        pub fn runAccumAvgNormals(
            ctx_ptr: *anyopaque,
            chunk_idx: usize,
            range_start: usize,
            range_end: usize,
        ) void {
            const stage: *AccumAvgNormalsStage = @ptrCast(@alignCast(ctx_ptr));
            const accum_start = chunk_idx * stage.node_normals_stride;
            const accum_end = accum_start + stage.node_normals_stride;
            accumAvgNodeNormalsRange(
                MT,
                stage.coords_nodes,
                stage.connect,
                stage.chunk_node_normals[accum_start..accum_end],
                range_start,
                range_end,
            );
        }

        const WriteVisAvgNormalsStage = struct {
            connect: *const meshio.Connect,
            vis_orig_elem_inds: []const usize,
            node_normals: []const F,
            prep_normals: *ndarray.NDArray(F),
        };

        pub fn runWriteVisAvgNormals(
            ctx_ptr: *anyopaque,
            chunk_idx: usize,
            range_start: usize,
            range_end: usize,
        ) void {
            _ = chunk_idx;
            const stage: *WriteVisAvgNormalsStage = @ptrCast(@alignCast(ctx_ptr));
            writeVisAvgNormalsRange(
                MT,
                stage.connect,
                stage.vis_orig_elem_inds,
                stage.node_normals,
                stage.prep_normals,
                range_start,
                range_end,
            );
        }
    };
}

//------------------------------------------------------------------------------------------
// Pipeline Integration Helpers
//------------------------------------------------------------------------------------------

pub fn prepVisNormalsThreaded(
    comptime MT: MeshType,
    allocator: std.mem.Allocator,
    coords_nodes: *const meshio.Coords,
    connect: *const meshio.Connect,
    vis_orig_elem_inds: []const usize,
    normal_type: shaderops.NormalType,
    chunk_exec: *pce.ParaChunkExecutor,
    elem_chunk_size: usize,
    vis_chunk_size: usize,
) !ndarray.MappedNDArray(F) {
    return try prepVisNormalsThreadedN(
        comptime MT.getNodesNum(),
        MT,
        allocator,
        coords_nodes,
        connect,
        vis_orig_elem_inds,
        normal_type,
        chunk_exec,
        elem_chunk_size,
        vis_chunk_size,
    );
}

pub fn prepVisNormalsThreadedN(
    comptime N: usize,
    comptime MT: MeshType,
    allocator: std.mem.Allocator,
    coords_nodes: *const meshio.Coords,
    connect: *const meshio.Connect,
    vis_orig_elem_inds: []const usize,
    normal_type: shaderops.NormalType,
    chunk_exec: *pce.ParaChunkExecutor,
    elem_chunk_size: usize,
    vis_chunk_size: usize,
) !ndarray.MappedNDArray(F) {
    var prep_normals = try initIdentityMappedNormals(
        N,
        allocator,
        vis_orig_elem_inds.len,
    );

    const Stages = NormalStages(MT);

    switch (normal_type) {
        .none => unreachable,
        .exact => {
            var exact_stage = Stages.ExactNormalsStage{
                .coords_nodes = coords_nodes,
                .connect = connect,
                .vis_orig_elem_inds = vis_orig_elem_inds,
                .prep_normals = &prep_normals.array,
            };
            pce.runStaticRange(
                chunk_exec,
                &exact_stage,
                Stages.runPrepVisExactNormals,
                vis_orig_elem_inds.len,
                vis_chunk_size,
            );
        },
        .avg => {
            const nodes_num = getConnectNodesNum(connect);
            const elem_chunks_num = pce.getStaticPartitionsNum(
                chunk_exec,
                connect.getElemsNum(),
                elem_chunk_size,
            );
            const node_normals_stride = nodes_num * 3;
            const chunk_node_normals = try allocator.alloc(
                F,
                elem_chunks_num * node_normals_stride,
            );
            defer allocator.free(chunk_node_normals);
            @memset(chunk_node_normals, 0.0);

            var accum_stage = Stages.AccumAvgNormalsStage{
                .coords_nodes = coords_nodes,
                .connect = connect,
                .chunk_node_normals = chunk_node_normals,
                .node_normals_stride = node_normals_stride,
            };
            pce.runStaticRange(
                chunk_exec,
                &accum_stage,
                Stages.runAccumAvgNormals,
                connect.getElemsNum(),
                elem_chunk_size,
            );

            const node_normals = try allocator.alloc(F, node_normals_stride);
            defer allocator.free(node_normals);
            @memset(node_normals, 0.0);

            for (0..elem_chunks_num) |cc| {
                const chunk_start = cc * node_normals_stride;
                const chunk_end = chunk_start + node_normals_stride;
                for (chunk_start..chunk_end) |ii| {
                    node_normals[ii - chunk_start] += chunk_node_normals[ii];
                }
            }

            var write_stage = Stages.WriteVisAvgNormalsStage{
                .connect = connect,
                .vis_orig_elem_inds = vis_orig_elem_inds,
                .node_normals = node_normals,
                .prep_normals = &prep_normals.array,
            };
            pce.runStaticRange(
                chunk_exec,
                &write_stage,
                Stages.runWriteVisAvgNormals,
                vis_orig_elem_inds.len,
                vis_chunk_size,
            );
        },
    }

    return prep_normals;
}

pub fn getConnectNodesNum(connect: *const meshio.Connect) usize {
    var max_node_idx: usize = 0;
    for (connect.table_mem) |node_idx| {
        max_node_idx = @max(max_node_idx, node_idx);
    }
    return max_node_idx + 1;
}

pub fn initIdentityMappedNormals(
    comptime N: usize,
    allocator: std.mem.Allocator,
    prep_count: usize,
) !ndarray.MappedNDArray(F) {
    const prep_normals = try ndarray.NDArray(F).initFlat(
        allocator,
        &[_]usize{ prep_count, 3, N },
    );
    const map = try allocator.alloc(usize, prep_count);
    for (0..prep_count) |pp| {
        map[pp] = pp;
    }

    return .{
        .array = prep_normals,
        .map = map,
    };
}
