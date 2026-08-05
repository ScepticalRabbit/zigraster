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
const matrix = @import("matstack.zig");
const meshio = @import("meshio.zig");
const rotation = @import("rotation.zig");
const vec = @import("vecstack.zig");
const F = buildconfig.F;

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const AxisAnchor = enum {
    min,
    center,
    max,
};

pub const Bounds3D = struct {
    min: [3]F,
    max: [3]F,
    center: [3]F,
    extent: [3]F,
};

pub const MeshGroupSpan = struct {
    mesh_start: usize,
    mesh_len: usize,
};

pub const MeshGroup = union(enum) {
    span: MeshGroupSpan,
    indices: []const usize,
};

pub const GridSpec = struct {
    gap: [3]F,
    max_divs: [3]usize,
};

pub const OverlapDirection = enum {
    negative,
    current,
    positive,
};

pub const BoundsOverlapSpec = struct {
    overlap_frac: [3]F,
    enabled_axes: [3]bool = .{ true, true, true },
    direction: [3]OverlapDirection = .{
        .current,
        .current,
        .current,
    },
    extra_offset: [3]F = .{ 0.0, 0.0, 0.0 },
};

pub const RadialAxis = enum {
    x,
    y,
    z,
};

pub const RadialSpec = struct {
    center: [3]F,
    radius: F,
    axis: RadialAxis = .z,
    angle_offset_rad: F = 0.0,
};

// --------------------------------------------------------------------------------------
// Public Entry-Point Func
// --------------------------------------------------------------------------------------

pub fn meshGroupSpan(
    mesh_start: usize,
    mesh_len: usize,
) MeshGroup {
    return .{
        .span = .{
            .mesh_start = mesh_start,
            .mesh_len = mesh_len,
        },
    };
}

pub fn meshGroupSingle(mesh_idx: usize) MeshGroup {
    return meshGroupSpan(mesh_idx, 1);
}

pub fn meshGroupLen(group: MeshGroup) usize {
    return switch (group) {
        .span => |span| span.mesh_len,
        .indices => |mesh_indices| mesh_indices.len,
    };
}

pub fn duplicateCoords(
    allocator: std.mem.Allocator,
    coords: meshio.Coords,
) !meshio.Coords {
    const coords_dup = try meshio.Coords.initAlloc(
        allocator,
        coords.mat.rows_num,
    );
    @memcpy(coords_dup.mem, coords.mem);
    return coords_dup;
}

pub fn duplicateMeshInstance(
    allocator: std.mem.Allocator,
    mesh_input: anytype,
) !@TypeOf(mesh_input) {
    var mesh_dup = mesh_input;
    mesh_dup.coords = try duplicateCoords(allocator, mesh_input.coords);
    return mesh_dup;
}

pub fn duplicateMeshInstances(
    comptime MeshType: type,
    allocator: std.mem.Allocator,
    mesh_inputs: []const MeshType,
) ![]MeshType {
    const mesh_dups = try allocator.alloc(MeshType, mesh_inputs.len);
    errdefer allocator.free(mesh_dups);

    var initialized_num: usize = 0;
    errdefer {
        for (0..initialized_num) |nn| {
            allocator.free(mesh_dups[nn].coords.mem);
        }
    }

    for (mesh_inputs, 0..) |mesh_input, nn| {
        mesh_dups[nn] = try duplicateMeshInstance(allocator, mesh_input);
        initialized_num += 1;
    }

    return mesh_dups;
}

pub fn duplicateMeshGroup(
    comptime MeshType: type,
    allocator: std.mem.Allocator,
    meshes: []const MeshType,
    group: MeshGroup,
) ![]MeshType {
    const mesh_num = meshGroupLen(group);
    const mesh_dups = try allocator.alloc(MeshType, mesh_num);
    errdefer allocator.free(mesh_dups);

    var initialized_num: usize = 0;
    errdefer {
        for (0..initialized_num) |nn| {
            allocator.free(mesh_dups[nn].coords.mem);
        }
    }

    for (0..mesh_num) |nn| {
        const mesh_idx = groupMeshIdx(group, nn);
        mesh_dups[nn] = try duplicateMeshInstance(allocator, meshes[mesh_idx]);
        initialized_num += 1;
    }

    return mesh_dups;
}

pub fn appendMeshGroupDuplicate(
    comptime MeshType: type,
    allocator: std.mem.Allocator,
    mesh_list: *std.ArrayList(MeshType),
    meshes: []const MeshType,
    group: MeshGroup,
) !MeshGroup {
    const mesh_dups = try duplicateMeshGroup(
        MeshType,
        allocator,
        meshes,
        group,
    );
    const group_start = mesh_list.items.len;
    try mesh_list.appendSlice(allocator, mesh_dups);
    return meshGroupSpan(group_start, mesh_dups.len);
}

pub fn boundsForCoords(coords: *const meshio.Coords) Bounds3D {
    std.debug.assert(coords.mat.rows_num > 0);

    var bounds = Bounds3D{
        .min = .{
            std.math.inf(F),
            std.math.inf(F),
            std.math.inf(F),
        },
        .max = .{
            -std.math.inf(F),
            -std.math.inf(F),
            -std.math.inf(F),
        },
        .center = .{ 0.0, 0.0, 0.0 },
        .extent = .{ 0.0, 0.0, 0.0 },
    };

    for (0..coords.mat.rows_num) |nn| {
        for (0..3) |dd| {
            const coord_val = coords.mat.get(nn, dd);
            bounds.min[dd] = @min(bounds.min[dd], coord_val);
            bounds.max[dd] = @max(bounds.max[dd], coord_val);
        }
    }

    for (0..3) |dd| {
        bounds.center[dd] = 0.5 * (bounds.min[dd] + bounds.max[dd]);
        bounds.extent[dd] = bounds.max[dd] - bounds.min[dd];
    }

    return bounds;
}

pub fn boundsForMeshes(meshes: anytype) Bounds3D {
    std.debug.assert(meshes.len > 0);

    var bounds = boundsForCoords(&meshes[0].coords);
    for (meshes[1..]) |mesh| {
        const mesh_bounds = boundsForCoords(&mesh.coords);
        bounds = mergeBounds(bounds, mesh_bounds);
    }

    return bounds;
}

pub fn boundsForMeshGroup(
    meshes: anytype,
    group: MeshGroup,
) Bounds3D {
    const mesh_num = meshGroupLen(group);
    std.debug.assert(mesh_num > 0);
    validateMeshGroup(meshes.len, group);

    var bounds = boundsForCoords(&meshes[groupMeshIdx(group, 0)].coords);
    for (1..mesh_num) |nn| {
        const mesh_idx = groupMeshIdx(group, nn);
        const mesh_bounds = boundsForCoords(&meshes[mesh_idx].coords);
        bounds = mergeBounds(bounds, mesh_bounds);
    }

    return bounds;
}

pub fn meanCenter(coords: *const meshio.Coords) vec.Vec3f {
    var center_world = vec.Vec3f.initZeros();
    const coords_num = coords.mat.rows_num;

    for (0..coords_num) |nn| {
        center_world.slice[0] += coords.mat.get(nn, 0);
        center_world.slice[1] += coords.mat.get(nn, 1);
        center_world.slice[2] += coords.mat.get(nn, 2);
    }

    return center_world.mulScal(
        1.0 / @as(F, @floatFromInt(coords_num)),
    );
}

pub fn boundsCenter(coords: *const meshio.Coords) vec.Vec3f {
    const bounds = boundsForCoords(coords);
    return vec.initVec3(
        F,
        bounds.center[0],
        bounds.center[1],
        bounds.center[2],
    );
}

pub fn boundsCenterOverMeshes(meshes: anytype) vec.Vec3f {
    const bounds = boundsForMeshes(meshes);
    return vec.initVec3(
        F,
        bounds.center[0],
        bounds.center[1],
        bounds.center[2],
    );
}

pub fn extentInRotatedFrame(
    cam_rot: rotation.Rotation,
    coords_world: *const meshio.Coords,
) [2]F {
    std.debug.assert(coords_world.mat.rows_num > 0);

    const world_to_cam_mat = matrix.Mat33Ops.inv(F, cam_rot.matrix);
    var coord_cam = world_to_cam_mat.mulVec(coords_world.getVec3(0));
    var bb_max = [_]F{ coord_cam.get(0), coord_cam.get(1) };
    var bb_min = [_]F{ coord_cam.get(0), coord_cam.get(1) };

    for (1..coords_world.mat.rows_num) |nn| {
        coord_cam = world_to_cam_mat.mulVec(coords_world.getVec3(nn));

        bb_max[0] = @max(bb_max[0], coord_cam.get(0));
        bb_min[0] = @min(bb_min[0], coord_cam.get(0));
        bb_max[1] = @max(bb_max[1], coord_cam.get(1));
        bb_min[1] = @min(bb_min[1], coord_cam.get(1));
    }

    return .{
        bb_max[0] - bb_min[0],
        bb_max[1] - bb_min[1],
    };
}

pub fn extentInRotatedFrameOverMeshes(
    cam_rot: rotation.Rotation,
    meshes: anytype,
) [2]F {
    std.debug.assert(meshes.len > 0);

    const world_to_cam_mat = matrix.Mat33Ops.inv(F, cam_rot.matrix);
    var is_first = true;
    var bb_max = [_]F{ 0.0, 0.0 };
    var bb_min = [_]F{ 0.0, 0.0 };

    for (meshes) |mesh| {
        for (0..mesh.coords.mat.rows_num) |nn| {
            const coord_cam = world_to_cam_mat.mulVec(
                mesh.coords.getVec3(nn),
            );
            if (is_first) {
                bb_max = .{ coord_cam.get(0), coord_cam.get(1) };
                bb_min = .{ coord_cam.get(0), coord_cam.get(1) };
                is_first = false;
            } else {
                bb_max[0] = @max(bb_max[0], coord_cam.get(0));
                bb_min[0] = @min(bb_min[0], coord_cam.get(0));
                bb_max[1] = @max(bb_max[1], coord_cam.get(1));
                bb_min[1] = @min(bb_min[1], coord_cam.get(1));
            }
        }
    }

    return .{
        bb_max[0] - bb_min[0],
        bb_max[1] - bb_min[1],
    };
}

pub fn transformCoords(
    transform: matrix.Mat44f,
    coords: *meshio.Coords,
) void {
    for (0..coords.mat.rows_num) |nn| {
        const coord_world = coords.getVec3(nn);
        const coord_out = matrix.Mat44Ops.mulVec3(F, transform, coord_world);
        coords.mat.set(nn, 0, coord_out.get(0));
        coords.mat.set(nn, 1, coord_out.get(1));
        coords.mat.set(nn, 2, coord_out.get(2));
    }
}

pub fn moveMesh(
    transform: matrix.Mat44f,
    mesh: anytype,
) void {
    transformCoords(transform, &mesh.coords);
}

pub fn moveMeshGroup(
    transform: matrix.Mat44f,
    meshes: anytype,
    group: MeshGroup,
) void {
    validateMeshGroup(meshes.len, group);
    const mesh_num = meshGroupLen(group);
    for (0..mesh_num) |nn| {
        const mesh_idx = groupMeshIdx(group, nn);
        moveMesh(transform, &meshes[mesh_idx]);
    }
}

pub fn translateCoords(
    coords: *meshio.Coords,
    translation: [3]F,
) void {
    for (0..coords.mat.rows_num) |nn| {
        coords.mat.set(nn, 0, coords.mat.get(nn, 0) + translation[0]);
        coords.mat.set(nn, 1, coords.mat.get(nn, 1) + translation[1]);
        coords.mat.set(nn, 2, coords.mat.get(nn, 2) + translation[2]);
    }
}

pub fn translateMesh(
    translation: [3]F,
    mesh: anytype,
) void {
    translateCoords(&mesh.coords, translation);
}

pub fn translateMeshes(
    meshes: anytype,
    translation: [3]F,
) void {
    for (meshes) |*mesh| {
        translateMesh(translation, mesh);
    }
}

pub fn translateMeshGroup(
    meshes: anytype,
    group: MeshGroup,
    translation: [3]F,
) void {
    validateMeshGroup(meshes.len, group);
    const mesh_num = meshGroupLen(group);
    for (0..mesh_num) |nn| {
        const mesh_idx = groupMeshIdx(group, nn);
        translateMesh(translation, &meshes[mesh_idx]);
    }
}

pub fn centerCoordsAt(
    coords: *meshio.Coords,
    targ_center: [3]F,
) void {
    const bounds = boundsForCoords(coords);
    translateCoords(coords, .{
        targ_center[0] - bounds.center[0],
        targ_center[1] - bounds.center[1],
        targ_center[2] - bounds.center[2],
    });
}

pub fn centerMeshAt(
    targ_center: [3]F,
    mesh: anytype,
) void {
    centerCoordsAt(&mesh.coords, targ_center);
}

pub fn centerMeshGroupAt(
    meshes: anytype,
    group: MeshGroup,
    targ_center: [3]F,
) void {
    const group_bounds = boundsForMeshGroup(meshes, group);
    translateMeshGroup(meshes, group, .{
        targ_center[0] - group_bounds.center[0],
        targ_center[1] - group_bounds.center[1],
        targ_center[2] - group_bounds.center[2],
    });
}

pub fn positionCoordsRelative(
    coords: *meshio.Coords,
    reference_bounds: Bounds3D,
    reference_anchor: [3]AxisAnchor,
    moving_anchor: [3]AxisAnchor,
    offset: [3]F,
) void {
    const moving_bounds = boundsForCoords(coords);
    var translation = [3]F{ 0.0, 0.0, 0.0 };

    for (0..3) |dd| {
        const reference_pos = anchorValue(
            reference_bounds,
            dd,
            reference_anchor[dd],
        );
        const moving_pos = anchorValue(
            moving_bounds,
            dd,
            moving_anchor[dd],
        );
        translation[dd] = reference_pos + offset[dd] - moving_pos;
    }

    translateCoords(coords, translation);
}

pub fn positionMeshRelative(
    reference_mesh: anytype,
    moving_mesh: anytype,
    reference_anchor: [3]AxisAnchor,
    moving_anchor: [3]AxisAnchor,
    offset: [3]F,
) void {
    const reference_bounds = boundsForCoords(&reference_mesh.coords);
    positionCoordsRelative(
        &moving_mesh.coords,
        reference_bounds,
        reference_anchor,
        moving_anchor,
        offset,
    );
}

pub fn alignMeshGroupToMeshGroup(
    meshes: anytype,
    fixed_group: MeshGroup,
    moving_group: MeshGroup,
    fixed_anchor: [3]AxisAnchor,
    moving_anchor: [3]AxisAnchor,
    offset: [3]F,
) void {
    const fixed_bounds = boundsForMeshGroup(meshes, fixed_group);
    const moving_bounds = boundsForMeshGroup(meshes, moving_group);
    var translation = [3]F{ 0.0, 0.0, 0.0 };

    for (0..3) |dd| {
        const fixed_pos = anchorValue(
            fixed_bounds,
            dd,
            fixed_anchor[dd],
        );
        const moving_pos = anchorValue(
            moving_bounds,
            dd,
            moving_anchor[dd],
        );
        translation[dd] = fixed_pos + offset[dd] - moving_pos;
    }

    translateMeshGroup(meshes, moving_group, translation);
}

pub fn overlapMeshGroupBounds(
    meshes: anytype,
    fixed_group: MeshGroup,
    moving_group: MeshGroup,
    spec: BoundsOverlapSpec,
) void {
    const fixed_bounds = boundsForMeshGroup(meshes, fixed_group);
    const moving_bounds = boundsForMeshGroup(meshes, moving_group);
    var translation = [3]F{
        spec.extra_offset[0],
        spec.extra_offset[1],
        spec.extra_offset[2],
    };

    for (0..3) |dd| {
        if (!spec.enabled_axes[dd]) continue;

        const desired_overlap = spec.overlap_frac[dd] *
            @min(fixed_bounds.extent[dd], moving_bounds.extent[dd]);
        const center_sep_mag = 0.5 * (fixed_bounds.extent[dd] + moving_bounds.extent[dd]) - desired_overlap;
        const current_sep = moving_bounds.center[dd] - fixed_bounds.center[dd];
        const sep_sign = overlapSign(current_sep, spec.direction[dd]);
        const targ_center = fixed_bounds.center[dd] +
            sep_sign * center_sep_mag + spec.extra_offset[dd];
        translation[dd] = targ_center - moving_bounds.center[dd];
    }

    translateMeshGroup(meshes, moving_group, translation);
}

pub fn arrangeMeshesGrid(
    meshes: anytype,
    spec: GridSpec,
) void {
    var max_extent = [3]F{ 0.0, 0.0, 0.0 };

    for (meshes) |mesh| {
        const bounds = boundsForCoords(&mesh.coords);
        for (0..3) |dd| {
            max_extent[dd] = @max(max_extent[dd], bounds.extent[dd]);
        }
    }

    const stride = [3]F{
        max_extent[0] + spec.gap[0],
        max_extent[1] + spec.gap[1],
        max_extent[2] + spec.gap[2],
    };

    for (meshes, 0..) |*mesh, nn| {
        const xx = nn % spec.max_divs[0];
        const yy = (nn / spec.max_divs[0]) % spec.max_divs[1];
        const zz = nn / (spec.max_divs[0] * spec.max_divs[1]);

        centerMeshAt(.{
            @as(F, @floatFromInt(xx)) * stride[0],
            @as(F, @floatFromInt(yy)) * stride[1],
            @as(F, @floatFromInt(zz)) * stride[2],
        }, mesh);
    }
}

pub fn arrangeMeshGroupsGrid(
    meshes: anytype,
    groups: []const MeshGroup,
    spec: GridSpec,
) void {
    std.debug.assert(groups.len > 0);

    var max_extent = [3]F{ 0.0, 0.0, 0.0 };
    for (groups) |group| {
        const bounds = boundsForMeshGroup(meshes, group);
        for (0..3) |dd| {
            max_extent[dd] = @max(max_extent[dd], bounds.extent[dd]);
        }
    }

    const stride = [3]F{
        max_extent[0] + spec.gap[0],
        max_extent[1] + spec.gap[1],
        max_extent[2] + spec.gap[2],
    };

    for (groups, 0..) |group, nn| {
        const xx = nn % spec.max_divs[0];
        const yy = (nn / spec.max_divs[0]) % spec.max_divs[1];
        const zz = nn / (spec.max_divs[0] * spec.max_divs[1]);

        centerMeshGroupAt(meshes, group, .{
            @as(F, @floatFromInt(xx)) * stride[0],
            @as(F, @floatFromInt(yy)) * stride[1],
            @as(F, @floatFromInt(zz)) * stride[2],
        });
    }
}

pub fn arrangeMeshesLine(
    meshes: anytype,
    axis: usize,
    gap: F,
) void {
    std.debug.assert(axis < 3);
    std.debug.assert(meshes.len > 0);

    var cursor: F = 0.0;
    for (meshes, 0..) |*mesh, nn| {
        const mesh_bounds = boundsForCoords(&mesh.coords);
        var center_targ = mesh_bounds.center;
        if (nn == 0) {
            center_targ[axis] = 0.0;
        } else {
            center_targ[axis] = cursor + 0.5 * mesh_bounds.extent[axis];
        }
        centerMeshAt(center_targ, mesh);
        cursor = center_targ[axis] + 0.5 * mesh_bounds.extent[axis] + gap;
    }
}

pub fn arrangeMeshGroupsLine(
    meshes: anytype,
    groups: []const MeshGroup,
    axis: usize,
    gap: F,
) void {
    std.debug.assert(axis < 3);
    std.debug.assert(groups.len > 0);

    var cursor: F = 0.0;
    for (groups, 0..) |group, nn| {
        const group_bounds = boundsForMeshGroup(meshes, group);
        var center_targ = group_bounds.center;
        if (nn == 0) {
            center_targ[axis] = 0.0;
        } else {
            center_targ[axis] = cursor + 0.5 * group_bounds.extent[axis];
        }
        centerMeshGroupAt(meshes, group, center_targ);
        cursor = center_targ[axis] + 0.5 * group_bounds.extent[axis] + gap;
    }
}

pub fn arrangeMeshesRadial(
    meshes: anytype,
    spec: RadialSpec,
) void {
    if (meshes.len == 0) return;

    const angle_step = 2.0 * std.math.pi /
        @as(F, @floatFromInt(meshes.len));

    for (meshes, 0..) |*mesh, nn| {
        const angle = spec.angle_offset_rad +
            @as(F, @floatFromInt(nn)) * angle_step;
        centerMeshAt(calcRadialCenter(spec, angle), mesh);
    }
}

pub fn arrangeMeshGroupsRadial(
    meshes: anytype,
    groups: []const MeshGroup,
    spec: RadialSpec,
) void {
    if (groups.len == 0) return;

    const angle_step = 2.0 * std.math.pi /
        @as(F, @floatFromInt(groups.len));

    for (groups, 0..) |group, nn| {
        const angle = spec.angle_offset_rad +
            @as(F, @floatFromInt(nn)) * angle_step;
        centerMeshGroupAt(meshes, group, calcRadialCenter(spec, angle));
    }
}

fn anchorValue(
    bounds: Bounds3D,
    axis: usize,
    anchor: AxisAnchor,
) F {
    return switch (anchor) {
        .min => bounds.min[axis],
        .center => bounds.center[axis],
        .max => bounds.max[axis],
    };
}

fn calcRadialCenter(
    spec: RadialSpec,
    angle: F,
) [3]F {
    var targ_center = spec.center;
    switch (spec.axis) {
        .x => {
            targ_center[1] += spec.radius * std.math.cos(angle);
            targ_center[2] += spec.radius * std.math.sin(angle);
        },
        .y => {
            targ_center[0] += spec.radius * std.math.cos(angle);
            targ_center[2] += spec.radius * std.math.sin(angle);
        },
        .z => {
            targ_center[0] += spec.radius * std.math.cos(angle);
            targ_center[1] += spec.radius * std.math.sin(angle);
        },
    }
    return targ_center;
}

fn groupMeshIdx(
    group: MeshGroup,
    group_idx: usize,
) usize {
    return switch (group) {
        .span => |span| span.mesh_start + group_idx,
        .indices => |mesh_indices| mesh_indices[group_idx],
    };
}

fn mergeBounds(
    bounds_a: Bounds3D,
    bounds_b: Bounds3D,
) Bounds3D {
    var bounds = Bounds3D{
        .min = undefined,
        .max = undefined,
        .center = undefined,
        .extent = undefined,
    };

    for (0..3) |dd| {
        bounds.min[dd] = @min(bounds_a.min[dd], bounds_b.min[dd]);
        bounds.max[dd] = @max(bounds_a.max[dd], bounds_b.max[dd]);
        bounds.center[dd] = 0.5 * (bounds.min[dd] + bounds.max[dd]);
        bounds.extent[dd] = bounds.max[dd] - bounds.min[dd];
    }

    return bounds;
}

fn overlapSign(
    current_sep: F,
    direction: OverlapDirection,
) F {
    return switch (direction) {
        .negative => -1.0,
        .positive => 1.0,
        .current => if (current_sep < 0.0) -1.0 else 1.0,
    };
}

fn validateMeshGroup(
    meshes_len: usize,
    group: MeshGroup,
) void {
    switch (group) {
        .span => |span| {
            std.debug.assert(span.mesh_len > 0);
            std.debug.assert(span.mesh_start + span.mesh_len <= meshes_len);
        },
        .indices => |mesh_indices| {
            std.debug.assert(mesh_indices.len > 0);
            for (mesh_indices) |mesh_idx| {
                std.debug.assert(mesh_idx < meshes_len);
            }
        },
    }
}

// --------------------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------------------

test "boundsForCoords and meanCenter" {
    var coords = try meshio.Coords.initAlloc(std.testing.allocator, 3);
    defer std.testing.allocator.free(coords.mem);

    coords.mat.set(0, 0, -1.0);
    coords.mat.set(0, 1, 1.0);
    coords.mat.set(0, 2, 2.0);
    coords.mat.set(1, 0, 3.0);
    coords.mat.set(1, 1, -2.0);
    coords.mat.set(1, 2, 4.0);
    coords.mat.set(2, 0, 2.0);
    coords.mat.set(2, 1, 5.0);
    coords.mat.set(2, 2, -3.0);

    const bounds = boundsForCoords(&coords);
    try std.testing.expectEqualDeep([3]F{ -1.0, -2.0, -3.0 }, bounds.min);
    try std.testing.expectEqualDeep([3]F{ 3.0, 5.0, 4.0 }, bounds.max);
    try std.testing.expectEqualDeep([3]F{ 1.0, 1.5, 0.5 }, bounds.center);
    try std.testing.expectEqualDeep([3]F{ 4.0, 7.0, 7.0 }, bounds.extent);

    const center_mean = meanCenter(&coords);
    try std.testing.expectApproxEqAbs(4.0 / 3.0, center_mean.get(0), 1e-12);
    try std.testing.expectApproxEqAbs(4.0 / 3.0, center_mean.get(1), 1e-12);
    try std.testing.expectApproxEqAbs(1.0, center_mean.get(2), 1e-12);
}

test "boundsForMeshGroup supps explicit indices" {
    const TestMesh = struct {
        coords: meshio.Coords,
    };

    var mesh_coords = [_]meshio.Coords{
        try meshio.Coords.initAlloc(std.testing.allocator, 1),
        try meshio.Coords.initAlloc(std.testing.allocator, 1),
        try meshio.Coords.initAlloc(std.testing.allocator, 1),
    };
    defer {
        for (mesh_coords) |coords| {
            std.testing.allocator.free(coords.mem);
        }
    }

    mesh_coords[0].mat.set(0, 0, -2.0);
    mesh_coords[0].mat.set(0, 1, 0.0);
    mesh_coords[0].mat.set(0, 2, 0.0);
    mesh_coords[1].mat.set(0, 0, 4.0);
    mesh_coords[1].mat.set(0, 1, 1.0);
    mesh_coords[1].mat.set(0, 2, 0.0);
    mesh_coords[2].mat.set(0, 0, 1.0);
    mesh_coords[2].mat.set(0, 1, 5.0);
    mesh_coords[2].mat.set(0, 2, 0.0);

    const meshes = [_]TestMesh{
        .{ .coords = mesh_coords[0] },
        .{ .coords = mesh_coords[1] },
        .{ .coords = mesh_coords[2] },
    };
    const mesh_indices = [_]usize{ 0, 2 };
    const bounds = boundsForMeshGroup(
        meshes[0..],
        .{ .indices = mesh_indices[0..] },
    );

    try std.testing.expectEqualDeep([3]F{ -2.0, 0.0, 0.0 }, bounds.min);
    try std.testing.expectEqualDeep([3]F{ 1.0, 5.0, 0.0 }, bounds.max);
}

test "translateCoords and centerCoordsAt" {
    var coords = try meshio.Coords.initAlloc(std.testing.allocator, 2);
    defer std.testing.allocator.free(coords.mem);

    coords.mat.set(0, 0, 0.0);
    coords.mat.set(0, 1, 0.0);
    coords.mat.set(0, 2, 0.0);
    coords.mat.set(1, 0, 2.0);
    coords.mat.set(1, 1, 4.0);
    coords.mat.set(1, 2, 6.0);

    translateCoords(&coords, .{ 1.0, -2.0, 3.0 });
    try std.testing.expectApproxEqAbs(1.0, coords.mat.get(0, 0), 1e-12);
    try std.testing.expectApproxEqAbs(2.0, coords.mat.get(1, 1), 1e-12);
    try std.testing.expectApproxEqAbs(9.0, coords.mat.get(1, 2), 1e-12);

    centerCoordsAt(&coords, .{ 0.0, 0.0, 0.0 });
    const bounds = boundsForCoords(&coords);
    try std.testing.expectEqualDeep([3]F{ 0.0, 0.0, 0.0 }, bounds.center);
}

test "overlapMeshGroupBounds overlaps selected axes" {
    const TestMesh = struct {
        coords: meshio.Coords,
    };

    var left_coords = try meshio.Coords.initAlloc(std.testing.allocator, 2);
    defer std.testing.allocator.free(left_coords.mem);
    var right_coords = try meshio.Coords.initAlloc(std.testing.allocator, 2);
    defer std.testing.allocator.free(right_coords.mem);

    left_coords.mat.set(0, 0, -1.0);
    left_coords.mat.set(0, 1, -1.0);
    left_coords.mat.set(0, 2, 0.0);
    left_coords.mat.set(1, 0, 1.0);
    left_coords.mat.set(1, 1, 1.0);
    left_coords.mat.set(1, 2, 0.0);

    right_coords.mat.set(0, 0, -1.0);
    right_coords.mat.set(0, 1, -1.0);
    right_coords.mat.set(0, 2, 5.0);
    right_coords.mat.set(1, 0, 1.0);
    right_coords.mat.set(1, 1, 1.0);
    right_coords.mat.set(1, 2, 5.0);

    var meshes = [_]TestMesh{
        .{ .coords = left_coords },
        .{ .coords = right_coords },
    };

    overlapMeshGroupBounds(
        meshes[0..],
        meshGroupSingle(0),
        meshGroupSingle(1),
        .{
            .overlap_frac = .{ 0.5, 0.5, 0.0 },
            .enabled_axes = .{ true, true, false },
            .direction = .{ .positive, .negative, .current },
        },
    );

    const left_bounds = boundsForMeshGroup(meshes[0..], meshGroupSingle(0));
    const right_bounds = boundsForMeshGroup(meshes[0..], meshGroupSingle(1));
    const overlap_x = 0.5 * (left_bounds.extent[0] + right_bounds.extent[0]) - @abs(right_bounds.center[0] - left_bounds.center[0]);
    const overlap_y = 0.5 * (left_bounds.extent[1] + right_bounds.extent[1]) - @abs(right_bounds.center[1] - left_bounds.center[1]);

    try std.testing.expectApproxEqAbs(1.0, overlap_x, 1e-12);
    try std.testing.expectApproxEqAbs(1.0, overlap_y, 1e-12);
    try std.testing.expectApproxEqAbs(5.0, right_bounds.center[2], 1e-12);
}

test "transformCoords applies affine transform" {
    var coords = try meshio.Coords.initAlloc(std.testing.allocator, 1);
    defer std.testing.allocator.free(coords.mem);

    coords.mat.set(0, 0, 1.0);
    coords.mat.set(0, 1, 2.0);
    coords.mat.set(0, 2, 3.0);

    var transform = matrix.Mat44f.initIdentity();
    transform.set(0, 0, 2.0);
    transform.set(1, 1, 3.0);
    transform.set(2, 2, 4.0);
    transform.set(0, 3, 5.0);
    transform.set(1, 3, 6.0);
    transform.set(2, 3, 7.0);

    transformCoords(transform, &coords);
    try std.testing.expectApproxEqAbs(7.0, coords.mat.get(0, 0), 1e-12);
    try std.testing.expectApproxEqAbs(12.0, coords.mat.get(0, 1), 1e-12);
    try std.testing.expectApproxEqAbs(19.0, coords.mat.get(0, 2), 1e-12);
}
