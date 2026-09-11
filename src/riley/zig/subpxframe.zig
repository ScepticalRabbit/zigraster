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
const MatSlice = @import("matslice.zig").MatSlice;

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

pub const SubpxFrameDomain = struct {
    fields_num: u8,
    image_w_subpx: usize,
    image_h_subpx: usize,
    halo_subpx: usize,
    storage_w_subpx: usize,
    storage_h_subpx: usize,

    pub fn init(
        fields_num: u8,
        pixels_num: [2]u32,
        sub_samp: u32,
        halo_subpx: usize,
    ) !SubpxFrameDomain {
        const image_w_subpx = try std.math.mul(
            usize,
            pixels_num[0],
            sub_samp,
        );
        const image_h_subpx = try std.math.mul(
            usize,
            pixels_num[1],
            sub_samp,
        );
        const halo_twice = try std.math.mul(usize, halo_subpx, 2);

        return .{
            .fields_num = fields_num,
            .image_w_subpx = image_w_subpx,
            .image_h_subpx = image_h_subpx,
            .halo_subpx = halo_subpx,
            .storage_w_subpx = try std.math.add(
                usize,
                image_w_subpx,
                halo_twice,
            ),
            .storage_h_subpx = try std.math.add(
                usize,
                image_h_subpx,
                halo_twice,
            ),
        };
    }
};

pub const SubpxTarget = struct {
    domain: SubpxFrameDomain,
    image: MatSlice(F),
    global_subx_min: i32,
    global_suby_min: i32,

    pub fn init(
        outer_alloc: std.mem.Allocator,
        domain: SubpxFrameDomain,
        global_subx_min: i32,
        global_suby_min: i32,
        background_value: F,
    ) !SubpxTarget {
        const samples_num = try std.math.mul(
            usize,
            domain.storage_w_subpx,
            domain.storage_h_subpx,
        );
        const total_num = try std.math.mul(
            usize,
            samples_num,
            domain.fields_num,
        );
        const image_mem = try outer_alloc.alloc(F, total_num);
        errdefer outer_alloc.free(image_mem);
        @memset(image_mem, background_value);

        return .{
            .domain = domain,
            .image = MatSlice(F).init(
                image_mem,
                domain.fields_num,
                samples_num,
            ),
            .global_subx_min = global_subx_min,
            .global_suby_min = global_suby_min,
        };
    }

    pub fn deinit(
        self: *SubpxTarget,
        outer_alloc: std.mem.Allocator,
    ) void {
        outer_alloc.free(self.image.slice);
        self.* = undefined;
    }

    pub inline fn flatIndex(
        self: *const SubpxTarget,
        global_subx: i32,
        global_suby: i32,
    ) usize {
        const local_subx = global_subx - self.global_subx_min;
        const local_suby = global_suby - self.global_suby_min;
        std.debug.assert(local_subx >= 0);
        std.debug.assert(local_suby >= 0);
        const local_x: usize = @intCast(local_subx);
        const local_y: usize = @intCast(local_suby);
        std.debug.assert(local_x < self.domain.storage_w_subpx);
        std.debug.assert(local_y < self.domain.storage_h_subpx);
        return local_y * self.domain.storage_w_subpx + local_x;
    }
};

pub const SubpxStripe = struct {
    target: SubpxTarget,
    core_suby_min: i32,
    core_suby_max: i32,

    pub fn init(
        outer_alloc: std.mem.Allocator,
        fields_num: u8,
        image_w_subpx: usize,
        core_suby_min: i32,
        core_suby_max: i32,
        halo_subpx_x: usize,
        halo_subpx_y: usize,
        background_value: F,
    ) !SubpxStripe {
        std.debug.assert(core_suby_min < core_suby_max);
        const core_h_subpx: usize = @intCast(core_suby_max - core_suby_min);
        const domain = try SubpxFrameDomain.init(
            fields_num,
            .{ @intCast(image_w_subpx), @intCast(core_h_subpx) },
            1,
            0,
        );
        const storage_w_subpx = try std.math.add(
            usize,
            domain.image_w_subpx,
            try std.math.mul(usize, halo_subpx_x, 2),
        );
        const storage_h_subpx = try std.math.add(
            usize,
            domain.image_h_subpx,
            try std.math.mul(usize, halo_subpx_y, 2),
        );
        const stripe_domain = SubpxFrameDomain{
            .fields_num = fields_num,
            .image_w_subpx = domain.image_w_subpx,
            .image_h_subpx = domain.image_h_subpx,
            .halo_subpx = 0,
            .storage_w_subpx = storage_w_subpx,
            .storage_h_subpx = storage_h_subpx,
        };

        return .{
            .target = try SubpxTarget.init(
                outer_alloc,
                stripe_domain,
                -@as(i32, @intCast(halo_subpx_x)),
                core_suby_min - @as(i32, @intCast(halo_subpx_y)),
                background_value,
            ),
            .core_suby_min = core_suby_min,
            .core_suby_max = core_suby_max,
        };
    }

    pub fn deinit(
        self: *SubpxStripe,
        outer_alloc: std.mem.Allocator,
    ) void {
        self.target.deinit(outer_alloc);
        self.* = undefined;
    }

    pub fn reset(
        self: *SubpxStripe,
        core_suby_min: i32,
        core_suby_max: i32,
        halo_subpx_y: usize,
        background_value: F,
    ) void {
        const core_h_subpx: usize = @intCast(core_suby_max - core_suby_min);
        std.debug.assert(core_suby_min < core_suby_max);
        std.debug.assert(
            core_h_subpx + 2 * halo_subpx_y <=
                self.target.domain.storage_h_subpx,
        );

        @memset(self.target.image.slice, background_value);
        self.target.global_suby_min = core_suby_min - @as(
            i32,
            @intCast(halo_subpx_y),
        );
        self.core_suby_min = core_suby_min;
        self.core_suby_max = core_suby_max;
    }
};

// --------------------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------------------

test "SubpxTarget maps global coordinates through outer halo" {
    const domain = try SubpxFrameDomain.init(1, .{ 3, 2 }, 4, 2);
    var target = try SubpxTarget.init(
        std.testing.allocator,
        domain,
        -2,
        -2,
        0.0,
    );
    defer target.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 0), target.flatIndex(-2, -2));
    try std.testing.expectEqual(@as(usize, 2), target.flatIndex(0, -2));
    try std.testing.expectEqual(
        domain.storage_w_subpx + 2,
        target.flatIndex(0, -1),
    );
}

test "SubpxStripe covers its core plus only stripe-edge halos" {
    var stripe = try SubpxStripe.init(
        std.testing.allocator,
        1,
        24,
        8,
        16,
        2,
        3,
        0.0,
    );
    defer stripe.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(i32, -2), stripe.target.global_subx_min);
    try std.testing.expectEqual(@as(i32, 5), stripe.target.global_suby_min);
    try std.testing.expectEqual(@as(usize, 28), stripe.target.domain.storage_w_subpx);
    try std.testing.expectEqual(@as(usize, 14), stripe.target.domain.storage_h_subpx);
    try std.testing.expectEqual(@as(i32, 8), stripe.core_suby_min);
    try std.testing.expectEqual(@as(i32, 16), stripe.core_suby_max);
}

test "SubpxStripe reset reuses storage for a shorter final stripe" {
    var stripe = try SubpxStripe.init(
        std.testing.allocator,
        1,
        12,
        0,
        8,
        1,
        2,
        0.0,
    );
    defer stripe.deinit(std.testing.allocator);

    stripe.target.image.slice[0] = 3.0;
    stripe.reset(8, 12, 2, -1.0);

    try std.testing.expectEqual(@as(i32, 6), stripe.target.global_suby_min);
    try std.testing.expectEqual(@as(i32, 8), stripe.core_suby_min);
    try std.testing.expectEqual(@as(i32, 12), stripe.core_suby_max);
    try std.testing.expectEqual(@as(F, -1.0), stripe.target.image.slice[0]);
}
