// --------------------------------------------------------------------------
// Float texture shader regression coverage for the all suite.
// --------------------------------------------------------------------------
const std = @import("std");
const buildconfig = @import("../riley/zig/buildconfig.zig");
const ndarray = @import("../riley/zig/ndarray.zig");
const shaderops = @import("../riley/zig/shaderops.zig");
const texops = @import("../riley/zig/textureops.zig");

const F = buildconfig.F;

test "float texture shader variants preserve fractional texels" {
    const allocator = std.testing.allocator;

    var grey = try texops.Tex(F, 1).init(allocator, 2, 2);
    defer grey.deinit(allocator);
    grey.setVal(0, 0, 0, 0.125);
    grey.setVal(0, 0, 1, 0.375);
    grey.setVal(0, 1, 0, 0.625);
    grey.setVal(0, 1, 1, 0.875);

    var rgb = try texops.Tex(F, 3).init(allocator, 1, 1);
    defer rgb.deinit(allocator);
    rgb.setVal(0, 0, 0, 0.125);
    rgb.setVal(1, 0, 0, 0.5);
    rgb.setVal(2, 0, 0, 0.875);

    const sampled = texops.sampScal(1, .{ .sample = .linear }, grey, 0.5, 0.5);
    try std.testing.expectEqual(@as(F, 0.5), sampled[0]);

    var uvs = try ndarray.NDArray(F).initFlat(allocator, &.{ 3, 2 });
    defer uvs.deinit(allocator);
    defer allocator.free(uvs.slice);

    const grey_shader: shaderops.ShaderInput = .{ .tex_f = .{
        .uvs = uvs,
        .tex = grey,
        .samp_cfg = .{ .sample = .linear },
    } };
    const rgb_shader: shaderops.ShaderInput = .{ .tex_rgb_f = .{
        .uvs = uvs,
        .tex = rgb,
        .samp_cfg = .{ .sample = .nearest },
    } };

    switch (grey_shader) {
        .tex_f => |shader| try std.testing.expectEqual(@as(F, 0.125), shader.tex.getVal(0, 0, 0)),
        else => unreachable,
    }
    switch (rgb_shader) {
        .tex_rgb_f => |shader| try std.testing.expectEqual(@as(F, 0.875), shader.tex.getVal(2, 0, 0)),
        else => unreachable,
    }
}

test "Lanczos2 supports direct and LUT sampling modes" {
    const allocator = std.testing.allocator;

    var texture = try texops.Tex(F, 1).init(allocator, 5, 5);
    defer texture.deinit(allocator);
    for (0..5) |row| {
        for (0..5) |col| {
            const texel = @as(F, @floatFromInt(row * 5 + col));
            texture.setVal(0, row, col, texel);
        }
    }

    const configs = [_]texops.TextureSampleConfig{
        .{ .sample = .lanczos2, .mode = .direct },
        .{ .sample = .lanczos2, .mode = .lut },
        .{ .sample = .lanczos2, .mode = .lut_lerp },
    };
    const u: F = 0.375;
    const v: F = 0.625;
    const direct = texops.sampScal(1, configs[0], texture, u, v)[0];
    const abs_tol: F = 2e-2;

    try std.testing.expectApproxEqAbs(@as(F, 1.0), texops.lanczos2Coeff(0.0), abs_tol);
    try std.testing.expectApproxEqAbs(@as(F, 0.0), texops.lanczos2Coeff(2.0), abs_tol);

    inline for (configs) |config| {
        const scalar = texops.sampScal(1, config, texture, u, v)[0];
        const one_lane = texops.sampOneLane(1, config, texture, u, v)[0];
        const wide = texops.sampWide(
            1,
            config,
            texture,
            @splat(u),
            @splat(v),
        );
        const wide_vals: [buildconfig.SimdWidth]F = wide[0];

        try std.testing.expectApproxEqAbs(direct, scalar, abs_tol);
        try std.testing.expectApproxEqAbs(scalar, one_lane, abs_tol);
        inline for (wide_vals) |wide_val| {
            try std.testing.expectApproxEqAbs(scalar, wide_val, abs_tol);
        }
    }
}
