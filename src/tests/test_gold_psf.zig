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
const F = buildconfig.F;
const Timestamp = std.Io.Clock.Timestamp;
const common = @import("../dev_support/tests.zig");
const suite = @import("../dev_support/psfsuite.zig");
const tcfg = @import("../dev_support/testconfig.zig");

const PSF_REL_TOL: F = if (F == f32) 1.0e-6 else 1.0e-9;
const PSF_ABS_TOL: F = if (F == f32) 1.0e-6 else 1.0e-9;

pub fn main() !void {
    std.debug.print(
        "Please use 'zig test -O ReleaseSafe src/test_gold_psf.zig' to run this test suite.\n",
        .{},
    );
}

test "Gold PSF Suite" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    std.debug.print("Running Gold PSF Tests...\n", .{});
    const suite_start = Timestamp.now(io, .awake);

    for (suite.distortion_cases) |distortion_case| {
        for (distortion_case.mesh_types) |mesh_type| {
            for (suite.shader_cases) |shader_case| {
                for (suite.psf_cases) |psf_case| {
                    _ = arena.reset(.retain_capacity);
                    const render_case = suite.RenderCase{
                        .distortion_case_name = distortion_case.name,
                        .mesh_type = mesh_type,
                        .shader_case = shader_case,
                        .psf_case = psf_case,
                    };
                    const case_dir_name = try suite.caseDirName(aa, render_case);
                    const gold_dir = try std.fmt.allocPrint(
                        aa,
                        "{s}/{s}",
                        .{ suite.gold_root, case_dir_name },
                    );
                    const result = try suite.renderCase(allocator, io, render_case, null);
                    defer {
                        allocator.free(result.slice);
                        var result_mut = result;
                        result_mut.deinit(allocator);
                    }

                    const frames_num = if (result.dims.len == 5) result.dims[1] else result.dims[0];
                    var first_err: ?anyerror = null;
                    for (0..frames_num) |frame_idx| {
                        const gold_path = try common.findGoldPath(
                            aa,
                            io,
                            gold_dir,
                            0,
                            frame_idx,
                            0,
                            false,
                        );

                        common.compareNDArrayToGold(
                            allocator,
                            io,
                            &result,
                            0,
                            frame_idx,
                            0,
                            1,
                            gold_path,
                            tcfg.REL_TOL,
                            tcfg.ABS_TOL,
                        ) catch |err| {
                            if (first_err == null) {
                                first_err = err;
                            }
                            const fail_dir_name = try std.fmt.allocPrint(
                                aa,
                                "psf_{s}",
                                .{case_dir_name},
                            );
                            try common.saveComparisonArtifactsFromResult(
                                aa,
                                io,
                                common.default_fails_root,
                                fail_dir_name,
                                &result,
                                0,
                                frame_idx,
                                0,
                                gold_path,
                                1,
                            );
                        };
                    }
                    if (first_err) |err| return err;
                }
            }
        }
    }

    const suite_end = Timestamp.now(io, .awake);
    const suite_ms = @as(
        F,
        @floatFromInt(suite_start.durationTo(suite_end).raw.nanoseconds),
    ) / 1e6;
    std.debug.print("Gold PSF Test Suite took {d:.3} ms\n", .{suite_ms});
}

test "PSF isotropic gaussian separable and non-separable agree" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const sep_case = suite.RenderCase{
        .distortion_case_name = "distort_bulge",
        .mesh_type = .tri6,
        .shader_case = suite.shader_cases[0],
        .psf_case = .{
            .tag = "gaussian_sep_cmp",
            .psf = .{ .gaussian = .{
                .sigma_px = 0.6,
                .supp_rad_px = 2.0,
                .separable = .yes,
            } },
        },
    };
    const nonsep_case = suite.RenderCase{
        .distortion_case_name = "distort_bulge",
        .mesh_type = .tri6,
        .shader_case = suite.shader_cases[0],
        .psf_case = .{
            .tag = "gaussian_nonsep_cmp",
            .psf = .{ .gaussian = .{
                .sigma_px = 0.6,
                .supp_rad_px = 2.0,
                .separable = .no,
            } },
        },
    };

    const result_sep = try suite.renderCase(allocator, io, sep_case, null);
    defer {
        allocator.free(result_sep.slice);
        var result_sep_mut = result_sep;
        result_sep_mut.deinit(allocator);
    }
    const result_nonsep = try suite.renderCase(allocator, io, nonsep_case, null);
    defer {
        allocator.free(result_nonsep.slice);
        var result_nonsep_mut = result_nonsep;
        result_nonsep_mut.deinit(allocator);
    }

    try suite.expectResultsApproxEq(
        &result_sep,
        &result_nonsep,
        PSF_REL_TOL,
        PSF_ABS_TOL,
    );
}

test "PSF gaussian checker render is invariant to tile size" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const render_case = suite.RenderCase{
        .distortion_case_name = "distort_shear",
        .mesh_type = .quad8,
        .shader_case = suite.shader_cases[0],
        .psf_case = suite.psf_cases[1],
    };

    const result_small = try suite.renderCase(allocator, io, render_case, suite.tile_size_small);
    defer {
        allocator.free(result_small.slice);
        var result_small_mut = result_small;
        result_small_mut.deinit(allocator);
    }
    const result_large = try suite.renderCase(allocator, io, render_case, suite.tile_size_large);
    defer {
        allocator.free(result_large.slice);
        var result_large_mut = result_large;
        result_large_mut.deinit(allocator);
    }

    try suite.expectResultsApproxEq(
        &result_small,
        &result_large,
        PSF_REL_TOL,
        PSF_ABS_TOL,
    );
}

test "PSF anisotropic checker render is invariant to tile size" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const render_case = suite.RenderCase{
        .distortion_case_name = "distort_bulge",
        .mesh_type = .quad8,
        .shader_case = suite.shader_cases[0],
        .psf_case = suite.psf_cases[3],
    };

    const result_small = try suite.renderCase(allocator, io, render_case, suite.tile_size_small);
    defer {
        allocator.free(result_small.slice);
        var result_small_mut = result_small;
        result_small_mut.deinit(allocator);
    }
    const result_large = try suite.renderCase(allocator, io, render_case, suite.tile_size_large);
    defer {
        allocator.free(result_large.slice);
        var result_large_mut = result_large;
        result_large_mut.deinit(allocator);
    }

    try suite.expectResultsApproxEq(
        &result_small,
        &result_large,
        PSF_REL_TOL,
        PSF_ABS_TOL,
    );
}

test "PSF gaussian constant render is invariant to tile size" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const render_case = suite.RenderCase{
        .distortion_case_name = "distort_shear",
        .mesh_type = .quad4ibi,
        .shader_case = suite.shader_cases[1],
        .psf_case = suite.psf_cases[1],
    };

    const result_small = try suite.renderCase(allocator, io, render_case, suite.tile_size_small);
    defer {
        allocator.free(result_small.slice);
        var result_small_mut = result_small;
        result_small_mut.deinit(allocator);
    }
    const result_large = try suite.renderCase(allocator, io, render_case, suite.tile_size_large);
    defer {
        allocator.free(result_large.slice);
        var result_large_mut = result_large;
        result_large_mut.deinit(allocator);
    }

    try suite.expectResultsApproxEq(
        &result_small,
        &result_large,
        PSF_REL_TOL,
        PSF_ABS_TOL,
    );
}

test "PSF halo-only bypass leaves the final image bitwise unchanged" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const render_case = suite.RenderCase{
        .distortion_case_name = "distort_bulge",
        .mesh_type = .quad8,
        .shader_case = suite.shader_cases[0],
        .psf_case = suite.psf_cases[0],
    };

    const result_base = try suite.renderCase(
        allocator,
        io,
        render_case,
        suite.tile_size_small,
    );
    defer {
        allocator.free(result_base.slice);
        var result_base_mut = result_base;
        result_base_mut.deinit(allocator);
    }
    const result_halo = try suite.renderCaseWithRasterHalo(
        allocator,
        io,
        render_case,
        suite.tile_size_small,
        3,
    );
    defer {
        allocator.free(result_halo.slice);
        var result_halo_mut = result_halo;
        result_halo_mut.deinit(allocator);
    }

    try std.testing.expectEqualSlices(F, result_base.slice, result_halo.slice);
}
