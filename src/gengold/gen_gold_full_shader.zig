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
const common = @import("gen_gold_full_common.zig");
const gk = @import("../riley/zig/geometrykernels.zig");
const iio = @import("../riley/zig/imageio.zig");
const imageops = @import("../riley/zig/imageops.zig");
const meshio = @import("../riley/zig/meshio.zig");
const mo = @import("../riley/zig/meshpipeline.zig");
const orch = @import("../dev_support/orchestration.zig");
const policy = @import("../dev_support/testpolicy.zig");
const rastcfg = @import("../riley/zig/rasterconfig.zig");
const riley = @import("../riley/zig/riley.zig");
const shaderops = @import("../riley/zig/shaderops_common.zig");
const tcfg = @import("../dev_support/testconfig.zig");
const texops = @import("../riley/zig/textureops.zig");

const F = buildconfig.F;
const CameraInput = camera.CameraInput;
const MeshInput = mo.MeshInput;

pub const FullNodalCase = struct {
    is_rgb: bool,
    scaling: imageops.ScaleStrategy,
    scale_over: shaderops.ScaleOver,
    normal_type: shaderops.NormalType,
};

pub const FullTexCase = struct {
    is_rgb: bool,
    dtype: enum { u8, u16, f64 },
    normal_type: shaderops.NormalType,
};

pub const FullFuncCase = struct {
    is_rgb: bool,
    builtin: shaderops.FuncShaderBuiltin,
    coord: shaderops.FuncCoordMode,
    normal_type: shaderops.NormalType,
};

pub const FullShaderCase = union(enum) {
    nodal: FullNodalCase,
    tex: FullTexCase,
    func: FullFuncCase,

    pub fn isRgb(self: FullShaderCase) bool {
        return switch (self) {
            .nodal => |n| n.is_rgb,
            .tex => |t| t.is_rgb,
            .func => |f| f.is_rgb,
        };
    }

    pub fn formatDirName(
        self: FullShaderCase,
        allocator: std.mem.Allocator,
        elem: gk.MeshType,
    ) ![]const u8 {
        const elem_str = switch (elem) {
            .tri3opt => "tri3",
            else => @tagName(elem),
        };
        switch (self) {
            .nodal => |n| {
                const colour_str = if (n.is_rgb) "rgb" else "mono";
                const scale_str = switch (n.scaling) {
                    .none => "scalenone",
                    .auto => switch (n.scale_over) {
                        .within_frames => "scalewithin",
                        .over_frames => "scaleover",
                    },
                    else => "scalenone",
                };
                const norm_str = switch (n.normal_type) {
                    .none => "normnone",
                    .avg => "normavg",
                    .exact => "normexact",
                };
                return std.fmt.allocPrint(
                    allocator,
                    "scene0_{s}_nodal_{s}_{s}_{s}",
                    .{ elem_str, colour_str, scale_str, norm_str },
                );
            },
            .tex => |t| {
                const colour_str = if (t.is_rgb) "rgb" else "mono";
                const dtype_str = @tagName(t.dtype);
                const norm_str = switch (t.normal_type) {
                    .none => "normnone",
                    .avg => "normavg",
                    .exact => "normexact",
                };
                return std.fmt.allocPrint(
                    allocator,
                    "scene0_{s}_tex_{s}_{s}_catmull_rom_direct_{s}",
                    .{ elem_str, colour_str, dtype_str, norm_str },
                );
            },
            .func => |f| {
                const colour_str = if (f.is_rgb) "rgb" else "mono";
                const builtin_str = @tagName(f.builtin);
                const coord_str = switch (f.coord) {
                    .para => "para",
                    .uv => "uv",
                    .world_reference => "world_ref",
                    .world_deformed => "world_def",
                };
                const norm_str = switch (f.normal_type) {
                    .none => "normnone",
                    .avg => "normavg",
                    .exact => "normexact",
                };
                return std.fmt.allocPrint(
                    allocator,
                    "scene0_{s}_func_{s}_{s}_{s}_{s}",
                    .{ elem_str, colour_str, builtin_str, coord_str, norm_str },
                );
            },
        }
    }
};

pub fn buildShaderCaseMeshes(
    mesh_type: gk.MeshType,
    prep: *const common.Scene0Prepared,
    textures: *const common.FullTextures,
    case: FullShaderCase,
) [2]MeshInput {
    switch (case) {
        .nodal => |n| {
            const sphere_field = if (n.is_rgb)
                prep.sphere_rgb
            else
                prep.sphere_temp;
            const cyl_field = if (n.is_rgb)
                prep.cylinder_rgb
            else
                prep.cylinder_temp;

            const sphere_shader = shaderops.ShaderInput{
                .nodal = .{
                    .field = sphere_field,
                    .bits = 8,
                    .scaling = n.scaling,
                    .scale_over = n.scale_over,
                    .normal_type = n.normal_type,
                },
            };
            const cyl_shader = shaderops.ShaderInput{
                .nodal = .{
                    .field = cyl_field,
                    .bits = 8,
                    .scaling = n.scaling,
                    .scale_over = n.scale_over,
                    .normal_type = n.normal_type,
                },
            };

            return [_]MeshInput{
                .{
                    .mesh_type = mesh_type,
                    .coords = prep.sphere_coords,
                    .connect = prep.sphere_connect,
                    .disp = prep.sphere_disp,
                    .shader = sphere_shader,
                },
                .{
                    .mesh_type = mesh_type,
                    .coords = prep.cylinder_coords,
                    .connect = prep.cylinder_connect,
                    .disp = prep.cylinder_disp,
                    .shader = cyl_shader,
                },
            };
        },
        .tex => |t| {
            const samp_cfg = texops.TexSampConfig{
                .sample = .cubic_catmull_rom,
                .mode = .direct,
            };

            const sphere_shader: shaderops.ShaderInput = if (t.is_rgb)
                switch (t.dtype) {
                    .u8 => .{
                        .tex_rgb_u8 = .{
                            .uvs = prep.sphere_uvs.array,
                            .tex = textures.tex_u8_rgb,
                            .samp_cfg = samp_cfg,
                            .normal_type = t.normal_type,
                        },
                    },
                    .u16 => .{
                        .tex_rgb_u16 = .{
                            .uvs = prep.sphere_uvs.array,
                            .tex = textures.tex_u16_rgb,
                            .samp_cfg = samp_cfg,
                            .normal_type = t.normal_type,
                        },
                    },
                    .f64 => .{
                        .tex_rgb_f = .{
                            .uvs = prep.sphere_uvs.array,
                            .tex = textures.tex_f64_rgb,
                            .samp_cfg = samp_cfg,
                            .normal_type = t.normal_type,
                        },
                    },
                }
            else
                switch (t.dtype) {
                    .u8 => .{
                        .tex_u8 = .{
                            .uvs = prep.sphere_uvs.array,
                            .tex = textures.tex_u8_mono,
                            .samp_cfg = samp_cfg,
                            .normal_type = t.normal_type,
                        },
                    },
                    .u16 => .{
                        .tex_u16 = .{
                            .uvs = prep.sphere_uvs.array,
                            .tex = textures.tex_u16_mono,
                            .samp_cfg = samp_cfg,
                            .normal_type = t.normal_type,
                        },
                    },
                    .f64 => .{
                        .tex_f = .{
                            .uvs = prep.sphere_uvs.array,
                            .tex = textures.tex_f64_mono,
                            .samp_cfg = samp_cfg,
                            .normal_type = t.normal_type,
                        },
                    },
                };

            const cyl_shader: shaderops.ShaderInput = if (t.is_rgb)
                switch (t.dtype) {
                    .u8 => .{
                        .tex_rgb_u8 = .{
                            .uvs = prep.cylinder_uvs.array,
                            .tex = textures.tex_u8_rgb,
                            .samp_cfg = samp_cfg,
                            .normal_type = t.normal_type,
                        },
                    },
                    .u16 => .{
                        .tex_rgb_u16 = .{
                            .uvs = prep.cylinder_uvs.array,
                            .tex = textures.tex_u16_rgb,
                            .samp_cfg = samp_cfg,
                            .normal_type = t.normal_type,
                        },
                    },
                    .f64 => .{
                        .tex_rgb_f = .{
                            .uvs = prep.cylinder_uvs.array,
                            .tex = textures.tex_f64_rgb,
                            .samp_cfg = samp_cfg,
                            .normal_type = t.normal_type,
                        },
                    },
                }
            else
                switch (t.dtype) {
                    .u8 => .{
                        .tex_u8 = .{
                            .uvs = prep.cylinder_uvs.array,
                            .tex = textures.tex_u8_mono,
                            .samp_cfg = samp_cfg,
                            .normal_type = t.normal_type,
                        },
                    },
                    .u16 => .{
                        .tex_u16 = .{
                            .uvs = prep.cylinder_uvs.array,
                            .tex = textures.tex_u16_mono,
                            .samp_cfg = samp_cfg,
                            .normal_type = t.normal_type,
                        },
                    },
                    .f64 => .{
                        .tex_f = .{
                            .uvs = prep.cylinder_uvs.array,
                            .tex = textures.tex_f64_mono,
                            .samp_cfg = samp_cfg,
                            .normal_type = t.normal_type,
                        },
                    },
                };

            return [_]MeshInput{
                .{
                    .mesh_type = mesh_type,
                    .coords = prep.sphere_coords,
                    .connect = prep.sphere_connect,
                    .disp = prep.sphere_disp,
                    .shader = sphere_shader,
                },
                .{
                    .mesh_type = mesh_type,
                    .coords = prep.cylinder_coords,
                    .connect = prep.cylinder_connect,
                    .disp = prep.cylinder_disp,
                    .shader = cyl_shader,
                },
            };
        },
        .func => |f| {
            const params = shaderops.FuncShaderParams{
                .settings = switch (f.builtin) {
                    .constant => .{ .constant = .{} },
                    .linear => .{ .linear = .{} },
                    .quadratic => .{ .quadratic = .{} },
                    .sinusoidal => .{ .sinusoidal = .{} },
                    .sinusoidal_approx => .{ .sinusoidal_approx = .{} },
                    .checker => .{ .checker = .{} },
                    .checker_smooth => .{ .checker_smooth = .{} },
                    .lambertian_normal_z => .{ .lambertian_normal_z = .{} },
                    .eggbox => .{ .eggbox = .{} },
                },
            };

            const sphere_uvs = if (f.coord == .uv)
                prep.sphere_uvs.array
            else
                null;
            const sphere_func_input = shaderops.FuncInput{
                .uvs = sphere_uvs,
                .coord_mode = f.coord,
                .builtin = f.builtin,
                .params = params,
                .normal_type = f.normal_type,
            };
            const sphere_shader: shaderops.ShaderInput = if (f.is_rgb)
                .{ .func_rgb = sphere_func_input }
            else
                .{ .func = sphere_func_input };

            const cyl_uvs = if (f.coord == .uv)
                prep.cylinder_uvs.array
            else
                null;
            const cyl_func_input = shaderops.FuncInput{
                .uvs = cyl_uvs,
                .coord_mode = f.coord,
                .builtin = f.builtin,
                .params = params,
                .normal_type = f.normal_type,
            };
            const cyl_shader: shaderops.ShaderInput = if (f.is_rgb)
                .{ .func_rgb = cyl_func_input }
            else
                .{ .func = cyl_func_input };

            return [_]MeshInput{
                .{
                    .mesh_type = mesh_type,
                    .coords = prep.sphere_coords,
                    .connect = prep.sphere_connect,
                    .disp = prep.sphere_disp,
                    .shader = sphere_shader,
                },
                .{
                    .mesh_type = mesh_type,
                    .coords = prep.cylinder_coords,
                    .connect = prep.cylinder_connect,
                    .disp = prep.cylinder_disp,
                    .shader = cyl_shader,
                },
            };
        },
    }
}

pub fn generateFullShaderCase(
    allocator: std.mem.Allocator,
    io: std.Io,
    mesh_type: gk.MeshType,
    prep: *const common.Scene0Prepared,
    textures: *const common.FullTextures,
    case: FullShaderCase,
    gold_dir_root: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    const case_name = try case.formatDirName(allocator, mesh_type);
    defer allocator.free(case_name);

    const out_dir_path = try std.fmt.allocPrint(
        allocator,
        "{s}/{s}",
        .{ gold_dir_root, case_name },
    );
    defer allocator.free(out_dir_path);

    var out_dir = try orch.openDirEnsured(io, out_dir_path);
    out_dir.close(io);

    const meshes = buildShaderCaseMeshes(mesh_type, prep, textures, case);

    const render_groups = [_]riley.RenderGroupSpec{
        .{ .io = io, .workers = @max(@as(u16, 1), config.total_threads) },
    };

    const images = try riley.raster(
        allocator,
        &render_groups,
        &[_]CameraInput{prep.camera_input},
        &meshes,
        config,
        out_dir_path,
    );

    if (images) |img| {
        allocator.free(img.slice);
    }
}

pub fn generateAllFullShaderCases(
    allocator: std.mem.Allocator,
    io: std.Io,
    gold_dir_root: []const u8,
    config: rastcfg.RasterConfig,
) !void {
    var textures = try common.FullTextures.init(allocator, io);
    defer textures.deinit(allocator);

    const mesh_types = [_]gk.MeshType{
        .tri3,
        .tri6,
        .quad4,
        .quad8,
        .quad9,
    };
    const normal_types = [_]shaderops.NormalType{
        .none,
        .avg,
        .exact,
    };
    const bool_values = [_]bool{ false, true };
    const builtins = [_]shaderops.FuncShaderBuiltin{
        .constant,
        .linear,
        .quadratic,
        .sinusoidal,
        .sinusoidal_approx,
        .checker,
        .checker_smooth,
        .lambertian_normal_z,
        .eggbox,
    };
    const coord_modes = [_]shaderops.FuncCoordMode{
        .para,
        .uv,
        .world_reference,
        .world_deformed,
    };
    const dtypes = [_]@typeInfo(FullTexCase).@"struct".fields[1].type{
        .u8,
        .u16,
        .f64,
    };

    for (mesh_types) |mesh_type| {
        var prep = try common.prepareScene0(allocator, io, mesh_type);
        defer prep.deinit(allocator);

        // 1. Nodal cases (18 per mesh)
        for (bool_values) |is_rgb| {
            for (normal_types) |norm| {
                // Scalenone
                try generateFullShaderCase(
                    allocator,
                    io,
                    mesh_type,
                    &prep,
                    &textures,
                    .{
                        .nodal = .{
                            .is_rgb = is_rgb,
                            .scaling = .none,
                            .scale_over = .over_frames,
                            .normal_type = norm,
                        },
                    },
                    gold_dir_root,
                    config,
                );
                // Scalewithin
                try generateFullShaderCase(
                    allocator,
                    io,
                    mesh_type,
                    &prep,
                    &textures,
                    .{
                        .nodal = .{
                            .is_rgb = is_rgb,
                            .scaling = .auto,
                            .scale_over = .within_frames,
                            .normal_type = norm,
                        },
                    },
                    gold_dir_root,
                    config,
                );
                // Scaleover
                try generateFullShaderCase(
                    allocator,
                    io,
                    mesh_type,
                    &prep,
                    &textures,
                    .{
                        .nodal = .{
                            .is_rgb = is_rgb,
                            .scaling = .auto,
                            .scale_over = .over_frames,
                            .normal_type = norm,
                        },
                    },
                    gold_dir_root,
                    config,
                );
            }
        }

        // 2. Texture cases (18 per mesh)
        for (bool_values) |is_rgb| {
            for (dtypes) |dtype| {
                for (normal_types) |norm| {
                    try generateFullShaderCase(
                        allocator,
                        io,
                        mesh_type,
                        &prep,
                        &textures,
                        .{
                            .tex = .{
                                .is_rgb = is_rgb,
                                .dtype = dtype,
                                .normal_type = norm,
                            },
                        },
                        gold_dir_root,
                        config,
                    );
                }
            }
        }

        // 3. Function cases (216 per mesh)
        for (bool_values) |is_rgb| {
            for (builtins) |builtin| {
                for (coord_modes) |coord| {
                    for (normal_types) |norm| {
                        try generateFullShaderCase(
                            allocator,
                            io,
                            mesh_type,
                            &prep,
                            &textures,
                            .{
                                .func = .{
                                    .is_rgb = is_rgb,
                                    .builtin = builtin,
                                    .coord = coord,
                                    .normal_type = norm,
                                },
                            },
                            gold_dir_root,
                            config,
                        );
                    }
                }
            }
        }
    }
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const allocator = arena.allocator();

    var config = tcfg.getRasterConfig(.gold_gen);
    config.save_strategy = .disk;
    config.image_save_opts = &[_]iio.ImageSaveOpts{
        .{ .format = .fimg, .bits = null, .scaling = .none },
        .{ .format = .bmp, .bits = 8, .scaling = .auto },
    };

    const gold_dir_root = policy.goldRoot(.full_shader);
    std.debug.print("Generating Full Suite: shader cases in {s}...\n", .{gold_dir_root});
    try generateAllFullShaderCases(
        allocator,
        io,
        gold_dir_root,
        config,
    );
    std.debug.print("Done full shader cases.\n", .{});
}
