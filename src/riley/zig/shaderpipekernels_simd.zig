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
const S = buildconfig.SimdWidth;
const VecSF = buildconfig.VecSF;
const VecSB = buildconfig.VecSB;

const MatSlice = @import("matslice.zig").MatSlice;
const shaderops = @import("shaderops.zig");
const shaderpipe = @import("shaderpipe.zig");
const CoordSpace = @import("geometrykernels.zig").CoordSpace;
const texops = @import("textureops.zig");
const TexSampConfig = texops.TexSampConfig;
const report = @import("report.zig");
const simdops = @import("simdops.zig");
const scal_impl = @import("shaderpipekernels_scalar.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

inline fn storeShadeSIMD(
    subpx_vals: []F,
    start_u: usize,
    ctx_shade: shaderops.ShadeContext,
    v_mask_active: VecSB,
    v_vals: VecSF,
) void {
    if (!ctx_shade.exclusive_subpx_target) {
        simdops.storeMaskedVecSF(
            subpx_vals,
            start_u,
            v_mask_active,
            v_vals,
        );
        return;
    }

    const mask_arr: [S]bool = v_mask_active;
    const vals_arr: [S]F = v_vals;
    inline for (0..S) |lane| {
        if (mask_arr[lane]) {
            subpx_vals[start_u + lane] = vals_arr[lane];
        }
    }
}

fn texSimdInterpMode(
    comptime C: comptime_int,
    comptime samp_cfg: TexSampConfig,
) buildconfig.SimdTexInterpMode {
    return buildconfig.TexSIMDPolicy.resolve(
        C,
        samp_cfg.sample == .linear,
        samp_cfg.mode == .lut or samp_cfg.mode == .lut_lerp,
    );
}

inline fn sampleMonoTexPayloadSIMD(
    samp_cfg_val: TexSampConfig,
    v_mask_active: VecSB,
    tex_payload: shaderpipe.MonoTexPayload,
    v_u: VecSF,
    v_v: VecSF,
) VecSF {
    return switch (samp_cfg_val.sample) {
        inline else => |samp_type| switch (samp_cfg_val.mode) {
            inline else => |mode_type| blk: {
                const samp_cfg = comptime (TexSampConfig{
                    .sample = samp_type,
                    .mode = mode_type,
                }).sanitize();
                const interp_mode = comptime texSimdInterpMode(1, samp_cfg);
                const vecs = switch (tex_payload) {
                    inline else => |tex| switch (interp_mode) {
                        .inner => texops.sampLanes(
                            1,
                            samp_cfg,
                            v_mask_active,
                            tex,
                            v_u,
                            v_v,
                        ),
                        .over_pixels => texops.sampWide(
                            1,
                            samp_cfg,
                            tex,
                            v_u,
                            v_v,
                        ),
                    },
                };
                break :blk vecs[0];
            },
        },
    };
}

inline fn sampleRgbTexPayloadSIMD(
    samp_cfg_val: TexSampConfig,
    v_mask_active: VecSB,
    tex_payload: shaderpipe.RgbTexPayload,
    v_u: VecSF,
    v_v: VecSF,
) [3]VecSF {
    return switch (samp_cfg_val.sample) {
        inline else => |samp_type| switch (samp_cfg_val.mode) {
            inline else => |mode_type| blk: {
                const samp_cfg = comptime (TexSampConfig{
                    .sample = samp_type,
                    .mode = mode_type,
                }).sanitize();
                const interp_mode = comptime texSimdInterpMode(3, samp_cfg);
                const vecs = switch (tex_payload) {
                    inline else => |tex| switch (interp_mode) {
                        .inner => texops.sampLanes(
                            3,
                            samp_cfg,
                            v_mask_active,
                            tex,
                            v_u,
                            v_v,
                        ),
                        .over_pixels => texops.sampWide(
                            3,
                            samp_cfg,
                            tex,
                            v_u,
                            v_v,
                        ),
                    },
                };
                break :blk vecs;
            },
        },
    };
}

pub fn MonoShaderPipeKern(comptime N: usize) type {
    return struct {
        pub inline fn shade(
            comptime coord_space: CoordSpace,
            ctx_shade: shaderops.ShadeContext,
            interp: shaderops.InterpData(N),
            pipe_buf: *const shaderpipe.LocalMonoPipeBuff(N),
            pipe: *const shaderpipe.MonoShaderPipePrepared,
            ctx_report: anytype,
            spx_img_scratch: *MatSlice(F),
        ) void {
            scal_impl.MonoShaderPipeKern(N).shade(
                coord_space,
                ctx_shade,
                interp,
                pipe_buf,
                pipe,
                ctx_report,
                spx_img_scratch,
            );
        }

        pub inline fn shadeSIMD(
            comptime coord_space: CoordSpace,
            ctx_shade: shaderops.ShadeContext,
            ctx_report: anytype,
            v_mask_active: VecSB,
            v_weights: [N]VecSF,
            v_xi: VecSF,
            v_eta: VecSF,
            v_nodes_inv_z: [N]VecSF,
            v_subpx_z: VecSF,
            pipe_buf: *const shaderpipe.LocalMonoPipeBuff(N),
            pipe: *const shaderpipe.MonoShaderPipePrepared,
            spx_image_scratch: *MatSlice(F),
        ) void {
            if (comptime @TypeOf(ctx_report).mode_tag == .full_stats) {
                // Record normals if needed
            }

            var state = shaderpipe.MonoPipeStateSIMD{ .value = @splat(0.0) };

            for (pipe.stages) |stage| {
                switch (stage) {
                    .nodal => |nodal| {
                        if (comptime coord_space == .raster) {
                            var v_weighted_sum: VecSF = @splat(0.0);
                            inline for (0..N) |nn| {
                                v_weighted_sum += v_weights[nn] *
                                    v_nodes_inv_z[nn] *
                                    @as(VecSF, @splat(pipe_buf.nodal_data[nn]));
                            }
                            state.value = (v_weighted_sum * v_subpx_z) *
                                @as(VecSF, @splat(nodal.scale_mul)) +
                                @as(VecSF, @splat(nodal.scale_add));
                        } else if (comptime coord_space == .clip_px_leng) {
                            var v_weighted_sum: VecSF = @splat(0.0);
                            inline for (0..N) |nn| {
                                v_weighted_sum += v_weights[nn] *
                                    @as(VecSF, @splat(pipe_buf.nodal_data[nn]));
                            }
                            state.value = v_weighted_sum *
                                @as(VecSF, @splat(nodal.scale_mul)) +
                                @as(VecSF, @splat(nodal.scale_add));
                        } else {
                            @panic("shadeSIMD not implemented for this coord_space");
                        }
                    },
                    .texture => |tex_stage| {
                        var v_tex_u: VecSF = @splat(0.0);
                        var v_tex_v: VecSF = @splat(0.0);
                        if (comptime coord_space == .raster) {
                            inline for (0..N) |nn| {
                                const v_inv_z = v_nodes_inv_z[nn];
                                const v_u0 = @as(VecSF, @splat(pipe_buf.uv_data[nn]));
                                const v_v0 = @as(VecSF, @splat(pipe_buf.uv_data[N + nn]));
                                v_tex_u += v_weights[nn] * v_u0 * v_inv_z;
                                v_tex_v += v_weights[nn] * v_v0 * v_inv_z;
                            }
                            v_tex_u *= v_subpx_z;
                            v_tex_v *= v_subpx_z;
                        } else {
                            inline for (0..N) |nn| {
                                v_tex_u += v_weights[nn] *
                                    @as(VecSF, @splat(pipe_buf.uv_data[nn]));
                                v_tex_v += v_weights[nn] *
                                    @as(VecSF, @splat(pipe_buf.uv_data[N + nn]));
                            }
                        }

                        const sampled_vec = sampleMonoTexPayloadSIMD(
                            tex_stage.samp_cfg,
                            v_mask_active,
                            tex_stage.tex,
                            v_tex_u,
                            v_tex_v,
                        );

                        state.value = sampled_vec *
                            @as(VecSF, @splat(tex_stage.scale_mul)) +
                            @as(VecSF, @splat(tex_stage.scale_add));
                    },
                    .function => |func_stage| {
                        var v_coord_0: VecSF = v_xi;
                        var v_coord_1: VecSF = v_eta;
                        switch (func_stage.coord_mode) {
                            .uv, .world_reference, .world_deformed => {
                                v_coord_0 = @splat(0.0);
                                v_coord_1 = @splat(0.0);
                                if (comptime coord_space == .raster) {
                                    inline for (0..N) |nn| {
                                        const v_inv_z = v_nodes_inv_z[nn];
                                        const v_fc0 = @as(
                                            VecSF,
                                            @splat(pipe_buf.func_coords[nn]),
                                        );
                                        const v_fc1 = @as(
                                            VecSF,
                                            @splat(pipe_buf.func_coords[N + nn]),
                                        );
                                        v_coord_0 += v_weights[nn] * v_fc0 * v_inv_z;
                                        v_coord_1 += v_weights[nn] * v_fc1 * v_inv_z;
                                    }
                                    v_coord_0 *= v_subpx_z;
                                    v_coord_1 *= v_subpx_z;
                                } else {
                                    inline for (0..N) |nn| {
                                        const v_fc0 = @as(
                                            VecSF,
                                            @splat(pipe_buf.func_coords[nn]),
                                        );
                                        const v_fc1 = @as(
                                            VecSF,
                                            @splat(pipe_buf.func_coords[N + nn]),
                                        );
                                        v_coord_0 += v_weights[nn] * v_fc0;
                                        v_coord_1 += v_weights[nn] * v_fc1;
                                    }
                                }
                            },
                            .para => {},
                        }

                        var normal_vecs = [3]VecSF{ @splat(0.0), @splat(0.0), @splat(1.0) };
                        if (func_stage.elem_normals != null) {
                            inline for (0..3) |cc| {
                                var sum: VecSF = @splat(0.0);
                                inline for (0..N) |nn| {
                                    sum += v_weights[nn] *
                                        @as(VecSF, @splat(pipe_buf.normals[cc * N + nn]));
                                }
                                normal_vecs[cc] = sum;
                            }
                        }

                        const coord = shaderops.FuncCoordSIMD{
                            .coord_0 = v_coord_0,
                            .coord_1 = v_coord_1,
                            .normal_x = normal_vecs[0],
                            .normal_y = normal_vecs[1],
                            .normal_z = normal_vecs[2],
                        };
                        const v_eval = @import("shaderops_simd.zig")
                            .evalFuncShaderGreyNormSIMD(
                            func_stage.builtin,
                            coord,
                            func_stage.params,
                        );
                        state.value = v_eval *
                            @as(VecSF, @splat(func_stage.scale_mul)) +
                            @as(VecSF, @splat(func_stage.scale_add));
                    },
                    .scale => |sc| {
                        state.value *= @as(VecSF, @splat(sc.factor));
                    },
                    .offset => |off| {
                        state.value += @as(VecSF, @splat(off.offset));
                    },
                    .linear_map => |lmap| {
                        state.value = state.value *
                            @as(VecSF, @splat(lmap.scale_mul)) +
                            @as(VecSF, @splat(lmap.scale_add));
                    },
                    .map_range => |mr| {
                        state.value = @as(VecSF, @splat(mr.out_min)) +
                            (state.value - @as(VecSF, @splat(mr.in_min))) /
                            @as(VecSF, @splat(mr.in_max - mr.in_min)) *
                            @as(VecSF, @splat(mr.out_max - mr.out_min));
                    },
                    .clamp => |cl| {
                        state.value = @min(
                            @max(state.value, @as(VecSF, @splat(cl.min))),
                            @as(VecSF, @splat(cl.max)),
                        );
                    },
                    .invert => |inv| {
                        state.value = @as(VecSF, @splat(inv.max_val)) - state.value;
                    },
                    .lut => |lut_stage| {
                        _ = lut_stage;
                    },
                }
            }

            const v_final = pipe.terminal.evalSIMD(&state);
            storeShadeSIMD(
                spx_image_scratch.slice,
                ctx_shade.scratch_idx,
                ctx_shade,
                v_mask_active,
                v_final,
            );
        }
    };
}

pub fn RgbShaderPipeKern(comptime N: usize) type {
    return struct {
        pub inline fn shade(
            comptime coord_space: CoordSpace,
            ctx_shade: shaderops.ShadeContext,
            interp: shaderops.InterpData(N),
            pipe_buf: *const shaderpipe.LocalRgbPipeBuff(N),
            pipe: *const shaderpipe.RgbShaderPipePrepared,
            ctx_report: anytype,
            spx_img_scratch: *MatSlice(F),
        ) void {
            scal_impl.RgbShaderPipeKern(N).shade(
                coord_space,
                ctx_shade,
                interp,
                pipe_buf,
                pipe,
                ctx_report,
                spx_img_scratch,
            );
        }

        pub inline fn shadeSIMD(
            comptime coord_space: CoordSpace,
            ctx_shade: shaderops.ShadeContext,
            ctx_report: anytype,
            v_mask_active: VecSB,
            v_weights: [N]VecSF,
            v_xi: VecSF,
            v_eta: VecSF,
            v_nodes_inv_z: [N]VecSF,
            v_subpx_z: VecSF,
            pipe_buf: *const shaderpipe.LocalRgbPipeBuff(N),
            pipe: *const shaderpipe.RgbShaderPipePrepared,
            spx_image_scratch: *MatSlice(F),
        ) void {
            if (comptime @TypeOf(ctx_report).mode_tag == .full_stats) {
                // Record normals if needed
            }

            var state = shaderpipe.RgbPipeStateSIMD{
                .r = @splat(0.0),
                .g = @splat(0.0),
                .b = @splat(0.0),
            };

            for (pipe.stages) |stage| {
                switch (stage) {
                    .constant => |c| {
                        state.r = @splat(c.value[0]);
                        state.g = @splat(c.value[1]);
                        state.b = @splat(c.value[2]);
                    },
                    .nodal => |nodal| {
                        inline for (0..3) |cc| {
                            const base = cc * N;
                            var v_field: VecSF = @splat(0.0);
                            if (comptime coord_space == .raster) {
                                inline for (0..N) |nn| {
                                    v_field += v_weights[nn] *
                                        v_nodes_inv_z[nn] *
                                        @as(
                                        VecSF,
                                        @splat(pipe_buf.nodal_data[base + nn]),
                                    );
                                }
                                v_field = (v_field * v_subpx_z) *
                                    @as(VecSF, @splat(nodal.scale_mul)) +
                                    @as(VecSF, @splat(nodal.scale_add));
                            } else if (comptime coord_space == .clip_px_leng) {
                                inline for (0..N) |nn| {
                                    v_field += v_weights[nn] *
                                        @as(
                                        VecSF,
                                        @splat(pipe_buf.nodal_data[base + nn]),
                                    );
                                }
                                v_field = v_field *
                                    @as(VecSF, @splat(nodal.scale_mul)) +
                                    @as(VecSF, @splat(nodal.scale_add));
                            }
                            if (cc == 0) state.r = v_field;
                            if (cc == 1) state.g = v_field;
                            if (cc == 2) state.b = v_field;
                        }
                    },
                    .texture => |tex_stage| {
                        var v_tex_u: VecSF = @splat(0.0);
                        var v_tex_v: VecSF = @splat(0.0);
                        if (comptime coord_space == .raster) {
                            inline for (0..N) |nn| {
                                const v_inv_z = v_nodes_inv_z[nn];
                                const v_u0 = @as(VecSF, @splat(pipe_buf.uv_data[nn]));
                                const v_v0 = @as(VecSF, @splat(pipe_buf.uv_data[N + nn]));
                                v_tex_u += v_weights[nn] * v_u0 * v_inv_z;
                                v_tex_v += v_weights[nn] * v_v0 * v_inv_z;
                            }
                            v_tex_u *= v_subpx_z;
                            v_tex_v *= v_subpx_z;
                        } else {
                            inline for (0..N) |nn| {
                                v_tex_u += v_weights[nn] *
                                    @as(VecSF, @splat(pipe_buf.uv_data[nn]));
                                v_tex_v += v_weights[nn] *
                                    @as(VecSF, @splat(pipe_buf.uv_data[N + nn]));
                            }
                        }

                        const sampled_vecs = sampleRgbTexPayloadSIMD(
                            tex_stage.samp_cfg,
                            v_mask_active,
                            tex_stage.tex,
                            v_tex_u,
                            v_tex_v,
                        );

                        state.r = sampled_vecs[0] *
                            @as(VecSF, @splat(tex_stage.scale_mul)) +
                            @as(VecSF, @splat(tex_stage.scale_add));
                        state.g = sampled_vecs[1] *
                            @as(VecSF, @splat(tex_stage.scale_mul)) +
                            @as(VecSF, @splat(tex_stage.scale_add));
                        state.b = sampled_vecs[2] *
                            @as(VecSF, @splat(tex_stage.scale_mul)) +
                            @as(VecSF, @splat(tex_stage.scale_add));
                    },
                    .function => |func_stage| {
                        var v_coord_0: VecSF = v_xi;
                        var v_coord_1: VecSF = v_eta;
                        switch (func_stage.coord_mode) {
                            .uv, .world_reference, .world_deformed => {
                                v_coord_0 = @splat(0.0);
                                v_coord_1 = @splat(0.0);
                                if (comptime coord_space == .raster) {
                                    inline for (0..N) |nn| {
                                        const v_inv_z = v_nodes_inv_z[nn];
                                        const v_fc0 = @as(
                                            VecSF,
                                            @splat(pipe_buf.func_coords[nn]),
                                        );
                                        const v_fc1 = @as(
                                            VecSF,
                                            @splat(pipe_buf.func_coords[N + nn]),
                                        );
                                        v_coord_0 += v_weights[nn] * v_fc0 * v_inv_z;
                                        v_coord_1 += v_weights[nn] * v_fc1 * v_inv_z;
                                    }
                                    v_coord_0 *= v_subpx_z;
                                    v_coord_1 *= v_subpx_z;
                                } else {
                                    inline for (0..N) |nn| {
                                        const v_fc0 = @as(
                                            VecSF,
                                            @splat(pipe_buf.func_coords[nn]),
                                        );
                                        const v_fc1 = @as(
                                            VecSF,
                                            @splat(pipe_buf.func_coords[N + nn]),
                                        );
                                        v_coord_0 += v_weights[nn] * v_fc0;
                                        v_coord_1 += v_weights[nn] * v_fc1;
                                    }
                                }
                            },
                            .para => {},
                        }

                        var normal_vecs = [3]VecSF{ @splat(0.0), @splat(0.0), @splat(1.0) };
                        if (func_stage.elem_normals != null) {
                            inline for (0..3) |cc| {
                                var sum: VecSF = @splat(0.0);
                                inline for (0..N) |nn| {
                                    sum += v_weights[nn] *
                                        @as(VecSF, @splat(pipe_buf.normals[cc * N + nn]));
                                }
                                normal_vecs[cc] = sum;
                            }
                        }

                        const coord = shaderops.FuncCoordSIMD{
                            .coord_0 = v_coord_0,
                            .coord_1 = v_coord_1,
                            .normal_x = normal_vecs[0],
                            .normal_y = normal_vecs[1],
                            .normal_z = normal_vecs[2],
                        };
                        const v_vals = @import("shaderops_simd.zig")
                            .evalFuncShaderRGBNormSIMD(
                            func_stage.builtin,
                            coord,
                            func_stage.params,
                        );
                        state.r = v_vals[0] *
                            @as(VecSF, @splat(func_stage.scale_mul)) +
                            @as(VecSF, @splat(func_stage.scale_add));
                        state.g = v_vals[1] *
                            @as(VecSF, @splat(func_stage.scale_mul)) +
                            @as(VecSF, @splat(func_stage.scale_add));
                        state.b = v_vals[2] *
                            @as(VecSF, @splat(func_stage.scale_mul)) +
                            @as(VecSF, @splat(func_stage.scale_add));
                    },
                    .scale => |sc| {
                        state.r *= @as(VecSF, @splat(sc.factor[0]));
                        state.g *= @as(VecSF, @splat(sc.factor[1]));
                        state.b *= @as(VecSF, @splat(sc.factor[2]));
                    },
                    .offset => |off| {
                        state.r += @as(VecSF, @splat(off.offset[0]));
                        state.g += @as(VecSF, @splat(off.offset[1]));
                        state.b += @as(VecSF, @splat(off.offset[2]));
                    },
                    .clamp => |cl| {
                        state.r = @min(
                            @max(state.r, @as(VecSF, @splat(cl.min[0]))),
                            @as(VecSF, @splat(cl.max[0])),
                        );
                        state.g = @min(
                            @max(state.g, @as(VecSF, @splat(cl.min[1]))),
                            @as(VecSF, @splat(cl.max[1])),
                        );
                        state.b = @min(
                            @max(state.b, @as(VecSF, @splat(cl.min[2]))),
                            @as(VecSF, @splat(cl.max[2])),
                        );
                    },
                    .linear_map => |lmap| {
                        state.r = state.r * @as(VecSF, @splat(lmap.scale_mul[0])) +
                            @as(VecSF, @splat(lmap.scale_add[0]));
                        state.g = state.g * @as(VecSF, @splat(lmap.scale_mul[1])) +
                            @as(VecSF, @splat(lmap.scale_add[1]));
                        state.b = state.b * @as(VecSF, @splat(lmap.scale_mul[2])) +
                            @as(VecSF, @splat(lmap.scale_add[2]));
                    },
                    .lut => |lut_stage| {
                        _ = lut_stage;
                    },
                }
            }

            const out_vecs = pipe.terminal.evalSIMD(&state);
            const px_stride = spx_image_scratch.cols_num;
            storeShadeSIMD(
                spx_image_scratch.slice,
                0 * px_stride + ctx_shade.scratch_idx,
                ctx_shade,
                v_mask_active,
                out_vecs[0],
            );
            storeShadeSIMD(
                spx_image_scratch.slice,
                1 * px_stride + ctx_shade.scratch_idx,
                ctx_shade,
                v_mask_active,
                out_vecs[1],
            );
            storeShadeSIMD(
                spx_image_scratch.slice,
                2 * px_stride + ctx_shade.scratch_idx,
                ctx_shade,
                v_mask_active,
                out_vecs[2],
            );
        }
    };
}

pub fn MultiShaderPipeKern(comptime N: usize, comptime C: usize) type {
    return struct {
        pub inline fn shade(
            comptime coord_space: CoordSpace,
            ctx_shade: shaderops.ShadeContext,
            interp: shaderops.InterpData(N),
            pipe_buf: *const shaderpipe.LocalMultiPipeBuff(N, C),
            pipe: *const shaderpipe.MultiShaderPipePrepared,
            ctx_report: anytype,
            spx_img_scratch: *MatSlice(F),
        ) void {
            scal_impl.MultiShaderPipeKern(N, C).shade(
                coord_space,
                ctx_shade,
                interp,
                pipe_buf,
                pipe,
                ctx_report,
                spx_img_scratch,
            );
        }

        pub inline fn shadeSIMD(
            comptime coord_space: CoordSpace,
            ctx_shade: shaderops.ShadeContext,
            ctx_report: anytype,
            v_mask_active: VecSB,
            v_weights: [N]VecSF,
            v_xi: VecSF,
            v_eta: VecSF,
            v_nodes_inv_z: [N]VecSF,
            v_subpx_z: VecSF,
            pipe_buf: *const shaderpipe.LocalMultiPipeBuff(N, C),
            pipe: *const shaderpipe.MultiShaderPipePrepared,
            spx_image_scratch: *MatSlice(F),
        ) void {
            _ = v_xi;
            _ = v_eta;
            if (comptime @TypeOf(ctx_report).mode_tag == .full_stats) {
                // Record normals if needed
            }

            var state = shaderpipe.MultiPipeStateSIMD(C){
                .channels = undefined,
            };
            const nodal = pipe.nodal;

            inline for (0..C) |cc| {
                const base = cc * N;
                var v_field: VecSF = @splat(0.0);
                if (comptime coord_space == .raster) {
                    inline for (0..N) |nn| {
                        v_field += v_weights[nn] *
                            v_nodes_inv_z[nn] *
                            @as(VecSF, @splat(pipe_buf.nodal_data[base + nn]));
                    }
                    v_field = (v_field * v_subpx_z) *
                        @as(VecSF, @splat(nodal.scale_mul)) +
                        @as(VecSF, @splat(nodal.scale_add));
                } else if (comptime coord_space == .clip_px_leng) {
                    inline for (0..N) |nn| {
                        v_field += v_weights[nn] *
                            @as(VecSF, @splat(pipe_buf.nodal_data[base + nn]));
                    }
                    v_field = v_field *
                        @as(VecSF, @splat(nodal.scale_mul)) +
                        @as(VecSF, @splat(nodal.scale_add));
                }
                state.channels[cc] = v_field;
            }

            const terminal = shaderpipe.MultiTerminal(C){};
            const out_vecs = terminal.evalSIMD(&state);
            const px_stride = spx_image_scratch.cols_num;

            inline for (0..C) |cc| {
                storeShadeSIMD(
                    spx_image_scratch.slice,
                    cc * px_stride + ctx_shade.scratch_idx,
                    ctx_shade,
                    v_mask_active,
                    out_vecs[cc],
                );
            }
        }
    };
}
