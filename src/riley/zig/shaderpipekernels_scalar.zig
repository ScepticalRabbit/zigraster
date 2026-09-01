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
const shaderops = @import("shaderops.zig");
const shaderpipe = @import("shaderpipe.zig");
const CoordSpace = @import("geometrykernels.zig").CoordSpace;
const texops = @import("textureops.zig");

// --------------------------------------------------------------------------------------
// Public Constants & Public Types
// --------------------------------------------------------------------------------------

inline fn sampleMonoTexPayloadScalar(
    samp_cfg_val: texops.TexSampConfig,
    tex_payload: shaderpipe.MonoTexPayload,
    u_val: F,
    v_val: F,
) F {
    return switch (samp_cfg_val.sample) {
        inline else => |samp_type| switch (samp_cfg_val.mode) {
            inline else => |mode_type| blk: {
                const samp_cfg = comptime (texops.TexSampConfig{
                    .sample = samp_type,
                    .mode = mode_type,
                }).sanitize();
                const sampled = switch (tex_payload) {
                    inline else => |tex| texops.sampScal(
                        1,
                        samp_cfg,
                        tex,
                        u_val,
                        v_val,
                    ),
                };
                break :blk sampled[0];
            },
        },
    };
}

inline fn sampleRgbTexPayloadScalar(
    samp_cfg_val: texops.TexSampConfig,
    tex_payload: shaderpipe.RgbTexPayload,
    u_val: F,
    v_val: F,
) [3]F {
    return switch (samp_cfg_val.sample) {
        inline else => |samp_type| switch (samp_cfg_val.mode) {
            inline else => |mode_type| blk: {
                const samp_cfg = comptime (texops.TexSampConfig{
                    .sample = samp_type,
                    .mode = mode_type,
                }).sanitize();
                break :blk switch (tex_payload) {
                    inline else => |tex| texops.sampScal(
                        3,
                        samp_cfg,
                        tex,
                        u_val,
                        v_val,
                    ),
                };
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
            if (comptime @TypeOf(ctx_report).mode_tag == .full_stats) {
                ctx_report.recordDepth(
                    ctx_shade.global_subx,
                    ctx_shade.global_suby,
                    1.0 / interp.sub_pixel_z,
                );
                const normal = pipe_buf.interpNormal(interp.weights);
                ctx_report.recordNormal(
                    ctx_shade.global_subx,
                    ctx_shade.global_suby,
                    normal[0],
                    normal[1],
                    normal[2],
                );
            }

            var state = shaderpipe.MonoPipeState{ .value = 0.0 };

            for (pipe.stages) |stage| {
                switch (stage) {
                    .nodal => |nodal| {
                        if (comptime coord_space == .raster) {
                            var sum: F = 0.0;
                            inline for (0..N) |nn| {
                                sum += interp.weights[nn] *
                                    interp.nodes_inv_z[nn] *
                                    pipe_buf.nodal_data[nn];
                            }
                            state.value = (sum * interp.sub_pixel_z) *
                                nodal.scale_mul + nodal.scale_add;
                        } else {
                            var sum: F = 0.0;
                            inline for (0..N) |nn| {
                                sum += interp.weights[nn] * pipe_buf.nodal_data[nn];
                            }
                            state.value = sum * nodal.scale_mul + nodal.scale_add;
                        }
                    },
                    .texture => |tex_stage| {
                        var u: F = 0.0;
                        var v: F = 0.0;
                        if (comptime coord_space == .raster) {
                            inline for (0..N) |nn| {
                                const inv_z = interp.nodes_inv_z[nn];
                                u += interp.weights[nn] * pipe_buf.uv_data[nn] * inv_z;
                                v += interp.weights[nn] *
                                    pipe_buf.uv_data[N + nn] * inv_z;
                            }
                            u *= interp.sub_pixel_z;
                            v *= interp.sub_pixel_z;
                        } else {
                            inline for (0..N) |nn| {
                                u += interp.weights[nn] * pipe_buf.uv_data[nn];
                                v += interp.weights[nn] * pipe_buf.uv_data[N + nn];
                            }
                        }

                        const sampled_val = sampleMonoTexPayloadScalar(
                            tex_stage.samp_cfg,
                            tex_stage.tex,
                            u,
                            v,
                        );
                        state.value = sampled_val * tex_stage.scale_mul +
                            tex_stage.scale_add;
                    },
                    .function => |func_stage| {
                        var coord_0: F = interp.xi;
                        var coord_1: F = interp.eta;
                        switch (func_stage.coord_mode) {
                            .uv, .world_reference, .world_deformed => {
                                coord_0 = 0.0;
                                coord_1 = 0.0;
                                if (comptime coord_space == .raster) {
                                    inline for (0..N) |nn| {
                                        const inv_z = interp.nodes_inv_z[nn];
                                        coord_0 += interp.weights[nn] *
                                            pipe_buf.func_coords[nn] * inv_z;
                                        coord_1 += interp.weights[nn] *
                                            pipe_buf.func_coords[N + nn] * inv_z;
                                    }
                                    coord_0 *= interp.sub_pixel_z;
                                    coord_1 *= interp.sub_pixel_z;
                                } else {
                                    inline for (0..N) |nn| {
                                        coord_0 += interp.weights[nn] *
                                            pipe_buf.func_coords[nn];
                                        coord_1 += interp.weights[nn] *
                                            pipe_buf.func_coords[N + nn];
                                    }
                                }
                            },
                            .para => {},
                        }
                        const norm = pipe_buf.interpNormal(interp.weights);
                        const coord = shaderops.FuncCoord{
                            .coord_0 = coord_0,
                            .coord_1 = coord_1,
                            .normal_x = norm[0],
                            .normal_y = norm[1],
                            .normal_z = norm[2],
                        };
                        const sc_comm = @import("shaderops_common.zig");
                        const val = sc_comm.evalFuncShaderBuiltinGreyNorm(
                            func_stage.builtin,
                            coord,
                            func_stage.params,
                        );
                        state.value = val * func_stage.scale_mul + func_stage.scale_add;
                    },
                    .scale => |sc| {
                        state.value *= sc.factor;
                    },
                    .offset => |off| {
                        state.value += off.offset;
                    },
                    .linear_map => |lmap| {
                        state.value = state.value * lmap.scale_mul + lmap.scale_add;
                    },
                    .map_range => |mr| {
                        state.value = mr.out_min + (state.value - mr.in_min) /
                            (mr.in_max - mr.in_min) * (mr.out_max - mr.out_min);
                    },
                    .clamp => |cl| {
                        state.value = @min(@max(state.value, cl.min), cl.max);
                    },
                    .invert => |inv| {
                        state.value = inv.max_val - state.value;
                    },
                    .lut => |lut_stage| {
                        if (lut_stage.lut.len > 0) {
                            const max_idx = lut_stage.lut.len - 1;
                            const pos = @max(0.0, @min(1.0, state.value)) *
                                @as(F, @floatFromInt(max_idx));
                            const idx_u = @as(usize, @intFromFloat(@floor(pos)));
                            const frac = pos - @as(F, @floatFromInt(idx_u));
                            const next_idx = @min(idx_u + 1, max_idx);
                            state.value = lut_stage.lut[idx_u] * (1.0 - frac) +
                                lut_stage.lut[next_idx] * frac;
                        }
                    },
                }
            }

            const final_val = pipe.terminal.eval(&state);
            spx_img_scratch.set(0, ctx_shade.scratch_idx, final_val);
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
            if (comptime @TypeOf(ctx_report).mode_tag == .full_stats) {
                ctx_report.recordDepth(
                    ctx_shade.global_subx,
                    ctx_shade.global_suby,
                    1.0 / interp.sub_pixel_z,
                );
                const normal = pipe_buf.interpNormal(interp.weights);
                ctx_report.recordNormal(
                    ctx_shade.global_subx,
                    ctx_shade.global_suby,
                    normal[0],
                    normal[1],
                    normal[2],
                );
            }

            var state = shaderpipe.RgbPipeState{ .value = .{ 0.0, 0.0, 0.0 } };

            for (pipe.stages) |stage| {
                switch (stage) {
                    .constant => |c| {
                        state.value = c.value;
                    },
                    .nodal => |nodal| {
                        inline for (0..3) |cc| {
                            const base = cc * N;
                            if (comptime coord_space == .raster) {
                                var sum: F = 0.0;
                                inline for (0..N) |nn| {
                                    sum += interp.weights[nn] *
                                        interp.nodes_inv_z[nn] *
                                        pipe_buf.nodal_data[base + nn];
                                }
                                state.value[cc] = (sum * interp.sub_pixel_z) *
                                    nodal.scale_mul + nodal.scale_add;
                            } else {
                                var sum: F = 0.0;
                                inline for (0..N) |nn| {
                                    sum += interp.weights[nn] *
                                        pipe_buf.nodal_data[base + nn];
                                }
                                state.value[cc] = sum * nodal.scale_mul +
                                    nodal.scale_add;
                            }
                        }
                    },
                    .texture => |tex_stage| {
                        var u: F = 0.0;
                        var v: F = 0.0;
                        if (comptime coord_space == .raster) {
                            inline for (0..N) |nn| {
                                const inv_z = interp.nodes_inv_z[nn];
                                u += interp.weights[nn] * pipe_buf.uv_data[nn] * inv_z;
                                v += interp.weights[nn] *
                                    pipe_buf.uv_data[N + nn] * inv_z;
                            }
                            u *= interp.sub_pixel_z;
                            v *= interp.sub_pixel_z;
                        } else {
                            inline for (0..N) |nn| {
                                u += interp.weights[nn] * pipe_buf.uv_data[nn];
                                v += interp.weights[nn] * pipe_buf.uv_data[N + nn];
                            }
                        }

                        const sampled = sampleRgbTexPayloadScalar(
                            tex_stage.samp_cfg,
                            tex_stage.tex,
                            u,
                            v,
                        );
                        inline for (0..3) |cc| {
                            state.value[cc] = sampled[cc] * tex_stage.scale_mul +
                                tex_stage.scale_add;
                        }
                    },
                    .function => |func_stage| {
                        var coord_0: F = interp.xi;
                        var coord_1: F = interp.eta;
                        switch (func_stage.coord_mode) {
                            .uv, .world_reference, .world_deformed => {
                                coord_0 = 0.0;
                                coord_1 = 0.0;
                                if (comptime coord_space == .raster) {
                                    inline for (0..N) |nn| {
                                        const inv_z = interp.nodes_inv_z[nn];
                                        coord_0 += interp.weights[nn] *
                                            pipe_buf.func_coords[nn] * inv_z;
                                        coord_1 += interp.weights[nn] *
                                            pipe_buf.func_coords[N + nn] * inv_z;
                                    }
                                    coord_0 *= interp.sub_pixel_z;
                                    coord_1 *= interp.sub_pixel_z;
                                } else {
                                    inline for (0..N) |nn| {
                                        coord_0 += interp.weights[nn] *
                                            pipe_buf.func_coords[nn];
                                        coord_1 += interp.weights[nn] *
                                            pipe_buf.func_coords[N + nn];
                                    }
                                }
                            },
                            .para => {},
                        }
                        const norm = pipe_buf.interpNormal(interp.weights);
                        const coord = shaderops.FuncCoord{
                            .coord_0 = coord_0,
                            .coord_1 = coord_1,
                            .normal_x = norm[0],
                            .normal_y = norm[1],
                            .normal_z = norm[2],
                        };
                        const sc_comm = @import("shaderops_common.zig");
                        const vals = sc_comm.evalFuncShaderBuiltinRGBNorm(
                            func_stage.builtin,
                            coord,
                            func_stage.params,
                        );
                        inline for (0..3) |cc| {
                            state.value[cc] = vals[cc] * func_stage.scale_mul +
                                func_stage.scale_add;
                        }
                    },
                    .scale => |sc| {
                        inline for (0..3) |cc| state.value[cc] *= sc.factor[cc];
                    },
                    .offset => |off| {
                        inline for (0..3) |cc| state.value[cc] += off.offset[cc];
                    },
                    .clamp => |cl| {
                        inline for (0..3) |cc| {
                            state.value[cc] = @min(
                                @max(state.value[cc], cl.min[cc]),
                                cl.max[cc],
                            );
                        }
                    },
                    .linear_map => |lmap| {
                        inline for (0..3) |cc| {
                            state.value[cc] = state.value[cc] * lmap.scale_mul[cc] +
                                lmap.scale_add[cc];
                        }
                    },
                    .lut => |lut_stage| {
                        _ = lut_stage;
                    },
                }
            }

            const final_val = pipe.terminal.eval(&state);
            inline for (0..3) |cc| {
                spx_img_scratch.set(cc, ctx_shade.scratch_idx, final_val[cc]);
            }
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
            if (comptime @TypeOf(ctx_report).mode_tag == .full_stats) {
                ctx_report.recordDepth(
                    ctx_shade.global_subx,
                    ctx_shade.global_suby,
                    1.0 / interp.sub_pixel_z,
                );
                const normal = pipe_buf.interpNormal(interp.weights);
                ctx_report.recordNormal(
                    ctx_shade.global_subx,
                    ctx_shade.global_suby,
                    normal[0],
                    normal[1],
                    normal[2],
                );
            }

            var state = shaderpipe.MultiPipeState(C){ .value = undefined };
            const nodal = pipe.nodal;

            inline for (0..C) |cc| {
                const base = cc * N;
                if (comptime coord_space == .raster) {
                    var sum: F = 0.0;
                    inline for (0..N) |nn| {
                        sum += interp.weights[nn] *
                            interp.nodes_inv_z[nn] *
                            pipe_buf.nodal_data[base + nn];
                    }
                    state.value[cc] = (sum * interp.sub_pixel_z) *
                        nodal.scale_mul + nodal.scale_add;
                } else {
                    var sum: F = 0.0;
                    inline for (0..N) |nn| {
                        sum += interp.weights[nn] * pipe_buf.nodal_data[base + nn];
                    }
                    state.value[cc] = sum * nodal.scale_mul + nodal.scale_add;
                }
            }

            const terminal = shaderpipe.MultiTerminal(C){};
            const final_val = terminal.eval(&state);
            inline for (0..C) |cc| {
                spx_img_scratch.set(cc, ctx_shade.scratch_idx, final_val[cc]);
            }
        }
    };
}
