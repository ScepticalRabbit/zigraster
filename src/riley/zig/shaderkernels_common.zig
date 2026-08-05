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
const eval_branch_quota = buildconfig.comptime_eval_branch_quota;
const shaderops = @import("shaderops.zig");
const MatSlice = @import("matslice.zig").MatSlice;
const texops = @import("textureops.zig");
const TexSampConfig = texops.TexSampConfig;
const CoordSpace = @import("geometrykernels.zig").CoordSpace;

// --------------------------------------------------------------------------------------
// Nodal Interp Shader
// --------------------------------------------------------------------------------------

pub inline fn shadeNodalScalComm(
    comptime N: usize,
    comptime coord_space: CoordSpace,
    ctx_shade: shaderops.ShadeContext,
    interp: shaderops.InterpData(N),
    shader_buf: *const shaderops.LocalShaderBuff(N),
    shader: *const shaderops.NodalPrepared,
    ctx_report: anytype,
    spx_img_scratch: *MatSlice(F),
) void {
    if (comptime @TypeOf(ctx_report).mode_tag == .full_stats) {
        ctx_report.recordDepth(
            ctx_shade.global_subx,
            ctx_shade.global_suby,
            1.0 / interp.sub_pixel_z,
        );
    }

    if (shader.elem_normals != null) {
        const normal = shader_buf.interpNormal(interp.weights);
        if (comptime @TypeOf(ctx_report).mode_tag == .full_stats) {
            ctx_report.recordNormal(
                ctx_shade.global_subx,
                ctx_shade.global_suby,
                normal[0],
                normal[1],
                normal[2],
            );
        }
    }

    if (comptime coord_space == CoordSpace.clip_px_leng) {
        shaderops.fillNodalClipScal(
            N,
            ctx_shade,
            interp,
            shader_buf,
            shader,
            spx_img_scratch,
        );
    } else {
        shaderops.fillNodalPerspScal(
            N,
            ctx_shade,
            interp,
            shader_buf,
            shader,
            spx_img_scratch,
        );
    }
}

// --------------------------------------------------------------------------------------
// Texture Shader
// --------------------------------------------------------------------------------------

pub inline fn shadeTexScalComm(
    comptime N: usize,
    comptime T: type,
    comptime C: usize,
    comptime coord_space: CoordSpace,
    ctx_shade: shaderops.ShadeContext,
    interp: shaderops.InterpData(N),
    shader_buf: *const shaderops.LocalShaderBuff(N),
    shader: *const shaderops.TexPrepared(T, C),
    ctx_report: anytype,
    spx_img_scratch: *MatSlice(F),
) void {
    if (comptime @TypeOf(ctx_report).mode_tag == .full_stats) {
        ctx_report.recordDepth(
            ctx_shade.global_subx,
            ctx_shade.global_suby,
            1.0 / interp.sub_pixel_z,
        );
    }

    if (shader.elem_normals != null) {
        const normal = shader_buf.interpNormal(interp.weights);
        if (comptime @TypeOf(ctx_report).mode_tag == .full_stats) {
            ctx_report.recordNormal(
                ctx_shade.global_subx,
                ctx_shade.global_suby,
                normal[0],
                normal[1],
                normal[2],
            );
        }
    }

    shadeTexScalDispatchSample(
        N,
        T,
        C,
        coord_space,
        shader.samp_cfg,
        ctx_shade,
        interp,
        shader_buf,
        shader,
        spx_img_scratch,
    );
}

inline fn shadeTexScalDispatchSample(
    comptime N: usize,
    comptime T: type,
    comptime C: usize,
    comptime coord_space: CoordSpace,
    samp_cfg: TexSampConfig,
    ctx_shade: shaderops.ShadeContext,
    interp: shaderops.InterpData(N),
    shader_buf: *const shaderops.LocalShaderBuff(N),
    shader: *const shaderops.TexPrepared(T, C),
    spx_img_scratch: *MatSlice(F),
) void {
    @setEvalBranchQuota(eval_branch_quota);
    switch (samp_cfg.sample) {
        inline else => |samp_type| shadeTexScalDispatchMode(
            N,
            T,
            C,
            coord_space,
            samp_type,
            samp_cfg.mode,
            ctx_shade,
            interp,
            shader_buf,
            shader,
            spx_img_scratch,
        ),
    }
}

inline fn shadeTexScalDispatchMode(
    comptime N: usize,
    comptime T: type,
    comptime C: usize,
    comptime coord_space: CoordSpace,
    comptime samp_type: texops.TexSamp,
    mode: texops.TexSampMode,
    ctx_shade: shaderops.ShadeContext,
    interp: shaderops.InterpData(N),
    shader_buf: *const shaderops.LocalShaderBuff(N),
    shader: *const shaderops.TexPrepared(T, C),
    spx_img_scratch: *MatSlice(F),
) void {
    switch (mode) {
        inline else => |mode_type| {
            const samp_cfg = comptime (TexSampConfig{
                .sample = samp_type,
                .mode = mode_type,
            }).sanitize();

            if (comptime coord_space == CoordSpace.clip_px_leng) {
                shaderops.fillTexClipScal(
                    N,
                    T,
                    C,
                    samp_cfg,
                    ctx_shade,
                    interp,
                    shader_buf,
                    shader,
                    spx_img_scratch,
                );
            } else {
                shaderops.fillTexPerspScal(
                    N,
                    T,
                    C,
                    samp_cfg,
                    ctx_shade,
                    interp,
                    shader_buf,
                    shader,
                    spx_img_scratch,
                );
            }
        },
    }
}

// --------------------------------------------------------------------------------------
// Function Shader
// --------------------------------------------------------------------------------------

pub inline fn shadeFuncScalComm(
    comptime N: usize,
    comptime C: usize,
    comptime coord_space: CoordSpace,
    ctx_shade: shaderops.ShadeContext,
    interp: shaderops.InterpData(N),
    shader_buf: *const shaderops.LocalShaderBuff(N),
    shader: *const shaderops.FuncPrepared,
    ctx_report: anytype,
    spx_img_scratch: *MatSlice(F),
) void {
    if (comptime @TypeOf(ctx_report).mode_tag == .full_stats) {
        ctx_report.recordDepth(
            ctx_shade.global_subx,
            ctx_shade.global_suby,
            1.0 / interp.sub_pixel_z,
        );
    }

    if (shader.elem_normals != null) {
        const normal = shader_buf.interpNormal(interp.weights);
        if (comptime @TypeOf(ctx_report).mode_tag == .full_stats) {
            ctx_report.recordNormal(
                ctx_shade.global_subx,
                ctx_shade.global_suby,
                normal[0],
                normal[1],
                normal[2],
            );
        }
    }

    if (comptime coord_space == CoordSpace.clip_px_leng) {
        shaderops.fillFuncClipScal(
            N,
            C,
            ctx_shade,
            interp,
            shader_buf,
            shader,
            spx_img_scratch,
        );
    } else {
        shaderops.fillFuncPerspScal(
            N,
            C,
            ctx_shade,
            interp,
            shader_buf,
            shader,
            spx_img_scratch,
        );
    }
}
