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
const cam = @import("camera.zig");
const geomkerns = @import("geometrykernels.zig");
const imageops = @import("imageops.zig");
const matslice = @import("matslice.zig");
const mo = @import("meshpipeline.zig");
const ndarray = @import("ndarray.zig");
const rastcfg = @import("rasterconfig.zig");
const report = @import("report.zig");
const shaderops = @import("shaderops_common.zig");
const texops = @import("textureops.zig");

const F = buildconfig.F;

// --------------------------------------------------------------------------------------
// File Constants & Tolerances
// --------------------------------------------------------------------------------------

const ROTATION_UNIT_TOLERANCE: F = 1.0e-2;
const ROTATION_ORTHO_TOLERANCE: F = 1.0e-2;
const ROTATION_DET_TOLERANCE: F = 1.0e-2;
const MIN_CAMERA_ROI_DIST: F = 1.0e-6;

// --------------------------------------------------------------------------------------
// Public Error Set & Summary Types
// --------------------------------------------------------------------------------------

pub const InputValidationError = error{
    NoRenderGroups,
    InvalidRenderGroupWorkers,
    FullStatsRequiresSingleRasterWorker,
    NoCameras,
    InvalidCameraPixels,
    InvalidCameraPixelSize,
    InvalidCameraFocalLength,
    InvalidCameraSubSample,
    InvalidCameraRoi,
    InvalidCameraRotation,
    InvalidCameraDistortion,
    InvalidCameraPsf,
    DistortionNotSuppedWithTri3Opt,
    NoMeshes,
    ZeroCoordinateCount,
    ZeroElementCount,
    InvalidCoordinateDimensions,
    InvalidConnectivityDimensions,
    InvalidDisplacementDimensions,
    InvalidNodalFieldDimensions,
    InvalidUvDimensions,
    InvalidTextureDimensions,
    InvalidTexSampleConfig,
    InvalidShaderBitDepth,
    InvalidScalingBounds,
    InvalidFuncShaderParams,
    MissingNodalField,
    MissingUvs,
    InvalidMeshType,
    InvalidTotalThreads,
    InvalidFrameBatchSize,
    InvalidGeomJobsInFlight,
    InvalidGeomWorkersPerJob,
    InvalidRasterWorkersPerJob,
    InvalidTileSizeMin,
    InvalidTileSizeMax,
    InvalidTileSizeRange,
    InvalidTileSizeOverride,
    InvalidGlobalSubpxTileSizeMin,
    InvalidGlobalSubpxTileSizeMax,
    InvalidGlobalSubpxTileSizeRange,
    InvalidGlobalSubpxTileSizeOverride,
    GlobalSubpxTileSizeNotAligned,
    InvalidGlobalSubpxStripeSizeMin,
    InvalidGlobalSubpxStripeSizeMax,
    InvalidGlobalSubpxStripeSizeRange,
    InvalidGlobalSubpxStripeSizeOverride,
    GlobalSubpxStripeSizeNotAligned,
    InvalidBackgroundValue,
    InvalidSaveFrameBuffCount,
    InvalidImageSaveOpts,
    InvalidFullStatsFormats,
    InvalidBenchCaptureBuff,
    InvalidRasterHaloPxOverride,
    UnsuppedImageModeFieldCount,
    InvalidOutputBuff,
    InvalidValidateInputMode,
    NoMeshFrames,
    NoOutputFields,
};

pub const ValidSummary = struct {
    num_time: usize,
    raw_num_fields: u8,
    out_num_fields: u8,
    img_dims: [5]usize,
};

pub const RenderSummary = ValidSummary;

// --------------------------------------------------------------------------------------
// Public Entry-Point Functions
// --------------------------------------------------------------------------------------

/// Performs fast validation (metadata, shapes, dimensions, cross-object compatibility)
/// without scanning large array payloads. Returns a summary of scene dimensions.
pub fn checkRenderInps(
    render_groups: anytype,
    cam_inps: []const cam.CameraInput,
    meshes: []const mo.MeshInput,
    config: rastcfg.RasterConfig,
    imgs_arr: ?*ndarray.NDArray(F),
    require_out_buff: bool,
    bench_capt: ?[]report.FrameBenchCapture,
) InputValidationError!ValidSummary {
    try checkTopLevelAndRenderGroups(render_groups, config);
    try checkRasterConfig(config);
    try checkMeshesMetadata(meshes, cam_inps);
    try checkCameras(cam_inps, config);

    const num_time = mo.countFrames(meshes);
    if (num_time == 0) {
        return error.NoMeshFrames;
    }

    const raw_num_fields = mo.countOutputFields(meshes);
    if (raw_num_fields == 0) {
        return error.NoOutputFields;
    }

    if (bench_capt) |capt| {
        if (capt.len != cam_inps.len * num_time) {
            return error.InvalidBenchCaptureBuff;
        }
    }

    const out_num_fields = try calcOutFieldsForImgSaveMode(
        config.image_save_mode,
        raw_num_fields,
    );
    const img_dims = calcAllFramesImgDims(
        cam_inps,
        num_time,
        out_num_fields,
    );
    try validOutBuffErr(
        config,
        imgs_arr,
        require_out_buff,
        img_dims,
    );

    return .{
        .num_time = num_time,
        .raw_num_fields = raw_num_fields,
        .out_num_fields = out_num_fields,
        .img_dims = img_dims,
    };
}

/// Computes metadata needed for allocations in `.off` mode without executing validation.
pub fn summariseRenderInpsAssumeValid(
    cam_inps: []const cam.CameraInput,
    meshes: []const mo.MeshInput,
    config: rastcfg.RasterConfig,
) ValidSummary {
    const num_time = if (meshes.len > 0) mo.countFrames(meshes) else 1;
    const raw_num_fields = if (meshes.len > 0) mo.countOutputFields(meshes) else 1;
    const out_num_fields = calcOutFieldsForImgSaveMode(
        config.image_save_mode,
        raw_num_fields,
    ) catch raw_num_fields;
    const img_dims = if (cam_inps.len > 0)
        calcAllFramesImgDims(cam_inps, num_time, out_num_fields)
    else
        [5]usize{ 1, num_time, out_num_fields, 1, 1 };

    return .{
        .num_time = num_time,
        .raw_num_fields = raw_num_fields,
        .out_num_fields = out_num_fields,
        .img_dims = img_dims,
    };
}

// --------------------------------------------------------------------------------------
// Top-Level & Render Group Checks
// --------------------------------------------------------------------------------------

fn checkTopLevelAndRenderGroups(
    render_groups: anytype,
    config: rastcfg.RasterConfig,
) InputValidationError!void {
    if (render_groups.len == 0) {
        return error.NoRenderGroups;
    }
    for (render_groups) |render_group| {
        if (render_group.workers == 0) {
            return error.InvalidRenderGroupWorkers;
        }
        if (config.report == .full_stats and
            @min(render_group.workers, config.max_raster_workers_per_job) > 1)
        {
            return error.FullStatsRequiresSingleRasterWorker;
        }
    }
}

fn checkRasterConfig(config: rastcfg.RasterConfig) InputValidationError!void {
    if (config.total_threads == 0) {
        return error.InvalidTotalThreads;
    }
    if (config.frame_batch_size_per_group == 0) {
        return error.InvalidFrameBatchSize;
    }
    if (config.max_geom_jobs_in_flight_per_group == 0) {
        return error.InvalidGeomJobsInFlight;
    }
    if (config.max_geom_workers_per_job == 0) {
        return error.InvalidGeomWorkersPerJob;
    }
    if (config.max_raster_workers_per_job == 0) {
        return error.InvalidRasterWorkersPerJob;
    }
    if (config.tile_size_min == 0) {
        return error.InvalidTileSizeMin;
    }
    if (config.tile_size_max == 0) {
        return error.InvalidTileSizeMax;
    }
    if (config.tile_size_min > config.tile_size_max) {
        return error.InvalidTileSizeRange;
    }
    if (config.tile_size_override) |tile_size_override| {
        if (tile_size_override < config.tile_size_min or
            tile_size_override > config.tile_size_max)
        {
            return error.InvalidTileSizeOverride;
        }
    }
    if (config.global_subpx_tile_size_min == 0) {
        return error.InvalidGlobalSubpxTileSizeMin;
    }
    if (config.global_subpx_tile_size_max == 0) {
        return error.InvalidGlobalSubpxTileSizeMax;
    }
    if (config.global_subpx_tile_size_min > config.global_subpx_tile_size_max) {
        return error.InvalidGlobalSubpxTileSizeRange;
    }
    if (config.global_subpx_tile_size_override) |tile_size_override| {
        if (tile_size_override < config.global_subpx_tile_size_min or
            tile_size_override > config.global_subpx_tile_size_max)
        {
            return error.InvalidGlobalSubpxTileSizeOverride;
        }
    }
    if (config.global_subpx_stripe_size_min == 0) {
        return error.InvalidGlobalSubpxStripeSizeMin;
    }
    if (config.global_subpx_stripe_size_max == 0) {
        return error.InvalidGlobalSubpxStripeSizeMax;
    }
    if (config.global_subpx_stripe_size_min > config.global_subpx_stripe_size_max) {
        return error.InvalidGlobalSubpxStripeSizeRange;
    }
    if (config.global_subpx_stripe_size_override) |stripe_size_override| {
        if (stripe_size_override < config.global_subpx_stripe_size_min or
            stripe_size_override > config.global_subpx_stripe_size_max)
        {
            return error.InvalidGlobalSubpxStripeSizeOverride;
        }
    }
    if (!std.math.isFinite(config.background_value)) {
        return error.InvalidBackgroundValue;
    }
    if (config.save_frame_buff_count == 0) {
        return error.InvalidSaveFrameBuffCount;
    }
    if ((config.save_strategy == .disk or config.save_strategy == .both) and
        config.image_save_opts.len == 0)
    {
        return error.InvalidImageSaveOpts;
    }
    if (config.report == .full_stats and config.full_stats_opts.formats.len == 0) {
        return error.InvalidFullStatsFormats;
    }
}

// --------------------------------------------------------------------------------------
// Mesh & Array Metadata Checks
// --------------------------------------------------------------------------------------

fn checkMeshesMetadata(
    meshes: []const mo.MeshInput,
    cam_inps: []const cam.CameraInput,
) InputValidationError!void {
    if (meshes.len == 0) {
        return error.NoMeshes;
    }

    for (meshes) |mesh| {
        if (mesh.coords.mat.rows_num == 0) {
            return error.ZeroCoordinateCount;
        }
        if (mesh.coords.mat.cols_num != 3 or
            mesh.coords.mem.len != mesh.coords.mat.rows_num * 3)
        {
            return error.InvalidCoordinateDimensions;
        }

        const expected_nodes_per_elem = mesh.mesh_type.getNodesNum();
        if (mesh.connect.table.rows_num == 0) {
            return error.ZeroElementCount;
        }
        if (mesh.connect.table.cols_num != expected_nodes_per_elem or
            mesh.connect.table_mem.len !=
                mesh.connect.table.rows_num * mesh.connect.table.cols_num)
        {
            return error.InvalidConnectivityDimensions;
        }

        if (mesh.disp) |disp_field| {
            if (disp_field.array.dims.len != 3 or
                disp_field.array.dims[0] == 0 or
                disp_field.array.dims[1] != mesh.coords.mat.rows_num or
                disp_field.array.dims[2] != 3 or
                disp_field.array_mem.len !=
                    disp_field.array.dims[0] * disp_field.array.dims[1] * disp_field.array.dims[2])
            {
                return error.InvalidDisplacementDimensions;
            }
        }

        if (mesh.mesh_type == .tri3opt) {
            for (cam_inps) |cam_inp| {
                if (!cam.isNoDistortion(cam_inp.distortion)) {
                    return error.DistortionNotSuppedWithTri3Opt;
                }
            }
        }

        try checkShaderMetadata(mesh.shader, mesh.coords.mat.rows_num);
    }
}

// --------------------------------------------------------------------------------------
// Camera & PSF Checks
// --------------------------------------------------------------------------------------

fn checkCameras(
    cam_inps: []const cam.CameraInput,
    config: rastcfg.RasterConfig,
) InputValidationError!void {
    if (cam_inps.len == 0) {
        return error.NoCameras;
    }
    for (cam_inps) |cam_inp| {
        try checkCamInp(cam_inp);
        try checkGlobalSubpxAlignment(config, cam_inp.sub_sample);
    }
}

fn checkCamInp(cam_inp: cam.CameraInput) InputValidationError!void {
    if (cam_inp.pixels_num[0] == 0 or cam_inp.pixels_num[1] == 0) {
        return error.InvalidCameraPixels;
    }
    if (!isFiniteSlice(&cam_inp.pixels_size) or
        cam_inp.pixels_size[0] <= 0.0 or
        cam_inp.pixels_size[1] <= 0.0)
    {
        return error.InvalidCameraPixelSize;
    }
    if (!std.math.isFinite(cam_inp.focal_length) or
        cam_inp.focal_length <= 0.0)
    {
        return error.InvalidCameraFocalLength;
    }
    if (cam_inp.sub_sample == 0) {
        return error.InvalidCameraSubSample;
    }
    if (!isFiniteVec3(cam_inp.pos_world) or
        !isFiniteVec3(cam_inp.roi_cent_world))
    {
        return error.InvalidCameraRoi;
    }

    const dx = cam_inp.pos_world.get(0) - cam_inp.roi_cent_world.get(0);
    const dy = cam_inp.pos_world.get(1) - cam_inp.roi_cent_world.get(1);
    const dz = cam_inp.pos_world.get(2) - cam_inp.roi_cent_world.get(2);
    const dist_sq = dx * dx + dy * dy + dz * dz;
    if (dist_sq < MIN_CAMERA_ROI_DIST * MIN_CAMERA_ROI_DIST) {
        return error.InvalidCameraRoi;
    }

    if (!isFiniteSlice(cam_inp.rot_world.matrix.slice[0..])) {
        return error.InvalidCameraRotation;
    }
    if (!isValidRotationMatrix(cam_inp.rot_world.matrix)) {
        return error.InvalidCameraRotation;
    }
    if (!isValidDistortion(cam_inp.distortion)) {
        return error.InvalidCameraDistortion;
    }
    if (!isValidPsf(cam_inp.psf)) {
        return error.InvalidCameraPsf;
    }
}

fn isValidRotationMatrix(mat: anytype) bool {
    const r00 = mat.get(0, 0);
    const r01 = mat.get(0, 1);
    const r02 = mat.get(0, 2);
    const r10 = mat.get(1, 0);
    const r11 = mat.get(1, 1);
    const r12 = mat.get(1, 2);
    const r20 = mat.get(2, 0);
    const r21 = mat.get(2, 1);
    const r22 = mat.get(2, 2);

    const norm_row0 = r00 * r00 + r01 * r01 + r02 * r02;
    const norm_row1 = r10 * r10 + r11 * r11 + r12 * r12;
    const norm_row2 = r20 * r20 + r21 * r21 + r22 * r22;

    if (@abs(norm_row0 - 1.0) > ROTATION_UNIT_TOLERANCE or
        @abs(norm_row1 - 1.0) > ROTATION_UNIT_TOLERANCE or
        @abs(norm_row2 - 1.0) > ROTATION_UNIT_TOLERANCE)
    {
        return false;
    }

    const dot01 = r00 * r10 + r01 * r11 + r02 * r12;
    const dot02 = r00 * r20 + r01 * r21 + r02 * r22;
    const dot12 = r10 * r20 + r11 * r21 + r12 * r22;

    if (@abs(dot01) > ROTATION_ORTHO_TOLERANCE or
        @abs(dot02) > ROTATION_ORTHO_TOLERANCE or
        @abs(dot12) > ROTATION_ORTHO_TOLERANCE)
    {
        return false;
    }

    const det = r00 * (r11 * r22 - r12 * r21) -
        r01 * (r10 * r22 - r12 * r20) +
        r02 * (r10 * r21 - r11 * r20);

    if (@abs(det - 1.0) > ROTATION_DET_TOLERANCE) {
        return false;
    }

    return true;
}

fn checkGlobalSubpxAlignment(
    config: rastcfg.RasterConfig,
    sub_samp: u32,
) InputValidationError!void {
    if (config.buffer_mode == .tile_local) return;

    const tile_size = config.global_subpx_tile_size_override orelse
        config.global_subpx_tile_size_min;
    if (@mod(tile_size, sub_samp) != 0) {
        return error.GlobalSubpxTileSizeNotAligned;
    }
    if (config.buffer_mode == .global_subpx_stripe) {
        const stripe_size = config.global_subpx_stripe_size_override orelse
            config.global_subpx_stripe_size_min;
        if (@mod(stripe_size, sub_samp) != 0) {
            return error.GlobalSubpxStripeSizeNotAligned;
        }
    }
}

// --------------------------------------------------------------------------------------
// Shader & Texture Metadata Checks
// --------------------------------------------------------------------------------------

fn checkShaderMetadata(
    shader: shaderops.ShaderInput,
    coords_num: usize,
) InputValidationError!void {
    switch (shader) {
        .tex_u8 => |tex_shader| {
            try checkTexMetadata(tex_shader.tex, 1);
            try checkTexUvMetadata(tex_shader.uvs, coords_num);
            try checkSampCfg(tex_shader.samp_cfg);
            try checkBits(tex_shader.bits);
            try checkScaling(tex_shader.scaling);
        },
        .tex_u16 => |tex_shader| {
            try checkTexMetadata(tex_shader.tex, 1);
            try checkTexUvMetadata(tex_shader.uvs, coords_num);
            try checkSampCfg(tex_shader.samp_cfg);
            try checkBits(tex_shader.bits);
            try checkScaling(tex_shader.scaling);
        },
        .tex_f => |tex_shader| {
            try checkTexMetadata(tex_shader.tex, 1);
            try checkTexUvMetadata(tex_shader.uvs, coords_num);
            try checkSampCfg(tex_shader.samp_cfg);
            try checkBits(tex_shader.bits);
            try checkScaling(tex_shader.scaling);
        },
        .tex_rgb_u8 => |tex_shader| {
            try checkTexMetadata(tex_shader.tex, 3);
            try checkTexUvMetadata(tex_shader.uvs, coords_num);
            try checkSampCfg(tex_shader.samp_cfg);
            try checkBits(tex_shader.bits);
            try checkScaling(tex_shader.scaling);
        },
        .tex_rgb_u16 => |tex_shader| {
            try checkTexMetadata(tex_shader.tex, 3);
            try checkTexUvMetadata(tex_shader.uvs, coords_num);
            try checkSampCfg(tex_shader.samp_cfg);
            try checkBits(tex_shader.bits);
            try checkScaling(tex_shader.scaling);
        },
        .tex_rgb_f => |tex_shader| {
            try checkTexMetadata(tex_shader.tex, 3);
            try checkTexUvMetadata(tex_shader.uvs, coords_num);
            try checkSampCfg(tex_shader.samp_cfg);
            try checkBits(tex_shader.bits);
            try checkScaling(tex_shader.scaling);
        },
        .nodal => |nodal_shader| {
            const field_dims = nodal_shader.field.array.dims;
            if (field_dims.len != 3 or
                field_dims[0] == 0 or
                field_dims[1] != coords_num or
                field_dims[2] == 0 or
                field_dims[2] > buildconfig.config.max_nodal_fields)
            {
                return error.InvalidNodalFieldDimensions;
            }
            try checkBits(nodal_shader.bits);
            try checkScaling(nodal_shader.scaling);
        },
        .func => |func_shader| {
            const require_uvs = func_shader.coord_mode == .uv;
            try checkOptionalUvMetadata(func_shader.uvs, coords_num, require_uvs);
            try checkBits(func_shader.bits);
            try checkScaling(func_shader.scaling);
            try checkFuncParams(func_shader.builtin, func_shader.params);
        },
        .func_rgb => |func_shader| {
            const require_uvs = func_shader.coord_mode == .uv;
            try checkOptionalUvMetadata(func_shader.uvs, coords_num, require_uvs);
            try checkBits(func_shader.bits);
            try checkScaling(func_shader.scaling);
            try checkFuncParams(func_shader.builtin, func_shader.params);
        },
    }
}

fn checkTexMetadata(tex: anytype, expected_channels: usize) InputValidationError!void {
    const dims = tex.array.dims;
    if (dims.len != 3 or
        dims[0] != expected_channels or
        dims[1] == 0 or
        dims[2] == 0 or
        tex.rows_num != dims[1] or
        tex.cols_num != dims[2])
    {
        return error.InvalidTextureDimensions;
    }
}

fn checkTexUvMetadata(
    uvs: ndarray.NDArray(F),
    coords_num: usize,
) InputValidationError!void {
    if (uvs.dims.len != 2 or
        uvs.dims[0] != coords_num or
        uvs.dims[1] != 2)
    {
        return error.InvalidUvDimensions;
    }
}

fn checkOptionalUvMetadata(
    uvs_opt: ?ndarray.NDArray(F),
    coords_num: usize,
    required: bool,
) InputValidationError!void {
    if (uvs_opt) |uvs| {
        if (uvs.dims.len != 2 or
            uvs.dims[0] != coords_num or
            uvs.dims[1] != 2)
        {
            return error.InvalidUvDimensions;
        }
    } else if (required) {
        return error.MissingUvs;
    }
}

fn checkSampCfg(samp_cfg: texops.TexSampConfig) InputValidationError!void {
    if (!samp_cfg.isValid()) {
        return error.InvalidTexSampleConfig;
    }
}

fn checkBits(bits_opt: ?u8) InputValidationError!void {
    if (bits_opt) |bits| {
        if (bits != 8 and bits != 10 and bits != 12 and
            bits != 14 and bits != 16)
        {
            return error.InvalidShaderBitDepth;
        }
    }
}

fn checkScaling(scaling: imageops.ScaleStrategy) InputValidationError!void {
    switch (scaling) {
        .none, .auto => {},
        .fixed => |bounds| {
            if (!isFiniteSlice(&bounds) or bounds[0] > bounds[1]) {
                return error.InvalidScalingBounds;
            }
        },
        .frac => |bounds| {
            if (!isFiniteSlice(&bounds) or
                bounds[0] < 0.0 or
                bounds[1] > 1.0 or
                bounds[0] > bounds[1])
            {
                return error.InvalidScalingBounds;
            }
        },
    }
}

fn checkFuncParams(
    builtin: shaderops.FuncShaderBuiltin,
    params: shaderops.FuncShaderParams,
) InputValidationError!void {
    if (!isFiniteSlice(&params.coord_scale) or
        params.coord_scale[0] == 0.0 or
        params.coord_scale[1] == 0.0 or
        !isFiniteSlice(&params.coord_offset) or
        !std.math.isFinite(params.output_scale) or
        !std.math.isFinite(params.output_offset))
    {
        return error.InvalidFuncShaderParams;
    }

    if (std.meta.activeTag(params.settings) != builtin) {
        return error.InvalidFuncShaderParams;
    }

    switch (params.settings) {
        .constant => |c| {
            if (!std.math.isFinite(c.value) or !isFiniteSlice(&c.value_rgb)) {
                return error.InvalidFuncShaderParams;
            }
        },
        .linear => |lin| {
            if (!isFiniteSlice(&lin.coeffs) or !isFinite2D(3, 3, &lin.coeffs_rgb)) {
                return error.InvalidFuncShaderParams;
            }
        },
        .quadratic => |quad| {
            if (!isFiniteSlice(&quad.coeffs) or !isFinite2D(3, 6, &quad.coeffs_rgb)) {
                return error.InvalidFuncShaderParams;
            }
        },
        .sinusoidal, .sinusoidal_approx => |sin| {
            if (!isFiniteSlice(&sin.wave_num_scalar) or
                !isFiniteSlice(&sin.wave_num_rgb) or
                !isFiniteSlice(&sin.amplitudes) or
                !isFiniteSlice(&sin.amplitudes_rgb) or
                !std.math.isFinite(sin.bias) or
                !isFiniteSlice(&sin.bias_rgb))
            {
                return error.InvalidFuncShaderParams;
            }
        },
        .checker => |chk| {
            if (!isFiniteSlice(&chk.levels) or !isFinite2D(2, 3, &chk.levels_rgb)) {
                return error.InvalidFuncShaderParams;
            }
        },
        .checker_smooth => |chk| {
            if (!std.math.isFinite(chk.frequency) or chk.frequency <= 0.0) {
                return error.InvalidFuncShaderParams;
            }
        },
        .lambertian_normal_z => |lam| {
            if (!isFiniteSlice(&lam.coeffs) or !isFinite2D(3, 2, &lam.coeffs_rgb)) {
                return error.InvalidFuncShaderParams;
            }
        },
        .eggbox => |egg| {
            if (!std.math.isFinite(egg.mean) or
                !std.math.isFinite(egg.contrast) or
                !isFiniteSlice(&egg.pitch) or
                egg.pitch[0] == 0.0 or
                egg.pitch[1] == 0.0 or
                !isFiniteSlice(&egg.phase))
            {
                return error.InvalidFuncShaderParams;
            }
        },
    }
}

// --------------------------------------------------------------------------------------
// Output & Save Checks
// --------------------------------------------------------------------------------------

pub fn calcOutFieldsForImgSaveMode(
    img_save_mode: rastcfg.ImageSaveMode,
    raw_num_fields: u8,
) InputValidationError!u8 {
    return switch (img_save_mode) {
        .multifield => raw_num_fields,
        .grey => switch (raw_num_fields) {
            1, 3 => 1,
            else => error.UnsuppedImageModeFieldCount,
        },
        .rgb => switch (raw_num_fields) {
            1, 3 => 3,
            else => error.UnsuppedImageModeFieldCount,
        },
    };
}

fn calcAllFramesImgDims(
    cam_inps: []const cam.CameraInput,
    num_time: usize,
    out_num_fields: u8,
) [5]usize {
    std.debug.assert(cam_inps.len > 0);

    var max_pix_num = cam_inps[0].pixels_num;
    for (cam_inps[1..]) |cam_inp| {
        max_pix_num[0] = @max(max_pix_num[0], cam_inp.pixels_num[0]);
        max_pix_num[1] = @max(max_pix_num[1], cam_inp.pixels_num[1]);
    }

    return .{
        cam_inps.len,
        num_time,
        @as(usize, out_num_fields),
        max_pix_num[1],
        max_pix_num[0],
    };
}

fn validOutBuffErr(
    config: rastcfg.RasterConfig,
    imgs_arr: ?*ndarray.NDArray(F),
    require_out_buff: bool,
    exp_dims: [5]usize,
) InputValidationError!void {
    if (config.save_strategy == .memory or config.save_strategy == .both) {
        if (imgs_arr) |imgs_arr_req| {
            try validAllFramesBuff(imgs_arr_req, exp_dims);
        } else if (require_out_buff) {
            return error.InvalidOutputBuff;
        }
    } else if (imgs_arr != null) {
        return error.InvalidOutputBuff;
    }
}

fn validAllFramesBuff(
    imgs_arr: *const ndarray.NDArray(F),
    exp_dims: [5]usize,
) InputValidationError!void {
    if (imgs_arr.dims.len != exp_dims.len) {
        return error.InvalidOutputBuff;
    }
    for (exp_dims, 0..) |exp_dim, dd| {
        if (imgs_arr.dims[dd] != exp_dim) {
            return error.InvalidOutputBuff;
        }
    }
}

// --------------------------------------------------------------------------------------
// Low-Level Distortion & PSF Validators
// --------------------------------------------------------------------------------------

fn isValidPolynomialMap(map: cam.PolynomialMap) bool {
    const term_count = map.order.termCount();
    return isFiniteSlice(map.coeffs_u[0..term_count]) and
        isFiniteSlice(map.coeffs_v[0..term_count]);
}

fn isValidBidirectionalPolynomial(poly: cam.BidirectionalPolynomial) bool {
    if (poly.forward_map == null and poly.inv_map == null) {
        return false;
    }
    if (poly.forward_map) |forward_map| {
        if (!isValidPolynomialMap(forward_map)) return false;
    }
    if (poly.inv_map) |inv_map| {
        if (!isValidPolynomialMap(inv_map)) return false;
    }
    return true;
}

fn isValidDistortion(distortion: cam.DistortionModel) bool {
    return switch (distortion) {
        .none => true,
        .brown_conrady => |bc| isFiniteSlice(&[_]F{
            bc.k1,
            bc.k2,
            bc.k3,
            bc.p1,
            bc.p2,
        }),
        .brown_conrady_ext => |bc| isFiniteSlice(&[_]F{
            bc.k1,
            bc.k2,
            bc.k3,
            bc.k4,
            bc.k5,
            bc.k6,
            bc.p1,
            bc.p2,
            bc.s1,
            bc.s2,
            bc.s3,
            bc.s4,
            bc.tau_x,
            bc.tau_y,
        }),
        .polynomial => |poly| isValidBidirectionalPolynomial(poly),
        .brown_conrady_polynomial => |chain| isFiniteSlice(&[_]F{
            chain.brown_conrady.k1,
            chain.brown_conrady.k2,
            chain.brown_conrady.k3,
            chain.brown_conrady.p1,
            chain.brown_conrady.p2,
        }) and isValidBidirectionalPolynomial(chain.polynomial),
        .brown_conrady_ext_polynomial => |chain| isFiniteSlice(&[_]F{
            chain.brown_conrady_ext.k1,
            chain.brown_conrady_ext.k2,
            chain.brown_conrady_ext.k3,
            chain.brown_conrady_ext.k4,
            chain.brown_conrady_ext.k5,
            chain.brown_conrady_ext.k6,
            chain.brown_conrady_ext.p1,
            chain.brown_conrady_ext.p2,
            chain.brown_conrady_ext.s1,
            chain.brown_conrady_ext.s2,
            chain.brown_conrady_ext.s3,
            chain.brown_conrady_ext.s4,
            chain.brown_conrady_ext.tau_x,
            chain.brown_conrady_ext.tau_y,
        }) and isValidBidirectionalPolynomial(chain.polynomial),
    };
}

fn isValidPsf(psf: cam.PointSpreadFunc) bool {
    return switch (psf) {
        .pixel_box => |box| std.math.isFinite(box.supp_rad_px) and
            box.supp_rad_px >= 0.0,
        .gaussian => |gauss| isFiniteSlice(&[_]F{
            gauss.sigma_px,
            gauss.supp_rad_px,
        }) and
            gauss.sigma_px > 0.0 and
            gauss.supp_rad_px >= 0.0,
        .anisotropic_gaussian => |gauss| isFiniteSlice(&[_]F{
            gauss.sigma_x_px,
            gauss.sigma_y_px,
            gauss.theta_rad,
            gauss.supp_rad_px,
        }) and
            gauss.sigma_x_px > 0.0 and
            gauss.sigma_y_px > 0.0 and
            gauss.supp_rad_px >= 0.0,
    };
}

// --------------------------------------------------------------------------------------
// Generic Low-Level Helpers
// --------------------------------------------------------------------------------------

fn isFiniteSlice(vals: []const F) bool {
    for (vals) |value| {
        if (!std.math.isFinite(value)) {
            return false;
        }
    }
    return true;
}

fn isFinite2D(comptime N1: usize, comptime N2: usize, arr: *const [N1][N2]F) bool {
    for (arr) |*row| {
        if (!isFiniteSlice(row)) {
            return false;
        }
    }
    return true;
}

fn isFiniteVec3(vec_val: anytype) bool {
    return isFiniteSlice(vec_val.slice[0..]);
}
