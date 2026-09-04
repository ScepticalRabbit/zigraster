# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
import os
import platform
from pathlib import Path

from riley import data

# Add DLL directory to the search path on Windows to avoid "DLL load failed"
if platform.system().lower() == "windows":
    _current_dir = Path(__file__).resolve().parent
    # Search zig/ and cython/ subdirectories where DLLs are located
    for _sub_dir in ("zig", "cython"):
        _dll_dir = _current_dir / _sub_dir
        if _dll_dir.is_dir():
            try:
                os.add_dll_directory(str(_dll_dir))
            except AttributeError:
                # Fallback for older Python versions
                pass

from riley.cython.riley import (
    BufferMode,
    Camera,
    CameraCoordSys,
    CameraInput,
    EFrameFit,
    FrameFitMode,
    FuncCoordMode,
    FuncShaderBuiltin,
    FuncShaderParams,
    GeometrySchedulingMode,
    HullMode,
    ImageFormat,
    ImageSaveMode,
    Mesh,
    MeshType,
    FunctionShader,
    NodalShader,
    TextureShader,
    NewtonSeedMode,
    NewtonSeedReuse,
    NormalType,
    PsfType,
    RasterConfig,
    RenderMode,
    ReportMode,
    SaveStrategy,
    ScaleOver,
    ScaleStrategy,
    SubPixelCenterMap,
    TextureSample,
    TextureSampleMode,
    calc_pixel_resolution,
    coverage_to_fov_scale,
    fov_scale_to_coverage,
    load_camera,
    load_stereo_pair,
    pos_fill_frame_from_rot,
    pos_fill_frame_from_rot_over_meshes,
    pos_frame_coords,
    pos_frame_mesh,
    pos_frame_meshes,
    pos_orbit_cam,
    pos_stereo_pair,
    raster,
    roi_cent_from_coords,
    roi_cent_over_meshes,
    save_camera,
    save_stereo_pair,
)
from riley.python.rileyconfig import create_raster_config
from riley.python.textureio import (
    ETextureCoercion,
    load_texture_mono_u8,
    load_texture_mono_u16,
    load_texture_rgb_u8,
    load_texture_rgb_u16,
)
from riley.python.meshconv import (
    ConnectConvention,
    EConnectAxis,
    EdgeNode,
    EElementType,
    ENodeOrder,
    FaceNode,
    MeshConvErr,
    MeshGeometry,
    MeshVerifyErr,
    MeshVerifyIssue,
    UserTopology,
    convert_mesh,
    extract_surface,
    verify_mesh,
)
from riley.python.meshio import create_mesh, load_csv
from riley.python.uvtools import (
    EPlanarProjMode,
    EProjPlane,
    ProjPlane,
    project_uvs_planar_bbox,
    project_uvs_planar_centered,
)

__all__ = [
    "BufferMode",
    "Camera",
    "CameraCoordSys",
    "CameraInput",
    "ConnectConvention",
    "EConnectAxis",
    "EElementType",
    "EFrameFit",
    "ENodeOrder",
    "EPlanarProjMode",
    "EProjPlane",
    "EdgeNode",
    "FaceNode",
    "FrameFitMode",
    "FuncCoordMode",
    "FuncShaderBuiltin",
    "FuncShaderParams",
    "FunctionShader",
    "GeometrySchedulingMode",
    "HullMode",
    "ImageFormat",
    "ImageSaveMode",
    "Mesh",
    "MeshConvErr",
    "MeshGeometry",
    "MeshType",
    "NodalShader",
    "MeshVerifyErr",
    "MeshVerifyIssue",
    "NewtonSeedMode",
    "NewtonSeedReuse",
    "NormalType",
    "ProjPlane",
    "PsfType",
    "RasterConfig",
    "RenderMode",
    "ReportMode",
    "SaveStrategy",
    "ScaleOver",
    "ScaleStrategy",
    "SubPixelCenterMap",
    "TextureSample",
    "TextureSampleMode",
    "TextureShader",
    "ETextureCoercion",
    "UserTopology",
    "calc_pixel_resolution",
    "convert_mesh",
    "coverage_to_fov_scale",
    "create_raster_config",
    "create_mesh",
    "data",
    "extract_surface",
    "fov_scale_to_coverage",
    "load_camera",
    "load_csv",
    "load_stereo_pair",
    "load_texture_mono_u8",
    "load_texture_mono_u16",
    "load_texture_rgb_u8",
    "load_texture_rgb_u16",
    "pos_fill_frame_from_rot",
    "pos_fill_frame_from_rot_over_meshes",
    "pos_frame_coords",
    "pos_frame_mesh",
    "pos_frame_meshes",
    "pos_orbit_cam",
    "pos_stereo_pair",
    "project_uvs_planar_bbox",
    "project_uvs_planar_centered",
    "raster",
    "roi_cent_from_coords",
    "roi_cent_over_meshes",
    "save_camera",
    "save_stereo_pair",
    "verify_mesh",
]
