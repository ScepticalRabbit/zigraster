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
    # Search zig/ and cyth/ subdirectories where DLLs are located
    for _sub_dir in ("zig", "cyth"):
        _dll_dir = _current_dir / _sub_dir
        if _dll_dir.is_dir():
            try:
                os.add_dll_directory(str(_dll_dir))
            except AttributeError:
                # Fallback for older Python versions
                pass

from riley.cyth.riley import (
    Camera,
    CameraInput,
    CameraCoordSys,
    HullMode,
    Mesh,
    MeshInput,
    MeshType,
    GeometrySchedulingMode,
    ImageSaveMode,
    NewtonSeedMode,
    NewtonSeedReuse,
    NormalType,
    RasterConfig,
    RenderMode,
    ReportMode,
    SaveStrategy,
    ScaleOver,
    ScaleStrategy,
    ShaderType,
    TextureStorage,
    SubPixelCenterMap,
    FuncShaderBuiltin,
    FuncCoordMode,
    FuncShaderParams,
    TextureSample,
    TextureSampleMode,
    PsfType,
    ImageFormat,
    load_camera,
    load_stereo_pair,
    pos_fill_frame_from_rot,
    pos_fill_frame_from_rot_over_meshes,
    raster,
    roi_cent_from_coords,
    roi_cent_over_meshes,
    save_camera,
    save_stereo_pair,
)
from riley.python.enums import (
    ConnectCsvOrientation,
    ConnectIndexing,
    CoordCsvOrientation,
    FieldCsvOrientation,
    PlanarProjectionMode,
    ProjectionPlane,
)
from riley.python.helpers import create_raster_config, load_texture
from riley.python.meshio import (
    load_connect_csv,
    load_coord_csv,
    load_disp_csvs,
    load_field_csv,
    load_field_csvs,
    load_sim_csvs,
)
from riley.python.meshtools import (
    enforce_mesh_convention,
    extract_surface_mesh,
    project_uvs_planar_bbox,
    project_uvs_planar_centered,
)

__all__ = [
    "Camera",
    "CameraInput",
    "CameraCoordSys",
    "ConnectCsvOrientation",
    "ConnectIndexing",
    "CoordCsvOrientation",
    "data",
    "FieldCsvOrientation",
    "PlanarProjectionMode",
    "ProjectionPlane",
    "HullMode",
    "Mesh",
    "MeshInput",
    "MeshType",
    "GeometrySchedulingMode",
    "ImageSaveMode",
    "NewtonSeedMode",
    "NewtonSeedReuse",
    "NormalType",
    "create_raster_config",
    "RasterConfig",
    "RenderMode",
    "ReportMode",
    "SaveStrategy",
    "ScaleOver",
    "ScaleStrategy",
    "ShaderType",
    "TextureStorage",
    "SubPixelCenterMap",
    "FuncShaderBuiltin",
    "FuncCoordMode",
    "FuncShaderParams",
    "TextureSample",
    "TextureSampleMode",
    "PsfType",
    "ImageFormat",
    "enforce_mesh_convention",
    "extract_surface_mesh",
    "load_connect_csv",
    "load_coord_csv",
    "load_disp_csvs",
    "load_field_csv",
    "load_field_csvs",
    "load_sim_csvs",
    "load_texture",
    "load_camera",
    "load_stereo_pair",
    "project_uvs_planar_bbox",
    "project_uvs_planar_centered",
    "pos_fill_frame_from_rot",
    "pos_fill_frame_from_rot_over_meshes",
    "raster",
    "roi_cent_from_coords",
    "roi_cent_over_meshes",
    "save_camera",
    "save_stereo_pair",
]
