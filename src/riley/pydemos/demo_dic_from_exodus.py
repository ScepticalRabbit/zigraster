# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import netCDF4
import numpy as np

import riley
from riley.pydemos.common import (
    first_last_frame_indices,
    make_demo_out_dir,
    select_frames,
)


def load_surface_sim(
    exodus_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = netCDF4.Dataset(str(exodus_path))
    coords_raw = np.column_stack((
        dataset.variables["coordx"][:],
        dataset.variables["coordy"][:],
        dataset.variables["coordz"][:],
    ))
    connect_raw = np.asarray(
        dataset.variables["connect1"][:],
        dtype=np.int64,
    )

    convention = riley.ConnectConvention(
        elem_type=riley.EElementType.HEX20,
        elem_axis=riley.EConnectAxis.ROW,
        index_base=1,
        node_order=riley.ENodeOrder.EXODUS,
    )
    mesh_vol = riley.convert_mesh(coords_raw, connect_raw, convention)
    mesh_surf = riley.extract_surface(mesh_vol)

    coord_to_idx = {
        tuple(coord): idx for idx, coord in enumerate(mesh_vol.coords)
    }
    surf_node_idxs = np.array(
        [coord_to_idx[tuple(coord)] for coord in mesh_surf.coords],
        dtype=np.int64,
    )

    disp_x = np.asarray(
        dataset.variables["vals_nod_var1"][:],
        dtype=np.float64,
    )
    disp_y = np.asarray(
        dataset.variables["vals_nod_var2"][:],
        dtype=np.float64,
    )
    disp_z = np.asarray(
        dataset.variables["vals_nod_var3"][:],
        dtype=np.float64,
    )

    disp = np.zeros(
        (disp_x.shape[0], surf_node_idxs.shape[0], 3),
        dtype=np.float64,
    )
    disp[:, :, 0] = disp_x[:, surf_node_idxs]
    disp[:, :, 1] = disp_y[:, surf_node_idxs]
    disp[:, :, 2] = disp_z[:, surf_node_idxs]
    return mesh_surf.coords, mesh_surf.connect, disp


def main() -> None:
    exodus_path = riley.data.platehole_exodus_path()
    texture_path = riley.data.speckle_texture_path()
    out_dir = make_demo_out_dir("demo-dicuq-from-exodus")
    pixels_num = (2464, 2056)
    pixels_size = (3.45e-6, 3.45e-6)
    focal_length = 50.0e-3
    fov_scale_factor = 0.65
    sub_sample = 2
    stereo_angle_deg = 20.0
    total_threads = 8
    distortion_model = {
        "distortion_model": 1,
        "distortion_k1": -0.2,
        "distortion_k2": 0.1,
        "distortion_k3": 0.0,
        "distortion_p1": 0.0001,
        "distortion_p2": -0.0001,
    }

    coords, connect, disp = load_surface_sim(exodus_path)
    frame_indices = first_last_frame_indices(disp.shape[0])
    disp = select_frames(disp, frame_indices)
    uvs = riley.project_uvs_planar_centered(
        coords,
        pixels_num,
        uv_span_max=0.8,
        proj_plane=(
            np.array((0.0, 0.0, -1.0), dtype=np.float64),
            np.array((0.0, 0.0, 0.0), dtype=np.float64),
        ),
    )
    texture = riley.load_texture_u8(texture_path)

    mesh = riley.Mesh(
        mesh_type=riley.MeshType.quad8,
        coords=coords,
        connect=connect,
        disp=disp,
        shader_type=riley.ShaderType.tex,
        uvs=uvs,
        texture=texture,
        sample=riley.TextureSample.cubic_catmull_rom,
        sample_mode=riley.TextureSampleMode.lut_lerp,
        bits=8,
        scaling_type=riley.ScaleStrategy.none,
    )

    roi_pos = riley.roi_cent_from_coords(coords)
    camera_0_pos = riley.pos_frame_coords(
        coords,
        pixels_num,
        pixels_size,
        focal_length,
        (0.0, 0.0, 0.0),
        fov_scale=fov_scale_factor,
    )
    camera_0 = riley.Camera(
        pixels_num=pixels_num,
        pixels_size=pixels_size,
        pos_world=camera_0_pos,
        rot_world=(0.0, 0.0, 0.0),
        roi_cent_world=roi_pos,
        focal_length=focal_length,
        sub_sample=sub_sample,
        **distortion_model,
    )
    camera_1_rot = (0.0, np.deg2rad(stereo_angle_deg), 0.0)
    camera_1_pos = riley.pos_frame_coords(
        coords,
        pixels_num,
        pixels_size,
        focal_length,
        camera_1_rot,
        fov_scale=fov_scale_factor,
    )
    camera_1 = riley.Camera(
        pixels_num=pixels_num,
        pixels_size=pixels_size,
        pos_world=camera_1_pos,
        rot_world=camera_1_rot,
        roi_cent_world=roi_pos,
        focal_length=focal_length,
        sub_sample=sub_sample,
        **distortion_model,
    )

    config = riley.create_raster_config(
        num_frames=disp.shape[0],
        total_threads=total_threads,
        save_strategy=riley.SaveStrategy.disk,
    )
    config.background_value = 128.0
    config.tile_size_max = 128
    config.save_scaling = riley.ScaleStrategy.none

    start_time = perf_counter()
    riley.raster([mesh], [camera_0, camera_1], config, out_dir=str(out_dir))
    elapsed_time = perf_counter() - start_time
    print(f"render time: {elapsed_time:.6f} s")

    riley.save_stereo_pair(
        str(out_dir),
        "stereo_data_opengl.csv",
        camera_0,
        camera_1,
    )
    riley.save_stereo_pair(
        str(out_dir),
        "stereo_data_opencv.csv",
        replace(camera_0, coord_sys=riley.CameraCoordSys.opencv),
        replace(camera_1, coord_sys=riley.CameraCoordSys.opencv),
    )
    print(f"rendered dicuq from exodus to {out_dir}")


if __name__ == "__main__":
    main()
