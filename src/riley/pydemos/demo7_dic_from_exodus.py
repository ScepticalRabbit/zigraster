# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

import riley


def main() -> None:
    # --------------------------------------------------------------------------
    # 1. Setup paths and parameters
    # --------------------------------------------------------------------------
    exodus_path = riley.data.platehole_exodus_path()
    texture_path = riley.data.speckle_texture_path()
    out_dir = Path.cwd() / "out_riley_py" / "demo7_dic_from_exodus"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)

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

    # --------------------------------------------------------------------------
    # 2. Load Exodus simulation, project UVs, and build mesh
    # --------------------------------------------------------------------------
    sim: riley.ExodusSim = riley.load_exodus(
        exodus_path,
        disp_keys=("disp_x", "disp_y", "disp_z"),
    )
    block = sim.elem_blocks["connect1"]
    assert sim.disp is not None

    frame_indices = riley.frames_first_last_idxs(sim.disp[0].shape[1])
    disp = tuple(item[:, frame_indices] for item in sim.disp)

    uvs = riley.project_uvs_planar_centered(
        sim.coords,
        pixels_num,
        uv_span_max=0.8,
        proj_plane=(
            np.array((0.0, 0.0, -1.0), dtype=np.float64),
            np.array((0.0, 0.0, 0.0), dtype=np.float64),
        ),
    )
    texture = riley.load_texture_mono_u8(texture_path)

    mesh: riley.Mesh = riley.create_mesh(
        convention=riley.ConnectConvention(
            elem_type=block.elem_type,
            elem_axis=riley.EConnectAxis.ROW,
            index_base=1,
            node_order=riley.ENodeOrder.EXODUS,
        ),
        mesh_type=riley.MeshType.quad8,
        coords=sim.coords,
        connect=block.connect,
        disp=disp,
        shader=riley.TextureShader(uvs=uvs, texture=texture),
    )
    coords = mesh.coords

    # --------------------------------------------------------------------------
    # 3. Create stereo cameras
    # --------------------------------------------------------------------------
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

    camera_1_rot = (0.0, float(np.deg2rad(stereo_angle_deg)), 0.0)
    camera_1_pos = riley.pos_frame_coords(
        coords,
        pixels_num,
        pixels_size,
        focal_length,
        camera_1_rot,
        fov_scale=fov_scale_factor,
    )

    camera_1 = copy.deepcopy(camera_0)
    camera_1.pos_world = camera_1_pos
    camera_1.rot_world = camera_1_rot

    # --------------------------------------------------------------------------
    # 4. Configure raster engine and render
    # --------------------------------------------------------------------------
    config = riley.create_raster_config(
        num_frames=mesh.disp.shape[0],
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

    # --------------------------------------------------------------------------
    # 5. Export stereo calibration data
    # --------------------------------------------------------------------------
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
