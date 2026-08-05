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
from time import perf_counter

import numpy as np
import riley

from riley.pydemos.common import make_demo_out_dir


def main() -> None:
    data_dir = riley.data.platehole_csv_case_path()
    texture_path = riley.data.speckle_texture_path()
    out_dir = make_demo_out_dir("demo-dicuq")
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

    coords, connect, uvs, disp = riley.load_sim_csvs(data_dir)
    texture = riley.load_texture(texture_path)

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
    camera_0_pos = riley.pos_fill_frame_from_rot(
        coords,
        pixels_num,
        pixels_size,
        focal_length,
        (0.0, 0.0, 0.0),
        fov_scale_factor,
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
    camera_1_pos = riley.pos_fill_frame_from_rot(
        coords,
        pixels_num,
        pixels_size,
        focal_length,
        camera_1_rot,
        fov_scale_factor,
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
        num_frames=2,
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

    riley.save_stereo_pair(str(out_dir), "stereo_data_opengl.csv", camera_0, camera_1)
    riley.save_stereo_pair(
        str(out_dir),
        "stereo_data_opencv.csv",
        replace(camera_0, coord_sys=riley.CameraCoordSys.opencv),
        replace(camera_1, coord_sys=riley.CameraCoordSys.opencv),
    )
    print(f"rendered dicuq to {out_dir}")


if __name__ == "__main__":
    main()
