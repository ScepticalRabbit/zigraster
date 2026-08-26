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


def project_nodes_to_uv(
    coords: np.ndarray,
    camera: riley.Camera,
) -> np.ndarray:
    """Project 3D world coordinates to normalized UV space [0, 1] x [0, 1]."""
    alpha_z, beta_y, gamma_x = camera.rot_world
    rot_mat = np.zeros((3, 3), dtype=np.float64)
    rot_mat[0, 0] = np.cos(alpha_z) * np.cos(beta_y)
    rot_mat[0, 1] = (
        np.cos(alpha_z) * np.sin(beta_y) * np.sin(gamma_x)
        - np.sin(alpha_z) * np.cos(gamma_x)
    )
    rot_mat[0, 2] = (
        np.cos(alpha_z) * np.sin(beta_y) * np.cos(gamma_x)
        + np.sin(alpha_z) * np.sin(gamma_x)
    )
    rot_mat[1, 0] = np.sin(alpha_z) * np.cos(beta_y)
    rot_mat[1, 1] = (
        np.sin(alpha_z) * np.sin(beta_y) * np.sin(gamma_x)
        + np.cos(alpha_z) * np.cos(gamma_x)
    )
    rot_mat[1, 2] = (
        np.sin(alpha_z) * np.sin(beta_y) * np.cos(gamma_x)
        - np.cos(alpha_z) * np.sin(gamma_x)
    )
    rot_mat[2, 0] = -np.sin(beta_y)
    rot_mat[2, 1] = np.cos(beta_y) * np.sin(gamma_x)
    rot_mat[2, 2] = np.cos(beta_y) * np.cos(gamma_x)

    pos_w = np.array(camera.pos_world, dtype=np.float64)
    roi_w = np.array(camera.roi_cent_world, dtype=np.float64)
    image_dist = float(np.linalg.norm(pos_w - roi_w))

    sensor_w = camera.pixels_num[0] * camera.pixels_size[0]
    sensor_h = camera.pixels_num[1] * camera.pixels_size[1]
    image_dim_x = (image_dist / camera.focal_length) * sensor_w
    image_dim_y = (image_dist / camera.focal_length) * sensor_h

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rot_mat
    c2w[:3, 3] = pos_w
    w2c = np.linalg.inv(c2w)

    coords_hom = np.hstack([coords, np.ones((coords.shape[0], 1))])
    coords_cam = (w2c @ coords_hom.T).T[:, :3]

    cx = image_dist * coords_cam[:, 0] / (-coords_cam[:, 2])
    cy = image_dist * coords_cam[:, 1] / (-coords_cam[:, 2])

    cx = 2.0 * cx / image_dim_x
    cy = 2.0 * cy / image_dim_y

    u = (cx + 1.0) * 0.5
    v = (1.0 - cy) * 0.5

    uvs = np.zeros((coords.shape[0], 2), dtype=np.float64)
    uvs[:, 0] = u
    uvs[:, 1] = v
    return uvs


def main() -> None:
    data_dir = riley.data.platehole_csv_case_path()
    texture_path = riley.data.speckle_texture_path()
    out_dir = make_demo_out_dir("demo-imagewarp2d")
    pixels_num = (2464, 2056)
    pixels_size = (3.45e-6, 3.45e-6)
    focal_length = 50.0e-3
    fov_scale_factor = 0.65
    sub_sample = 2
    stereo_angle_deg = 20.0
    total_threads = 8
    vertical_translation_y = 0.005  # 5mm vertical shift (half hole diameter)

    distortion_model = {
        "distortion_model": 1,
        "distortion_k1": -0.2,
        "distortion_k2": 0.1,
        "distortion_k3": 0.0,
        "distortion_p1": 0.0001,
        "distortion_p2": -0.0001,
    }

    coords, connect, _, _ = riley.load_sim_csvs(data_dir)

    # 2 Frames: Frame 0 = undeformed reference, Frame 1 = vertical shift
    disp = np.zeros((2, coords.shape[0], 3), dtype=np.float64)
    disp[1, :, 1] = vertical_translation_y

    texture = riley.load_texture_u8(texture_path)

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

    # Projected UV mapping from reference camera (Camera 0)
    uvs_projected = project_nodes_to_uv(coords, camera_0)

    mesh = riley.Mesh(
        mesh_type=riley.MeshType.quad8,
        coords=coords,
        connect=connect,
        disp=disp,
        shader_type=riley.ShaderType.tex,
        uvs=uvs_projected,
        texture=texture,
        sample=riley.TextureSample.cubic_catmull_rom,
        sample_mode=riley.TextureSampleMode.lut_lerp,
        bits=8,
        scaling_type=riley.ScaleStrategy.none,
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
    print(f"rendered imagewarp2d to {out_dir}")


if __name__ == "__main__":
    main()
