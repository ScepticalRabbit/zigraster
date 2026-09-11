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
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

import riley

FRAMES_MAX = 8

MATCHED_ROI = (0.0125, 0.0175, 0.0005)
MATCHED_CAM0_POS = (0.0125, 0.0175, 0.160864856482)
MATCHED_CAM1_POS = (0.067348011198, 0.0175, 0.151193672270)


def create_stereo_cameras(
    roi_pos: tuple[float, float, float] | np.ndarray,
) -> tuple[riley.Camera, riley.Camera]:
    """Create stereo camera pair matching the DICUQ demo parameters."""
    pixels_num = (2464, 2056)
    pixels_size = (3.45e-6, 3.45e-6)
    focal_length = 50.0e-3
    stereo_angle_deg = 20.0
    sub_sample = 2

    # Brown-Conrady distortion
    distortion_model = {
        "distortion_model": 1,
        "distortion_k1": -0.2,
        "distortion_k2": 0.1,
        "distortion_k3": 0.0,
        "distortion_p1": 0.0001,
        "distortion_p2": -0.0001,
    }

    # Camera 0: face on
    cam0_rot = (0.0, 0.0, 0.0)
    camera_0 = riley.Camera(
        pixels_num=pixels_num,
        pixels_size=pixels_size,
        pos_world=MATCHED_CAM0_POS,
        rot_world=cam0_rot,
        roi_cent_world=tuple(roi_pos),
        focal_length=focal_length,
        sub_sample=sub_sample,
        **distortion_model,
    )

    # Camera 1: stereo angle 
    cam1_rot = (0.0, float(np.deg2rad(stereo_angle_deg)), 0.0)
    camera_1 = copy.deepcopy(camera_0)
    camera_1.pos_world = MATCHED_CAM1_POS
    camera_1.rot_world = cam1_rot

    return camera_0, camera_1


def main() -> None:
    # --------------------------------------------------------------------------
    # 1. Setup paths and parameters
    # --------------------------------------------------------------------------
    data_dir = riley.data.stereocal_case_path()
    texture_path = riley.data.cal_target_texture_path()
    out_dir = Path.cwd() / "out_riley_py" / "demo8_stereocal"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    total_threads = 8

    # --------------------------------------------------------------------------
    # 2. Load calibration plate mesh, sample frames, and apply texture
    # --------------------------------------------------------------------------
    coords = riley.load_csv(data_dir / "coords.csv")
    connect = riley.load_csv(data_dir / "connect.csv", dtype=np.int64)
    uvs = riley.load_csv(data_dir / "uvs.csv")
    disp_components = tuple(
        riley.load_csv(data_dir / f"field_disp_{axis}.csv")
        for axis in "xyz"
    )

    frame_indices = riley.frames_evenly_spaced_idxs(
        disp_components[0].shape[1],
        FRAMES_MAX,
    )
    disp_components = tuple(item[:, frame_indices] for item in disp_components)
    texture = riley.load_texture_mono_u8(texture_path)

    # Shift calibration plate to match the DICUQ specimen center
    roi_pos_orig = riley.roi_cent_from_coords(coords)
    roi_shift = np.array(MATCHED_ROI) - np.array(roi_pos_orig)
    coords = coords + roi_shift
    roi_pos = riley.roi_cent_from_coords(coords)

    # --------------------------------------------------------------------------
    # 3. Create, save, and reload stereo camera pair
    # --------------------------------------------------------------------------
    camera_0, camera_1 = create_stereo_cameras(roi_pos)

    stereo_file = "stereo_data_opengl.csv"
    riley.save_stereo_pair(str(out_dir), stereo_file, camera_0, camera_1)
    camera_0, camera_1 = riley.load_stereo_pair(str(out_dir), stereo_file)

    # --------------------------------------------------------------------------
    # 4. Build mesh and raster configuration
    # --------------------------------------------------------------------------
    mesh = riley.create_mesh(
        convention=riley.ConnectConvention(
            riley.EElemType.TRI3,
            riley.EConnectAxis.ROW,
            0,
            riley.ENodeOrder.RILEY,
        ),
        mesh_type=riley.MeshType.tri3,
        coords=coords,
        connect=connect,
        disp=disp_components,
        shader=riley.TextureShader(uvs=uvs, texture=texture),
    )

    config = riley.create_raster_config(
        num_frames=mesh.disp.shape[0],
        total_threads=total_threads,
        save_strategy=riley.SaveStrategy.disk,
    )
    config.background_value = 128.0

    # --------------------------------------------------------------------------
    # 5. Render stereocal poses
    # --------------------------------------------------------------------------
    start_time = perf_counter()
    riley.raster([mesh], [camera_0, camera_1], config, out_dir=str(out_dir))
    elapsed_time = perf_counter() - start_time
    print(f"Riley render time: {elapsed_time:.6f} s")
    print(f"rendered stereocal to {out_dir}")


if __name__ == "__main__":
    main()
