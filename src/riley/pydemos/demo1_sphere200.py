# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

import riley


def main() -> None:
    # --------------------------------------------------------------------------
    # 1. Setup paths and parameters
    # --------------------------------------------------------------------------
    data_dir = riley.data.sphere200_case_path()
    texture_path = riley.data.speckle_texture_path()
    out_dir = Path.cwd() / "out_riley_py" / "demo1_sphere200"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)

    pixels_num = (800, 500)
    pixels_size = (5.3e-6, 5.3e-6)
    focal_length = 50.0e-3
    rot_world = (0.0, 0.0, 0.0)
    frame_fill = 1.0

    # --------------------------------------------------------------------------
    # 2. Load mesh data and texture shader
    # --------------------------------------------------------------------------
    coords = riley.load_csv(data_dir / "coords.csv")
    connect = riley.load_csv(data_dir / "connect.csv", dtype=np.int64)
    uvs = riley.load_csv(data_dir / "uvs.csv")
    texture = riley.load_texture_mono_u8(texture_path)

    convention = riley.ConnectConvention(
        riley.EElemType.TRI6,
        riley.EConnectAxis.ROW,
        0,
        riley.ENodeOrder.RILEY,
    )
    shader = riley.TextureShader(uvs=uvs, texture=texture)

    mesh = riley.create_mesh(
        convention=convention,
        mesh_type=riley.MeshType.tri6,
        coords=coords,
        connect=connect,
        shader=shader,
    )

    # --------------------------------------------------------------------------
    # 3. Position and create camera
    # --------------------------------------------------------------------------
    roi_cent_world = riley.roi_cent_from_coords(coords)
    pos_world = riley.pos_frame_coords(
        coords,
        pixels_num,
        pixels_size,
        focal_length,
        rot_world,
        fov_scale=frame_fill,
    )

    camera = riley.Camera(
        pixels_num=pixels_num,
        pixels_size=pixels_size,
        pos_world=pos_world,
        rot_world=rot_world,
        roi_cent_world=roi_cent_world,
        focal_length=focal_length,
        sub_sample=2,
        coord_sys=riley.CameraCoordSys.opengl,
    )

    # --------------------------------------------------------------------------
    # 4. Configure raster engine
    # --------------------------------------------------------------------------
    config = riley.create_raster_config(
        num_frames=1,
        total_threads=4,
        save_strategy=riley.SaveStrategy.disk,
    )

    # --------------------------------------------------------------------------
    # 5. Render sphere scene
    # --------------------------------------------------------------------------
    start_time = perf_counter()
    image_array = riley.raster(mesh, camera, config, out_dir=str(out_dir))
    elapsed_time = perf_counter() - start_time
    print(f"Riley render time: {elapsed_time:.6f} s")

    if image_array is None:
        print(f"rendered disk output to {out_dir}")
    else:
        print(
            f"rendered image array with shape {image_array.shape} "
            f"to {out_dir}"
        )


if __name__ == "__main__":
    main()
