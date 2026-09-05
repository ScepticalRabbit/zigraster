# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
from __future__ import annotations

from time import perf_counter
from pathlib import Path
import shutil
import numpy as np

import riley

RASTER_THREADS = 8


def main() -> None:
    data_dir = riley.data.sphere200_case_path()
    texture_path = riley.data.speckle_texture_path()
    out_dir_root = Path.cwd() / "out-riley-py" / "demo-psf"
    shutil.rmtree(out_dir_root, ignore_errors=True)
    out_dir_root.mkdir(parents=True)
    pixels_num = (800, 500)
    pixels_size = (5.3e-6, 5.3e-6)
    focal_length = 50.0e-3
    rot_world = (0.0, 0.0, 0.0)

    coords = riley.load_csv(data_dir / "coords.csv")
    connect = riley.load_csv(data_dir / "connect.csv", dtype=np.int64)
    uvs = riley.load_csv(data_dir / "uvs.csv")
    texture = riley.load_texture_mono_u8(texture_path)
    convention = riley.ConnectConvention(
        riley.EElemType.TRI6, riley.EConnectAxis.ROW, 0,
        riley.ENodeOrder.RILEY,
    )
    roi_cent_world = riley.roi_cent_from_coords(coords)
    pos_world = riley.pos_frame_coords(
        coords,
        pixels_num,
        pixels_size,
        focal_length,
        rot_world,
        fov_scale=1.0,
    )
    mesh = riley.create_mesh(
        convention=convention,
        mesh_type=riley.MeshType.tri6,
        coords=coords,
        connect=connect,
        shader=riley.TextureShader(uvs=uvs, texture=texture),
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
        psf_type=riley.PsfType.gaussian,
        psf_sigma_x=1.0,
        psf_support_rad=3.0,
        psf_separable=1,
    )

    for mode in (
        riley.BufferMode.global_subpx_full,
        riley.BufferMode.global_subpx_stripe,
    ):
        out_dir = out_dir_root / mode.name
        out_dir.mkdir(parents=True, exist_ok=True)
        config = riley.create_raster_config(
            num_frames=1,
            total_threads=RASTER_THREADS,
            save_strategy=riley.SaveStrategy.disk,
        )
        config.buffer_mode = mode

        print(f"Rendering {mode.name} with {RASTER_THREADS} raster threads...")
        start_time = perf_counter()
        image_array = riley.raster(mesh, camera, config, out_dir=str(out_dir))
        elapsed_time = perf_counter() - start_time
        print(f"{mode.name}: {elapsed_time:.6f} s")
        if image_array is not None:
            print(
                f"rendered image array with shape {image_array.shape} "
                f"to {out_dir}"
            )


if __name__ == "__main__":
    main()
