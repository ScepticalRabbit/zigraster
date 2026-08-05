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

import riley

from riley.pydemos.common import make_demo_out_dir


def main() -> None:
    data_dir = riley.data.sphere200_case_path()
    texture_path = riley.data.speckle_texture_path()
    out_dir = make_demo_out_dir("demo-sphere200")
    pixels_num = (800, 500)
    pixels_size = (5.3e-6, 5.3e-6)
    focal_length = 50.0e-3
    rot_world = (0.0, 0.0, 0.0)
    frame_fill = 1.0

    coords, connect, uvs, _ = riley.load_sim_csvs(data_dir)
    texture = riley.load_texture(texture_path)

    roi_cent_world = riley.roi_cent_from_coords(coords)
    pos_world = riley.pos_fill_frame_from_rot(
        coords,
        pixels_num,
        pixels_size,
        focal_length,
        rot_world,
        frame_fill,
    )

    mesh = riley.Mesh(
        mesh_type=riley.MeshType.tri6,
        coords=coords,
        connect=connect,
        uvs=uvs,
        texture=texture,
        sample=riley.TextureSample.cubic_catmull_rom,
        sample_mode=riley.TextureSampleMode.lut_lerp,
        bits=8,
        scaling_type=riley.ScaleStrategy.none,
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
    config = riley.create_raster_config(
        num_frames=1,
        total_threads=4,
        save_strategy=riley.SaveStrategy.disk,
    )

    start_time = perf_counter()
    image_array = riley.raster(mesh, camera, config, out_dir=str(out_dir))
    elapsed_time = perf_counter() - start_time
    print(f"Riley render time: {elapsed_time:.6f} s")

    if image_array is None:
        print(f"rendered disk output to {out_dir}")
    else:
        print(f"rendered image array with shape {image_array.shape} to {out_dir}")


if __name__ == "__main__":
    main()
