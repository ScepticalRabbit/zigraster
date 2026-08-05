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

import numpy as np
import riley

from riley.pydemos.common import make_demo_out_dir


def main() -> None:
    data_dir = riley.data.stereocal_case_path()
    texture_path = riley.data.cal_target_texture_path()
    out_dir = make_demo_out_dir("demo-stereocal")
    dicuq_camera_dir = Path.cwd() / "out-riley-py" / "demo-dicuq"
    total_threads = 8

    coords, connect, uvs, disp = riley.load_sim_csvs(data_dir)
    texture = riley.load_texture(texture_path)

    camera_0, camera_1 = riley.load_stereo_pair(
        str(dicuq_camera_dir),
        "stereo_data_opengl.csv",
    )

    roi_pos = np.asarray(riley.roi_cent_from_coords(coords), dtype=np.float64)
    target_roi = np.asarray(camera_0.roi_cent_world, dtype=np.float64)
    roi_shift = target_roi - roi_pos
    coords = np.ascontiguousarray(coords + roi_shift, dtype=np.float64)
    roi_pos = riley.roi_cent_from_coords(coords)
    camera_0 = replace(camera_0, roi_cent_world=roi_pos)
    camera_1 = replace(camera_1, roi_cent_world=roi_pos)

    mesh = riley.Mesh(
        mesh_type=riley.MeshType.tri3,
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

    config = riley.create_raster_config(
        num_frames=2,
        total_threads=total_threads,
        save_strategy=riley.SaveStrategy.disk,
    )
    config.background_value = 128.0

    start_time = perf_counter()
    riley.raster([mesh], [camera_0, camera_1], config, out_dir=str(out_dir))
    elapsed_time = perf_counter() - start_time
    print(f"Riley render time: {elapsed_time:.6f} s")
    print(f"rendered stereocal to {out_dir}")


if __name__ == "__main__":
    main()
