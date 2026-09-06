from pathlib import Path
import shutil

import numpy as np

import riley


def main() -> None:
    coords = np.array(
        ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    connect = np.array(((0, 1, 2),), dtype=np.int64)
    convention = riley.ConnectConvention(
        riley.EElemType.TRI3,
        riley.EConnectAxis.ROW,
        0,
        riley.ENodeOrder.RILEY,
    )
    shader = riley.FunctionShader(
        builtin=riley.FuncShaderBuiltin.checker,
        coord_mode=riley.FuncCoordMode.world_reference,
        params=riley.FuncShaderParams(coord_scale=(4.0, 4.0)),
        scaling_type=riley.ScaleStrategy.auto,
    )
    mesh = riley.create_mesh(
        convention,
        riley.MeshType.tri3opt,
        coords,
        connect,
        shader,
    )
    pixels_num = (512, 512)
    pixels_size = (0.02, 0.02)
    focal_length = 1.0
    rotation = (0.0, 0.0, 0.0)
    target = riley.roi_cent_from_coords(coords)
    position = riley.pos_frame_mesh(
        mesh,
        pixels_num,
        pixels_size,
        focal_length,
        rotation,
        target=target,
    )
    camera = riley.Camera(
        pixels_num=pixels_num,
        pixels_size=pixels_size,
        pos_world=position,
        rot_world=rotation,
        roi_cent_world=target,
        focal_length=focal_length,
        sub_sample=1,
    )
    config = riley.create_raster_config(
        1,
        save_strategy=riley.SaveStrategy.disk,
    )
    config.image_save_mode = riley.ImageSaveMode.grey
    config.save_scaling = riley.ScaleStrategy.none
    out_dir = Path.cwd() / "out_riley_py" / "demo0_quickstart"
    shutil.rmtree(out_dir, ignore_errors=True)
    riley.raster(mesh, camera, config, out_dir=str(out_dir))


if __name__ == "__main__":
    main()
