from pathlib import Path
import shutil

import numpy as np

import riley


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "shapes" / "cylinder_surf" / "tri6"
OUT_DIR = ROOT_DIR / "out_riley_py" / "check_cylinders"
PIXELS_NUM = (1024, 1024)
PIXELS_SIZE = (5.3e-6, 5.3e-6)
FOCAL_LENGTH = 50.0e-3


def render_check(
    case_name: str,
    rotation: tuple[float, float, float],
    sub_sample: int,
    use_distortion: bool,
) -> None:
    coords = riley.load_csv(DATA_DIR / "coords.csv")
    connect = riley.load_csv(DATA_DIR / "connect.csv", dtype=np.int64)
    uvs = riley.load_csv(DATA_DIR / "uvs.csv")
    texture = riley.load_texture_mono_u8(
        ROOT_DIR / "texture" / "speck128_mono_u8.png"
    )
    convention = riley.ConnectConvention(
        riley.EElemType.TRI6,
        riley.EConnectAxis.ROW,
        0,
        riley.ENodeOrder.RILEY,
    )
    shaders = (
        (
            "checker",
            riley.FunctionShader(
                builtin=riley.FuncShaderBuiltin.checker,
                coord_mode=riley.FuncCoordMode.world_reference,
                params=riley.FuncShaderParams(coord_scale=(1000.0, 1000.0)),
                channels=1,
                bits=8,
                scaling_type=riley.ScaleStrategy.auto,
                normal_type=riley.NormalType.none,
            ),
        ),
        (
            "texture",
            riley.TextureShader(
                uvs=uvs,
                texture=texture,
                sample=riley.TextureSample.linear,
                sample_mode=riley.TextureSampleMode.direct,
                bits=8,
                scaling_type=riley.ScaleStrategy.auto,
                normal_type=riley.NormalType.none,
            ),
        ),
    )
    target = riley.roi_cent_from_coords(coords)
    position = riley.pos_frame_mesh(
        riley.create_mesh(
            convention,
            riley.MeshType.tri6,
            coords,
            connect,
            shaders[0][1],
        ),
        PIXELS_NUM,
        PIXELS_SIZE,
        FOCAL_LENGTH,
        rotation,
        fov_scale=1.05,
        target=target,
    )
    camera_options = {}
    if use_distortion:
        camera_options = {
            "distortion_model": 1,
            "distortion_k1": -0.12,
            "distortion_k2": 0.035,
            "distortion_p1": 0.0002,
            "distortion_p2": -0.0001,
        }
    camera = riley.Camera(
        pixels_num=PIXELS_NUM,
        pixels_size=PIXELS_SIZE,
        pos_world=position,
        rot_world=rotation,
        roi_cent_world=target,
        focal_length=FOCAL_LENGTH,
        sub_sample=sub_sample,
        **camera_options,
    )
    case_dir = OUT_DIR / case_name
    shutil.rmtree(case_dir, ignore_errors=True)
    for shader_name, shader in shaders:
        mesh = riley.create_mesh(
            convention,
            riley.MeshType.tri6,
            coords,
            connect,
            shader,
        )
        config = riley.create_raster_config(
            num_frames=1,
            total_threads=4,
            save_strategy=riley.SaveStrategy.disk,
        )
        config.image_save_mode = riley.ImageSaveMode.grey
        config.save_bits = 8
        config.save_format = riley.ImageFormat.bmp
        config.save_scaling = riley.ScaleStrategy.none
        config.background_value = 0.5 * (2**8 - 1)
        riley.raster(
            mesh,
            camera,
            config,
            out_dir=str(case_dir / shader_name),
        )
