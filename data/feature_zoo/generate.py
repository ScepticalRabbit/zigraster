from pathlib import Path

import numpy as np

import riley


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "feature_zoo"

CASES = (
    ("cube_quad9", "cube", "hex27", riley.EElemType.HEX27,
     riley.MeshType.quad9),
    ("cube_tri6", "cube", "tet10", riley.EElemType.TET10,
     riley.MeshType.tri6),
    ("cylinder_quad8", "cylinder", "hex20", riley.EElemType.HEX20,
     riley.MeshType.quad8),
    ("cylinder_tri6", "cylinder", "tet10", riley.EElemType.TET10,
     riley.MeshType.tri6),
    ("plate_quad4ibi", "platewithhole", "hex8", riley.EElemType.HEX8,
     riley.MeshType.quad4ibi),
    ("plate_quad4newton", "platewithhole", "hex8", riley.EElemType.HEX8,
     riley.MeshType.quad4newton),
    ("plate_tri3", "platewithhole", "tet4", riley.EElemType.TET4,
     riley.MeshType.tri3),
)


def save_csv(path: Path, values: np.ndarray, integer: bool = False) -> None:
    fmt = "%d" if integer else "%.17g"
    np.savetxt(path, values, delimiter=",", fmt=fmt)


def generate_case(
    name: str,
    shape: str,
    source_name: str,
    elem_type: riley.EElemType,
    mesh_type: riley.MeshType,
) -> None:
    coords = riley.load_csv(riley.data.shape_coords_path(shape, source_name))
    connect = riley.load_csv(
        riley.data.shape_connectivity_path(shape, source_name),
        dtype=np.int64,
    )
    disp = tuple(
        riley.load_csv(riley.data.shape_disp_path(shape, source_name, axis))
        for axis in "xyz"
    )
    temperature = riley.load_csv(
        riley.data.shape_temperature_path(shape, source_name)
    )
    convention = riley.ConnectConvention(
        elem_type,
        riley.EConnectAxis.ROW,
        0,
        riley.ENodeOrder.RILEY,
    )
    mesh = riley.create_mesh(
        convention,
        mesh_type,
        coords,
        connect,
        riley.NodalShader(temperature),
        disp=disp,
    )
    uvs = riley.project_uvs_planar_centered(
        mesh.coords,
        (128, 128),
        uv_span_max=0.9,
        proj_plane=riley.EProjPlane.XY,
    )

    case_dir = OUT_DIR / name
    case_dir.mkdir(parents=True, exist_ok=True)
    save_csv(case_dir / "coords.csv", mesh.coords)
    save_csv(case_dir / "connect.csv", mesh.connect, integer=True)
    save_csv(case_dir / "uvs.csv", uvs)
    save_csv(case_dir / "temperature.csv", mesh.shader.field[:, :, 0].T)
    if mesh.disp is None:
        raise RuntimeError("Feature-zoo source has no displacement field.")
    for index, axis in enumerate("xyz"):
        save_csv(case_dir / f"disp_{axis}.csv", mesh.disp[:, :, index].T)


def main() -> None:
    for case in CASES:
        generate_case(*case)
    texture = riley.load_texture_rgb_u16(
        ROOT / "texture" / "speck128_rgb_u16.png"
    )
    header = f"FIMG\n{texture.shape[2]} {texture.shape[1]} 3\n".encode()
    payload = np.ascontiguousarray(texture, dtype="<f8").tobytes()
    (OUT_DIR / "texture_rgb_u16.fimg").write_bytes(header + payload)


if __name__ == "__main__":
    main()
