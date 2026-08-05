from pathlib import Path
import sys
import numpy as np
import pyvale.mooseherder as mh

from extract_surface_mesh import extract_surf_mesh

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import riley

def main() -> None:
    #---------------------------------------------------------------------------
    # NOTE: exodus connectivity starts at 1 - need to subtract 1!
    #---------------------------------------------------------------------------

    base_dir = Path(__file__).resolve().parent
    num_meshes = 7
    frame_counts = (1, 63)
    for num_frames in frame_counts:
        for mm in range(1, num_meshes):
            mesh_num = mm
            sim_name = f"platehole3d_{mesh_num}mr_{num_frames}f"

            sim_file = base_dir / f"{sim_name}.e"

            save_path = base_dir / sim_name
            if not save_path.is_dir():
                save_path.mkdir(parents=True, exist_ok=True)

            sim_data = mh.ExodusLoader(sim_file).load_all_sim_data()
            mesh_world = extract_surf_mesh(sim_data)

            uvs = riley.project_uvs_planar_centered(
                mesh_world.coords,
                (2464, 2056),
                uv_span_max=0.8,
                projection_plane=(
                    np.array((0.0, 0.0, -1.0), dtype=np.float64),
                    np.array((0.0, 0.0, 0.0), dtype=np.float64),
                ),
            )

            connect_keys = sorted(mesh_world.connect.keys())
            if len(connect_keys) != 1:
                raise ValueError(
                    f"{sim_name} extracted {len(connect_keys)} surface connectivity tables; "
                    "the CSV export currently expects one table.",
                )

            connect = mesh_world.connect[connect_keys[0]].T - 1
            z_face_stats = _get_broad_face_orientation_stats(mesh_world.coords, connect)

            print(80 * "-")
            print(f"MESH: {sim_name}") 
            print()
            print(f"{sim_data.coords.shape=}")
            print(f"{sim_data.connect['connect1'].shape=}")
            print()
            print(f"{mesh_world.coords.shape=}")
            print(f"{mesh_world.connect[connect_keys[0]].T.shape=}")
            print(f"{mesh_world.node_vars['disp_x'].shape=}")
            print()
            print(f"{np.max(connect)=},{np.min(connect)=}")
            print()
            print(f"{np.min(mesh_world.coords,axis=0)=}")
            print(f"{np.max(mesh_world.coords,axis=0)=}")
            print()
            print(f"{np.min(uvs,axis=0)=}")
            print(f"{np.max(uvs,axis=0)=}")
            print()
            print(f"{z_face_stats=}")
            print(80 * "-")

            np.savetxt(save_path / 'coords.csv', mesh_world.coords, delimiter=',')
            np.savetxt(save_path / 'connect.csv', connect, delimiter=',')
            np.savetxt(save_path / 'field_disp_x.csv',
                        mesh_world.node_vars['disp_x'], delimiter=',')
            np.savetxt(save_path / 'field_disp_y.csv',
                        mesh_world.node_vars['disp_y'], delimiter=',')
            np.savetxt(save_path / 'field_disp_z.csv',
                        mesh_world.node_vars['disp_z'], delimiter=',')
            np.savetxt(save_path / 'uvs.csv', uvs, delimiter=',')
        
def _get_broad_face_orientation_stats(
    coords: np.ndarray,
    connect: np.ndarray,
) -> dict[str, int]:
    corners = connect[:, :4]
    face_coords = coords[corners]
    normals = np.cross(
        face_coords[:, 1, :] - face_coords[:, 0, :],
        face_coords[:, 2, :] - face_coords[:, 0, :],
    )
    z_centroids = face_coords[:, :, 2].mean(axis=1)
    z_min = np.min(z_centroids)
    z_max = np.max(z_centroids)
    tol = max(1.0e-12, 1.0e-6 * max(abs(z_min), abs(z_max), 1.0))

    is_z_min = np.abs(z_centroids - z_min) <= tol
    is_z_max = np.abs(z_centroids - z_max) <= tol

    return {
        "z_min_pos": int(np.sum(normals[is_z_min, 2] > 0.0)),
        "z_min_neg": int(np.sum(normals[is_z_min, 2] < 0.0)),
        "z_max_pos": int(np.sum(normals[is_z_max, 2] > 0.0)),
        "z_max_neg": int(np.sum(normals[is_z_max, 2] < 0.0)),
    }


if __name__ == "__main__":
    main()
