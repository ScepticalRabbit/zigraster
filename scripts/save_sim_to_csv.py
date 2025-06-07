from pathlib import Path
import numpy as np
import pyvale as pyv
import mooseherder as mh


def main() -> None:
    sim_path = Path.cwd() / "data"

    sim_file = sim_path / "cylinder" / "cylinder_m1_out.e"
    #sim_file = sim_path / "block" / "case25_out.e"

    sim_data = mh.ExodusReader(sim_file).read_all_sim_data()

    field_keys = ("disp_y",)
    disp_keys = ("disp_x","disp_y","disp_z")

    mesh_world = pyv.create_render_mesh(sim_data,
                                        field_render_keys=field_keys,
                                        sim_spat_dim=3,
                                        field_disp_keys=disp_keys)

    save_path = Path.home() / "zigraster" / "data"

    print(80*"-")
    print(f"{mesh_world.coords.shape=}")
    print(f"{mesh_world.connectivity.shape=}")
    print(f"{mesh_world.fields_render.shape=}")
    print(80*"-")

    np.savetxt(save_path/'coords.csv',mesh_world.coords, delimiter=',')
    np.savetxt(save_path/'connectivity.csv',mesh_world.connectivity, delimiter=',')
    np.savetxt(save_path/'field_disp_x.csv',mesh_world.fields_disp[:,:,0], delimiter=',')
    np.savetxt(save_path/'field_disp_y.csv',mesh_world.fields_disp[:,:,1], delimiter=',')
    np.savetxt(save_path/'field_disp_z.csv',mesh_world.fields_disp[:,:,2], delimiter=',')

    num_frames = mesh_world.fields_disp.shape[1]
    for ff in range(num_frames):
        save_file = save_path / f"field_disp_frame{ff}.csv"
        np.savetxt(save_file,mesh_world.fields_disp[:,ff,:],delimiter=",")


if __name__ == "__main__":
    main()