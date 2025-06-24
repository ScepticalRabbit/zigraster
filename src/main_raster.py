import numpy as np
import time
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale as pyv
import zigraster.cyth.zraster as zr


def main() -> None:
    print()
    print(80*"=")
    print("RASTER ZIG FILE (should be *.so on Linux):")
    print(zr.__file__)
    print(80*"=")
    print()

    sim_path = pyv.DataSet.render_simple_block_path()
    #sim_path = pyv.DataSet.render_mechanical_3d_path()
    sim_data = mh.ExodusReader(sim_path).read_all_sim_data()

    disp_comps = ("disp_x","disp_y","disp_z")

    # Scale m -> mm
    sim_data = pyv.scale_length_units(1000.0,sim_data,disp_comps)

    print()
    print(f"{np.max(np.abs(sim_data.node_vars['disp_x']))=}")
    print(f"{np.max(np.abs(sim_data.node_vars['disp_y']))=}")
    print(f"{np.max(np.abs(sim_data.node_vars['disp_z']))=}")
    print()

    # Extracts the surface mesh from a full 3d simulation for rendering
    render_mesh = pyv.create_render_mesh(sim_data,
                                        ("disp_y","disp_x","disp_z"),
                                        sim_spat_dim=3,
                                        field_disp_keys=disp_comps)


    pixel_num = np.array((960,1280),dtype=np.int32)
    pixel_size = np.array((5.3e-3,5.3e-3),dtype=np.float64)
    focal_leng: float = 50.0
    cam_rot = Rotation.from_euler("ZYX",(0.0,-30.0,-10.0),degrees=True)
    fov_scale_factor: float = 1.1

    (roi_pos_world,
    cam_pos_world) = pyv.CameraTools.pos_fill_frame(
        coords_world=render_mesh.coords,
        pixel_num=pixel_num,
        pixel_size=pixel_size,
        focal_leng=focal_leng,
        cam_rot=cam_rot,
        frame_fill=fov_scale_factor,
    )

    cam_data = pyv.CameraData(
        pixels_num=pixel_num,
        pixels_size=pixel_size,
        pos_world=cam_pos_world,
        rot_world=cam_rot,
        roi_cent_world=roi_pos_world,
        focal_length=focal_leng,
        sub_samp=2,
        back_face_removal=True,
    )

    zr.set_camera(cam_data)


    # print()
    # print(80*"-")
    # print("MESH DATA:")
    # print(80*"-")
    # print("connectivity.shape=(num_elems,num_nodes_per_elem)")
    # print(f"{render_mesh.connectivity.shape=}")
    # print()
    # print("coords.shape=(num_nodes,coord[x,y,z])")
    # print(f"{render_mesh.coords.shape=}")
    # print()
    # print("fields.shape=(num_coords,num_time_steps,num_components)")
    # print(f"{render_mesh.fields_render.shape=}")
    # if render_mesh.fields_disp is not None:
    #     print(f"{render_mesh.fields_disp.shape=}")
    # print(80*"-")
    # print()

    # print(80*"-")
    # print("CAMERA DATA:")
    # print(80*"-")
    # print(f"{cam_data.image_dist=}")
    # print(f"{cam_data.roi_cent_world=}")
    # print(f"{cam_data.pos_world=}")
    # print()
    # print("World to camera matrix:")
    # print(cam_data.world_to_cam_mat)
    # print(80*"-")
    # print()

    # print(80*"-")
    # total_frames = render_mesh.fields_render.shape[1]*render_mesh.fields_render.shape[2]
    # print(f"Time steps to render: {render_mesh.fields_render.shape[1]}")
    # print(f"Fields to render: {render_mesh.fields_render.shape[2]}")
    # print(f"Total frames to render: {total_frames}")
    # print(80*"-")






if __name__ == "__main__":
    main()