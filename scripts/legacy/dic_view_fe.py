from pathlib import Path
import numpy as np
import pyvista as pv
import pyvale.sensorsim as sens
import pyvale.mooseherder as mh

def main() -> None:
    #---------------------------------------------------------------------------
    # Loading in the exodus simulation output 
    
    script_path = Path(__file__).resolve().parent
    print(f"{script_path=}")
     
    output_path = script_path.parent / 'data' / 'FE'
    print(f"{output_path=}")

    exodus_name = "platehole3d_6mr_63f.e"
    
    output_exodus = output_path / exodus_name
    exodus_reader = mh.ExodusLoader(output_exodus)
    
    print("\nReading exodus file with ExodusReader:")
    print(f"{output_exodus=}\n")

    sim_data = exodus_reader.load_all_sim_data()

    disp_keys = ("disp_x","disp_y","disp_z")
    sim_data = sens.scale_length_units(scale=1000.0,
                                       sim_data=sim_data,
                                       disp_keys=disp_keys)

    # Prints out the fields of our dataclass so we can see what we have.
    print("SimData from 'load_all':")
    sens.simtools.print_sim_data(sim_data)

    component = "disp_y"
    time_step = -1
        
    sim_vis = sens.simdata_to_pyvista_vis(sim_data,
                                          sens.EDim.THREED)
    sim_vis[component] = sim_data.node_vars[component][:,time_step]

    vis_opts = sens.VisOptsSimSensors()
    pv_plot = sens.create_pv_plotter(vis_opts)
    pv_plot.add_mesh(sim_vis,
                     scalars=component,
                     label="sim-data",
                     show_edges=True,
                     show_scalar_bar=True,
                     scalar_bar_args={"title":component},)

    pv_plot.camera_position = "xy"
    
    # Set to False to show an interactive plot instead of saving the figure
    pv_plot.off_screen = False
    if pv_plot.off_screen:
        pv_plot.screenshot(output_path/f"TODO.png")
    else:
        pv_plot.show()

if __name__ == "__main__":
    main()
