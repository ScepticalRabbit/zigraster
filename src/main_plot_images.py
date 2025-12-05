from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotopts as po

def main() -> None:
    data_path = Path.cwd() / "raster-out"

    field_num = 1
    frame_num = 8
    
    image_path = data_path / f"image_out_field{field_num}_frame{frame_num}.csv"
    image_subpx_path = data_path / f"imagesp_field{field_num}_frame{frame_num}.csv"
    depth_subpx_path = data_path / f"depthsp_frame{frame_num}.csv"

    if image_path.is_file():
        print(f"Found image file: {image_path.resolve()}")
        image_buff = pd.read_csv(image_path,header=None)
        image_buff = image_buff.to_numpy()

    if image_subpx_path.is_file():
        print(f"Found subpx image file: {image_subpx_path.resolve()}")
        image_subpx_buff = pd.read_csv(image_subpx_path,header=None)
        image_subpx_buff = image_subpx_buff.to_numpy()

    if depth_subpx_path.is_file():
        print(f"Found depth subpx file: {depth_subpx_path.resolve()}")
        depth_subpx_buff = pd.read_csv(depth_subpx_path,header=None)
        depth_subpx_buff = depth_subpx_buff.to_numpy()

        if image_subpx_path.is_file(): 
            image_subpx_buff[depth_subpx_buff > 1e4] = np.nan
            
        depth_subpx_buff[depth_subpx_buff > 1e4] = np.nan

    #---------------------------------------------------------------------------
    plot_opts = po.PlotOptsGeneral()

    if depth_subpx_path.is_file():
        (fig, ax) = plt.subplots(figsize=plot_opts.single_fig_size_square,
                                layout='constrained')
        fig.set_dpi(plot_opts.resolution)
        cset = plt.imshow(depth_subpx_buff,
                        cmap=plt.get_cmap(plot_opts.cmap_seq))
                        #origin='lower')
        ax.set_aspect('equal','box')
        fig.colorbar(cset)
        ax.set_title(f"Depth Subpx buffer",fontsize=plot_opts.font_head_size)
        ax.set_xlabel(r"x ($px$)",
                    fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
        ax.set_ylabel(r"y ($px$)",
                    fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)


    if image_subpx_path.is_file():
        (fig, ax) = plt.subplots(figsize=plot_opts.single_fig_size_square,
                                layout='constrained')
        fig.set_dpi(plot_opts.resolution)
        cset = plt.imshow(image_subpx_buff,
                        cmap=plt.get_cmap(plot_opts.cmap_seq))
                        #origin='lower')
        ax.set_aspect('equal','box')
        fig.colorbar(cset)
        ax.set_title(f"Image Subpx buffer",fontsize=plot_opts.font_head_size)
        ax.set_xlabel(r"x ($px$)",
                    fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
        ax.set_ylabel(r"y ($px$)",
                    fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)

    if image_path.is_file():
        (fig, ax) = plt.subplots(figsize=plot_opts.single_fig_size_square,
                                layout='constrained')
        fig.set_dpi(plot_opts.resolution)
        cset = plt.imshow(image_buff,
                        cmap=plt.get_cmap(plot_opts.cmap_seq))
                        #origin='lower')
        ax.set_aspect('equal','box')
        fig.colorbar(cset)
        ax.set_title(f"Field Image",fontsize=plot_opts.font_head_size)
        ax.set_xlabel(r"x ($px$)",
                    fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
        ax.set_ylabel(r"y ($px$)",
                    fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)

    plt.show()




if __name__ == "__main__":
    main()
