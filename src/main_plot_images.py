from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvale as pyv

def main() -> None:
    data_path = Path.cwd() / "raster-out"
    image_path = data_path / "image.csv"
    depth_path = data_path / "depth.csv"

    image_buff = pd.read_csv(image_path)
    image_buff = image_buff.to_numpy()

    depth_buff = pd.read_csv(depth_path)
    depth_buff = depth_buff.to_numpy()

    image_buff[depth_buff > 1e5] = np.nan
    depth_buff[depth_buff > 1e5] = np.nan

    #---------------------------------------------------------------------------
    plot_opts = pyv.PlotOptsGeneral()

    (fig, ax) = plt.subplots(figsize=plot_opts.single_fig_size_square,
                            layout='constrained')
    fig.set_dpi(plot_opts.resolution)
    cset = plt.imshow(depth_buff,
                    cmap=plt.get_cmap(plot_opts.cmap_seq))
                    #origin='lower')
    ax.set_aspect('equal','box')
    fig.colorbar(cset)
    ax.set_title(f"Depth buffer",fontsize=plot_opts.font_head_size)
    ax.set_xlabel(r"x ($px$)",
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    ax.set_ylabel(r"y ($px$)",
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)

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