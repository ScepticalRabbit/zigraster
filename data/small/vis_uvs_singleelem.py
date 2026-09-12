import pyvista as pv
import numpy as np
from pathlib import Path

def visualize_case(case_name):
    case_dir = Path("data/small") / case_name
    if not case_dir.exists():
        print(f"Error: {case_dir} does not exist.")
        return

    coords = np.loadtxt(case_dir / "coords.csv", delimiter=",")
    connect = np.loadtxt(case_dir / "connectivity.csv", delimiter=",", dtype=int)
    uvs = np.loadtxt(case_dir / "uvs.csv", delimiter=",")

    # Determine element type from name
    if "tri3" in case_name:
        cell_type = pv.CellType.TRIANGLE
        nodes_per_elem = 3
    elif "tri6" in case_name:
        cell_type = pv.CellType.QUADRATIC_TRIANGLE
        nodes_per_elem = 6
    elif "quad4" in case_name:
        cell_type = pv.CellType.QUAD
        nodes_per_elem = 4
    elif "quad8" in case_name:
        cell_type = pv.CellType.QUADRATIC_QUAD
        nodes_per_elem = 8
    elif "quad9" in case_name:
        cell_type = pv.CellType.BIQUADRATIC_QUAD
        nodes_per_elem = 9
    else:
        print(f"Unsupported mesh type for {case_name}")
        return

    # PyVista UnstructuredGrid cells: [n_nodes, id1, id2, ..., n_nodes, id1, id2, ...]
    if connect.ndim == 1:
        cells = np.concatenate(([nodes_per_elem], connect))
        num_cells = 1
    else:
        cells = np.hstack([np.full((connect.shape[0], 1), nodes_per_elem), connect]).ravel()
        num_cells = connect.shape[0]

    cell_types = np.full(num_cells, cell_type, dtype=np.uint8)
    
    mesh = pv.UnstructuredGrid(cells, cell_types, coords)
    mesh.active_texture_coordinates = uvs

    tex_path = Path("texture/speckle_mono.tiff")
    if not tex_path.exists():
        print(f"Error: Texture {tex_path} not found.")
        return
    texture = pv.read_texture(str(tex_path))

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, texture=texture, show_edges=True)
    plotter.add_text(f"UV Projection: {case_name}", font_size=12)
    plotter.show()

def main():
    # User Configurable Parameter: which case to visualize
    CASE_TO_VISUALIZE = "tri6_single"
    
    visualize_case(CASE_TO_VISUALIZE)

if __name__ == "__main__":
    main()
