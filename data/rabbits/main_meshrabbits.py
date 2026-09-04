import os
import gmsh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from riley.python import meshconv
from svgpathtools import svg2paths
from enum import Enum
import argparse
import csv

# --- CONFIGURATION CONSTANTS ---
TARGET_LENGTH = 1.0
EDGE_FRACTION = 0.07 # Slightly finer for better ear definition
MIN_SEGMENT_LENGTH = 0.001 # Even less aggressive simplification to preserve detail
STEPS_ON_PATH = 400
TRI_FACT = 0.6

class ElemType(Enum):
    TRI3 = "tri3"
    TRI6 = "tri6"
    QUAD4 = "quad4"
    QUAD8 = "quad8"
    QUAD9 = "quad9"


RILEY_ELEMENT_TYPES = {
    ElemType.TRI3: meshconv.EElementType.TRI3,
    ElemType.TRI6: meshconv.EElementType.TRI6,
    ElemType.QUAD4: meshconv.EElementType.QUAD4,
    ElemType.QUAD8: meshconv.EElementType.QUAD8,
    ElemType.QUAD9: meshconv.EElementType.QUAD9,
}

def mesh_rabbit_smooth(target_length=TARGET_LENGTH, edge_fraction=EDGE_FRACTION, elem_type=ElemType.TRI3):
    gmsh.initialize()
    gmsh.model.add(f"rabbit_{elem_type.value}")

    # --- SVG Path Extraction & Simplification ---
    paths, _ = svg2paths('outline/rabbitoutline.svg')
    path = paths[0]
    
    # Pre-calculate normalization bounding box
    # Use high-res sampling just for the bbox
    sample_pts = np.array([path.point(t/500) for t in range(501)])
    min_x, max_x = np.min(sample_pts.real), np.max(sample_pts.real)
    min_y, max_y = np.min(sample_pts.imag), np.max(sample_pts.imag)
    scale = target_length / (max_x - min_x)
    
    def transform(p):
        return (p.real - min_x) * scale, -(p.imag - min_y) * scale

    # --- Geometry Simplification ---
    # Strategy: Re-sample the path with a minimum distance constraint.
    # This effectively "merges" short segments and smoothes the boundary.
    simplified_points = []
    last_pt = None
    
    # Step along the path and collect points that are far enough apart
    num_steps: int = STEPS_ON_PATH    
    for i in range(num_steps):
        t = i / num_steps
        curr_p = path.point(t)
        curr_trans = transform(curr_p)
        
        if last_pt is None:
            simplified_points.append(curr_trans)
            last_pt = curr_trans
        else:
            dist = np.sqrt((curr_trans[0]-last_pt[0])**2 + (curr_trans[1]-last_pt[1])**2)
            if dist > MIN_SEGMENT_LENGTH:
                simplified_points.append(curr_trans)
                last_pt = curr_trans

    # Ensure the loop is closed by adding the first point if needed
    # but GMSH addSpline handles the closure if we repeat the first tag.
    
    lc = target_length * edge_fraction
    if elem_type == ElemType.TRI3 or elem_type == ElemType.TRI6:
        lc = lc*TRI_FACT
        
    gmsh_points = []
    for pt in simplified_points:
        gmsh_points.append(gmsh.model.geo.addPoint(pt[0], pt[1], 0, lc))
    
    # Close the loop
    gmsh_points.append(gmsh_points[0])
    
    # Create a single continuous spline for the whole rabbit
    spline = gmsh.model.geo.addSpline(gmsh_points)
    
    loop = gmsh.model.geo.addCurveLoop([spline])
    surface = gmsh.model.geo.addPlaneSurface([loop])
    
    gmsh.model.geo.synchronize()

    # --- Meshing Configuration ---
    # Gmsh Mesh Algorithm Numerical Constants Reference
    # ------------------------------------------------
    # To set these in Python (using gmsh-sdk):
    # gmsh.option.setNumber("Mesh.Algorithm", <value>)
    
    # --- 2D Surface Mesh Algorithms (Mesh.Algorithm) ---
    # 1:  MeshAdapt (Robust, local modifications)
    # 2:  Automatic (Delaunay or MeshAdapt)
    # 3:  Initial Mesh (Initial triangulation only)
    # 5:  Delaunay (Standard, fast)
    # 6:  Frontal-Delaunay (Better element quality)
    # 7:  BAMG (Anisotropic mesh generator)
    # 8:  Frontal-Delaunay for Quads (Recombination optimized)
    # 9:  Packing of Parallelograms (High-quality quads)
    # 11: Quasi-structured Quad (Structured-like quad layout)
    if "quad" in elem_type.value:
        gmsh.model.mesh.setRecombine(2, surface)
        # Use subdivision to guarantee 100% quads
        gmsh.option.setNumber("Mesh.Algorithm", 1) 
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1) # All Quads
    else:
        gmsh.option.setNumber("Mesh.Algorithm", 1)
    
    if elem_type in [ElemType.TRI6, ElemType.QUAD8, ElemType.QUAD9]:
        order = 2
    else:
        order = 1
        
    if elem_type == ElemType.QUAD8:
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
    elif elem_type == ElemType.QUAD9:
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)

    # --- Save Mesh ---
    out_dir = f"rabbit_{elem_type.value}"
    os.makedirs(out_dir, exist_ok=True)
    msh_path = os.path.join(out_dir, f"rabbit_{elem_type.value}.msh")
    gmsh.write(msh_path)

    # --- Data Export ---
    export_mesh_data(elem_type)
    
    gmsh.finalize()

def export_mesh_data(elem_type):
    # Get all elements of dimension 2
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
    
    # Collect ALL node tags used in these elements
    used_node_tags = set()
    all_connect_tags = []
    for i in range(len(elem_types)):
        name, dim, order, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(elem_types[i])
        e_nodes = elem_node_tags[i].reshape((-1, num_nodes))
        for en in e_nodes:
            all_connect_tags.append(en.tolist())
            for tag in en:
                used_node_tags.add(tag)

    # Get coordinate data for ONLY the used nodes
    # We need to preserve the mapping from tag to new index
    sorted_used_tags = sorted(list(used_node_tags))
    node_map = {tag: i for i, tag in enumerate(sorted_used_tags)}
    
    # Efficiently get coordinates for these specific tags
    coords_dict = {}
    for tag in sorted_used_tags:
        coord, _, _, _ = gmsh.model.mesh.getNode(tag)
        coords_dict[tag] = coord

    final_nodes = np.array([coords_dict[tag] for tag in sorted_used_tags])
    source_connect = np.asarray(
        [[node_map[tag] for tag in en] for en in all_connect_tags],
        dtype=np.int64,
    )
    convention = meshconv.ConnectConvention(
        RILEY_ELEMENT_TYPES[elem_type],
        meshconv.EConnectAxis.ROW,
        0,
        node_order=meshconv.ENodeOrder.VTK,
        material_normal_hint=(0.0, 0.0, 1.0),
    )
    mesh = meshconv.convert_mesh(final_nodes, source_connect, convention)
    meshconv.verify_mesh(mesh)
    final_connect = mesh.connect

    out_dir = f"rabbit_{elem_type.value}"
    os.makedirs(out_dir, exist_ok=True)
    
    # Coords CSV
    with open(os.path.join(out_dir, "coords.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        for row in final_nodes:
            writer.writerow([f"{x:.18e}" for x in row])

    # Connectivity CSV
    with open(os.path.join(out_dir, "connectivity.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        for row in final_connect:
            writer.writerow(row)

    # UVs
    x_coords, y_coords = final_nodes[:, 0], final_nodes[:, 1]
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    range_x = max_x - min_x
    u = 0.25 + (x_coords - min_x) / range_x * 0.5
    min_y = np.min(y_coords)
    v = (y_coords - min_y) / range_x * 0.5
    
    with open(os.path.join(out_dir, "uvs.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        for uu, vv in zip(u, v):
            writer.writerow([f"{uu:.18e}", f"{vv:.18e}"])

    # Visualization
    plot_smooth_mesh(out_dir, elem_type, final_nodes, final_connect)
    print(f"Exported smooth data to {out_dir}")

def plot_smooth_mesh(data_dir, elem_type, coords, connect):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # For visualization, we always draw lines between the corner nodes.
    # In GMSH: 
    # Tri3: 3 nodes (all corners)
    # Tri6: 6 nodes (first 3 are corners)
    # Quad4: 4 nodes (all corners)
    # Quad8: 8 nodes (first 4 are corners)
    # Quad9: 9 nodes (first 4 are corners)
    num_corners = 3 if "tri" in elem_type.value else 4
    
    polygons = []
    corner_indices = set()
    for en in connect:
        poly_nodes = [coords[idx][:2] for idx in en[:num_corners]]
        polygons.append(poly_nodes)
        for idx in en[:num_corners]:
            corner_indices.add(idx)

    coll = PolyCollection(polygons, facecolors='none', edgecolors='black', linewidths=0.8)
    ax.add_collection(coll)
    
    # Plot ALL nodes, but color corners and midside nodes differently
    all_indices = np.arange(len(coords))
    is_corner = np.array([i in corner_indices for i in all_indices])
    
    num_corner = np.sum(is_corner)
    num_midside = np.sum(~is_corner)
    print(f"  Vis: {elem_type.value} - Corners: {num_corner}, Midside: {num_midside}")

    # Corner nodes: Lime Green
    ax.scatter(coords[is_corner, 0], coords[is_corner, 1], 
               s=16, c='limegreen', marker='o', 
               edgecolors='black', linewidths=0.6, zorder=4)
    
    # Midside nodes: Smaller, lighter to show they belong to the element
    if not np.all(is_corner):
        ax.scatter(coords[~is_corner, 0], coords[~is_corner, 1], 
                   s=8, c='white', marker='o', 
                   edgecolors='black', linewidths=0.3, zorder=3)
    
    ax.set_aspect('equal')
    plt.axis('off')
    
    # Save in two locations:
    # 1. Alongside the mesh CSVs
    png_path_local = os.path.join(data_dir, f"rabbit_{elem_type.value}.png")
    plt.savefig(png_path_local, dpi=300, bbox_inches='tight')
    
    # 2. In the central smooth visualization directory
    vis_dir = "vis/"
    os.makedirs(vis_dir, exist_ok=True)
    png_path_central = os.path.join(vis_dir, f"rabbit_{elem_type.value}.png")
    plt.savefig(png_path_central, dpi=300, bbox_inches='tight')
    
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=float, default=TARGET_LENGTH)
    parser.add_argument("--edge", type=float, default=EDGE_FRACTION)
    args = parser.parse_args()

    for etype in ElemType:
        print(f"Processing smooth {etype.value}...")
        mesh_rabbit_smooth(args.length, args.edge, etype)
