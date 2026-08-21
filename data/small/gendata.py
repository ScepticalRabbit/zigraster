import numpy as np
import os
from pathlib import Path

from riley.python import meshconv

# Coordinate System: Right-handed Cartesian (X right, Y up, Z towards viewer).
# Vertex Winding: All elements MUST follow Counter-Clockwise (CCW) winding.
# This ensures positive signed area calculation in the rasterizer, which is 
# critical for correct shape function interpolation and weight distribution.

def save_case(base_dir, name, coords, connect, disp_x, disp_y, disp_z):
    mesh = meshconv.MeshData(
        coords=np.ascontiguousarray(coords, dtype=np.float64),
        connect={"connect1": np.ascontiguousarray(connect, dtype=np.int64)},
        mesh_type="surface",
    )
    connect = meshconv.enforce_mesh_convention(mesh).connect["connect1"]
    out_dir = Path(base_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "coords.csv", coords, delimiter=",")
    np.savetxt(out_dir / "connectivity.csv", connect.astype(int), 
               delimiter=",", fmt='%d')
    np.savetxt(out_dir / "field_disp_x.csv", disp_x, delimiter=",")
    np.savetxt(out_dir / "field_disp_y.csv", disp_y, delimiter=",")
    np.savetxt(out_dir / "field_disp_z.csv", disp_z, delimiter=",")

def compute_disps(coords, frame0_params, frame1_params):
    num_nodes = coords.shape[0]
    xmin, ymin, _ = np.min(coords, axis=0)
    xmax, ymax, _ = np.max(coords, axis=0)
    
    xrng = xmax - xmin if xmax > xmin else 1.0
    yrng = ymax - ymin if ymax > ymin else 1.0
    
    disps_x = np.zeros((num_nodes, 2))
    disps_y = np.zeros((num_nodes, 2))
    disps_z = np.zeros((num_nodes, 2))
    
    for i, (ux_min, ux_max, uy_min, uy_max, uz_min, uz_max) in enumerate(
        [frame0_params, frame1_params]):
        for j in range(num_nodes):
            x, y, _ = coords[j]
            # Normalised local coords [0, 1]
            xi = (x - xmin) / xrng
            eta = (y - ymin) / yrng
            
            disps_x[j, i] = ux_min + (ux_max - ux_min) * xi
            disps_y[j, i] = uy_min + (uy_max - uy_min) * eta
            
            # Z gradient from bottom-left (0,0) to top-right (1,1)
            # Use average of normalized x and y for a diagonal gradient
            zeta = (xi + eta) / 2.0
            disps_z[j, i] = uz_min + (uz_max - uz_min) * zeta
            
    return disps_x, disps_y, disps_z

def move_midside(v1, v2, centroid, offset):
    # straight midside
    m = (v1 + v2) / 2.0
    # direction from centroid to m
    dir = m - centroid
    dist = np.linalg.norm(dir)
    if dist < 1e-9:
        # Fallback if m is at centroid (unlikely for boundary edge)
        return m
    unit_dir = dir / dist
    # move away from centroid if offset > 0, towards if offset < 0
    return m + unit_dir * offset

def compute_uvs(coords, u_range=(0.4, 0.6), v_range=(0.4, 0.6)):
    """
    Project 3D coordinates to 2D UV coordinates within a specified range.
    Assumes projection along the Z-axis onto the XY plane.
    """
    num_nodes = coords.shape[0]
    xmin, ymin, _ = np.min(coords, axis=0)
    xmax, ymax, _ = np.max(coords, axis=0)
    
    xrng = xmax - xmin if xmax > xmin else 1.0
    yrng = ymax - ymin if ymax > ymin else 1.0
    
    uvs = np.zeros((num_nodes, 2))
    for j in range(num_nodes):
        x, y, _ = coords[j]
        # Normalize to [0, 1] then scale to [u_min, u_max]
        uvs[j, 0] = u_range[0] + (u_range[1] - u_range[0]) * (x - xmin) / xrng
        uvs[j, 1] = v_range[0] + (v_range[1] - v_range[0]) * (y - ymin) / yrng
    return uvs

def save_uvs(base_dir, name, uvs):
    out_dir = Path(base_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "uvs.csv", uvs, delimiter=",")
