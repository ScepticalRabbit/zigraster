import numpy as np
from pathlib import Path
import gendata

from riley.python import meshconv

# Coordinate System: Right-handed Cartesian (X right, Y up, Z towards viewer).
# Vertex Winding: All elements MUST follow Counter-Clockwise (CCW) winding.
# This ensures positive signed area calculation in the rasterizer, which is 
# critical for correct shape function interpolation and weight distribution.
# Connectivity Order:
# Tri6: [c0, c1, c2, m01, m12, m20] (CCW)
# Quad8: [c0, c1, c2, c3, m01, m12, m23, m30] (CCW)
# Quad9: [c0, c1, c2, c3, m01, m12, m23, m30, center] (CCW)

def generate_twoelems(base_dir, length, d_shift, frame0, frame1):
    """
    Generates two-element test cases with consistent Counter-Clockwise (CCW) winding.
    Left element bulges IN, Right element bulges OUT.
    """
    h = np.sqrt(3) / 2 * length
    
    # --- Triangles ---
    v0 = np.array([0.0, length/2.0, 0.0])
    v1 = np.array([h, 0.0, 0.0])
    v2 = np.array([0.0, -length/2.0, 0.0])
    v3 = np.array([-h, 0.0, 0.0])
    
    c_right = (v0 + v1 + v2) / 3.0
    c_left = (v0 + v2 + v3) / 3.0
    
    # Tri3
    coords_tri3 = np.array([v0, v1, v2, v3])
    # Right: [1, 0, 2], Left: [2, 0, 3] (CCW)
    connect_tri3 = np.array([[1, 0, 2], [2, 0, 3]])
    dx, dy, dz = gendata.compute_disps(coords_tri3, frame0, frame1)
    gendata.save_case(base_dir, "tri3_twoelems", coords_tri3, connect_tri3, meshconv.EElementType.TRI3, dx, dy, dz)
    
    # Tri6
    m10_r = gendata.move_midside(v1, v0, c_right, d_shift)
    m02_r = gendata.move_midside(v0, v2, c_right, d_shift)
    m21_r = gendata.move_midside(v2, v1, c_right, d_shift)
    m20_l = gendata.move_midside(v2, v0, c_left, -d_shift)
    m03_l = gendata.move_midside(v0, v3, c_left, -d_shift)
    m32_l = gendata.move_midside(v3, v2, c_left, -d_shift)
    
    coords_tri6 = np.array([v0, v1, v2, v3, m10_r, m02_r, m21_r, m20_l, m03_l, m32_l])
    connect_tri6 = np.array([
        [1, 0, 2, 4, 5, 6],
        [2, 0, 3, 7, 8, 9]
    ])
    dx, dy, dz = gendata.compute_disps(coords_tri6, frame0, frame1)
    gendata.save_case(base_dir, "tri6_twoelems", coords_tri6, connect_tri6, meshconv.EElementType.TRI6, dx, dy, dz)
    
    # --- Quads ---
    vq0 = np.array([-length, length/2.0, 0.0])
    vq1 = np.array([0.0, length/2.0, 0.0])
    vq2 = np.array([length, length/2.0, 0.0])
    vq3 = np.array([-length, -length/2.0, 0.0])
    vq4 = np.array([0.0, -length/2.0, 0.0])
    vq5 = np.array([length, -length/2.0, 0.0])
    
    cq_left = np.array([-length/2.0, 0.0, 0.0])
    cq_right = np.array([length/2.0, 0.0, 0.0])
    
    # Quad4 (No bulges)
    coords_q4 = np.array([vq0, vq1, vq2, vq3, vq4, vq5])
    # Left: [3, 4, 1, 0], Right: [4, 5, 2, 1] (CCW)
    connect_q4 = np.array([[3, 4, 1, 0], [4, 5, 2, 1]])
    dx, dy, dz = gendata.compute_disps(coords_q4, frame0, frame1)
    gendata.save_case(base_dir, "quad4_twoelems", coords_q4, connect_q4, meshconv.EElementType.QUAD4, dx, dy, dz)
    
    # Quad8
    mq34_l = gendata.move_midside(vq3, vq4, cq_left, -d_shift)
    mq41_l = gendata.move_midside(vq4, vq1, cq_left, -d_shift)
    mq10_l = gendata.move_midside(vq1, vq0, cq_left, -d_shift)
    mq03_l = gendata.move_midside(vq0, vq3, cq_left, -d_shift)
    
    mq45_r = gendata.move_midside(vq4, vq5, cq_right, d_shift)
    mq52_r = gendata.move_midside(vq5, vq2, cq_right, d_shift)
    mq21_r = gendata.move_midside(vq2, vq1, cq_right, d_shift)
    mq14_r = gendata.move_midside(vq1, vq4, cq_right, d_shift)
    
    coords_q8 = np.array([vq0, vq1, vq2, vq3, vq4, vq5, 
                          mq34_l, mq41_l, mq10_l, mq03_l,
                          mq45_r, mq52_r, mq21_r, mq14_r])
    
    connect_q8 = np.array([
        [3, 4, 1, 0, 6, 7, 8, 9],
        [4, 5, 2, 1, 10, 11, 12, 13]
    ])
    dx, dy, dz = gendata.compute_disps(coords_q8, frame0, frame1)
    gendata.save_case(base_dir, "quad8_twoelems", coords_q8, connect_q8, meshconv.EElementType.QUAD8, dx, dy, dz)
    
    # Quad9
    coords_q9 = np.vstack([coords_q8, [cq_left, cq_right]])
    connect_q9 = np.array([
        [3, 4, 1, 0, 6, 7, 8, 9, 14],
        [4, 5, 2, 1, 10, 11, 12, 13, 15]
    ])
    dx, dy, dz = gendata.compute_disps(coords_q9, frame0, frame1)
    gendata.save_case(base_dir, "quad9_twoelems", coords_q9, connect_q9, meshconv.EElementType.QUAD9, dx, dy, dz)

def main():
    base_dir = "."
    L = 10.0
    D = 1.0
    FRAME0_PARAMS = (-1e-6, 1e-6, -1e-6, 1e-6, -1e-6, 1e-6)
    FRAME1_PARAMS = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
    generate_twoelems(base_dir, L, D, FRAME0_PARAMS, FRAME1_PARAMS)
    print(f"Generated two-element cases in {base_dir}/")

if __name__ == "__main__":
    main()
