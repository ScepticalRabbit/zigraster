import numpy as np
from riley.python import meshconv
from pathlib import Path
import gendata
import gen_data_twoelems

# Coordinate System: Right-handed Cartesian (X right, Y up, Z towards viewer).
# Vertex Winding: All elements MUST follow Counter-Clockwise (CCW) winding.
# This ensures positive signed area calculation in the rasterizer, which is 
# critical for correct shape function interpolation and weight distribution.

def generate_fullscreen(base_dir, width, height, frame0, frame1):
    # Tri3 Full Screen (2 elements)
    coords_tri3 = np.array([
        [0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0]
    ], dtype=float)
    # 0,1,2 (CCW), 0,2,3 (CCW)
    connect_tri3 = np.array([[0, 1, 2], [0, 2, 3]])
    dx, dy, dz = gendata.compute_disps(coords_tri3, frame0, frame1)
    gendata.save_case(base_dir, "tri3_fullscreen", meshconv.EElementType.TRI3, coords_tri3, connect_tri3, dx, dy, dz)

    # Tri6 Full Screen (2 elements)
    coords_tri6 = np.array([
        [0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0],
        [width/2, 0, 0], [width, height/2, 0], [width/2, height, 0], [0, height/2, 0],
        [width/2, height/2, 0]
    ], dtype=float)
    # Elem 0: corners 0,1,2, midsides 4,5,8 (CCW)
    # Elem 1: corners 0,2,3, midsides 8,6,7 (CCW)
    connect_tri6 = np.array([
        [0, 1, 2, 4, 5, 8],
        [0, 2, 3, 8, 6, 7]
    ])
    dx, dy, dz = gendata.compute_disps(coords_tri6, frame0, frame1)
    gendata.save_case(base_dir, "tri6_fullscreen", meshconv.EElementType.TRI6, coords_tri6, connect_tri6, dx, dy, dz)

    # Quad4 Full Screen (1 element)
    coords_quad4 = coords_tri3.copy()
    # 0,1,2,3 (CCW)
    connect_quad4 = np.array([[0, 1, 2, 3]])
    dx, dy, dz = gendata.compute_disps(coords_quad4, frame0, frame1)
    gendata.save_case(base_dir, "quad4_fullscreen", meshconv.EElementType.QUAD4, coords_quad4, connect_quad4, dx, dy, dz)

    # Quad8 Full Screen (1 element)
    coords_quad8 = np.array([
        [0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0],
        [width/2, 0, 0], [width, height/2, 0], [width/2, height, 0], [0, height/2, 0]
    ], dtype=float)
    # 0,1,2,3, 4,5,6,7 (CCW)
    connect_quad8 = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    dx, dy, dz = gendata.compute_disps(coords_quad8, frame0, frame1)
    gendata.save_case(base_dir, "quad8_fullscreen", meshconv.EElementType.QUAD8, coords_quad8, connect_quad8, dx, dy, dz)

    # Quad9 Full Screen (1 element)
    coords_quad9 = np.array([
        [0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0],
        [width/2, 0, 0], [width, height/2, 0], [width/2, height, 0], [0, height/2, 0],
        [width/2, height/2, 0]
    ], dtype=float)
    # 0,1,2,3, 4,5,6,7, 8 (CCW)
    connect_quad9 = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]])
    dx, dy, dz = gendata.compute_disps(coords_quad9, frame0, frame1)
    gendata.save_case(base_dir, "quad9_fullscreen", meshconv.EElementType.QUAD9, coords_quad9, connect_quad9, dx, dy, dz)

def generate_singleelem(base_dir, length, d_shift, frame0, frame1):
    """
    Generates single-element test cases with consistent Counter-Clockwise (CCW) winding.
    CCW winding is REQUIRED for correct area and weight calculation in the rasterizer.
    """
    h = np.sqrt(3) / 2 * length
    tri_centroid = np.array([length/2, h/3, 0])
    v_tri = np.array([
        [0, 0, 0],
        [length, 0, 0],
        [length/2, h, 0]
    ])

    # Tri3 Single: 0,1,2 (CCW)
    dx, dy, dz = gendata.compute_disps(v_tri, frame0, frame1)
    gendata.save_case(base_dir, "tri3_single", meshconv.EElementType.TRI3, v_tri, np.array([[0, 1, 2]]), dx, dy, dz)

    # Tri6 Single: 0,1,2, 3,4,5 (CCW)
    m01 = (v_tri[0] + v_tri[1]) / 2.0
    m12 = gendata.move_midside(v_tri[1], v_tri[2], tri_centroid, d_shift)
    m20 = gendata.move_midside(v_tri[2], v_tri[0], tri_centroid, -d_shift)
    coords_tri6 = np.vstack([v_tri, [m01, m12, m20]])
    dx, dy, dz = gendata.compute_disps(coords_tri6, frame0, frame1)
    gendata.save_case(base_dir, "tri6_single", meshconv.EElementType.TRI6, coords_tri6, np.array([[0, 1, 2, 3, 4, 5]]), dx, dy, dz)

    # Quad Single (Square)
    v_quad = np.array([
        [0, 0, 0], [length, 0, 0], [length, length, 0], [0, length, 0]
    ])
    quad_centroid = np.array([length/2, length/2, 0])

    # Quad4 Single: 0,1,2,3 (CCW)
    dx, dy, dz = gendata.compute_disps(v_quad, frame0, frame1)
    gendata.save_case(base_dir, "quad4_single", meshconv.EElementType.QUAD4, v_quad, np.array([[0, 1, 2, 3]]), dx, dy, dz)

    # Quad8 Single: 0,1,2,3, 4,5,6,7 (CCW)
    m01_q = (v_quad[0] + v_quad[1]) / 2.0
    m12_q = gendata.move_midside(v_quad[1], v_quad[2], quad_centroid, d_shift)
    m23_q = gendata.move_midside(v_quad[2], v_quad[3], quad_centroid, -d_shift)
    m30_q = (v_quad[3] + v_quad[0]) / 2.0
    coords_quad8 = np.vstack([v_quad, [m01_q, m12_q, m23_q, m30_q]])
    dx, dy, dz = gendata.compute_disps(coords_quad8, frame0, frame1)
    gendata.save_case(base_dir, "quad8_single", meshconv.EElementType.QUAD8, coords_quad8, np.array([[0, 1, 2, 3, 4, 5, 6, 7]]), dx, dy, dz)

    # Quad9 Single: 0,1,2,3, 4,5,6,7, 8 (CCW)
    coords_quad9 = np.vstack([coords_quad8, [quad_centroid]])
    dx, dy, dz = gendata.compute_disps(coords_quad9, frame0, frame1)
    gendata.save_case(base_dir, "quad9_single", meshconv.EElementType.QUAD9, coords_quad9, np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]]), dx, dy, dz)

def generate_uvs(base_dir, u_range, v_range):
    cases = [
        "tri3_fullscreen", "tri6_fullscreen", "quad4_fullscreen", "quad8_fullscreen", "quad9_fullscreen",
        "tri3_single", "tri6_single", "quad4_single", "quad8_single", "quad9_single",
        "tri3_twoelems", "tri6_twoelems", "quad4_twoelems", "quad8_twoelems", "quad9_twoelems"
    ]
    for case in cases:
        case_dir = Path(base_dir) / case
        if not case_dir.exists():
            print(f"Skipping {case}: {case_dir} does not exist")
            continue
        print(f"Generating UVs for {case}...")
        coords = np.loadtxt(case_dir / "coords.csv", delimiter=",")
        uvs = gendata.compute_uvs(coords, u_range, v_range)
        gendata.save_uvs(base_dir, case, uvs)

def main():
    base_dir = "."
    WIDTH = 16.0
    HEIGHT = 10.0
    L = 10.0
    D = 1.0
    FRAME0_PARAMS = (-1e-6, 1e-6, -1e-6, 1e-6, -1e-6, 1e-6)
    FRAME1_PARAMS = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
    U_RANGE = (0.4, 0.6)
    V_RANGE = (0.4, 0.6)

    generate_fullscreen(base_dir, WIDTH, HEIGHT, FRAME0_PARAMS, FRAME1_PARAMS)
    generate_singleelem(base_dir, L, D, FRAME0_PARAMS, FRAME1_PARAMS)
    gen_data_twoelems.generate_twoelems(base_dir, L, D, FRAME0_PARAMS, FRAME1_PARAMS)
    generate_uvs(base_dir, U_RANGE, V_RANGE)
    print(f"Generated all cases in {base_dir}/")

if __name__ == "__main__":
    main()
