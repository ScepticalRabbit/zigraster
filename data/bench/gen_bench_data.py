import numpy as np
import os

from riley.python import meshconv


ELEMENT_TYPES = {
    "tri3": meshconv.EElementType.TRI3,
    "tri6": meshconv.EElementType.TRI6,
    "quad4ibi": meshconv.EElementType.QUAD4,
    "quad4newton": meshconv.EElementType.QUAD4,
    "quad8": meshconv.EElementType.QUAD8,
    "quad9": meshconv.EElementType.QUAD9,
}

def save_csv(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, data, delimiter=',', fmt='%.10f' if data.dtype == np.float64 else '%d')

def save_surface_mesh(
    out_dir,
    elem_type,
    coords,
    connect,
    material_normal_hint=(0.0, 0.0, 1.0),
):
    convention = meshconv.ConnectConvention(
        elem_type,
        meshconv.EConnectAxis.ROW,
        0,
        node_order=meshconv.ENodeOrder.RILEY,
        material_normal_hint=material_normal_hint,
    )
    mesh = meshconv.convert_mesh(coords, connect, convention)
    meshconv.verify_mesh(mesh)
    save_csv(f"{out_dir}/coords.csv", mesh.coords)
    save_csv(f"{out_dir}/connect.csv", mesh.connect)

WIDTH = 16.0
HEIGHT = 10.0
SPHERE_CENTER = np.array((0.0, 0.0, 5.0), dtype=np.float64)
SPHERE_ORIENT_TOL = 1.0e-12


def verify_sphere_orientation(
    coords: np.ndarray,
    connect: np.ndarray,
    elem_type: meshconv.EElementType,
) -> None:
    """Verify that every non-collapsed sphere element faces outwards."""
    corner_count = 3 if elem_type in (
        meshconv.EElementType.TRI3,
        meshconv.EElementType.TRI6,
    ) else 4
    corners = coords[connect[:, :corner_count]]
    edge_a = corners[:, 1] - corners[:, 0]
    edge_b = corners[:, 2] - corners[:, 0]
    face_normals = np.cross(edge_a, edge_b)
    face_centres = np.mean(corners, axis=1)
    radial_vectors = face_centres - SPHERE_CENTER
    orient_metrics = np.einsum(
        "ij,ij->i",
        face_normals,
        radial_vectors,
    )
    noncollapsed = np.linalg.norm(face_normals, axis=1) > SPHERE_ORIENT_TOL
    inward = noncollapsed & (orient_metrics <= SPHERE_ORIENT_TOL)
    if np.any(inward):
        elem_idxs = np.flatnonzero(inward)
        raise ValueError(
            "Sphere contains inward-facing elements at rows "
            f"{elem_idxs.tolist()}."
        )

def compute_uvs(coords, u_range=(0.4, 0.6), v_range=(0.4, 0.6)):
    xmin, ymin, _ = np.min(coords, axis=0)
    xmax, ymax, _ = np.max(coords, axis=0)
    xrng, yrng = max(xmax - xmin, 1.0), max(ymax - ymin, 1.0)
    uvs = np.zeros((len(coords), 2))
    for j in range(len(coords)):
        x, y, _ = coords[j]
        uvs[j, 0] = u_range[0] + (u_range[1] - u_range[0]) * (x - xmin) / xrng
        uvs[j, 1] = v_range[0] + (v_range[1] - v_range[0]) * (y - ymin) / yrng
    return uvs

def compute_rgb_fields(coords):
    xmin, ymin, _ = np.min(coords, axis=0)
    xmax, ymax, _ = np.max(coords, axis=0)
    xrng, yrng = max(xmax - xmin, 1.0), max(ymax - ymin, 1.0)
    fields = np.zeros((len(coords), 3))
    for j in range(len(coords)):
        xn = (coords[j, 0] - xmin) / xrng
        yn = (coords[j, 1] - ymin) / yrng
        # Purely linear gradient experiment
        r = xn
        g = yn
        b = 1.0 - (xn + yn) / 2.0
        fields[j] = [r, g, b]
    return fields

def generate_fullscreen(etype, out_dir):
    if "tri" in etype:
        coords = np.array([[0, 0, 0], [WIDTH, 0, 0], [WIDTH, HEIGHT, 0], [0, HEIGHT, 0]], dtype=float)
        if etype == "tri6":
            coords = np.array([
                [0, 0, 0], [WIDTH, 0, 0], [WIDTH, HEIGHT, 0], [0, HEIGHT, 0],
                [WIDTH/2, 0, 0], [WIDTH, HEIGHT/2, 0], [WIDTH/2, HEIGHT, 0], [0, HEIGHT/2, 0],
                [WIDTH/2, HEIGHT/2, 0]
            ], dtype=float)
            connect = np.array([[0, 1, 2, 4, 5, 8], [0, 2, 3, 8, 6, 7]])
        else:
            connect = np.array([[0, 1, 2], [0, 2, 3]])
    else:
        coords = np.array([[0, 0, 0], [WIDTH, 0, 0], [WIDTH, HEIGHT, 0], [0, HEIGHT, 0]], dtype=float)
        if etype in ["quad8", "quad9"]:
            coords = np.array([
                [0, 0, 0], [WIDTH, 0, 0], [WIDTH, HEIGHT, 0], [0, HEIGHT, 0],
                [WIDTH/2, 0, 0], [WIDTH, HEIGHT/2, 0], [WIDTH/2, HEIGHT, 0], [0, HEIGHT/2, 0],
                [WIDTH/2, HEIGHT/2, 0]
            ], dtype=float)
            if etype == "quad8": connect = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
            else: connect = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]])
        else:
            connect = np.array([[0, 1, 2, 3]])
    save_surface_mesh(out_dir, ELEMENT_TYPES[etype], coords, connect)
    save_csv(f"{out_dir}/field.csv", compute_rgb_fields(coords))
    save_csv(f"{out_dir}/uvs.csv", compute_uvs(coords))

def generate_grid(etype, out_dir, N=320):
    is_higher = etype in ["tri6", "quad8", "quad9"]
    step = 2 if is_higher else 1
    xn, yn = N * step, N * step
    x, y = np.linspace(0, WIDTH, xn + 1), np.linspace(0, HEIGHT, yn + 1)
    xv, yv = np.meshgrid(x, y)
    coords = np.stack([xv.flatten(), yv.flatten(), np.zeros_like(xv.flatten())], axis=1)
    conn = []
    for jj in range(0, yn, step):
        for ii in range(0, xn, step):
            i0 = jj * (xn + 1) + ii
            i1 = jj * (xn + 1) + ii + step
            i2 = (jj + step) * (xn + 1) + ii + step
            i3 = (jj + step) * (xn + 1) + ii
            if etype == "tri3":
                conn.append([i0, i1, i2])
                conn.append([i0, i2, i3])
            elif "quad" in etype and not is_higher:
                conn.append([i0, i1, i2, i3])
            elif etype == "tri6":
                m01, m12, m23, m30 = i0+1, i1+(xn+1), i3+1, i0+(xn+1)
                m02 = i0+(xn+1)+1
                conn.append([i0, i1, i2, m01, m12, m02])
                conn.append([i0, i2, i3, m02, m23, m30])
            elif etype in ["quad8", "quad9"]:
                m01, m12, m23, m30 = i0+1, i1+(xn+1), i3+1, i0+(xn+1)
                q8 = [i0, i1, i2, i3, m01, m12, m23, m30]
                if etype == "quad9": q8.append(i0+(xn+1)+1)
                conn.append(q8)
    save_surface_mesh(
        out_dir,
        ELEMENT_TYPES[etype],
        coords,
        np.array(conn),
    )
    save_csv(f"{out_dir}/field.csv", compute_rgb_fields(coords))
    save_csv(f"{out_dir}/uvs.csv", compute_uvs(coords))

def generate_sphere(etype, out_dir, N_target):
    # side is the number of elements per side of the grid
    side = int(np.sqrt(N_target)) + 1
    
    # For high order elements, we need a grid that provides mid-nodes
    is_high = etype in ["tri6", "quad8", "quad9"]
    grid_side = side * 2 if is_high else side
    rows, cols = grid_side + 1, grid_side + 1
    
    v_vals = np.linspace(0, np.pi, rows)
    # Move seam to the back by using -pi to pi
    u_vals = np.linspace(-np.pi, np.pi, cols)
    
    coords = []
    uvs = []
    fields = []
    
    for r, v in enumerate(v_vals):
        for c, u in enumerate(u_vals):
            x = np.cos(u) * np.sin(v)
            y = np.sin(u) * np.sin(v)
            z = np.cos(v) + 5.0
            coords.append([x, y, z])
            # Normalize u from [-pi, pi] to [0, 1]
            uu = (u + np.pi) / (2 * np.pi)
            vv = v / np.pi
            # Zoom in: map [0, 1] to [0.4, 0.6]
            uu = 0.4 + 0.2 * uu
            vv = 0.4 + 0.2 * vv
            uvs.append([uu, vv])
            fields.append([uu, vv, 1.0 - (uu + vv) / 2.0])
            
    coords = np.array(coords)
    uvs = np.array(uvs)
    fields = np.array(fields)
    
    conn = []
    step = 2 if is_high else 1
    for r in range(0, grid_side, step):
        for c in range(0, grid_side, step):
            # Base grid indices for this element's corners
            # i0 (0,0), i1 (1,0), i2 (1,1), i3 (0,1)
            # Winding for outward normal: i0, i3, i2, i1
            i0 = r * cols + c
            i1 = r * cols + (c + step)
            i2 = (r + step) * cols + (c + step)
            i3 = (r + step) * cols + c
            
            if etype == "tri3":
                conn.append([i0, i3, i2])
                conn.append([i0, i2, i1])
            elif etype in ["quad4ibi", "quad4newton"]:
                conn.append([i0, i3, i2, i1])
            elif etype == "tri6":
                # Tri 1 (i0, i3, i2): m03, m32, m20(diag)
                v0, v1, v2 = i0, i3, i2
                m01 = (r + 1) * cols + c
                m12 = (r + 2) * cols + (c + 1)
                m20 = (r + 1) * cols + (c + 1) # diagonal
                conn.append([v0, v1, v2, m01, m12, m20])
                # Tri 2 (i0, i2, i1): m02(diag), m21, m10
                v0, v1, v2 = i0, i2, i1
                m01 = (r + 1) * cols + (c + 1) # diagonal
                m12 = (r + 1) * cols + (c + 2)
                m20 = r * cols + (c + 1)
                conn.append([v0, v1, v2, m01, m12, m20])
            elif etype in ["quad8", "quad9"]:
                # Source grid edges in increasing parameter directions.
                m01 = r * cols + (c + 1)
                m12 = (r + 1) * cols + (c + 2)
                m23 = (r + 2) * cols + (c + 1)
                m30 = (r + 1) * cols + c
                # Riley corners wind outwards. Midsides follow those edges.
                q = [i0, i3, i2, i1, m30, m23, m12, m01]
                if etype == "quad9":
                    q.append((r + 1) * cols + (c + 1))
                conn.append(q)
                
    connect = np.asarray(conn, dtype=np.int64)
    elem_type = ELEMENT_TYPES[etype]
    verify_sphere_orientation(coords, connect, elem_type)
    save_surface_mesh(
        out_dir,
        elem_type,
        coords,
        connect,
        material_normal_hint=None,
    )
    save_csv(f"{out_dir}/uvs.csv", uvs)
    save_csv(f"{out_dir}/field.csv", fields)

if __name__ == "__main__":
    elements = [
        "tri3",
        "tri6",
        "quad4ibi",
        "quad4newton",
        "quad8",
        "quad9",
    ]
    for et in elements:
        print(f"Generating data for {et}...")
        generate_fullscreen(et, f"data/bench/{et}_fullraster")
        generate_grid(et, f"data/bench/{et}_geom", N=320)
        for size_str, target_elems in [
            ("1e3", 1000),
            ("1e4", 10000),
            ("1e5", 100000),
            ("1e6", 1000000),
        ]:
            if "tri" in et:
                N = int(round(np.sqrt(target_elems / 2)))
            else:
                N = int(round(np.sqrt(target_elems)))
            generate_grid(et, f"data/bench/{et}_geom_{size_str}", N=N)

        if et != "quad4ibi":
            generate_sphere(et, f"data/bench/{et}_sphere200", 200)
            generate_sphere(et, f"data/bench/{et}_sphere2000", 2000)
