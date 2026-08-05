import numpy as np
import os

def save_csv(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(
        path,
        data,
        delimiter=",",
        fmt="%.10f" if data.dtype == np.float64 else "%d",
    )

WIDTH = 16.0
HEIGHT = 10.0
TILT_Z = 5.0

def compute_uvs(coords):
    xmin, ymin, _ = np.min(coords, axis=0)
    xmax, ymax, _ = np.max(coords, axis=0)
    xrng = max(xmax - xmin, 1.0)
    yrng = max(ymax - ymin, 1.0)
    uvs = np.zeros((len(coords), 2))
    for j in range(len(coords)):
        x, y, _ = coords[j]
        uvs[j, 0] = 0.4 + 0.2 * (x - xmin) / xrng
        uvs[j, 1] = 0.4 + 0.2 * (y - ymin) / yrng
    return uvs

def compute_rgb_fields(coords):
    xmin, ymin, _ = np.min(coords, axis=0)
    xmax, ymax, _ = np.max(coords, axis=0)
    xrng = max(xmax - xmin, 1.0)
    yrng = max(ymax - ymin, 1.0)
    fields = np.zeros((len(coords), 3))
    for j in range(len(coords)):
        xn = (coords[j, 0] - xmin) / xrng
        yn = (coords[j, 1] - ymin) / yrng
        r = xn
        g = yn
        b = 1.0 - (xn + yn) / 2.0
        fields[j] = [r, g, b]
    return fields

def apply_tilt(coords):
    tilted = coords.copy()
    tilted[:, 2] = TILT_Z * (coords[:, 0] / WIDTH)
    return tilted

def generate_fullscreen_tilt(etype, out_dir):
    if "tri" in etype:
        coords = np.array(
            [
                [0, 0, 0],
                [WIDTH, 0, 0],
                [WIDTH, HEIGHT, 0],
                [0, HEIGHT, 0]
            ],
            dtype=float,
        )
        if etype == "tri6":
            coords = np.array(
                [
                    [0, 0, 0],
                    [WIDTH, 0, 0],
                    [WIDTH, HEIGHT, 0],
                    [0, HEIGHT, 0],
                    [WIDTH / 2, 0, 0],
                    [WIDTH, HEIGHT / 2, 0],
                    [WIDTH / 2, HEIGHT, 0],
                    [0, HEIGHT / 2, 0],
                    [WIDTH / 2, HEIGHT / 2, 0],
                ],
                dtype=float,
            )
            connect = np.array([[0, 1, 2, 4, 5, 8], [0, 2, 3, 8, 6, 7]])
        else:
            connect = np.array([[0, 1, 2], [0, 2, 3]])
    else:
        coords = np.array(
            [
                [0, 0, 0],
                [WIDTH, 0, 0],
                [WIDTH, HEIGHT, 0],
                [0, HEIGHT, 0]
            ],
            dtype=float,
        )
        if etype in ["quad8", "quad9"]:
            coords = np.array(
                [
                    [0, 0, 0],
                    [WIDTH, 0, 0],
                    [WIDTH, HEIGHT, 0],
                    [0, HEIGHT, 0],
                    [WIDTH / 2, 0, 0],
                    [WIDTH, HEIGHT / 2, 0],
                    [WIDTH / 2, HEIGHT, 0],
                    [0, HEIGHT / 2, 0],
                    [WIDTH / 2, HEIGHT / 2, 0],
                ],
                dtype=float,
            )
            if etype == "quad8":
                connect = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
            else:
                connect = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]])
        else:
            connect = np.array([[0, 1, 2, 3]])
    
    tilted_coords = apply_tilt(coords)
    save_csv(f"{out_dir}/coords.csv", tilted_coords)
    save_csv(f"{out_dir}/connect.csv", connect)
    save_csv(f"{out_dir}/field.csv", compute_rgb_fields(tilted_coords))
    save_csv(f"{out_dir}/uvs.csv", compute_uvs(tilted_coords))

if __name__ == "__main__":
    elements = [
        "tri3",
        "tri6",
        "quad4ibi",
        "quad4newton",
        "quad8",
        "quad9"
    ]
    for et in elements:
        print(f"Generating tilted data for {et}...")
        generate_fullscreen_tilt(et, f"data/tilt/{et}_fullraster")
