import argparse
import csv
import os
from enum import Enum

import gmsh
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from riley.python import meshconv
from svgpathtools import svg2paths


TARGET_LENGTH = 1.0
FEEBS_SCALE = 0.85
EDGE_FRACTION = 0.07
MIN_SEGMENT_LENGTH = 0.005
STEPS_ON_PATH = 500
TRI_FACT = 0.6
QUAD_FACT = 0.5
QUAD_ALGORITHM = 8
QUAD_HIGH_ORDER_OPTIMIZE = 2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTLINE_DIR = os.path.join(SCRIPT_DIR, "outline")
VIS_DIR = os.path.join(SCRIPT_DIR, "vis")


class ElemType(Enum):
    TRI3 = "tri3"
    TRI6 = "tri6"
    QUAD4 = "quad4"
    QUAD8 = "quad8"
    QUAD9 = "quad9"


class BunnyName(Enum):
    FEEBS = "feebs"
    RILEY = "riley"


SVG_NAMES = {
    BunnyName.FEEBS: "Feebs_ToMesh.svg",
    BunnyName.RILEY: "Riley_ToMesh.svg",
}

RILEY_ELEMENT_TYPES = {
    ElemType.TRI3: meshconv.EElementType.TRI3,
    ElemType.TRI6: meshconv.EElementType.TRI6,
    ElemType.QUAD4: meshconv.EElementType.QUAD4,
    ElemType.QUAD8: meshconv.EElementType.QUAD8,
    ElemType.QUAD9: meshconv.EElementType.QUAD9,
}


def load_outline_path(svg_name):
    svg_path = os.path.join(OUTLINE_DIR, svg_name)
    paths, _ = svg2paths(svg_path)
    return paths[0]


def get_path_bbox(path):
    sample_points = np.array([path.point(tt / 500) for tt in range(501)])
    min_x = np.min(sample_points.real)
    max_x = np.max(sample_points.real)
    min_y = np.min(sample_points.imag)
    max_y = np.max(sample_points.imag)
    return min_x, max_x, min_y, max_y


def build_simplified_points(path, target_length):
    min_x, max_x, min_y, _ = get_path_bbox(path)
    scale = target_length / (max_x - min_x)

    def transform(point):
        x_coord = (point.real - min_x) * scale
        y_coord = -(point.imag - min_y) * scale
        return x_coord, y_coord

    simplified_points = []
    last_point = None

    for step_index in range(STEPS_ON_PATH):
        path_param = step_index / STEPS_ON_PATH
        current_point = path.point(path_param)
        transformed_point = transform(current_point)

        if last_point is None:
            simplified_points.append(transformed_point)
            last_point = transformed_point
            continue

        distance = np.sqrt(
            (transformed_point[0] - last_point[0]) ** 2
            + (transformed_point[1] - last_point[1]) ** 2
        )
        if distance > MIN_SEGMENT_LENGTH:
            simplified_points.append(transformed_point)
            last_point = transformed_point

    return flip_points_horizontal(simplified_points)


def flip_points_horizontal(points):
    x_coords = [point[0] for point in points]
    min_x = min(x_coords)
    max_x = max(x_coords)

    flipped_points = []
    for x_coord, y_coord in points:
        flipped_x_coord = max_x + min_x - x_coord
        flipped_points.append((flipped_x_coord, y_coord))

    return flipped_points


def get_target_length_by_bunny():
    return {
        BunnyName.RILEY: TARGET_LENGTH,
        BunnyName.FEEBS: TARGET_LENGTH * FEEBS_SCALE,
    }


def get_mesh_size(target_length, edge_fraction, elem_type):
    mesh_size = target_length * edge_fraction
    if elem_type in [ElemType.TRI3, ElemType.TRI6]:
        return mesh_size * TRI_FACT
    if elem_type in [ElemType.QUAD4, ElemType.QUAD8, ElemType.QUAD9]:
        return mesh_size * QUAD_FACT
    return mesh_size


def get_element_order(elem_type):
    if elem_type in [ElemType.TRI6, ElemType.QUAD8, ElemType.QUAD9]:
        return 2
    return 1


def mesh_bunny(
    bunny_name,
    elem_type,
    target_length=TARGET_LENGTH,
    edge_fraction=EDGE_FRACTION,
):
    gmsh.initialize()
    gmsh.model.add(f"{bunny_name.value}_{elem_type.value}")

    outline_path = load_outline_path(SVG_NAMES[bunny_name])
    simplified_points = build_simplified_points(outline_path, target_length)

    mesh_size = get_mesh_size(target_length, edge_fraction, elem_type)
    gmsh_point_tags = []
    for point in simplified_points:
        gmsh_point_tags.append(
            gmsh.model.geo.addPoint(point[0], point[1], 0, mesh_size)
        )

    gmsh_point_tags.append(gmsh_point_tags[0])
    spline_tag = gmsh.model.geo.addSpline(gmsh_point_tags)
    loop_tag = gmsh.model.geo.addCurveLoop([spline_tag])
    surface_tag = gmsh.model.geo.addPlaneSurface([loop_tag])

    gmsh.model.geo.synchronize()

    if "quad" in elem_type.value:
        gmsh.model.mesh.setRecombine(2, surface_tag)
        gmsh.option.setNumber("Mesh.Algorithm", QUAD_ALGORITHM)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
    else:
        gmsh.option.setNumber("Mesh.Algorithm", 1)

    if elem_type == ElemType.QUAD8:
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
    elif elem_type == ElemType.QUAD9:
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)

    gmsh.model.mesh.generate(2)

    if "quad" in elem_type.value:
        gmsh.model.mesh.optimize("Relocate2D")

    order = get_element_order(elem_type)
    gmsh.model.mesh.setOrder(order)
    if "quad" in elem_type.value and order == 2:
        gmsh.option.setNumber(
            "Mesh.HighOrderOptimize",
            QUAD_HIGH_ORDER_OPTIMIZE,
        )
        gmsh.model.mesh.optimize("HighOrder")

    out_dir = os.path.join(SCRIPT_DIR, f"{bunny_name.value}_{elem_type.value}")
    os.makedirs(out_dir, exist_ok=True)
    msh_path = os.path.join(
        out_dir,
        f"{bunny_name.value}_{elem_type.value}.msh",
    )
    gmsh.write(msh_path)

    mesh_data = export_mesh_data(out_dir, bunny_name, elem_type)
    gmsh.finalize()
    return mesh_data


def export_mesh_data(out_dir, bunny_name, elem_type):
    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(2)

    used_node_tags = set()
    all_connectivity = []
    for type_index, gmsh_elem_type in enumerate(elem_types):
        _, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(
            gmsh_elem_type
        )
        element_nodes = elem_node_tags[type_index].reshape((-1, num_nodes))
        for element_node_tags_row in element_nodes:
            all_connectivity.append(element_node_tags_row.tolist())
            for node_tag in element_node_tags_row:
                used_node_tags.add(node_tag)

    sorted_node_tags = sorted(used_node_tags)
    node_map = {
        node_tag: node_index
        for node_index, node_tag in enumerate(sorted_node_tags)
    }

    coords_by_tag = {}
    for node_tag in sorted_node_tags:
        coords, _, _, _ = gmsh.model.mesh.getNode(node_tag)
        coords_by_tag[node_tag] = coords

    final_coords = np.array([coords_by_tag[node_tag] for node_tag in sorted_node_tags])
    source_connectivity = np.asarray([
        [node_map[node_tag] for node_tag in element_node_tags_row]
        for element_node_tags_row in all_connectivity
    ], dtype=np.int64)
    convention = meshconv.ConnectConvention(
        RILEY_ELEMENT_TYPES[elem_type],
        meshconv.EConnectAxis.ROW,
        0,
        node_order=meshconv.ENodeOrder.VTK,
        material_normal_hint=(0.0, 0.0, 1.0),
    )
    mesh = meshconv.convert_mesh(
        final_coords,
        source_connectivity,
        convention,
    )
    meshconv.verify_mesh(mesh)
    final_connectivity = mesh.connect

    write_coords(out_dir, final_coords)
    write_connectivity(out_dir, final_connectivity)
    write_uvs(out_dir, final_coords)

    plot_mesh(
        out_dir,
        bunny_name,
        elem_type,
        final_coords,
        final_connectivity,
    )
    print(f"Exported mesh data to {out_dir}")

    return {
        "coords": final_coords,
        "connectivity": final_connectivity,
    }


def write_coords(out_dir, coords):
    with open(os.path.join(out_dir, "coords.csv"), "w", newline="") as file_obj:
        writer = csv.writer(file_obj)
        for row in coords:
            writer.writerow([f"{value:.18e}" for value in row])


def write_connectivity(out_dir, connectivity):
    with open(
        os.path.join(out_dir, "connectivity.csv"),
        "w",
        newline="",
    ) as file_obj:
        writer = csv.writer(file_obj)
        for row in connectivity:
            writer.writerow(row)


def write_uvs(out_dir, coords):
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    min_x = np.min(x_coords)
    max_x = np.max(x_coords)
    range_x = max_x - min_x
    min_y = np.min(y_coords)

    uu = 0.25 + (x_coords - min_x) / range_x * 0.5
    vv = (y_coords - min_y) / range_x * 0.5

    with open(os.path.join(out_dir, "uvs.csv"), "w", newline="") as file_obj:
        writer = csv.writer(file_obj)
        for uu_value, vv_value in zip(uu, vv):
            writer.writerow([f"{uu_value:.18e}", f"{vv_value:.18e}"])


def get_num_corners(elem_type):
    if "tri" in elem_type.value:
        return 3
    return 4


def build_polygons(coords, connectivity, elem_type):
    num_corners = get_num_corners(elem_type)
    polygons = []
    corner_indices = set()

    for element_nodes in connectivity:
        polygon_nodes = [
            coords[node_index][:2]
            for node_index in element_nodes[:num_corners]
        ]
        polygons.append(polygon_nodes)
        for node_index in element_nodes[:num_corners]:
            corner_indices.add(node_index)

    return polygons, corner_indices


def draw_mesh(ax, coords, connectivity, elem_type, title=None):
    polygons, corner_indices = build_polygons(coords, connectivity, elem_type)
    collection = PolyCollection(
        polygons,
        facecolors="none",
        edgecolors="black",
        linewidths=0.8,
    )
    ax.add_collection(collection)

    all_indices = np.arange(len(coords))
    is_corner = np.array([node_index in corner_indices for node_index in all_indices])

    num_corner = np.sum(is_corner)
    num_midside = np.sum(~is_corner)
    if title is not None:
        print(
            f"  Vis: {title} - Corners: {num_corner}, "
            f"Midside: {num_midside}"
        )

    ax.scatter(
        coords[is_corner, 0],
        coords[is_corner, 1],
        s=16,
        c="limegreen",
        marker="o",
        edgecolors="black",
        linewidths=0.6,
        zorder=4,
    )

    if not np.all(is_corner):
        ax.scatter(
            coords[~is_corner, 0],
            coords[~is_corner, 1],
            s=8,
            c="white",
            marker="o",
            edgecolors="black",
            linewidths=0.3,
            zorder=3,
        )

    ax.set_aspect("equal")
    ax.axis("off")
    if title is not None:
        ax.set_title(title)


def plot_mesh(out_dir, bunny_name, elem_type, coords, connectivity):
    os.makedirs(VIS_DIR, exist_ok=True)

    figure, ax = plt.subplots(figsize=(10, 6))
    title = f"{bunny_name.value}_{elem_type.value}"
    draw_mesh(ax, coords, connectivity, elem_type, title=title)

    local_png_path = os.path.join(out_dir, f"{bunny_name.value}_{elem_type.value}.png")
    figure.savefig(local_png_path, dpi=300, bbox_inches="tight")

    central_png_path = os.path.join(
        VIS_DIR,
        f"{bunny_name.value}_{elem_type.value}.png",
    )
    figure.savefig(central_png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def offset_coords(coords, x_offset):
    offset_coords_array = np.array(coords, copy=True)
    offset_coords_array[:, 0] += x_offset
    return offset_coords_array


def plot_comparison(elem_type, feebs_mesh_data, riley_mesh_data):
    os.makedirs(VIS_DIR, exist_ok=True)

    feebs_coords = feebs_mesh_data["coords"]
    riley_coords = riley_mesh_data["coords"]

    feebs_width = np.max(feebs_coords[:, 0]) - np.min(feebs_coords[:, 0])
    riley_width = np.max(riley_coords[:, 0]) - np.min(riley_coords[:, 0])
    horizontal_gap = 0.2 * TARGET_LENGTH
    riley_offset = feebs_width + horizontal_gap

    figure, ax = plt.subplots(figsize=(12, 6))
    draw_mesh(
        ax,
        offset_coords(feebs_coords, 0.0),
        feebs_mesh_data["connectivity"],
        elem_type,
    )
    draw_mesh(
        ax,
        offset_coords(riley_coords, riley_offset),
        riley_mesh_data["connectivity"],
        elem_type,
    )

    min_y = min(np.min(feebs_coords[:, 1]), np.min(riley_coords[:, 1]))
    max_y = max(np.max(feebs_coords[:, 1]), np.max(riley_coords[:, 1]))
    text_y = max_y + 0.05 * max(TARGET_LENGTH, max_y - min_y)

    ax.text(
        feebs_width * 0.5,
        text_y,
        f"Feebs ({FEEBS_SCALE:.3f}x)",
        ha="center",
        va="bottom",
    )
    ax.text(
        riley_offset + riley_width * 0.5,
        text_y,
        "Riley (1.000x)",
        ha="center",
        va="bottom",
    )

    compare_png_path = os.path.join(
        VIS_DIR,
        f"bunnies_{elem_type.value}_compare.png",
    )
    figure.savefig(compare_png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def mesh_all_bunnies(target_length, edge_fraction):
    target_lengths = get_target_length_by_bunny()

    for elem_type in ElemType:
        print(f"Processing {elem_type.value}...")
        mesh_data_by_bunny = {}

        for bunny_name in [BunnyName.FEEBS, BunnyName.RILEY]:
            mesh_data_by_bunny[bunny_name] = mesh_bunny(
                bunny_name,
                elem_type,
                target_length=target_lengths[bunny_name],
                edge_fraction=edge_fraction,
            )

        plot_comparison(
            elem_type,
            mesh_data_by_bunny[BunnyName.FEEBS],
            mesh_data_by_bunny[BunnyName.RILEY],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=float, default=TARGET_LENGTH)
    parser.add_argument("--edge", type=float, default=EDGE_FRACTION)
    args = parser.parse_args()

    TARGET_LENGTH = args.length
    mesh_all_bunnies(args.length, args.edge)
