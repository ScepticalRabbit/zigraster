
from __future__ import annotations

from pathlib import Path
import numpy as np
import numpy.typing as npt

from riley.cython.riley import (
    FunctionShader,
    Mesh,
    MeshType,
    NodalShader,
    RileyShader,
    TextureShader,
)
from riley.python.meshconstants import (
    ELEM_FAMILY_MAP,
    ELEM_NODE_COUNT_MAP,
    ELEM_ORDER_MAP,
    RILEY_MESH_ELEM_TYPE_MAP,
    RILEY_TRI_STENCIL_MAP,
    RILEY_VOL_SURF_TYPE_MAP,
)
from riley.python.meshconv import (
    ConnectConvention,
    EElemType,
    MeshError,
    MeshGeometry,
    _extract_surface_with_node_idxs,
    _reduce_elem_order_with_node_idxs,
    _triangulate_with_node_idxs,
    convert_mesh,
    verify_mesh,
)

def load_csv(
    path: str | Path,
    dtype: npt.DTypeLike = np.float64,
    skip_rows: int = 0,
) -> np.ndarray:
    if skip_rows < 0:
        raise ValueError("skip_rows must be non-negative.")

    dtype_out = np.dtype(dtype)

    if np.issubdtype(dtype_out, np.integer):
        array = np.loadtxt(
            path, delimiter=",", dtype=np.float64, ndmin=2,
            skiprows=skip_rows,
        )

        if not np.all(np.isfinite(array)) or not np.all(
            array == array.astype(dtype_out)
        ):
            raise ValueError(
                f"CSV table '{path}' contains non-integer values."
            )

        return np.ascontiguousarray(array.astype(dtype_out))

    array = np.loadtxt(
        path, delimiter=",", dtype=dtype_out, ndmin=2, skiprows=skip_rows,
    )

    if np.issubdtype(dtype_out, np.floating) and not np.all(
        np.isfinite(array)
    ):
        raise ValueError(f"CSV table '{path}' contains non-finite values.")

    return np.ascontiguousarray(array)


def _prepare_component(
    values: np.ndarray,
    nodes_num: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(values)

    if array.ndim == 1:
        array = array[:, None]

    if array.ndim != 2 or array.shape[0] != nodes_num:
        raise ValueError(f"{name} must have shape (nodes, time).")

    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must have a floating-point dtype.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    return np.ascontiguousarray(array, dtype=np.float64)


def _prepare_disp(
    disp: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    nodes_num: int,
    source_node_idxs: np.ndarray,
) -> np.ndarray | None:

    if disp is None:
        return None

    if not isinstance(disp, tuple) or len(disp) != 3:
        raise TypeError("disp must be a tuple of (disp_x, disp_y, disp_z).")

    components_out = []
    for value, axis in zip(disp, "xyz", strict=True):
        component = _prepare_component(value, nodes_num, f"disp_{axis}")
        components_out.append(component)

    components = tuple(components_out)

    for item in components[1:]:
        if item.shape != components[0].shape:
            raise ValueError("Displacement components must have equal shapes.")

    stacked = np.stack(components, axis=2)

    return np.ascontiguousarray(
        stacked[source_node_idxs].transpose(1, 0, 2)
    )


def _prepare_uvs(
    values: np.ndarray,
    nodes_num: int,
    source_node_idxs: np.ndarray,
) -> np.ndarray:
    array = np.asarray(values)

    if array.shape != (nodes_num, 2):
        raise ValueError(f"uvs must have shape ({nodes_num}, 2).")

    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("uvs must have a floating-point dtype.")

    if not np.all(np.isfinite(array)):
        raise ValueError("uvs must contain only finite values.")

    return np.ascontiguousarray(array[source_node_idxs], dtype=np.float64)


def _prepare_shader(
    shader: RileyShader,
    nodes_num: int,
    source_node_idxs: np.ndarray,
) -> RileyShader:

    match shader:
        case TextureShader():
            texture = np.asarray(shader.texture)
            valid_dtype = texture.dtype in (
                np.dtype(np.uint8), np.dtype(np.uint16),
                np.dtype(np.float32), np.dtype(np.float64),
            )
            valid_shape = texture.ndim == 3 and texture.shape[0] in (1, 3)
            if not valid_shape or not valid_dtype:
                raise ValueError(
                    "texture must be a supported dtype with 1 or 3 channels."
                )
            if np.issubdtype(texture.dtype, np.floating):
                if not np.all(np.isfinite(texture)):
                    raise ValueError(
                        "floating texture must contain finite values."
                    )
                texture = np.ascontiguousarray(texture, dtype=np.float64)
            return TextureShader(
                _prepare_uvs(shader.uvs, nodes_num, source_node_idxs),
                np.ascontiguousarray(texture), shader.sample,
                shader.sample_mode, shader.bits, shader.scaling_type,
                shader.scaling_min, shader.scaling_max, shader.normal_type,
            )

        case NodalShader():
            values = np.asarray(shader.field)

            if values.ndim == 2:
                values = values[:, :, None]

            if values.ndim != 3 or values.shape[0] != nodes_num:
                raise ValueError(
                    "nodal field must have shape (nodes, time[, fields])."
                )

            if values.shape[2] not in (1, 3):
                raise ValueError(
                    "nodal field must contain one or three fields."
                )

            if not np.issubdtype(values.dtype, np.floating):
                raise TypeError("nodal field must have a floating-point dtype.")

            if not np.all(np.isfinite(values)):
                raise ValueError("nodal field must contain only finite values.")

            field_out = np.ascontiguousarray(
                values[source_node_idxs].transpose(1, 0, 2), dtype=np.float64,
            )

            return NodalShader(
                field=field_out,
                bits=shader.bits,
                scaling_type=shader.scaling_type,
                scaling_min=shader.scaling_min,
                scaling_max=shader.scaling_max,
                scale_over=shader.scale_over,
                normal_type=shader.normal_type,
            )

        case FunctionShader():
            if shader.channels not in (1, 3):
                raise ValueError(
                    "FunctionShader.channels must be one or three."
                )
            uvs_out = None
            if shader.uvs is not None:
                uvs_out = _prepare_uvs(
                    shader.uvs, nodes_num, source_node_idxs
                )
            return FunctionShader(
                builtin=shader.builtin,
                coord_mode=shader.coord_mode,
                params=shader.params,
                uvs=uvs_out,
                channels=shader.channels,
                bits=shader.bits,
                scaling_type=shader.scaling_type,
                scaling_min=shader.scaling_min,
                scaling_max=shader.scaling_max,
                normal_type=shader.normal_type,
            )

        case _:
            raise TypeError("shader must be a Riley shader object.")


def create_mesh(
    convention: ConnectConvention,
    mesh_type: MeshType,
    coords: np.ndarray,
    connect: np.ndarray,
    shader: RileyShader,
    disp: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> Mesh:
    if not isinstance(convention, ConnectConvention):
        raise TypeError("convention must be a ConnectConvention.")

    if not isinstance(mesh_type, MeshType):
        raise TypeError("mesh_type must be a MeshType member.")

    if not isinstance(shader, RileyShader):
        raise TypeError("shader must be a Riley supported shader.")

    source_elem = convention.elem_type
    target_elem = RILEY_MESH_ELEM_TYPE_MAP[mesh_type]
    surf_elem = RILEY_VOL_SURF_TYPE_MAP.get(source_elem, source_elem)

    if target_elem is not EElemType.TRI3:
        if ELEM_FAMILY_MAP[surf_elem] != ELEM_FAMILY_MAP[target_elem]:
            raise MeshError(
                f"Cannot change {source_elem.value} into {target_elem.value}."
            )
        if ELEM_ORDER_MAP[target_elem] > ELEM_ORDER_MAP[surf_elem]:
            raise MeshError(
                f"Cannot convert {source_elem.value} to {target_elem.value}."
            )

    coords_array = np.asarray(coords)
    nodes_num = coords_array.shape[0] if coords_array.ndim == 2 else 0

    # 1) Convert to standard Riley convention
    mesh = convert_mesh(coords, connect, convention)
    source_node_idxs = np.arange(mesh.coords.shape[0], dtype=np.uintp)

    # 2) Extract boundary surface if volume mesh
    if ELEM_FAMILY_MAP[source_elem] in ("tet", "hex"):
        mesh, source_node_idxs = _extract_surface_with_node_idxs(mesh)

    # 3) Tessellate to tri3 or reduce order if needed
    if target_elem is EElemType.TRI3:
        if mesh.elem_type is not EElemType.TRI3:
            mesh, source_node_idxs = _triangulate_with_node_idxs(
                mesh, source_node_idxs, mesh.elem_type
            )
    elif mesh.elem_type is not target_elem:
        mesh, source_node_idxs = _reduce_elem_order_with_node_idxs(
            mesh, source_node_idxs, target_elem
        )

    return Mesh(
        mesh_type,
        mesh.coords,
        np.ascontiguousarray(mesh.connect, dtype=np.uintp),
        _prepare_disp(disp, nodes_num, source_node_idxs),
        _prepare_shader(shader, nodes_num, source_node_idxs),
    )


__all__ = ["create_mesh", "load_csv"]
