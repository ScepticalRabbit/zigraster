"""Load numeric arrays and construct renderer-ready Riley meshes."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
import numpy as np
import numpy.typing as npt

from riley.cython.riley import (
    FunctionShader,
    Mesh,
    MeshType,
    NodalShader,
    TextureShader,
)
from riley.python.meshconstants import (
    ELEMENT_FAMILIES,
    ELEMENT_NODE_COUNTS,
    ELEMENT_ORDERS,
    RILEY_MESH_ELEMENT_TYPES,
)
from riley.python.meshconv import (
    ConnectConvention, EElementType, MeshConvErr, MeshGeometry,
    _extract_surface_with_node_idxs, convert_mesh, verify_mesh,
)

def load_csv(
    path: str | Path,
    dtype: npt.DTypeLike = np.float64,
    skip_rows: int = 0,
) -> np.ndarray:
    """Load a numeric CSV as a C-contiguous two-dimensional array.

    Parameters
    ----------
    path : str | Path
        CSV file to read.
    dtype : numpy.typing.DTypeLike, optional
        Required output dtype. Integer dtypes reject fractional and
        out-of-range values. The default is ``np.float64``.
    skip_rows : int, optional
        Number of leading rows to skip.

    Returns
    -------
    np.ndarray
        Parsed array. No mesh operation is performed.
    """
    if skip_rows < 0:
        raise ValueError("skip_rows must be non-negative.")
    dtype_out = np.dtype(dtype)
    if np.issubdtype(dtype_out, np.integer):
        values = np.loadtxt(path, delimiter=",", dtype=str, ndmin=2,
                            skiprows=skip_rows)
        try:
            limits = np.iinfo(dtype_out)
            integer_values = []
            for value in values.flat:
                decimal = Decimal(str(value))
                integer = int(decimal)
                if (
                    decimal != integer
                    or integer < limits.min
                    or integer > limits.max
                ):
                    raise ValueError
                integer_values.append(integer)
            array = np.asarray(integer_values, dtype=dtype_out)
            array = array.reshape(values.shape)
        except (InvalidOperation, OverflowError, ValueError) as error:
            raise ValueError(
                f"CSV table '{path}' contains a non-integer or an integer "
                f"outside {dtype_out}."
            ) from error
    else:
        array = np.loadtxt(path, delimiter=",", dtype=dtype_out, ndmin=2,
                           skiprows=skip_rows)
        is_inexact = np.issubdtype(dtype_out, np.inexact)
        contains_non_finite = is_inexact and not np.all(np.isfinite(array))
        if contains_non_finite:
            raise ValueError(f"CSV table '{path}' contains non-finite values.")
    return np.ascontiguousarray(array)


def _compact(
    mesh: MeshGeometry,
    source_idxs: np.ndarray,
    target: EElementType,
) -> tuple[MeshGeometry, np.ndarray]:
    connect = mesh.connect[:, :ELEMENT_NODE_COUNTS[target]]
    retained = np.unique(connect)
    remap = np.full(mesh.coords.shape[0], -1, dtype=np.int64)
    remap[retained] = np.arange(retained.size, dtype=np.int64)
    result = MeshGeometry(
        target,
        np.ascontiguousarray(mesh.coords[retained]),
        np.ascontiguousarray(remap[connect], dtype=np.uintp),
    )
    verify_mesh(result)
    return result, np.ascontiguousarray(source_idxs[retained])


def _prepare_geometry(
    convention: ConnectConvention,
    mesh_type: MeshType,
    coords: np.ndarray,
    connect: np.ndarray,
) -> tuple[MeshGeometry, np.ndarray]:
    mesh = convert_mesh(coords, connect, convention)
    source_idxs = np.arange(mesh.coords.shape[0], dtype=np.uintp)
    target = RILEY_MESH_ELEMENT_TYPES[mesh_type]
    source = mesh.elem_type
    if ELEMENT_FAMILIES[source] in ("tet", "hex"):
        required = "tri" if ELEMENT_FAMILIES[source] == "tet" else "quad"
        if ELEMENT_FAMILIES[target] != required:
            raise MeshConvErr(
                f"Cannot create {target.value} from {source.value}."
            )
        mesh, source_idxs = _extract_surface_with_node_idxs(mesh)
        source = mesh.elem_type
    if ELEMENT_FAMILIES[source] != ELEMENT_FAMILIES[target]:
        raise MeshConvErr(f"Cannot change {source.value} into {target.value}.")
    if ELEMENT_ORDERS[target] > ELEMENT_ORDERS[source]:
        raise MeshConvErr(f"Cannot elevate {source.value} to {target.value}.")
    if source is target:
        return mesh, source_idxs
    return _compact(mesh, source_idxs, target)


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
    source_idxs: np.ndarray,
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
    if any(item.shape != components[0].shape for item in components[1:]):
        raise ValueError("Displacement components must have equal shapes.")
    stacked = np.stack(components, axis=2)
    return np.ascontiguousarray(stacked[source_idxs].transpose(1, 0, 2))


def _prepare_uvs(
    values: np.ndarray,
    nodes_num: int,
    source_idxs: np.ndarray,
) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (nodes_num, 2):
        raise ValueError(f"uvs must have shape ({nodes_num}, 2).")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("uvs must have a floating-point dtype.")
    if not np.all(np.isfinite(array)):
        raise ValueError("uvs must contain only finite values.")
    return np.ascontiguousarray(array[source_idxs], dtype=np.float64)


def _prepare_shader(
    shader: TextureShader | NodalShader | FunctionShader,
    nodes_num: int,
    source_idxs: np.ndarray,
) -> TextureShader | NodalShader | FunctionShader:
    if isinstance(shader, TextureShader):
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
                raise ValueError("floating texture must contain finite values.")
            texture = np.ascontiguousarray(texture, dtype=np.float64)
        return TextureShader(
            _prepare_uvs(shader.uvs, nodes_num, source_idxs),
            np.ascontiguousarray(texture), shader.sample, shader.sample_mode,
            shader.bits, shader.scaling_type, shader.scaling_min,
            shader.scaling_max, shader.normal_type,
        )
    if isinstance(shader, NodalShader):
        values = np.asarray(shader.field)
        if values.ndim == 2:
            values = values[:, :, None]
        if values.ndim != 3 or values.shape[0] != nodes_num:
            raise ValueError(
                "nodal field must have shape (nodes, time[, fields])."
            )
        if values.shape[2] not in (1, 3):
            raise ValueError("nodal field must contain one or three fields.")
        if not np.issubdtype(values.dtype, np.floating):
            raise TypeError("nodal field must have a floating-point dtype.")
        if not np.all(np.isfinite(values)):
            raise ValueError("nodal field must contain only finite values.")
        field_out = np.ascontiguousarray(
            values[source_idxs].transpose(1, 0, 2), dtype=np.float64)
        return NodalShader(
            field=field_out,
            bits=shader.bits,
            scaling_type=shader.scaling_type,
            scaling_min=shader.scaling_min,
            scaling_max=shader.scaling_max,
            scale_over=shader.scale_over,
            normal_type=shader.normal_type,
        )
    if isinstance(shader, FunctionShader):
        if shader.channels not in (1, 3):
            raise ValueError("FunctionShader.channels must be one or three.")
        uvs_out = None
        if shader.uvs is not None:
            uvs_out = _prepare_uvs(shader.uvs, nodes_num, source_idxs)
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
    raise TypeError("shader must be a Riley shader object.")


def create_mesh(
    convention: ConnectConvention,
    mesh_type: MeshType,
    coords: np.ndarray,
    connect: np.ndarray,
    disp: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    shader: TextureShader | NodalShader | FunctionShader | None = None,
) -> Mesh:
    """Create a renderer-ready mesh from explicit user arrays.

    Parameters
    ----------
    convention : ConnectConvention
        Convention of the supplied connectivity table.
    mesh_type : MeshType
        Requested Riley surface renderer and order.
    coords : np.ndarray
        Node-major coordinates with shape ``(nodes, 3)``.
    connect : np.ndarray
        A single-topology connectivity table.
    disp : tuple[np.ndarray, np.ndarray, np.ndarray] | None, optional
        Node-major x, y and z components shaped ``(nodes, time)``.
    shader : TextureShader | NodalShader | FunctionShader, optional
        Shader whose nodal arrays use the same source nodes. Required.

    Returns
    -------
    Mesh
        A Riley-standard surface mesh with ABI-ready arrays.
    """
    if not isinstance(convention, ConnectConvention):
        raise TypeError("convention must be a ConnectConvention.")
    if not isinstance(mesh_type, MeshType):
        raise TypeError("mesh_type must be a MeshType member.")
    if shader is None:
        raise ValueError("shader is required.")
    coords_array = np.asarray(coords)
    nodes_num = coords_array.shape[0] if coords_array.ndim == 2 else 0
    geometry, source_idxs = _prepare_geometry(
        convention, mesh_type, coords, connect)
    return Mesh(
        mesh_type, geometry.coords,
        np.ascontiguousarray(geometry.connect, dtype=np.uintp),
        _prepare_disp(disp, nodes_num, source_idxs),
        _prepare_shader(shader, nodes_num, source_idxs),
    )


__all__ = ["create_mesh", "load_csv"]
