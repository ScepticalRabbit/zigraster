# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------

from enum import Flag, auto
from pathlib import Path

import numpy as np
from PIL import Image


class ETextureCoercion(Flag):
    """Flags specifying allowed automatic texture conversions during loading.

    Members
    -------
    NONE
        No conversions allowed; file format must match target precisely.
    RGB_TO_MONO
        Convert 3-channel RGB image to 1-channel greyscale.
    MONO_TO_RGB
        Replicate 1-channel greyscale image across 3 RGB channels.
    U8_TO_U16
        Scale 8-bit unsigned integer depth to 16-bit unsigned integer depth.
    U16_TO_U8
        Scale 16-bit unsigned integer depth down to 8-bit unsigned integer.
    """

    NONE = 0
    RGB_TO_MONO = auto()
    MONO_TO_RGB = auto()
    U8_TO_U16 = auto()
    U16_TO_U8 = auto()


def _load_texture(
    texture_path: str | Path,
    channels: int,
    dtype: np.dtype,
    coercion: ETextureCoercion,
) -> np.ndarray:
    """Load an image file and convert it into a CHW texture array.

    Parameters
    ----------
    texture_path : str or pathlib.Path
        Path to the texture file on disk.
    channels : int
        Target number of channels (1 for monochrome, 3 for RGB).
    dtype : numpy.dtype
        Target NumPy integer dtype (np.uint8 or np.uint16).
    coercion : ETextureCoercion
        Allowed channel and bit-depth coercion flags.

    Returns
    -------
    numpy.ndarray
        Array of shape `(C, H, W)` and dtype matching `dtype`, where
        `C` is channels (1 or 3), `H` is image height (rows), and `W` is
        image width (columns).

    Raises
    ------
    TypeError
        If `coercion` is not an `ETextureCoercion` member.
    FileNotFoundError
        If `texture_path` does not exist on disk.
    ValueError
        If the image contains invalid channels/modes, has an unsupported
        bit depth, or requires coercions not permitted by `coercion`.
    """

    # Pre-load verification
    if not isinstance(coercion, ETextureCoercion):
        raise TypeError("coercion must be an ETextureCoercion member.")

    path = Path(texture_path)
    if not path.is_file():
        raise FileNotFoundError(f"Texture file not found: {path}")

    # Load image data
    with Image.open(path) as image_in:
        if image_in.mode in ("P", "PA", "RGBA", "LA"):
            raise ValueError(
                "Texture must not contain palette or alpha channels."
            )
        texture = np.asarray(image_in)

    # Post-load verification
    if texture.ndim not in (2, 3):
        raise ValueError("Texture must be a monochrome or RGB image.")
    if texture.ndim == 3 and texture.shape[2] != 3:
        raise ValueError("Texture must not contain palette or alpha channels.")
    if texture.dtype not in (np.uint8, np.uint16):
        is_mono_u16 = texture.ndim == 2 and np.issubdtype(
            texture.dtype,
            np.signedinteger,
        )
        if not is_mono_u16:
            raise ValueError(
                "Texture must decode to uint8 or uint16 sample values."
            )
        in_range = (texture >= 0) & (
            texture <= np.iinfo(np.uint16).max
        )
        if not np.all(in_range):
            raise ValueError("Texture sample values lie outside uint16 range.")
        texture = texture.astype(np.uint16)

    # Channel coercion
    source_channels = 1 if texture.ndim == 2 else texture.shape[2]
    if source_channels != channels:
        if source_channels == 3 and channels == 1:
            if ETextureCoercion.RGB_TO_MONO not in coercion:
                raise ValueError(
                    "RGB texture requires ETextureCoercion.RGB_TO_MONO."
                )

            weights = np.array((0.299, 0.587, 0.114), dtype=np.float64)
            mono = np.rint(texture @ weights)
            texture = mono.astype(texture.dtype)

        elif source_channels == 1 and channels == 3:
            if ETextureCoercion.MONO_TO_RGB not in coercion:
                raise ValueError(
                    "Monochrome texture requires ETextureCoercion.MONO_TO_RGB."
                )

            texture = np.repeat(texture[:, :, None], 3, axis=2)

        else:
            raise ValueError("Unsupported texture channel conversion.")

    # Dtype coercion
    if texture.dtype != dtype:
        if texture.dtype == np.uint8 and dtype == np.dtype(np.uint16):
            if ETextureCoercion.U8_TO_U16 not in coercion:
                raise ValueError(
                    "uint8 texture requires ETextureCoercion.U8_TO_U16."
                )

            texture = texture.astype(np.uint16) * np.uint16(257)

        elif texture.dtype == np.uint16 and dtype == np.dtype(np.uint8):
            if ETextureCoercion.U16_TO_U8 not in coercion:
                raise ValueError(
                    "uint16 texture requires ETextureCoercion.U16_TO_U8."
                )

            rounded = (texture.astype(np.uint32) + 128) // 257
            texture = rounded.astype(np.uint8)

        else:
            raise ValueError("Unsupported texture dtype conversion.")

    if channels == 1:
        texture = texture[None, :, :]
    else:
        texture = np.moveaxis(texture, 2, 0)

    return np.ascontiguousarray(texture, dtype=dtype)


def load_texture_mono_u8(
    texture_path: str | Path,
    coercion: ETextureCoercion = ETextureCoercion.NONE,
) -> np.ndarray:
    """Load an 8-bit monochrome texture from file.

    Parameters
    ----------
    texture_path : str or pathlib.Path
        Path to the texture file on disk.
    coercion : ETextureCoercion, default=ETextureCoercion.NONE
        Allowed channel and bit-depth coercion flags.

    Returns
    -------
    numpy.ndarray
        Array of shape `(1, H, W)` and dtype `np.uint8`, where dimension 0
        is channels (1), dimension 1 is height `H` (rows), and dimension 2
        is width `W` (columns).

    Raises
    ------
    TypeError
        If `coercion` is not an `ETextureCoercion` member.
    FileNotFoundError
        If `texture_path` does not exist on disk.
    ValueError
        If the image cannot be loaded as 8-bit monochrome without
        prohibited coercions.
    """
    return _load_texture(texture_path, 1, np.dtype(np.uint8), coercion)


def load_texture_rgb_u8(
    texture_path: str | Path,
    coercion: ETextureCoercion = ETextureCoercion.NONE,
) -> np.ndarray:
    """Load an 8-bit RGB texture from file.

    Parameters
    ----------
    texture_path : str or pathlib.Path
        Path to the texture file on disk.
    coercion : ETextureCoercion, default=ETextureCoercion.NONE
        Allowed channel and bit-depth coercion flags.

    Returns
    -------
    numpy.ndarray
        Array of shape `(3, H, W)` and dtype `np.uint8`, where dimension 0
        is color channels `(R, G, B)`, dimension 1 is height `H` (rows),
        and dimension 2 is width `W` (columns).

    Raises
    ------
    TypeError
        If `coercion` is not an `ETextureCoercion` member.
    FileNotFoundError
        If `texture_path` does not exist on disk.
    ValueError
        If the image cannot be loaded as 8-bit RGB without prohibited
        coercions.
    """
    return _load_texture(texture_path, 3, np.dtype(np.uint8), coercion)


def load_texture_mono_u16(
    texture_path: str | Path,
    coercion: ETextureCoercion = ETextureCoercion.NONE,
) -> np.ndarray:
    """Load a 16-bit monochrome texture from file.

    Parameters
    ----------
    texture_path : str or pathlib.Path
        Path to the texture file on disk.
    coercion : ETextureCoercion, default=ETextureCoercion.NONE
        Allowed channel and bit-depth coercion flags.

    Returns
    -------
    numpy.ndarray
        Array of shape `(1, H, W)` and dtype `np.uint16`, where dimension 0
        is channels (1), dimension 1 is height `H` (rows), and dimension 2
        is width `W` (columns).

    Raises
    ------
    TypeError
        If `coercion` is not an `ETextureCoercion` member.
    FileNotFoundError
        If `texture_path` does not exist on disk.
    ValueError
        If the image cannot be loaded as 16-bit monochrome without
        prohibited coercions.
    """
    return _load_texture(texture_path, 1, np.dtype(np.uint16), coercion)


__all__ = [
    "ETextureCoercion",
    "load_texture_mono_u8",
    "load_texture_mono_u16",
    "load_texture_rgb_u8",
]
