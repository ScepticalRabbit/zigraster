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

import cv2
import numpy as np
from PIL import Image


class ETextureCoercion(Flag):

    NONE = 0
    RGB_TO_MONO = auto()
    MONO_TO_RGB = auto()
    U8_TO_U16 = auto()
    U16_TO_U8 = auto()


def _load_texture_array(texture_path: str | Path) -> np.ndarray:
    path = Path(texture_path)
    if not path.is_file():
        raise FileNotFoundError(f"Texture file not found: {path}")

    texture: np.ndarray | None = None
    try:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is not None:
            if raw.ndim == 3 and raw.shape[2] == 3:
                texture = raw[:, :, ::-1]
            else:
                texture = raw
    except (cv2.error, OSError, ValueError):
        texture = None

    if texture is None:
        with Image.open(path) as image_in:
            texture = np.asarray(image_in)

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
    return np.ascontiguousarray(texture)


def _coerce_channels(
    texture: np.ndarray,
    channels: int,
    coercion: ETextureCoercion,
) -> np.ndarray:
    source_channels = 1 if texture.ndim == 2 else texture.shape[2]
    if source_channels == channels:
        return texture
    if source_channels == 3 and channels == 1:
        if ETextureCoercion.RGB_TO_MONO not in coercion:
            raise ValueError(
                "RGB texture requires ETextureCoercion.RGB_TO_MONO."
            )
        weights = np.array((0.299, 0.587, 0.114), dtype=np.float64)
        mono = np.rint(texture @ weights)
        return mono.astype(texture.dtype)
    if source_channels == 1 and channels == 3:
        if ETextureCoercion.MONO_TO_RGB not in coercion:
            raise ValueError(
                "Monochrome texture requires ETextureCoercion.MONO_TO_RGB."
            )
        return np.repeat(texture[:, :, None], 3, axis=2)
    raise ValueError("Unsupported texture channel conversion.")


def _coerce_dtype(
    texture: np.ndarray,
    dtype: np.dtype,
    coercion: ETextureCoercion,
) -> np.ndarray:
    if texture.dtype == dtype:
        return texture
    if texture.dtype == np.uint8 and dtype == np.dtype(np.uint16):
        if ETextureCoercion.U8_TO_U16 not in coercion:
            raise ValueError(
                "uint8 texture requires ETextureCoercion.U8_TO_U16."
            )
        return texture.astype(np.uint16) * np.uint16(257)
    if texture.dtype == np.uint16 and dtype == np.dtype(np.uint8):
        if ETextureCoercion.U16_TO_U8 not in coercion:
            raise ValueError(
                "uint16 texture requires ETextureCoercion.U16_TO_U8."
            )
        rounded = (texture.astype(np.uint32) + 128) // 257
        return rounded.astype(np.uint8)
    raise ValueError("Unsupported texture dtype conversion.")


def _load_texture(
    texture_path: str | Path,
    channels: int,
    dtype: np.dtype,
    coercion: ETextureCoercion,
) -> np.ndarray:
    if not isinstance(coercion, ETextureCoercion):
        raise TypeError("coercion must be an ETextureCoercion member.")
    texture = _load_texture_array(texture_path)
    texture = _coerce_channels(texture, channels, coercion)
    texture = _coerce_dtype(texture, dtype, coercion)
    if channels == 1:
        texture = texture[None, :, :]
    else:
        texture = np.moveaxis(texture, 2, 0)
    return np.ascontiguousarray(texture, dtype=dtype)


def load_texture_mono_u8(
    texture_path: str | Path,
    coercion: ETextureCoercion = ETextureCoercion.NONE,
) -> np.ndarray:
    return _load_texture(texture_path, 1, np.dtype(np.uint8), coercion)


def load_texture_rgb_u8(
    texture_path: str | Path,
    coercion: ETextureCoercion = ETextureCoercion.NONE,
) -> np.ndarray:
    return _load_texture(texture_path, 3, np.dtype(np.uint8), coercion)


def load_texture_mono_u16(
    texture_path: str | Path,
    coercion: ETextureCoercion = ETextureCoercion.NONE,
) -> np.ndarray:
    return _load_texture(texture_path, 1, np.dtype(np.uint16), coercion)


def load_texture_rgb_u16(
    texture_path: str | Path,
    coercion: ETextureCoercion = ETextureCoercion.NONE,
) -> np.ndarray:
    return _load_texture(texture_path, 3, np.dtype(np.uint16), coercion)


__all__ = [
    "ETextureCoercion",
    "load_texture_mono_u8",
    "load_texture_mono_u16",
    "load_texture_rgb_u8",
    "load_texture_rgb_u16",
]
