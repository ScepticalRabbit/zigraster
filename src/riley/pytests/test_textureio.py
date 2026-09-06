# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
import pytest

import riley
from riley.python.textureio import ETextureCoercion

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TEX_DIR = _REPO_ROOT / "texture"

_VALID_TEST_TEXTURES = (
    ("speck128_mono_u8.bmp", 1, np.uint8, riley.load_texture_mono_u8),
    ("speck128_mono_u8.tiff", 1, np.uint8, riley.load_texture_mono_u8),
    ("speck128_mono_u8.png", 1, np.uint8, riley.load_texture_mono_u8),
    ("speck128_mono_u16.tiff", 1, np.uint16, riley.load_texture_mono_u16),
    ("speck128_mono_u16.png", 1, np.uint16, riley.load_texture_mono_u16),
    ("speck128_rgb_u8.bmp", 3, np.uint8, riley.load_texture_rgb_u8),
    ("speck128_rgb_u8.tiff", 3, np.uint8, riley.load_texture_rgb_u8),
    ("speck128_rgb_u8.png", 3, np.uint8, riley.load_texture_rgb_u8),
    ("speck128_rgb_u16.tiff", 3, np.uint16, riley.load_texture_rgb_u16),
    ("speck128_rgb_u16.png", 3, np.uint16, riley.load_texture_rgb_u16),
)

_RGB_TEXTURES = (
    "speck128_rgb_u8.bmp",
    "speck128_rgb_u8.tiff",
    "speck128_rgb_u8.png",
    "speck128_rgb_u16.tiff",
    "speck128_rgb_u16.png",
)

_MONO_TEXTURES = (
    "speck128_mono_u8.bmp",
    "speck128_mono_u8.tiff",
    "speck128_mono_u8.png",
    "speck128_mono_u16.tiff",
    "speck128_mono_u16.png",
)


@pytest.mark.parametrize(
    ("filename", "channels", "expected_dtype", "loader_fn"),
    _VALID_TEST_TEXTURES,
)
def test_strict_loading_valid_textures(
    filename: str,
    channels: int,
    expected_dtype: np.dtype,
    loader_fn: Callable[[Path], np.ndarray],
) -> None:
    path = _TEX_DIR / filename
    texture = loader_fn(path)
    assert texture.shape == (channels, 128, 128)
    assert texture.dtype == expected_dtype
    assert texture.flags.c_contiguous


def test_cross_format_data_consistency_mono_u8() -> None:
    bmp_tex = riley.load_texture_mono_u8(_TEX_DIR / "speck128_mono_u8.bmp")
    tif_tex = riley.load_texture_mono_u8(_TEX_DIR / "speck128_mono_u8.tiff")
    png_tex = riley.load_texture_mono_u8(_TEX_DIR / "speck128_mono_u8.png")

    np.testing.assert_array_equal(bmp_tex, tif_tex)
    np.testing.assert_array_equal(bmp_tex, png_tex)


def test_cross_format_data_consistency_rgb_u8() -> None:
    bmp_tex = riley.load_texture_rgb_u8(_TEX_DIR / "speck128_rgb_u8.bmp")
    tif_tex = riley.load_texture_rgb_u8(_TEX_DIR / "speck128_rgb_u8.tiff")
    png_tex = riley.load_texture_rgb_u8(_TEX_DIR / "speck128_rgb_u8.png")

    np.testing.assert_array_equal(bmp_tex, tif_tex)
    np.testing.assert_array_equal(bmp_tex, png_tex)


def test_cross_format_data_consistency_mono_u16() -> None:
    tif_tex = riley.load_texture_mono_u16(_TEX_DIR / "speck128_mono_u16.tiff")
    png_tex = riley.load_texture_mono_u16(_TEX_DIR / "speck128_mono_u16.png")
    mono_u8 = riley.load_texture_mono_u8(_TEX_DIR / "speck128_mono_u8.png")

    np.testing.assert_array_equal(tif_tex, png_tex)
    expected_u16 = mono_u8.astype(np.uint16) * np.uint16(257)
    np.testing.assert_array_equal(tif_tex, expected_u16)


def test_cross_format_data_consistency_rgb_u16() -> None:
    tif_tex = riley.load_texture_rgb_u16(_TEX_DIR / "speck128_rgb_u16.tiff")
    png_tex = riley.load_texture_rgb_u16(_TEX_DIR / "speck128_rgb_u16.png")
    rgb_u8 = riley.load_texture_rgb_u8(_TEX_DIR / "speck128_rgb_u8.png")

    np.testing.assert_array_equal(tif_tex, png_tex)
    expected_u16 = rgb_u8.astype(np.uint16) * np.uint16(257)
    np.testing.assert_array_equal(tif_tex, expected_u16)


@pytest.mark.parametrize("filename", _RGB_TEXTURES)
def test_explicit_rgb_to_mono_coercion(filename: str) -> None:
    path = _TEX_DIR / filename
    is_u16 = "u16" in filename
    coercion = ETextureCoercion.RGB_TO_MONO

    if is_u16:
        tex_mono = riley.load_texture_mono_u16(path, coercion=coercion)
        tex_rgb = riley.load_texture_rgb_u16(path)
        assert tex_mono.dtype == np.uint16
    else:
        tex_mono = riley.load_texture_mono_u8(path, coercion=coercion)
        tex_rgb = riley.load_texture_rgb_u8(path)
        assert tex_mono.dtype == np.uint8

    assert tex_mono.shape == (1, 128, 128)
    weights = np.array((0.299, 0.587, 0.114), dtype=np.float64)
    rgb_hwc = np.ascontiguousarray(np.moveaxis(tex_rgb, 0, -1))
    expected_mono = np.rint(
        np.einsum("ijk,k->ij", rgb_hwc, weights)
    ).astype(tex_mono.dtype)[None, :, :]
    np.testing.assert_allclose(tex_mono, expected_mono, atol=1)


@pytest.mark.parametrize("filename", _MONO_TEXTURES)
def test_explicit_mono_to_rgb_coercion(filename: str) -> None:
    path = _TEX_DIR / filename
    is_u16 = "u16" in filename
    coercion = ETextureCoercion.MONO_TO_RGB

    if is_u16:
        tex_rgb = riley.load_texture_rgb_u16(path, coercion=coercion)
        tex_mono = riley.load_texture_mono_u16(path)
        assert tex_rgb.dtype == np.uint16
    else:
        tex_rgb = riley.load_texture_rgb_u8(path, coercion=coercion)
        tex_mono = riley.load_texture_mono_u8(path)
        assert tex_rgb.dtype == np.uint8

    assert tex_rgb.shape == (3, 128, 128)
    assert np.array_equal(tex_rgb[0], tex_mono[0])
    assert np.array_equal(tex_rgb[1], tex_mono[0])
    assert np.array_equal(tex_rgb[2], tex_mono[0])


@pytest.mark.parametrize(
    "filename",
    (
        "speck128_mono_u8.bmp",
        "speck128_mono_u8.tiff",
        "speck128_mono_u8.png",
    ),
)
def test_explicit_mono_u8_to_u16_coercion(filename: str) -> None:
    path = _TEX_DIR / filename
    coercion = ETextureCoercion.U8_TO_U16
    tex_u16 = riley.load_texture_mono_u16(path, coercion=coercion)
    tex_u8 = riley.load_texture_mono_u8(path)

    assert tex_u16.shape == (1, 128, 128)
    assert tex_u16.dtype == np.uint16
    expected = tex_u8.astype(np.uint16) * np.uint16(257)
    np.testing.assert_array_equal(tex_u16, expected)


@pytest.mark.parametrize(
    "filename",
    (
        "speck128_mono_u16.tiff",
        "speck128_mono_u16.png",
    ),
)
def test_explicit_mono_u16_to_u8_coercion(filename: str) -> None:
    path = _TEX_DIR / filename
    coercion = ETextureCoercion.U16_TO_U8
    tex_u8 = riley.load_texture_mono_u8(path, coercion=coercion)
    tex_u16 = riley.load_texture_mono_u16(path)

    assert tex_u8.shape == (1, 128, 128)
    assert tex_u8.dtype == np.uint8
    expected = ((tex_u16.astype(np.uint32) + 128) // 257).astype(np.uint8)
    np.testing.assert_array_equal(tex_u8, expected)


@pytest.mark.parametrize(
    "filename",
    (
        "speck128_rgb_u8.bmp",
        "speck128_rgb_u8.tiff",
        "speck128_rgb_u8.png",
    ),
)
def test_explicit_rgb_u8_to_u16_coercion(filename: str) -> None:
    path = _TEX_DIR / filename
    coercion = ETextureCoercion.U8_TO_U16
    tex_u16 = riley.load_texture_rgb_u16(path, coercion=coercion)
    tex_u8 = riley.load_texture_rgb_u8(path)

    assert tex_u16.shape == (3, 128, 128)
    assert tex_u16.dtype == np.uint16
    expected = tex_u8.astype(np.uint16) * np.uint16(257)
    np.testing.assert_array_equal(tex_u16, expected)


@pytest.mark.parametrize(
    "filename",
    (
        "speck128_rgb_u16.tiff",
        "speck128_rgb_u16.png",
    ),
)
def test_explicit_rgb_u16_to_u8_coercion(filename: str) -> None:
    path = _TEX_DIR / filename
    coercion = ETextureCoercion.U16_TO_U8
    tex_u8 = riley.load_texture_rgb_u8(path, coercion=coercion)
    tex_u16 = riley.load_texture_rgb_u16(path)

    assert tex_u8.shape == (3, 128, 128)
    assert tex_u8.dtype == np.uint8
    expected = ((tex_u16.astype(np.uint32) + 128) // 257).astype(np.uint8)
    np.testing.assert_array_equal(tex_u8, expected)


def test_compound_coercion_rgb_u8_to_mono_u16() -> None:
    path = _TEX_DIR / "speck128_rgb_u8.png"
    coercion = (
        ETextureCoercion.RGB_TO_MONO | ETextureCoercion.U8_TO_U16
    )
    tex = riley.load_texture_mono_u16(path, coercion=coercion)
    assert tex.shape == (1, 128, 128)
    assert tex.dtype == np.uint16


def test_compound_coercion_rgb_u16_to_mono_u8() -> None:
    path = _TEX_DIR / "speck128_rgb_u16.png"
    coercion = (
        ETextureCoercion.RGB_TO_MONO | ETextureCoercion.U16_TO_U8
    )
    tex = riley.load_texture_mono_u8(path, coercion=coercion)
    assert tex.shape == (1, 128, 128)
    assert tex.dtype == np.uint8


def test_compound_coercion_mono_u8_to_rgb_u16() -> None:
    path = _TEX_DIR / "speck128_mono_u8.png"
    coercion = (
        ETextureCoercion.MONO_TO_RGB | ETextureCoercion.U8_TO_U16
    )
    tex = riley.load_texture_rgb_u16(path, coercion=coercion)
    assert tex.shape == (3, 128, 128)
    assert tex.dtype == np.uint16


def test_compound_coercion_mono_u16_to_rgb_u8() -> None:
    path = _TEX_DIR / "speck128_mono_u16.png"
    coercion = (
        ETextureCoercion.MONO_TO_RGB | ETextureCoercion.U16_TO_U8
    )
    tex = riley.load_texture_rgb_u8(path, coercion=coercion)
    assert tex.shape == (3, 128, 128)
    assert tex.dtype == np.uint8


def test_path_string_and_path_object_loading() -> None:
    path_obj = _TEX_DIR / "speck128_mono_u8.png"
    path_str = str(path_obj)

    tex1 = riley.load_texture_mono_u8(path_obj)
    tex2 = riley.load_texture_mono_u8(path_str)
    np.testing.assert_array_equal(tex1, tex2)

