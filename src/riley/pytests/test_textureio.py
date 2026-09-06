"""Tests for strict texture loading and raster configuration."""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import riley


def test_texture_rgb_u8_is_channel_first_and_contiguous(tmp_path: Path) -> None:
    path = tmp_path / "rgb.png"
    source = np.array([[(255, 0, 0), (0, 255, 0)]], dtype=np.uint8)
    Image.fromarray(source).save(path)
    texture = riley.load_texture_rgb_u8(path)
    assert texture.shape == (3, 1, 2)
    assert texture.dtype == np.uint8
    assert texture.flags.c_contiguous
    np.testing.assert_array_equal(texture, np.moveaxis(source, 2, 0))


def test_texture_explicit_channel_and_depth_coercion(tmp_path: Path) -> None:
    path = tmp_path / "rgb.png"
    Image.fromarray(np.full((2, 2, 3), 255, dtype=np.uint8)).save(path)
    coercion = (riley.ETextureCoercion.RGB_TO_MONO
                | riley.ETextureCoercion.U8_TO_U16)
    texture = riley.load_texture_mono_u16(path, coercion)
    assert texture.shape == (1, 2, 2)
    assert texture.dtype == np.uint16
    assert np.all(texture == 65535)
