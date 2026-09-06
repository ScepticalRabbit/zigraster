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

import numpy as np
from PIL import Image
import pytest

import riley
from riley.python.textureio import ETextureCoercion

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TEX_DIR = _REPO_ROOT / "texture"


def test_rgb_loaded_as_mono_without_coercion_raises() -> None:
    path = _TEX_DIR / "speck128_rgb_u8.png"
    with pytest.raises(ValueError, match="RGB_TO_MONO"):
        riley.load_texture_mono_u8(path)


def test_mono_loaded_as_rgb_without_coercion_raises() -> None:
    path = _TEX_DIR / "speck128_mono_u8.png"
    with pytest.raises(ValueError, match="MONO_TO_RGB"):
        riley.load_texture_rgb_u8(path)


def test_u8_loaded_as_u16_without_coercion_raises() -> None:
    path = _TEX_DIR / "speck128_mono_u8.png"
    with pytest.raises(ValueError, match="U8_TO_U16"):
        riley.load_texture_mono_u16(path)


def test_u16_loaded_as_u8_without_coercion_raises() -> None:
    path = _TEX_DIR / "speck128_mono_u16.png"
    with pytest.raises(ValueError, match="U16_TO_U8"):
        riley.load_texture_mono_u8(path)


def test_compound_mismatch_missing_depth_coercion_raises() -> None:
    path = _TEX_DIR / "speck128_rgb_u8.png"
    coercion = ETextureCoercion.RGB_TO_MONO
    with pytest.raises(ValueError, match="U8_TO_U16"):
        riley.load_texture_mono_u16(path, coercion=coercion)


def test_compound_mismatch_missing_channel_coercion_raises() -> None:
    path = _TEX_DIR / "speck128_rgb_u8.png"
    coercion = ETextureCoercion.U8_TO_U16
    with pytest.raises(ValueError, match="RGB_TO_MONO"):
        riley.load_texture_mono_u16(path, coercion=coercion)


@pytest.mark.parametrize("invalid_coercion", (None, "NONE", 0, 1, []))
def test_invalid_coercion_type_raises(invalid_coercion: object) -> None:
    path = _TEX_DIR / "speck128_mono_u8.png"
    with pytest.raises(
        TypeError, match="coercion must be an ETextureCoercion member"
    ):
        riley.load_texture_mono_u8(
            path, coercion=invalid_coercion  # type: ignore[arg-type]
        )


def test_nonexistent_file_raises() -> None:
    non_existent = _TEX_DIR / "non_existent_texture.png"
    with pytest.raises(FileNotFoundError):
        riley.load_texture_mono_u8(non_existent)


def test_directory_path_raises() -> None:
    with pytest.raises(FileNotFoundError):
        riley.load_texture_mono_u8(_TEX_DIR)


def test_corrupted_empty_file_raises(tmp_path: Path) -> None:
    empty_file = tmp_path / "corrupted.png"
    empty_file.write_bytes(b"")
    with pytest.raises(Exception):
        riley.load_texture_mono_u8(empty_file)


def test_rgba_4_channel_image_raises(tmp_path: Path) -> None:
    path = tmp_path / "rgba.png"
    rgba_data = np.full((16, 16, 4), 255, dtype=np.uint8)
    Image.fromarray(rgba_data, mode="RGBA").save(path)
    with pytest.raises(
        ValueError, match="palette or alpha channels"
    ):
        riley.load_texture_rgb_u8(path)


def test_palette_image_raises(tmp_path: Path) -> None:
    path = tmp_path / "palette.png"
    im = Image.fromarray(np.zeros((16, 16), dtype=np.uint8), mode="L")
    im_palette = im.convert("P")
    im_palette.save(path)
    with pytest.raises(ValueError):
        riley.load_texture_mono_u8(path)

