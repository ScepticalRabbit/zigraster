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


def test_texture_loader_is_strict_by_default(tmp_path: Path) -> None:
    path = tmp_path / "rgb.png"
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match="RGB_TO_MONO"):
        riley.load_texture_mono_u8(path)


def test_texture_explicit_channel_and_depth_coercion(tmp_path: Path) -> None:
    path = tmp_path / "rgb.png"
    Image.fromarray(np.full((2, 2, 3), 255, dtype=np.uint8)).save(path)
    coercion = (riley.ETextureCoercion.RGB_TO_MONO
                | riley.ETextureCoercion.U8_TO_U16)
    texture = riley.load_texture_mono_u16(path, coercion)
    assert texture.shape == (1, 2, 2)
    assert texture.dtype == np.uint16
    assert np.all(texture == 65535)


@pytest.mark.parametrize(
    ("num_frames", "total_threads"),
    ((0, 1), (-1, 1), (1, 0), (1, -1)),
)
def test_create_raster_config_rejects_non_positive_counts(
    num_frames: int,
    total_threads: int,
) -> None:
    with pytest.raises(ValueError):
        riley.create_raster_config(num_frames, total_threads)


def test_create_raster_config_uses_enum_default() -> None:
    config = riley.create_raster_config(4, 8)
    assert config.save_strategy is riley.SaveStrategy.both
    assert config.max_raster_workers_per_job == 2


def test_create_raster_config_rejects_raw_strategy_integer() -> None:
    with pytest.raises(TypeError, match="SaveStrategy"):
        riley.create_raster_config(1, save_strategy=2)
