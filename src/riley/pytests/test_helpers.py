"""Tests for Python convenience helpers."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import riley


def test_load_texture_converts_to_contiguous_greyscale(tmp_path: Path) -> None:
    texture_path = tmp_path / "texture.png"
    Image.fromarray(
        np.array([[(255, 0, 0), (0, 255, 0)]], dtype=np.uint8),
    ).save(texture_path)

    texture = riley.load_texture(texture_path)

    assert texture.shape == (1, 2)
    assert texture.dtype == np.uint8
    assert texture.flags.c_contiguous


@pytest.mark.parametrize(
    ("num_frames", "total_threads"),
    [(0, 1), (-1, 1), (1, 0), (1, -1)],
)
def test_create_raster_config_rejects_non_positive_counts(
    num_frames: int,
    total_threads: int,
) -> None:
    with pytest.raises(ValueError):
        riley.create_raster_config(num_frames, total_threads)


def test_create_raster_config_uses_enums_and_balances_workers() -> None:
    config = riley.create_raster_config(
        num_frames=4,
        total_threads=8,
        save_strategy=riley.SaveStrategy.memory,
    )

    assert config.render_mode is riley.RenderMode.offline
    assert config.save_strategy is riley.SaveStrategy.memory
    assert config.max_raster_workers_per_job == 2
