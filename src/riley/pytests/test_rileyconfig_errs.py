"""Tests for strict texture loading and raster configuration."""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import riley


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


def test_create_raster_config_rejects_raw_strategy_integer() -> None:
    with pytest.raises(TypeError, match="SaveStrategy"):
        riley.create_raster_config(1, save_strategy=2)


def test_create_raster_config_rejects_raw_validate_input_int() -> None:
    with pytest.raises(TypeError, match="ValidateInput"):
        riley.create_raster_config(1, validate_input=1)


def test_create_raster_config_rejects_invalid_validate_input_type() -> None:
    with pytest.raises(TypeError, match="ValidateInput"):
        riley.create_raster_config(1, validate_input="fast")

