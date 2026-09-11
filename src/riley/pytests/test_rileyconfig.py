"""Tests for strict texture loading and raster configuration."""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import riley


def test_create_raster_config_uses_enum_default() -> None:
    config = riley.create_raster_config(4, 8)
    assert config.save_strategy is riley.SaveStrategy.both
    assert config.max_raster_workers_per_job == 2
    assert config.validate_input is riley.ValidateInput.fast


def test_create_raster_config_accepts_validate_input_modes() -> None:
    for mode in (
        riley.ValidateInput.off,
        riley.ValidateInput.fast,
        riley.ValidateInput.full,
    ):
        config = riley.create_raster_config(1, 1, validate_input=mode)
        assert config.validate_input is mode
