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
