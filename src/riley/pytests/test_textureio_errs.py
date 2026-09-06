"""Tests for strict texture loading and raster configuration."""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import riley


def test_texture_loader_is_strict_by_default(tmp_path: Path) -> None:
    path = tmp_path / "rgb.png"
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match="RGB_TO_MONO"):
        riley.load_texture_mono_u8(path)
