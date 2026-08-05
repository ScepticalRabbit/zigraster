import numpy as np
import pytest

from riley.cyth import riley as bindings


def test_texture_storage_accepts_explicit_u8_u16_and_float() -> None:
    u8 = np.zeros((2, 2), dtype=np.uint8)
    u16 = np.zeros((2, 2), dtype=np.uint16)
    f32 = np.full((2, 2), 0.125, dtype=np.float32)

    assert bindings._contig_texture(u8, 1, bindings.TextureStorage.u8).dtype == np.uint8
    assert bindings._contig_texture(u16, 1, bindings.TextureStorage.u16).dtype == np.uint16
    float_texture = bindings._contig_texture(
        f32,
        1,
        bindings.TextureStorage.floating,
    )
    assert float_texture.dtype == np.float64
    assert float_texture[0, 0, 0] == pytest.approx(0.125)


def test_texture_storage_rejects_mismatched_dtype() -> None:
    with pytest.raises(ValueError, match="uint16"):
        bindings._contig_texture(
            np.zeros((2, 2), dtype=np.uint8),
            1,
            bindings.TextureStorage.u16,
        )
