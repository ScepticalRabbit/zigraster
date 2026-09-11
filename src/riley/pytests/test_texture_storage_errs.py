import numpy as np
import pytest

from riley.cython import riley as bindings


def test_texture_storage_rejects_mismatched_dtype() -> None:
    with pytest.raises(ValueError, match="uint16"):
        bindings._contig_texture(
            np.zeros((2, 2), dtype=np.uint8),
            1,
            bindings.TextureStorage.u16,
        )
