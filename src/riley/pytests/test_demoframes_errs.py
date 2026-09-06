# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import pytest

from riley.pydemos.demoframes import (
    evenly_spaced_frame_indices,
    first_last_frame_indices,
    select_frames,
)


@pytest.mark.parametrize("frames_num", (0, -1))
def test_frame_index_helpers_reject_empty_sequences(frames_num: int) -> None:
    with pytest.raises(ValueError, match="At least one"):
        first_last_frame_indices(frames_num)

    with pytest.raises(ValueError, match="At least one"):
        evenly_spaced_frame_indices(frames_num, 8)
