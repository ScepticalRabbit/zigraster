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


def test_first_last_indices_retain_both_endpoints() -> None:
    indices = first_last_frame_indices(64)

    np.testing.assert_array_equal(indices, (0, 63))


def test_first_last_indices_do_not_duplicate_one_frame() -> None:
    indices = first_last_frame_indices(1)

    np.testing.assert_array_equal(indices, (0,))


def test_even_selection_caps_and_spans_the_source_sequence() -> None:
    indices = evenly_spaced_frame_indices(100, 8)

    np.testing.assert_array_equal(indices, (0, 14, 28, 42, 56, 70, 84, 99))


def test_even_selection_retains_short_sequences() -> None:
    indices = evenly_spaced_frame_indices(3, 8)

    np.testing.assert_array_equal(indices, (0, 1, 2))


def test_select_frames_returns_an_independent_contiguous_copy() -> None:
    source = np.arange(24, dtype=np.float64).reshape(4, 3, 2)

    selected = select_frames(source, np.array((0, 3), dtype=np.intp))
    source[0, 0, 0] = -1.0

    assert selected.flags.c_contiguous
    np.testing.assert_array_equal(selected[0], np.arange(6).reshape(3, 2))
    np.testing.assert_array_equal(selected[1], source[3])
