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

import riley
from riley.python.frameops import (
    frames_evenly_spaced_idxs,
    frames_first_last_idxs,
    frames_select,
)


def test_first_last_idxs_retain_both_endpoints() -> None:
    indices = riley.frames_first_last_idxs(64)
    np.testing.assert_array_equal(indices, (0, 63))


def test_first_last_idxs_single_frame() -> None:
    indices = riley.frames_first_last_idxs(1)
    np.testing.assert_array_equal(indices, (0,))


def test_evenly_spaced_idxs_spans_sequence() -> None:
    indices = riley.frames_evenly_spaced_idxs(100, 8)
    np.testing.assert_array_equal(
        indices,
        (0, 14, 28, 42, 56, 70, 84, 99),
    )


def test_evenly_spaced_idxs_short_sequence() -> None:
    indices = riley.frames_evenly_spaced_idxs(3, 8)
    np.testing.assert_array_equal(indices, (0, 1, 2))


def test_evenly_spaced_idxs_single_requested() -> None:
    indices = riley.frames_evenly_spaced_idxs(10, 1)
    np.testing.assert_array_equal(indices, (0,))


def test_frames_select_returns_contiguous_copy() -> None:
    source = np.arange(24, dtype=np.float64).reshape(4, 3, 2)
    selected = riley.frames_select(source, np.array((0, 3), dtype=np.intp))
    source[0, 0, 0] = -1.0

    assert selected.flags.c_contiguous
    np.testing.assert_array_equal(
        selected[0],
        np.arange(6).reshape(3, 2),
    )
    np.testing.assert_array_equal(selected[1], source[3])


@pytest.mark.parametrize("frames_num", (0, -1))
def test_frameops_reject_invalid_frames_num(frames_num: int) -> None:
    with pytest.raises(ValueError, match="At least one"):
        frames_first_last_idxs(frames_num)

    with pytest.raises(ValueError, match="At least one"):
        frames_evenly_spaced_idxs(frames_num, 8)


def test_evenly_spaced_rejects_invalid_max() -> None:
    with pytest.raises(ValueError, match="frame limit must be positive"):
        frames_evenly_spaced_idxs(10, 0)


def test_frames_select_rejects_empty_data() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        frames_select(np.empty((0, 3, 2)), np.array((0,), dtype=np.intp))


def test_frames_select_rejects_empty_indices() -> None:
    with pytest.raises(ValueError, match="non-empty 1D array"):
        frames_select(np.ones((4, 3)), np.empty((0,), dtype=np.intp))


def test_frames_select_rejects_out_of_bounds() -> None:
    with pytest.raises(IndexError, match="out of bounds"):
        frames_select(np.ones((4, 3)), np.array((0, 4), dtype=np.intp))

    with pytest.raises(IndexError, match="out of bounds"):
        frames_select(np.ones((4, 3)), np.array((-1,), dtype=np.intp))
