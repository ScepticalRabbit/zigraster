# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
from __future__ import annotations

from enum import Enum


class CoordCsvOrientation(Enum):
    node_major = "node_major"
    coord_major = "coord_major"


class ConnectCsvOrientation(Enum):
    elem_major = "elem_major"
    node_major = "node_major"


class FieldCsvOrientation(Enum):
    frame_major = "frame_major"
    node_major = "node_major"


class ConnectIndexing(Enum):
    auto = "auto"
    zero_based = "zero_based"
    one_based = "one_based"


class ProjectionPlane(Enum):
    xy = "xy"
    yz = "yz"
    xz = "xz"


class PlanarProjectionMode(Enum):
    best = "best"
    fit_x = "fit_x"
    fit_y = "fit_y"


__all__ = [
    "ConnectCsvOrientation",
    "ConnectIndexing",
    "CoordCsvOrientation",
    "FieldCsvOrientation",
    "PlanarProjectionMode",
    "ProjectionPlane",
]
