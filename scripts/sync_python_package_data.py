# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
from __future__ import annotations

import shutil
from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DATA_ROOT = PROJECT_ROOT / "src" / "riley" / "data"
MANIFEST_PATH = PROJECT_ROOT / "scripts" / "python_package_data.toml"


def _load_sync_entries() -> list[tuple[str, str]]:
    with MANIFEST_PATH.open("rb") as manifest_file:
        manifest = tomllib.load(manifest_file)

    entries = manifest.get("copy", [])
    sync_entries: list[tuple[str, str]] = []
    for entry in entries:
        src_rel = entry["source"]
        dst_rel = entry["dest"]
        sync_entries.append((src_rel, dst_rel))
    return sync_entries


def sync_python_package_data() -> None:
    PACKAGE_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(PACKAGE_DATA_ROOT / "__pycache__", ignore_errors=True)

    for src_rel, dst_rel in _load_sync_entries():
        src_path = PROJECT_ROOT / src_rel
        dst_path = PACKAGE_DATA_ROOT / dst_rel
        if not src_path.is_file():
            if dst_path.is_file():
                # sdists already contain the synced package data under
                # src/riley/data, so wheel builds from sdist do not need the
                # original repo-side source paths.
                continue
            raise FileNotFoundError(
                f"Required package data source is missing: {src_path}",
            )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)


def main() -> None:
    sync_python_package_data()


if __name__ == "__main__":
    main()
