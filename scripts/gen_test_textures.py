# --------------------------------------------------------------------------
# Riley: A High Performance Rasteriser for DIC UQ
#
# Copyright (c) 2025-2026 scepticalrabbit (Lloyd Fletcher)
# Licensed under the MIT License (see LICENSE file for details)
#
# Authors: scepticalrabbit (Lloyd Fletcher)
# --------------------------------------------------------------------------
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_CROP_SIZE: int = 128
COLOR_MODES: tuple[str, ...] = ("mono", "rgb")
FORMATS: tuple[str, ...] = ("bmp", "tiff", "png")

DEFAULT_SOURCE_DIR = Path("texture")
DEFAULT_OUTPUT_DIR = Path("texture")
MONO_SOURCE_FILENAME = "speckle.bmp"
RGB_SOURCE_FILENAME = "speckle_rgb.bmp"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _crop_texture(image_array: np.ndarray, crop_size: int) -> np.ndarray:
    height, width = image_array.shape[:2]
    if height < crop_size or width < crop_size:
        raise ValueError(
            f"Image dimensions ({width}x{height}) smaller than crop size "
            f"({crop_size}x{crop_size})."
        )
    return image_array[:crop_size, :crop_size]


def _convert_depth(
    image_array: np.ndarray,
    bit_depth: str,
) -> np.ndarray:
    if bit_depth == "u8":
        if image_array.dtype == np.uint8:
            return image_array
        if image_array.dtype == np.uint16:
            rounded = (image_array.astype(np.uint32) + 128) // 257
            return rounded.astype(np.uint8)
        raise ValueError(f"Unsupported input dtype: {image_array.dtype}")

    if bit_depth == "u16":
        if image_array.dtype == np.uint16:
            return image_array
        if image_array.dtype == np.uint8:
            return image_array.astype(np.uint16) * np.uint16(257)
        raise ValueError(f"Unsupported input dtype: {image_array.dtype}")

    raise ValueError(f"Unsupported bit depth: {bit_depth}")


def save_texture_image(
    image_array: np.ndarray,
    output_path: Path,
) -> None:
    output_path = Path(output_path)
    extension = output_path.suffix.lower().lstrip(".")
    is_u16 = image_array.dtype == np.uint16

    if extension == "bmp":
        if is_u16:
            raise ValueError("BMP format does not support 16-bit textures.")
        Image.fromarray(image_array).save(output_path)
    elif extension in ("tiff", "tif", "png"):
        Image.fromarray(image_array).save(output_path)
    else:
        raise ValueError(f"Unsupported texture format: {extension}")


def generate_test_textures(
    source_dir: Path | None = None,
    output_dir: Path | None = None,
    crop_size: int = DEFAULT_CROP_SIZE,
    formats: tuple[str, ...] = FORMATS,
) -> list[Path]:
    repo = _repo_root()
    if source_dir is None:
        source_dir = repo / DEFAULT_SOURCE_DIR
    else:
        source_dir = Path(source_dir).resolve()

    if output_dir is None:
        output_dir = repo / DEFAULT_OUTPUT_DIR
    else:
        output_dir = Path(output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    mono_path = source_dir / MONO_SOURCE_FILENAME
    rgb_path = source_dir / RGB_SOURCE_FILENAME

    if not mono_path.is_file():
        raise FileNotFoundError(f"Source mono file not found: {mono_path}")
    if not rgb_path.is_file():
        raise FileNotFoundError(f"Source RGB file not found: {rgb_path}")

    with Image.open(mono_path) as im_mono:
        raw_mono = np.asarray(im_mono)
    with Image.open(rgb_path) as im_rgb:
        raw_rgb = np.asarray(im_rgb)

    cropped_mono = _crop_texture(raw_mono, crop_size)
    cropped_rgb = _crop_texture(raw_rgb, crop_size)

    generated_paths: list[Path] = []

    # Mono u8 and mono u16
    for depth in ("u8", "u16"):
        depth_array = _convert_depth(cropped_mono, depth)
        for fmt in formats:
            if fmt == "bmp" and depth == "u16":
                continue
            filename = f"speck{crop_size}_mono_{depth}.{fmt}"
            out_path = output_dir / filename
            save_texture_image(depth_array, out_path)
            generated_paths.append(out_path)

    # RGB u8 only
    rgb_u8 = _convert_depth(cropped_rgb, "u8")
    for fmt in formats:
        filename = f"speck{crop_size}_rgb_u8.{fmt}"
        out_path = output_dir / filename
        save_texture_image(rgb_u8, out_path)
        generated_paths.append(out_path)

    return generated_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate test textures from standard speckle patterns."
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=DEFAULT_CROP_SIZE,
        help="Size (pixels) of square crop (default: 128)",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Directory containing source speckle files (default: ./texture)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for generated textures (default: ./texture)",
    )
    args = parser.parse_args()

    paths = generate_test_textures(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        crop_size=args.crop_size,
    )

    print(f"Generated {len(paths)} test textures:")
    total_size = 0
    for path in paths:
        size = path.stat().st_size
        total_size += size
        print(f"  {path.name:<25} ({size:>6} bytes)")
    print(f"Total size: {total_size} bytes ({total_size / 1024:.2f} KB)")


if __name__ == "__main__":
    main()
