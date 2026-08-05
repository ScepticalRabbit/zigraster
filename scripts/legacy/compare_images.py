#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Image paths as constants
# CAM0_NODIST = "out/demo-dicuq-nodist/cam0_frame0_field0.bmp"
# CAM0_DIST = "out/demo-dicuq-dist/cam0_frame0_field0.bmp"
# CAM1_NODIST = "out/demo-dicuq-nodist/cam1_frame0_field0.bmp"
# CAM1_DIST = "out/demo-dicuq-dist/cam1_frame0_field0.bmp"
# OUTPUT_DIFF = "out/image_diff.png"

CAM0_NODIST = "out/test-nodist/cam0_frame0_field0.bmp"
CAM0_DIST = "out/test-dist/cam0_frame0_field0.bmp"
CAM1_NODIST = "out/test-nodist/cam1_frame0_field0.bmp"
CAM1_DIST = "out/test-dist/cam1_frame0_field0.bmp"
OUTPUT_DIFF = "out/image_diff_downloaded.png"


def load_gray_image(path_str: str) -> np.ndarray:
    """Loads an image and converts it to a float numpy array."""
    path = pathlib.Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path_str}")
    with Image.open(path) as img:
        # Convert to grayscale float array
        return np.asarray(img.convert("L"), dtype=float)


def main() -> None:
    print("Loading images...")
    cam0_nodist = load_gray_image(CAM0_NODIST)
    cam0_dist = load_gray_image(CAM0_DIST)
    cam1_nodist = load_gray_image(CAM1_NODIST)
    cam1_dist = load_gray_image(CAM1_DIST)

    print("Computing differences...")
    diff0 = np.abs(cam0_nodist - cam0_dist)
    diff1 = np.abs(cam1_nodist - cam1_dist)

    print("Generating matplotlib plots...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Camera 0 difference
    im0 = axes[0].imshow(diff0, cmap="viridis")
    axes[0].set_title("Camera 0: Absolute Difference")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot Camera 1 difference
    im1 = axes[1].imshow(diff1, cmap="viridis")
    axes[1].set_title("Camera 1: Absolute Difference")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = pathlib.Path(OUTPUT_DIFF)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved difference image to {OUTPUT_DIFF}")


if __name__ == "__main__":
    main()
