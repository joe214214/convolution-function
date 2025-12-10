"""Image and kernel I/O helpers based on OpenCV and NumPy."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def load_image(path: str, channels: int, dtype: np.dtype) -> np.ndarray:
    """Load an image using OpenCV and convert to the desired channel count."""
    img_path = Path(path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
    image = cv2.imread(str(img_path), flag)
    if image is None:
        raise RuntimeError(f"Failed to read image at {path}")
    if channels == 1 and image.ndim == 2:
        arr = image.astype(dtype, copy=False)
        return arr[np.newaxis, ...]
    if channels == 1:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray.astype(dtype, copy=False)[np.newaxis, ...]
    # OpenCV loads color as BGR; convert to RGB for reproducibility.
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(dtype, copy=False).transpose(2, 0, 1)
    if channels == 3:
        return arr
    if channels < 3:
        return arr[:channels]
    pad = np.zeros((channels - 3, arr.shape[1], arr.shape[2]), dtype=dtype)
    return np.concatenate([arr, pad], axis=0)


def save_image(path: str, tensor: np.ndarray) -> None:
    """Save tensor to disk. Accepts shape (H,W) or (C,H,W) (C=1/3)."""
    arr = tensor
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        arr = tensor[0]
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(Path(path)), arr)


def generate_synthetic(
    batch: int,
    channels: int,
    height: int,
    width: int,
    dtype: np.dtype,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.random((batch, channels, height, width), dtype=np.float64)
    return data.astype(dtype, copy=False)


def generate_kernel(
    cout: int, cin: int, k: int, dtype: np.dtype, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    kernel = rng.standard_normal((cout, cin, k, k), dtype=np.float64)
    return kernel.astype(dtype, copy=False)
