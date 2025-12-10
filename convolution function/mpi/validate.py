"""Validation utilities against reference convolutions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from kernels import output_hw

try:  # pragma: no cover - optional dependency
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False
    njit = None  # type: ignore

try:  # pragma: no cover - optional dependencies
    import scipy.signal

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - fallback
    SCIPY_AVAILABLE = False

try:  # pragma: no cover - optional dependencies
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - fallback
    TORCH_AVAILABLE = False


@dataclass
class ValidationResult:
    passed: bool
    max_abs_err: float
    max_rel_err: float
    details: Dict[str, float]


def _naive_conv_python(
    image: np.ndarray,
    kernel: np.ndarray,
    stride: int,
    padding: int,
    bias: np.ndarray,
    out: np.ndarray,
) -> None:
    batch, cin, h, w = image.shape
    cout = kernel.shape[0]
    k = kernel.shape[-1]
    hout, wout = out.shape[-2], out.shape[-1]
    for n in range(batch):
        for co in range(cout):
            bias_val = bias[co] if bias.size else 0.0
            for oy in range(hout):
                base_y = oy * stride - padding
                for ox in range(wout):
                    base_x = ox * stride - padding
                    acc = bias_val
                    for ci in range(cin):
                        for ky in range(k):
                            iy = base_y + ky
                            if 0 <= iy < h:
                                for kx in range(k):
                                    ix = base_x + kx
                                    if 0 <= ix < w:
                                        acc += image[n, ci, iy, ix] * kernel[co, ci, ky, kx]
                    out[n, co, oy, ox] = acc


if NUMBA_AVAILABLE:

    @njit(cache=True)  # type: ignore[misc]
    def _naive_conv_batch_numba(image, kernel, stride, padding, bias, out):
        batch, cin, h, w = image.shape
        cout = kernel.shape[0]
        k = kernel.shape[-1]
        hout, wout = out.shape[-2], out.shape[-1]
        for n in range(batch):
            for co in range(cout):
                bias_val = bias[co] if bias.size else 0.0
                for oy in range(hout):
                    base_y = oy * stride - padding
                    for ox in range(wout):
                        base_x = ox * stride - padding
                        acc = bias_val
                        for ci in range(cin):
                            for ky in range(k):
                                iy = base_y + ky
                                if 0 <= iy < h:
                                    for kx in range(k):
                                        ix = base_x + kx
                                        if 0 <= ix < w:
                                            acc += image[n, ci, iy, ix] * kernel[co, ci, ky, kx]
                        out[n, co, oy, ox] = acc


def naive_conv_batch(
    image: np.ndarray,
    kernel: np.ndarray,
    stride: int,
    padding: int,
    bias: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Reference implementation accelerated with Numba when available."""
    batch, _, h, w = image.shape
    k = kernel.shape[-1]
    hout, wout = output_hw(h, w, k, stride, padding)
    out = np.zeros((batch, kernel.shape[0], hout, wout), dtype=image.dtype)
    bias_arr = (
        np.ascontiguousarray(bias, dtype=image.dtype)
        if bias is not None
        else np.zeros((kernel.shape[0],), dtype=image.dtype)
    )
    if NUMBA_AVAILABLE:
        _naive_conv_batch_numba(
            np.ascontiguousarray(image),
            np.ascontiguousarray(kernel),
            stride,
            padding,
            bias_arr,
            out,
        )
    else:
        _naive_conv_python(
            image,
            kernel,
            stride,
            padding,
            bias_arr,
            out,
        )
    return out


def scipy_conv(
    image: np.ndarray,
    kernel: np.ndarray,
    stride: int,
    padding: int,
) -> np.ndarray:
    if not SCIPY_AVAILABLE:  # pragma: no cover - optional dependency guard
        raise RuntimeError("SciPy is not installed")
    batch, cin, h, w = image.shape
    cout = kernel.shape[0]
    k = kernel.shape[-1]
    pad_width = ((0, 0), (0, 0), (padding, padding), (padding, padding))
    padded = np.pad(image, pad_width, mode="constant")
    out = np.zeros(
        (batch, cout, (h + 2 * padding - k) // stride + 1, (w + 2 * padding - k) // stride + 1),
        dtype=image.dtype,
    )
    for n in range(batch):
        for co in range(cout):
            acc = np.zeros(out.shape[-2:], dtype=image.dtype)
            for ci in range(cin):
                ref = scipy.signal.correlate2d(
                    padded[n, ci], kernel[co, ci], mode="valid", boundary="fill", fillvalue=0.0
                )
                acc += ref[::stride, ::stride]
            out[n, co] = acc
    return out


def torch_conv(
    image: np.ndarray,
    kernel: np.ndarray,
    stride: int,
    padding: int,
    bias: Optional[np.ndarray] = None,
) -> np.ndarray:
    if not TORCH_AVAILABLE:  # pragma: no cover - optional dependency guard
        raise RuntimeError("PyTorch is not installed")
    device = torch.device("cpu")
    x = torch.from_numpy(image).to(device)
    w = torch.from_numpy(kernel).to(device)
    b = torch.from_numpy(bias).to(device) if bias is not None else None
    out = F.conv2d(x, w, bias=b, stride=stride, padding=padding)
    return out.cpu().numpy()


def validate_tensor(
    candidate: np.ndarray,
    reference: np.ndarray,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> ValidationResult:
    diff = np.abs(candidate - reference)
    max_abs = float(np.max(diff)) if diff.size else 0.0
    denom = np.maximum(np.abs(reference), 1e-12)
    rel = diff / denom
    max_rel = float(np.max(rel)) if rel.size else 0.0
    ref_abs_max = float(np.max(np.abs(reference))) if reference.size else 0.0
    passed = max_abs <= atol + rtol * ref_abs_max
    details = {"max_abs": max_abs, "max_rel": max_rel}
    return ValidationResult(passed=passed, max_abs_err=max_abs, max_rel_err=max_rel, details=details)
