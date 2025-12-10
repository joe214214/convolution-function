"""Convolution kernels implemented in pure Python with optional Numba JIT."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:  # pragma: no cover - runtime capability detection
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path
    NUMBA_AVAILABLE = False
    njit = None  # type: ignore
    prange = range  # type: ignore

USE_NUMBA = NUMBA_AVAILABLE


def set_numba_enabled(enabled: bool) -> None:
    """Globally enable/disable Numba accelerated kernels."""
    global USE_NUMBA
    USE_NUMBA = bool(enabled and NUMBA_AVAILABLE)


def output_hw(h_in: int, w_in: int, kernel: int, stride: int, padding: int) -> Tuple[int, int]:
    hout = (h_in + 2 * padding - kernel) // stride + 1
    wout = (w_in + 2 * padding - kernel) // stride + 1
    if hout <= 0 or wout <= 0:
        raise ValueError("Convolution output has non-positive dimension")
    return hout, wout


def _conv2d_block_python(
    image: np.ndarray,
    kernel: np.ndarray,
    stride: int,
    padding: int,
    out: np.ndarray,
    out_offset_y: int,
    out_offset_x: int,
    in_offset_y: int,
    in_offset_x: int,
    halo: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    bias: Optional[np.ndarray],
) -> None:
    batch, cin, h_buf, w_buf = image.shape
    cout = kernel.shape[0]
    k = kernel.shape[-1]

    bias_arr = bias if bias is not None else None

    for n in range(batch):
        for co in range(cout):
            bias_val = bias_arr[co] if bias_arr is not None else 0.0
            for oy in range(y0, y1):
                base_y_global = (out_offset_y + oy) * stride - padding
                base_y_local = base_y_global - in_offset_y + halo
                for ox in range(x0, x1):
                    base_x_global = (out_offset_x + ox) * stride - padding
                    base_x_local = base_x_global - in_offset_x + halo
                    acc = bias_val
                    for ci in range(cin):
                        for ky in range(k):
                            iy = base_y_local + ky
                            if 0 <= iy < h_buf:
                                for kx in range(k):
                                    ix = base_x_local + kx
                                    if 0 <= ix < w_buf:
                                        acc += image[n, ci, iy, ix] * kernel[co, ci, ky, kx]
                    out[n, co, oy, ox] = acc


if NUMBA_AVAILABLE:

    @njit(parallel=True, fastmath=True)  # type: ignore[misc]
    def _conv2d_block_numba(
        image,
        kernel,
        stride: int,
        padding: int,
        out,
        out_offset_y: int,
        out_offset_x: int,
        in_offset_y: int,
        in_offset_x: int,
        halo: int,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        bias,
    ):
        batch, cin, h_buf, w_buf = image.shape
        cout = kernel.shape[0]
        k = kernel.shape[-1]
        for n in prange(batch):
            for co in range(cout):
                bias_val = bias[co] if bias.size > 0 else 0.0
                for oy in range(y0, y1):
                    base_y_global = (out_offset_y + oy) * stride - padding
                    base_y_local = base_y_global - in_offset_y + halo
                    for ox in range(x0, x1):
                        base_x_global = (out_offset_x + ox) * stride - padding
                        base_x_local = base_x_global - in_offset_x + halo
                        acc = bias_val
                        for ci in range(cin):
                            for ky in range(k):
                                iy = base_y_local + ky
                                if 0 <= iy < h_buf:
                                    for kx in range(k):
                                        ix = base_x_local + kx
                                        if 0 <= ix < w_buf:
                                            acc += (
                                                image[n, ci, iy, ix] * kernel[co, ci, ky, kx]
                                            )
                        out[n, co, oy, ox] = acc


def conv2d_block(
    image: np.ndarray,
    kernel: np.ndarray,
    stride: int,
    padding: int,
    out: np.ndarray,
    out_offset_y: int,
    out_offset_x: int,
    in_offset_y: int,
    in_offset_x: int,
    halo: int,
    y_slice: Optional[slice] = None,
    x_slice: Optional[slice] = None,
    bias: Optional[np.ndarray] = None,
) -> None:
    """Compute a block of the convolution output for a batched tensor."""
    if y_slice is None:
        y_slice = slice(0, out.shape[-2])
    if x_slice is None:
        x_slice = slice(0, out.shape[-1])
    y0, y1 = y_slice.start, y_slice.stop
    x0, x1 = x_slice.start, x_slice.stop
    if y1 <= y0 or x1 <= x0:
        return

    if USE_NUMBA:
        bias_arr = bias if bias is not None else np.empty((0,), dtype=out.dtype)
        _conv2d_block_numba(
            image,
            kernel,
            stride,
            padding,
            out,
            out_offset_y,
            out_offset_x,
            in_offset_y,
            in_offset_x,
            halo,
            y0,
            y1,
            x0,
            x1,
            bias_arr,
        )
    else:
        _conv2d_block_python(
            image,
            kernel,
            stride,
            padding,
            out,
            out_offset_y,
            out_offset_x,
            in_offset_y,
            in_offset_x,
            halo,
            y0,
            y1,
            x0,
            x1,
            bias,
        )


def conv2d_single_dispatch(
    image: np.ndarray, kernel: np.ndarray, stride: int, padding: int, out: np.ndarray
) -> None:
    """Backward compatibility helper for single-channel validation."""
    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("conv2d_single_dispatch expects 2D inputs")
    batch = image[np.newaxis, np.newaxis, ...]
    weight = kernel[np.newaxis, np.newaxis, ...]
    out_full = out[np.newaxis, np.newaxis, ...]
    conv2d_block(
        image=batch,
        kernel=weight,
        stride=stride,
        padding=padding,
        out=out_full,
        out_offset_y=0,
        out_offset_x=0,
        in_offset_y=0,
        in_offset_x=0,
        halo=(kernel.shape[0] - 1) // 2,
    )
    out[:, :] = out_full[0, 0]
