"""Domain decomposition utilities for MPI-based 2D convolutions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from mpi4py import MPI


@dataclass(frozen=True)
class LocalSlices:
    """Describe the portion of the global tensor owned by a rank.

    All slices are expressed in global index space using Python slice
    semantics (start inclusive, stop exclusive). The `halo` field encodes the
    symmetric halo width used in both spatial dimensions.
    """

    out_y: slice
    out_x: slice
    in_y: slice
    in_x: slice
    required_in_y: Tuple[int, int]
    required_in_x: Tuple[int, int]
    halo: int
    interior_out_y: slice
    interior_out_x: slice


def make_cart(comm: MPI.Comm, px: int, py: int) -> MPI.Cartcomm:
    """Create a 2D Cartesian communicator with optional periodicity disabled."""
    if px * py != comm.Get_size():
        raise ValueError(
            f"px*py ({px}*{py}) must equal comm size {comm.Get_size()}"
        )
    dims = (px, py)
    periods = (False, False)
    reorder = False
    return comm.Create_cart(dims=dims, periods=periods, reorder=reorder)


def _split_range(length: int, parts: int, index: int) -> Tuple[int, int]:
    """Evenly split `length` elements into `parts` segments and pick one."""
    base = length // parts
    remainder = length % parts
    start = index * base + min(index, remainder)
    stop = start + base
    if index < remainder:
        stop += 1
    return start, stop


def _compute_interior_slice(
    out_slice: slice, in_slice: slice, stride: int, padding: int, kernel: int
) -> slice:
    """Compute the local interior slice indices for a single dimension."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    if kernel <= 0:
        raise ValueError("kernel size must be positive")

    global_start, global_stop = out_slice.start, out_slice.stop

    # y dimension
    iy0 = in_slice.start
    iy1 = in_slice.stop
    min_oy = ((iy0 + padding) + stride - 1) // stride
    max_oy = (iy1 + padding - kernel) // stride
    int_o0 = max(global_start, min_oy)
    int_o1 = min(global_stop, max_oy + 1)

    if int_o1 <= int_o0:
        return slice(0, 0)
    return slice(int_o0 - global_start, int_o1 - global_start)


def decompose_2d(
    global_out_hw: Tuple[int, int],
    global_in_hw: Tuple[int, int],
    stride: int,
    padding: int,
    kernel: int,
    px: int,
    py: int,
    coords: Tuple[int, int],
    halo: int,
) -> LocalSlices:
    """Partition the global output domain onto a rank in a px x py grid."""
    gout_h, gout_w = global_out_hw
    gin_h, gin_w = global_in_hw
    ry, rx = coords

    oy0, oy1 = _split_range(gout_h, px, ry)
    ox0, ox1 = _split_range(gout_w, py, rx)

    # Compute the input footprint needed by this rank (excluding halo padding).
    in_y_start = oy0 * stride - padding
    in_y_stop = (oy1 - 1) * stride - padding + kernel if oy1 > oy0 else 0
    in_x_start = ox0 * stride - padding
    in_x_stop = (ox1 - 1) * stride - padding + kernel if ox1 > ox0 else 0

    if oy1 == oy0 or ox1 == ox0:
        in_y = slice(0, 0)
        in_x = slice(0, 0)
    else:
        in_y0 = max(0, in_y_start)
        in_y1 = min(gin_h, in_y_stop)
        in_x0 = max(0, in_x_start)
        in_x1 = min(gin_w, in_x_stop)
        in_y = slice(in_y0, in_y1)
        in_x = slice(in_x0, in_x1)

    if oy1 == oy0 or ox1 == ox0:
        interior_out_y = slice(0, 0)
        interior_out_x = slice(0, 0)
    else:
        interior_out_y = _compute_interior_slice(
            slice(oy0, oy1), in_y, stride=stride, padding=padding, kernel=kernel
        )
        interior_out_x = _compute_interior_slice(
            slice(ox0, ox1), in_x, stride=stride, padding=padding, kernel=kernel
        )

    req_in_y = (in_y_start, in_y_stop)
    req_in_x = (in_x_start, in_x_stop)

    return LocalSlices(
        out_y=slice(oy0, oy1),
        out_x=slice(ox0, ox1),
        in_y=in_y,
        in_x=in_x,
        required_in_y=req_in_y,
        required_in_x=req_in_x,
        halo=halo,
        interior_out_y=interior_out_y,
        interior_out_x=interior_out_x,
    )
