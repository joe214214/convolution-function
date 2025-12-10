"""MPI communication helpers for halo exchange and data redistribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from mpi4py import MPI

from domain import LocalSlices

TAG_SCATTER = 1001
TAG_GATHER = 1002


@dataclass
class HaloExchangeHandle:
    """Represents an in-flight halo exchange with staged communication."""

    cart: MPI.Cartcomm
    buffer: np.ndarray
    halo: int
    _vertical_reqs: List[MPI.Request]
    _horizontal_reqs: List[MPI.Request]
    _neighbors: Tuple[int, int, int, int]
    _vertical_recv_pairs: List[Tuple[np.ndarray, np.ndarray]]
    _horizontal_recv_pairs: List[Tuple[np.ndarray, np.ndarray]]
    _send_buffers: List[np.ndarray]

    def wait(self) -> None:
        """Complete the staged halo exchange."""
        if not self._vertical_reqs and not self._horizontal_reqs:
            return
        if self._vertical_reqs:
            MPI.Request.Waitall(self._vertical_reqs)
            self._vertical_reqs.clear()
            for buf, target in self._vertical_recv_pairs:
                np.copyto(target, buf)
            self._vertical_recv_pairs.clear()
        self._start_horizontal_phase()
        if self._horizontal_reqs:
            MPI.Request.Waitall(self._horizontal_reqs)
            self._horizontal_reqs.clear()
            for buf, target in self._horizontal_recv_pairs:
                np.copyto(target, buf)
            self._horizontal_recv_pairs.clear()
        self._send_buffers.clear()

    def _start_horizontal_phase(self) -> None:
        if self._horizontal_reqs or self.halo <= 0:
            return
        up, down, left, right = self._neighbors
        halo = self.halo
        buf = self.buffer
        ndim = buf.ndim
        if ndim < 2:
            raise ValueError("buffer must be at least 2D")

        prefix = (slice(None),) * (ndim - 2)
        full_rows = slice(0, buf.shape[-2])

        def view(y: slice, x: slice) -> np.ndarray:
            return buf[prefix + (y, x)]

        left_recv = view(full_rows, slice(0, halo))
        left_send = view(full_rows, slice(halo, 2 * halo))
        right_recv = view(full_rows, slice(buf.shape[-1] - halo, buf.shape[-1]))
        right_send = view(full_rows, slice(buf.shape[-1] - 2 * halo, buf.shape[-1] - halo))

        if left != MPI.PROC_NULL:
            recv_buf = np.empty_like(left_recv)
            self._horizontal_recv_pairs.append((recv_buf, left_recv))
            self._horizontal_reqs.append(self.cart.Irecv(recv_buf, source=left))
            send_buf = np.ascontiguousarray(left_send)
            self._send_buffers.append(send_buf)
            self._horizontal_reqs.append(self.cart.Isend(send_buf, dest=left))
        else:
            left_recv[...] = 0.0

        if right != MPI.PROC_NULL:
            recv_buf = np.empty_like(right_recv)
            self._horizontal_recv_pairs.append((recv_buf, right_recv))
            self._horizontal_reqs.append(self.cart.Irecv(recv_buf, source=right))
            send_buf = np.ascontiguousarray(right_send)
            self._send_buffers.append(send_buf)
            self._horizontal_reqs.append(self.cart.Isend(send_buf, dest=right))
        else:
            right_recv[...] = 0.0


def _neighbor_ranks(cart: MPI.Cartcomm) -> Tuple[int, int, int, int]:
    up, down = cart.Shift(0, 1)
    left, right = cart.Shift(1, 1)
    return up, down, left, right


def post_halo_exchange(
    cart: MPI.Cartcomm,
    buffer: np.ndarray,
    halo: int,
) -> HaloExchangeHandle:
    """Kick off halo exchange in two stages (vertical then horizontal)."""
    if halo <= 0:
        return HaloExchangeHandle(
            cart=cart,
            buffer=buffer,
            halo=halo,
            _vertical_reqs=[],
            _horizontal_reqs=[],
            _neighbors=_neighbor_ranks(cart),
            _vertical_recv_pairs=[],
            _horizontal_recv_pairs=[],
            _send_buffers=[],
        )

    buf = buffer
    ndim = buf.ndim
    prefix = (slice(None),) * (ndim - 2)
    up, down, left, right = _neighbor_ranks(cart)

    def view(y: slice, x: slice) -> np.ndarray:
        return buf[prefix + (y, x)]

    top_recv = view(slice(0, halo), slice(halo, buf.shape[-1] - halo))
    top_send = view(slice(halo, 2 * halo), slice(halo, buf.shape[-1] - halo))
    bottom_recv = view(
        slice(buf.shape[-2] - halo, buf.shape[-2]), slice(halo, buf.shape[-1] - halo)
    )
    bottom_send = view(
        slice(buf.shape[-2] - 2 * halo, buf.shape[-2] - halo),
        slice(halo, buf.shape[-1] - halo),
    )

    vertical_reqs: List[MPI.Request] = []
    vertical_recv_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    send_buffers: List[np.ndarray] = []
    if up != MPI.PROC_NULL:
        recv_buf = np.empty_like(top_recv)
        vertical_recv_pairs.append((recv_buf, top_recv))
        vertical_reqs.append(cart.Irecv(recv_buf, source=up))
        send_buf = np.ascontiguousarray(top_send)
        send_buffers.append(send_buf)
        vertical_reqs.append(cart.Isend(send_buf, dest=up))
    else:
        top_recv[...] = 0.0

    if down != MPI.PROC_NULL:
        recv_buf = np.empty_like(bottom_recv)
        vertical_recv_pairs.append((recv_buf, bottom_recv))
        vertical_reqs.append(cart.Irecv(recv_buf, source=down))
        send_buf = np.ascontiguousarray(bottom_send)
        send_buffers.append(send_buf)
        vertical_reqs.append(cart.Isend(send_buf, dest=down))
    else:
        bottom_recv[...] = 0.0

    return HaloExchangeHandle(
        cart=cart,
        buffer=buffer,
        halo=halo,
        _vertical_reqs=vertical_reqs,
        _horizontal_reqs=[],
        _neighbors=(up, down, left, right),
        _vertical_recv_pairs=vertical_recv_pairs,
        _horizontal_recv_pairs=[],
        _send_buffers=send_buffers,
    )


def allocate_local_input(
    batch: int, channels: int, local_info: LocalSlices, dtype: np.dtype
) -> np.ndarray:
    """Allocate a zero-initialised buffer including halo margins."""
    halo = local_info.halo
    h_local = max(local_info.in_y.stop - local_info.in_y.start, 0)
    w_local = max(local_info.in_x.stop - local_info.in_x.start, 0)
    shape = (batch, channels, h_local + 2 * halo, w_local + 2 * halo)
    return np.zeros(shape, dtype=dtype)


def scatter_input(
    comm: MPI.Comm,
    all_infos: Sequence[LocalSlices],
    local_info: LocalSlices,
    global_input: Optional[np.ndarray],
    batch: int,
    channels: int,
    dtype: np.dtype,
) -> np.ndarray:
    """Distribute input subdomains to ranks."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    halo = local_info.halo
    h_local = max(local_info.in_y.stop - local_info.in_y.start, 0)
    w_local = max(local_info.in_x.stop - local_info.in_x.start, 0)
    buffer = allocate_local_input(batch, channels, local_info, dtype)
    center = buffer[:, :, halo : halo + h_local, halo : halo + w_local]

    if rank == 0:
        if global_input is None:
            raise ValueError("Root rank requires global_input for scattering")
        if h_local and w_local:
            center[...] = np.ascontiguousarray(global_input[:, :, local_info.in_y, local_info.in_x])
        for dest in range(1, size):
            info = all_infos[dest]
            h_dest = max(info.in_y.stop - info.in_y.start, 0)
            w_dest = max(info.in_x.stop - info.in_x.start, 0)
            if h_dest == 0 or w_dest == 0:
                comm.Send(np.empty((0,), dtype=dtype), dest=dest, tag=TAG_SCATTER)
                continue
            block = np.ascontiguousarray(global_input[:, :, info.in_y, info.in_x])
            comm.Send(block, dest=dest, tag=TAG_SCATTER)
    else:
        if h_local == 0 or w_local == 0:
            comm.Recv(np.empty((0,), dtype=dtype), source=0, tag=TAG_SCATTER)
        else:
            recv = np.empty((batch, channels, h_local, w_local), dtype=dtype)
            comm.Recv(recv, source=0, tag=TAG_SCATTER)
            center[...] = recv
    return buffer


def gather_output(
    comm: MPI.Comm,
    all_infos: Sequence[LocalSlices],
    local_info: LocalSlices,
    local_output: np.ndarray,
    global_shape: Tuple[int, int, int, int],
    dtype: np.dtype,
) -> Optional[np.ndarray]:
    """Gather distributed outputs on the root rank."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        global_out = np.zeros(global_shape, dtype=dtype)
        oy_slice = local_info.out_y
        ox_slice = local_info.out_x
        if local_output.size:
            global_out[:, :, oy_slice, ox_slice] = np.ascontiguousarray(local_output)
        for src in range(1, size):
            info = all_infos[src]
            h = max(info.out_y.stop - info.out_y.start, 0)
            w = max(info.out_x.stop - info.out_x.start, 0)
            if h == 0 or w == 0:
                comm.Recv(np.empty((0,), dtype=dtype), source=src, tag=TAG_GATHER)
                continue
            recv = np.empty(
                (global_shape[0], global_shape[1], h, w),
                dtype=dtype,
            )
            comm.Recv(recv, source=src, tag=TAG_GATHER)
            global_out[:, :, info.out_y, info.out_x] = recv
        return global_out
    h = local_output.shape[-2] if local_output.ndim >= 3 else 0
    w = local_output.shape[-1] if local_output.ndim >= 3 else 0
    if h == 0 or w == 0:
        comm.Send(np.empty((0,), dtype=dtype), dest=0, tag=TAG_GATHER)
    else:
        comm.Send(np.ascontiguousarray(local_output), dest=0, tag=TAG_GATHER)
    return None
