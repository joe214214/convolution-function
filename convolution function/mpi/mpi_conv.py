"""MPI entry point for distributed 2D convolution."""

from __future__ import annotations

import argparse
from datetime import datetime
import math
import socket
from pathlib import Path
from statistics import median
from typing import Optional, Sequence, Tuple

import numpy as np
from mpi4py import MPI

import bench
from comm import gather_output, post_halo_exchange, scatter_input
from domain import LocalSlices, decompose_2d, make_cart
from io_utils import generate_kernel, generate_synthetic, load_image, save_image
from kernels import conv2d_block, output_hw, set_numba_enabled
from validate import (
    SCIPY_AVAILABLE,
    TORCH_AVAILABLE,
    naive_conv_batch,
    scipy_conv,
    torch_conv,
    validate_tensor,
)


def _suggest_grid(size: int) -> tuple[int, int]:
    """Return a near-square process grid for `size` ranks."""
    best = (1, size)
    min_diff = size
    for px in range(1, int(size**0.5) + 1):
        if size % px == 0:
            py = size // px
            diff = abs(px - py)
            if diff < min_diff:
                best = (px, py)
                min_diff = diff
    return best


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distributed 2D convolution")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=str, help="Path to input image (read on rank 0)")
    src.add_argument(
        "--synthetic",
        nargs=2,
        metavar=("H", "W"),
        type=int,
        help="Generate synthetic input of size HxW",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        help="Path to .npy kernel (rank 0 loads); overrides --kernel-size",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=3,
        help="Squared kernel size to generate when --kernel missing",
    )
    parser.add_argument("--stride", type=int, default=1, help="Spatial stride")
    parser.add_argument(
        "--padding",
        type=str,
        default="valid",
        help="Padding: integer value or keywords valid/same",
    )
    parser.add_argument("--px", type=int, help="Process grid rows")
    parser.add_argument("--py", type=int, help="Process grid cols")
    parser.add_argument("--cin", type=int, default=1, help="Input channels")
    parser.add_argument("--cout", type=int, default=1, help="Output channels")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=("float32", "float64"),
        help="Computation floating point precision",
    )
    parser.add_argument(
        "--bias",
        type=str,
        help="Optional path to .npy bias vector of length Cout",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        help="Reference single-rank runtime (seconds) for speedup reporting",
    )
    parser.add_argument(
        "--save-output",
        type=str,
        help="Optional path to save the gathered output (rank 0 writes)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Optional CSV log file for benchmark rows (rank 0 appends)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run correctness check vs naive reference",
    )
    parser.add_argument(
        "--check-scipy",
        action="store_true",
        help="Additionally compare against scipy.signal.convolve2d",
    )
    parser.add_argument(
        "--check-torch",
        action="store_true",
        help="Additionally compare against torch.nn.functional.conv2d",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for synthetic inputs and generated kernels",
    )
    parser.add_argument(
        "--numba-disable",
        action="store_true",
        help="Force use of pure Python kernels (debugging aid)",
    )
    return parser


def _resolve_padding(padding: str | int, kernel: int) -> int:
    if isinstance(padding, int):
        return padding
    if padding.lower() == "valid":
        return 0
    if padding.lower() == "same":
        return kernel // 2
    try:
        return int(padding)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported padding value {padding}") from exc


def _halo_radius(kernel: int) -> int:
    if kernel <= 1:
        return 0
    if kernel % 2 == 1:
        return (kernel - 1) // 2
    return max(kernel // 2 - 1, 0)


def _normalise_kernel(arr: np.ndarray, cout: int, cin: int, dtype: np.dtype) -> np.ndarray:
    kernel = np.asarray(arr, dtype=dtype, order="C")
    if kernel.ndim == 2:
        if cout != 1 or cin != 1:
            raise ValueError("2D kernel only valid for single in/out channel")
        kernel = kernel[np.newaxis, np.newaxis, ...]
    elif kernel.ndim == 3:
        if kernel.shape[0] != cin:
            raise ValueError("3D kernel must have shape (Cin, K, K)")
        kernel = kernel[np.newaxis, ...]
    elif kernel.ndim == 4:
        pass
    else:
        raise ValueError("Kernel must be 2D, 3D, or 4D array")
    if kernel.shape[0] != cout or kernel.shape[1] != cin:
        raise ValueError(f"Kernel shape mismatch: expected ({cout},{cin},K,K), got {kernel.shape}")
    if kernel.shape[2] != kernel.shape[3]:
        raise ValueError("Kernel must be square")
    return kernel


def _prepare_input(
    args: argparse.Namespace,
    dtype: np.dtype,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    if args.synthetic:
        h, w = map(int, args.synthetic)
        input_data = generate_synthetic(args.batch, args.cin, h, w, dtype, args.seed)
    else:
        tensor = load_image(args.image, args.cin, dtype)
        h, w = tensor.shape[-2:]
        input_data = np.repeat(tensor[np.newaxis, ...], args.batch, axis=0)
    if args.kernel:
        kernel_raw = np.load(args.kernel)
        kernel = _normalise_kernel(kernel_raw, args.cout, args.cin, dtype)
        ksize = int(kernel.shape[-1])
    else:
        ksize = args.kernel_size
        kernel = generate_kernel(args.cout, args.cin, ksize, dtype, args.seed + 1)
    if args.bias:
        bias = np.load(args.bias).astype(dtype, copy=False)
        if bias.shape != (args.cout,):
            raise ValueError(f"Bias vector must have length {args.cout}")
    else:
        bias = np.zeros((args.cout,), dtype=dtype)
    input_data = np.ascontiguousarray(input_data, dtype=dtype)
    kernel = np.ascontiguousarray(kernel, dtype=dtype)
    bias = np.ascontiguousarray(bias, dtype=dtype)
    return input_data, kernel, bias, h, w


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.stride <= 0:
        raise SystemExit("Stride must be positive")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if args.px and args.py:
        px, py = args.px, args.py
    elif args.px:
        if size % args.px != 0:
            raise SystemExit("--px must divide world size")
        px, py = args.px, size // args.px
    elif args.py:
        if size % args.py != 0:
            raise SystemExit("--py must divide world size")
        py, px = args.py, size // args.py
    else:
        px, py = _suggest_grid(size)

    cart = make_cart(comm, px=px, py=py)
    cart_rank = cart.Get_rank()
    coords = tuple(cart.Get_coords(cart_rank))

    dtype = np.dtype(args.dtype)
    set_numba_enabled(not args.numba_disable)

    global_input = None
    kernel = None
    bias = None
    height = width = 0

    if rank == 0:
        global_input, kernel, bias, height, width = _prepare_input(args, dtype)
        kernel_size = int(kernel.shape[-1])
        padding = _resolve_padding(args.padding, kernel_size)
    else:
        kernel_size = 0
        padding = 0

    kernel_size = comm.bcast(kernel_size, root=0)
    padding = comm.bcast(padding, root=0)
    height, width = comm.bcast((height, width), root=0)
    kernel = np.asarray(comm.bcast(kernel if rank == 0 else None, root=0), dtype=dtype, order="C")
    bias = np.asarray(comm.bcast(bias if rank == 0 else None, root=0), dtype=dtype)

    halo = _halo_radius(kernel_size)
    stride = args.stride

    gout_hw = output_hw(height, width, kernel_size, stride, padding)
    gin_hw = (height, width)

    all_infos: list[LocalSlices] = []
    for r in range(size):
        coords_r = tuple(cart.Get_coords(r))
        info = decompose_2d(
            global_out_hw=gout_hw,
            global_in_hw=gin_hw,
            stride=stride,
            padding=padding,
            kernel=kernel_size,
            px=px,
            py=py,
            coords=coords_r,
            halo=halo,
        )
        all_infos.append(info)
    local_info = all_infos[rank]

    batch = args.batch
    cin = args.cin
    cout = args.cout

    comm_time = 0.0
    comp_time = 0.0

    t_total_start = MPI.Wtime()

    t0 = MPI.Wtime()
    local_input = scatter_input(
        comm,
        all_infos,
        local_info,
        global_input if rank == 0 else None,
        batch=batch,
        channels=cin,
        dtype=dtype,
    )
    comm_time += MPI.Wtime() - t0

    local_out_h = max(local_info.out_y.stop - local_info.out_y.start, 0)
    local_out_w = max(local_info.out_x.stop - local_info.out_x.start, 0)
    local_output = np.zeros((batch, cout, local_out_h, local_out_w), dtype=dtype)

    halo_handle = post_halo_exchange(cart, local_input, halo)

    interior_y = local_info.interior_out_y
    interior_x = local_info.interior_out_x
    full_y = slice(0, local_out_h)
    full_x = slice(0, local_out_w)

    if interior_y.stop is None:
        interior_y = slice(0, 0)
    if interior_x.stop is None:
        interior_x = slice(0, 0)

    t_comp = MPI.Wtime()
    if interior_y.stop - interior_y.start > 0 and interior_x.stop - interior_x.start > 0:
        conv2d_block(
            image=local_input,
            kernel=kernel,
            stride=stride,
            padding=padding,
            out=local_output,
            out_offset_y=local_info.out_y.start,
            out_offset_x=local_info.out_x.start,
            in_offset_y=local_info.in_y.start,
            in_offset_x=local_info.in_x.start,
            halo=halo,
            y_slice=interior_y,
            x_slice=interior_x,
            bias=bias,
        )
    comp_time += MPI.Wtime() - t_comp

    t_wait = MPI.Wtime()
    halo_handle.wait()
    comm_time += MPI.Wtime() - t_wait

    def _compute_region(y_slice: slice, x_slice: slice) -> None:
        if y_slice.stop - y_slice.start <= 0 or x_slice.stop - x_slice.start <= 0:
            return
        conv2d_block(
            image=local_input,
            kernel=kernel,
            stride=stride,
            padding=padding,
            out=local_output,
            out_offset_y=local_info.out_y.start,
            out_offset_x=local_info.out_x.start,
            in_offset_y=local_info.in_y.start,
            in_offset_x=local_info.in_x.start,
            halo=halo,
            y_slice=y_slice,
            x_slice=x_slice,
            bias=bias,
        )

    t_border = MPI.Wtime()
    top_slice = slice(0, interior_y.start) if interior_y.start else slice(0, 0)
    bottom_slice = (
        slice(interior_y.stop, local_out_h) if interior_y.stop < local_out_h else slice(0, 0)
    )
    left_slice = slice(0, interior_x.start) if interior_x.start else slice(0, 0)
    right_slice = (
        slice(interior_x.stop, local_out_w) if interior_x.stop < local_out_w else slice(0, 0)
    )

    _compute_region(top_slice, full_x)
    _compute_region(bottom_slice, full_x)
    _compute_region(
        slice(interior_y.start, interior_y.stop),
        left_slice,
    )
    _compute_region(
        slice(interior_y.start, interior_y.stop),
        right_slice,
    )
    comp_time += MPI.Wtime() - t_border

    gather_needed_root = bool(
        rank == 0 and (args.save_output or args.check or args.check_scipy or args.check_torch)
    )
    gather_needed = comm.bcast(gather_needed_root, root=0)

    global_output = None
    if gather_needed:
        t_gather = MPI.Wtime()
        global_output = gather_output(
            comm,
            all_infos,
            local_info,
            local_output,
            global_shape=(batch, cout, gout_hw[0], gout_hw[1]),
            dtype=dtype,
        )
        comm_time += MPI.Wtime() - t_gather
    else:
        # still ensure sends happen to avoid deadlock
        gather_output(
            comm,
            all_infos,
            local_info,
            local_output,
            global_shape=(batch, cout, gout_hw[0], gout_hw[1]),
            dtype=dtype,
        )

    t_total = MPI.Wtime() - t_total_start

    all_comp = comm.gather(comp_time, root=0)
    all_comm = comm.gather(comm_time, root=0)

    if rank == 0 and gather_needed and global_output is not None:
        validation_results = []
        if args.check:
            ref = naive_conv_batch(global_input, kernel, stride=stride, padding=padding, bias=bias)
            validation_results.append(("naive", validate_tensor(global_output, ref)))
        if args.check_scipy:
            if SCIPY_AVAILABLE:
                ref = scipy_conv(global_input, kernel, stride=stride, padding=padding)
                validation_results.append(("scipy", validate_tensor(global_output, ref)))
            else:
                validation_results.append(("scipy", None))
        if args.check_torch:
            if TORCH_AVAILABLE:
                ref = torch_conv(global_input, kernel, stride=stride, padding=padding, bias=bias)
                validation_results.append(("torch", validate_tensor(global_output, ref)))
            else:
                validation_results.append(("torch", None))
    else:
        validation_results = None

    if rank == 0:
        comp_median = median(all_comp) if all_comp else 0.0
        comm_median = median(all_comm) if all_comm else 0.0
        baseline = args.baseline if args.baseline else math.nan
        speedup = baseline / t_total if baseline and baseline > 0 else math.nan
        efficiency = speedup / size if not math.isnan(speedup) else math.nan

        print(
            f"[rank0] total={t_total:.4f}s comp_med={comp_median:.4f}s "
            f"comm_med={comm_median:.4f}s speedup={speedup:.3f} eff={efficiency:.3f}"
        )
        if validation_results:
            for name, result in validation_results:
                if result is None:
                    print(f"  validation {name}: skipped (dependency missing)")
                else:
                    status = "OK" if result.passed else "FAIL"
                    print(
                        f"  validation {name}: {status} "
                        f"max_abs={result.max_abs_err:.3e} max_rel={result.max_rel_err:.3e}"
                    )

        if args.save_output and global_output is not None:
            out_path = Path(args.save_output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tensor = global_output[0, : min(3, cout)]
            save_image(str(out_path), tensor)

        if args.csv:
            row = {
                "ts": datetime.utcnow().isoformat(),
                "H": height,
                "W": width,
                "K": kernel_size,
                "S": stride,
                "P": padding,
                "Cin": cin,
                "Cout": cout,
                "batch": batch,
                "ranks": size,
                "px": px,
                "py": py,
                "halo": halo,
                "t_total": t_total,
                "t_comp": comp_median,
                "t_comm": comm_median,
                "speedup": speedup,
                "eff": efficiency,
                "host": socket.gethostname(),
            }
            bench.append_csv(
                args.csv,
                fieldnames=[
                    "ts",
                    "H",
                    "W",
                    "K",
                    "S",
                    "P",
                    "Cin",
                    "Cout",
                    "batch",
                    "ranks",
                    "px",
                    "py",
                    "halo",
                    "t_total",
                    "t_comp",
                    "t_comm",
                    "speedup",
                    "eff",
                    "host",
                ],
                rows=[row],
            )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
