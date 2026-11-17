# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""LayerNorm kernel builder using the @flyc.kernel API.

LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta

Two paths:
  - Vector path (16-bit dtype, N % VEC_WIDTH == 0): buffer-backed tiled copy
    over N / VEC_WIDTH tiles, input cached in registers across both passes.
  - Scalar path (fp32, or N not divisible by VEC_WIDTH): 2-pass scalar
    implementation.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, full
from flydsl.runtime.device import get_rocm_arch
from kernels.common.kernels_common import atomic_add, dtype_to_elem_type, get_warp_size
from kernels.norm.rmsnorm_common import validate_norm_operand_dtypes

try:
    import torch
except ImportError:
    torch = None

KERNEL_NAME = "layernorm"

EPS = 1e-5

BLOCK_THREADS = 256
WARP_SIZE = get_warp_size()
VEC_WIDTH = 8


def _load_vec(copy_atom, vec_width, elem_dtype, div_tensor, idx):
    r = fx.make_rmem_tensor(vec_width, elem_dtype)
    fx.copy(copy_atom, fx.slice(div_tensor, (None, idx)), r)
    return fx.memref_load_vec(r)


def _store_vec(copy_atom, vec_width, elem_dtype, val, div_tensor, idx):
    r = fx.make_rmem_tensor(vec_width, elem_dtype)
    fx.memref_store_vec(val, r)
    fx.copy(copy_atom, r, fx.slice(div_tensor, (None, idx)))


def _to_elem_scalar(dtype_str: str, elem_dtype, y):
    if const_expr(dtype_str == "f32"):
        return y
    return y.to(elem_dtype)


def _to_elem_vec(dtype_str: str, elem_dtype, use_hw_cvt_bf16: bool, y):
    if const_expr(dtype_str == "bf16"):
        if const_expr(use_hw_cvt_bf16):
            return y.to(elem_dtype)
        u = y.bitcast(fx.Uint32)
        upper = u >> 16
        lsb = upper & 1
        bias = lsb + 0x7FFF
        u_round = y.bitcast(fx.Uint32) + bias
        bf16_bits = u_round >> 16
        even = bf16_bits.shuffle(bf16_bits, [0, 2, 4, 6])
        odd = bf16_bits.shuffle(bf16_bits, [1, 3, 5, 7])
        odd_sh = odd << 16
        packed = even | odd_sh
        return packed.bitcast(elem_dtype)
    if const_expr(dtype_str == "f32"):
        return y
    return y.to(elem_dtype)


def _quant_dtype_to_elem_type(dtype_str: str):
    if dtype_str in ("i8", "int8"):
        return fx.Int8
    raise ValueError(f"unsupported quant dtype: {dtype_str!r} (expected 'i8' or 'int8')")


def _quant_dtype_max(dtype_str: str) -> float:
    if dtype_str in ("i8", "int8"):
        return 127.0
    raise ValueError(f"unsupported quant dtype: {dtype_str!r} (expected 'i8' or 'int8')")


def build_layernorm_module(N: int, dtype_str: str, store_stats: bool = False, eps: float = EPS):
    arch = get_rocm_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    elem_bits = 32 if dtype_str == "f32" else 16
    USE_VEC_N = elem_bits <= 16 and N % VEC_WIDTH == 0
    VEC_TILES = N // VEC_WIDTH if USE_VEC_N else 0
    NUM_VEC_ITERS = (VEC_TILES + BLOCK_THREADS - 1) // BLOCK_THREADS if USE_VEC_N else 0

    BlockReduce = fx.coop.BlockReduce[fx.Float32x2, BLOCK_THREADS]
    SharedStorage = BlockReduce.SharedStorage

    # ── GPU kernel ────────────────────────────────────────────────────────
    @flyc.kernel
    def layernorm_kernel(
        Input: fx.Pointer,
        Gamma: fx.Pointer,
        Beta: fx.Pointer,
        Output: fx.Pointer,
        Mean: fx.Pointer,
        Rstd: fx.Pointer,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        fm_fast = arith.FastMathFlags.fast
        eps_c = eps

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        # The wrapper guarantees contiguous [M, N] rows and contiguous [N]
        # affine parameters. Reconstruct only this block's static views so the
        # kernel ABI carries one address per argument, without shape/stride
        # metadata for layouts already fixed by the contract.
        row_layout = fx.make_layout(N, 1)
        scalar_layout = fx.make_layout(1, 1)
        bid_i64 = fx.Int64(bid)
        row_offset = bid_i64 * N
        Input_buf = fx.rocdl.make_buffer_tensor((Input + row_offset).view(row_layout))
        Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma.view(row_layout))
        Beta_buf = fx.rocdl.make_buffer_tensor(Beta.view(row_layout))
        Output_buf = fx.rocdl.make_buffer_tensor((Output + row_offset).view(row_layout))

        if const_expr(store_stats):
            mean_row = fx.rocdl.make_buffer_tensor((Mean + bid_i64).view(scalar_layout))
            rstd_row = fx.rocdl.make_buffer_tensor((Rstd + bid_i64).view(scalar_layout))

        # ── helpers: wave / block reduction ───────────────────────────────

        def block_reduce_add2(val0, val1):
            with fx.fastmath(fm_fast):
                values = fx.Vector.from_elements([val0, val1], fx.Float32)
                total = BlockReduce(values, fx.ReductionOp.ADD, storage=lds)
            return total[0], total[1]

        def compute_mean_rstd(sum_val, sumsq_val):
            inv_n = 1.0 / float(N)
            mean = sum_val * inv_n
            mean_sq = sumsq_val * inv_n
            mean2 = mean * mean
            var = mean_sq - mean2
            is_neg = var < 0.0
            var = is_neg.select(0.0, var)
            var_eps = var + eps_c
            rstd = fmath.rsqrt(var_eps, fastmath=fm_fast)
            return mean, rstd

        # ==================================================================
        # Vector path for every complete f16/bf16 vec8 row.
        # ==================================================================
        if const_expr(USE_VEC_N):
            c_zero_f = fx.Float32(0.0)
            thread_sum = c_zero_f
            thread_sumsq = c_zero_f
            in_local = []

            # ── Layout API: buffer-backed tensors + tiled access ─────
            in_div = fx.logical_divide(Input_buf, fx.make_layout(VEC_WIDTH, 1))
            out_div = fx.logical_divide(Output_buf, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VEC_WIDTH, 1))
            beta_div = fx.logical_divide(Beta_buf, fx.make_layout(VEC_WIDTH, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)

            # ── Pass 1: load input, accumulate sum / sumsq ───────────────
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                is_valid = idx < VEC_TILES
                idx_safe = is_valid.select(idx, 0)
                vec = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx_safe)
                in_local.append(vec)
                x = vec.to(fx.Float32)

                x2 = x * x
                red = x.reduce(ReductionOp.ADD, fastmath=fm_fast)
                red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sum = thread_sum + is_valid.select(red, c_zero_f)
                thread_sumsq = thread_sumsq + is_valid.select(red2, c_zero_f)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean, rstd = compute_mean_rstd(sum_val, sumsq_val)

            if const_expr(store_stats):
                if tid == 0:
                    mean_row[0] = mean
                    rstd_row[0] = rstd

            # ── Pass 2: normalize + affine + store ───────────────────────
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                if idx < VEC_TILES:
                    g = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, idx).to(fx.Float32)
                    b = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, beta_div, idx).to(fx.Float32)
                    x = in_local[tile_i].to(fx.Float32)
                    y = (x - mean) * rstd
                    y = y * g + b
                    out_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, y)
                    _store_vec(copy_atom, VEC_WIDTH, elem_dtype, out_e, out_div, idx)

        else:
            # ==============================================================
            # Generic path: 2-pass scalar implementation for arbitrary N
            # ==============================================================
            c_zero_f = fx.Float32(0.0)
            thread_sum = c_zero_f
            thread_sumsq = c_zero_f

            # ── Pass 1: sum + sumsq ──────────────────────────────────────
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = Input_buf[idx_safe]
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                x2 = x * x
                x_safe = is_valid.select(x, c_zero_f)
                x2_safe = is_valid.select(x2, c_zero_f)
                thread_sum = thread_sum + x_safe
                thread_sumsq = thread_sumsq + x2_safe

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean, rstd = compute_mean_rstd(sum_val, sumsq_val)

            if const_expr(store_stats):
                if tid == 0:
                    mean_row[0] = mean
                    rstd_row[0] = rstd

            # ── Pass 2: normalize + affine + store ───────────────────────
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    x_e = Input_buf[idx]
                    g_e = Gamma_buf[idx]
                    b_e = Beta_buf[idx]
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                    diff = x - mean
                    norm = diff * rstd
                    scaled = norm * g
                    y = scaled + b
                    y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                    Output_buf[idx] = y_e

    # ── JIT host launcher ─────────────────────────────────────────────────
    if store_stats:

        @flyc.jit
        def launch_layernorm(
            Input: fx.Pointer,
            Gamma: fx.Pointer,
            Beta: fx.Pointer,
            Output: fx.Pointer,
            Mean: fx.Pointer,
            Rstd: fx.Pointer,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = layernorm_kernel(Input, Gamma, Beta, Output, Mean, Rstd)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_layernorm

    @flyc.jit
    def launch_layernorm(
        Input: fx.Pointer,
        Gamma: fx.Pointer,
        Beta: fx.Pointer,
        Output: fx.Pointer,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = layernorm_kernel(Input, Gamma, Beta, Output, Gamma, Gamma)
        launcher.launch(
            grid=(m_in, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_layernorm


def build_layernorm_bwd_module(N: int, dtype_str: str):
    """Fused LayerNorm backward: grid=(M,), one block per row.

    With x_hat = (x - mean)*rstd, wdy = dy*gamma:
      c1 = mean_N(wdy) ; c2 = mean_N(wdy * x_hat)
    Pass 2:
      dx = (wdy - c1 - x_hat*c2) * rstd            -> DX (elem dtype);
      dgamma_elem = dy * x_hat (fp32)              -> atomicAdd into DGamma[idx];
      dbias_elem  = dy        (fp32)               -> atomicAdd into DBias[idx].
    eps is baked into Rstd by the forward, so it is not needed here.

    Perf follow-ups (deferred; correctness-complete as-is): generic scalar path
    only — a vectorized fast path (mirroring the forward) and caching x/dy/gamma
    between pass 1 and pass 2 would cut global traffic.
    """
    BlockReduce = fx.coop.BlockReduce[fx.Float32x2, BLOCK_THREADS]
    SharedStorage = BlockReduce.SharedStorage

    @flyc.kernel
    def layernorm_bwd_kernel(
        Input: fx.Pointer,
        Gamma: fx.Pointer,
        DY: fx.Pointer,
        Mean: fx.Pointer,
        Rstd: fx.Pointer,
        DX: fx.Pointer,
        DGamma: fx.Pointer,
        DBias: fx.Pointer,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        fm_fast = arith.FastMathFlags.fast
        n_float = float(N)
        c_zero_f = fx.Float32(0.0)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        def block_reduce_add2(val0, val1):
            with fx.fastmath(fm_fast):
                values = fx.Vector.from_elements([val0, val1], fx.Float32)
                total = BlockReduce(values, fx.ReductionOp.ADD, storage=lds)
            return total[0], total[1]

        # All layouts are fixed by the wrapper's contiguous contract. Rebuild
        # only the row/scalar views used by this block so the kernel ABI carries
        # one address per tensor and no dynamic shape/stride operands.
        row_layout = fx.make_layout(N, 1)
        scalar_layout = fx.make_layout(1, 1)
        bid_i64 = fx.Int64(bid)
        row_offset = bid_i64 * N
        row_in = fx.rocdl.make_buffer_tensor((Input + row_offset).view(row_layout))
        gamma = fx.rocdl.make_buffer_tensor(Gamma.view(row_layout))
        row_dy = fx.rocdl.make_buffer_tensor((DY + row_offset).view(row_layout))
        mean_row = fx.rocdl.make_buffer_tensor((Mean + bid_i64).view(scalar_layout))
        rstd_row = fx.rocdl.make_buffer_tensor((Rstd + bid_i64).view(scalar_layout))
        row_dx = fx.rocdl.make_buffer_tensor((DX + row_offset).view(row_layout))

        mean = mean_row[0]
        rstd = rstd_row[0]
        dgamma_view = DGamma.view(fx.make_layout(N, 1))
        dbias_view = DBias.view(fx.make_layout(N, 1))

        # Pass 1: c1 = mean(wdy) ; c2 = mean(wdy * x_hat)
        thread_c1 = c_zero_f
        thread_c2 = c_zero_f
        for base in range_constexpr(0, N, BLOCK_THREADS):
            idx = tid + base
            is_valid = idx < N
            idx_safe = is_valid.select(idx, 0)
            x_e = row_in[idx_safe]
            dy_e = row_dy[idx_safe]
            g_e = gamma[idx_safe]
            x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
            dy = dy_e if dtype_str == "f32" else dy_e.to(fx.Float32)
            g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
            x_hat = (x - mean) * rstd
            wdy = dy * g
            thread_c1 = thread_c1 + is_valid.select(wdy, c_zero_f)
            thread_c2 = thread_c2 + is_valid.select(wdy * x_hat, c_zero_f)

        sum_c1, sum_c2 = block_reduce_add2(thread_c1, thread_c2)
        c1 = sum_c1 / n_float
        c2 = sum_c2 / n_float

        # Pass 2: dx = (wdy - c1 - x_hat*c2) * rstd ; dgamma += dy*x_hat ; dbias += dy
        for base in range_constexpr(0, N, BLOCK_THREADS):
            idx = tid + base
            if idx < N:
                x_e = row_in[idx]
                dy_e = row_dy[idx]
                g_e = gamma[idx]
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                dy = dy_e if dtype_str == "f32" else dy_e.to(fx.Float32)
                g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                x_hat = (x - mean) * rstd
                wdy = dy * g
                dx = (wdy - c1 - x_hat * c2) * rstd
                dx_e = dx if dtype_str == "f32" else dx.to(elem_dtype)
                row_dx[idx] = dx_e

                dgamma = dy * x_hat
                atomic_add(dgamma_view, idx, dgamma, dtype_bytes=4)
                atomic_add(dbias_view, idx, dy, dtype_bytes=4)

    @flyc.jit
    def launch_layernorm_bwd(
        Input: fx.Pointer,
        Gamma: fx.Pointer,
        DY: fx.Pointer,
        Mean: fx.Pointer,
        Rstd: fx.Pointer,
        DX: fx.Pointer,
        DGamma: fx.Pointer,
        DBias: fx.Pointer,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = layernorm_bwd_kernel(Input, Gamma, DY, Mean, Rstd, DX, DGamma, DBias)
        launcher.launch(grid=(m_in, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch_layernorm_bwd


def build_fused_add_layernorm_module(N: int, dtype_str: str, eps: float = EPS):
    arch = get_rocm_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    elem_bits = 32 if dtype_str == "f32" else 16
    USE_VEC_N = elem_bits <= 16 and N % VEC_WIDTH == 0
    VEC_TILES = N // VEC_WIDTH if USE_VEC_N else 0
    NUM_VEC_ITERS = (VEC_TILES + BLOCK_THREADS - 1) // BLOCK_THREADS if USE_VEC_N else 0

    BlockReduce = fx.coop.BlockReduce[fx.Float32x2, BLOCK_THREADS]
    SharedStorage = BlockReduce.SharedStorage

    @flyc.kernel
    def fused_add_layernorm_kernel(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        Output: fx.Tensor,
        ResidualOut: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        fm_fast = arith.FastMathFlags.fast
        eps_c = eps

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        def block_reduce_add2(val0, val1):
            with fx.fastmath(fm_fast):
                values = fx.Vector.from_elements([val0, val1], fx.Float32)
                total = BlockReduce(values, fx.ReductionOp.ADD, storage=lds)
            return total[0], total[1]

        def compute_mean_rstd(sum_val, sumsq_val):
            inv_n = 1.0 / float(N)
            mean = sum_val * inv_n
            mean_sq = sumsq_val * inv_n
            var = mean_sq - mean * mean
            var = (var < 0.0).select(0.0, var)
            return mean, fmath.rsqrt(var + eps_c, fastmath=fm_fast)

        # ==================================================================
        # Vector path for every complete f16/bf16 vec8 row.
        # ==================================================================
        if const_expr(USE_VEC_N):
            c_zero_f = fx.Float32(0.0)
            thread_sum = c_zero_f
            thread_sumsq = c_zero_f
            added_local = []

            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            ResidualIn_buf = fx.rocdl.make_buffer_tensor(ResidualIn)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Beta_buf = fx.rocdl.make_buffer_tensor(Beta)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            ResidualOut_buf = fx.rocdl.make_buffer_tensor(ResidualOut)

            row_in = fx.slice(Input_buf, (bid, None))
            row_residual_in = fx.slice(ResidualIn_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))
            row_residual_out = fx.slice(ResidualOut_buf, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            residual_in_div = fx.logical_divide(row_residual_in, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VEC_WIDTH, 1))
            beta_div = fx.logical_divide(Beta_buf, fx.make_layout(VEC_WIDTH, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(VEC_WIDTH, 1))
            residual_out_div = fx.logical_divide(row_residual_out, fx.make_layout(VEC_WIDTH, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)

            # Pass 1: add residual, cache/store it, and accumulate sum/sumsq.
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                is_valid = idx < VEC_TILES
                idx_safe = is_valid.select(idx, 0)
                x = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx_safe).to(fx.Float32)
                residual = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, residual_in_div, idx_safe).to(fx.Float32)
                added_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, x + residual)
                added_local.append(added_e)
                added = added_e.to(fx.Float32)
                added2 = added * added
                sum_val = added.reduce(ReductionOp.ADD, fastmath=fm_fast)
                sumsq_val = added2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sum = thread_sum + is_valid.select(sum_val, c_zero_f)
                thread_sumsq = thread_sumsq + is_valid.select(sumsq_val, c_zero_f)
                if idx < VEC_TILES:
                    _store_vec(copy_atom, VEC_WIDTH, elem_dtype, added_e, residual_out_div, idx)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean, rstd = compute_mean_rstd(sum_val, sumsq_val)

            # Pass 2: normalize + affine + store, reusing cached added values.
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                if idx < VEC_TILES:
                    added = added_local[tile_i].to(fx.Float32)
                    g = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, idx).to(fx.Float32)
                    b = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, beta_div, idx).to(fx.Float32)
                    y = (added - mean) * rstd
                    y = y * g + b
                    y_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, y)
                    _store_vec(copy_atom, VEC_WIDTH, elem_dtype, y_e, out_div, idx)

        else:
            # ==============================================================
            # Generic path: scalar 2-pass implementation for arbitrary N
            # ==============================================================
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            ResidualIn_buf = fx.rocdl.make_buffer_tensor(ResidualIn)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Beta_buf = fx.rocdl.make_buffer_tensor(Beta)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            ResidualOut_buf = fx.rocdl.make_buffer_tensor(ResidualOut)

            row_in = fx.slice(Input_buf, (bid, None))
            row_residual_in = fx.slice(ResidualIn_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))
            row_residual_out = fx.slice(ResidualOut_buf, (bid, None))

            c_zero_f = fx.Float32(0.0)
            thread_sum = c_zero_f
            thread_sumsq = c_zero_f

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = row_in[idx_safe]
                r_e = row_residual_in[idx_safe]
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                residual = r_e if dtype_str == "f32" else r_e.to(fx.Float32)
                added_e = _to_elem_scalar(dtype_str, elem_dtype, x + residual)
                added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                added_safe = is_valid.select(added, c_zero_f)
                thread_sum = thread_sum + added_safe
                thread_sumsq = thread_sumsq + is_valid.select(added * added, c_zero_f)
                if idx < N:
                    row_residual_out[idx] = added_e

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean, rstd = compute_mean_rstd(sum_val, sumsq_val)

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    added_e = row_residual_out[idx]
                    g_e = Gamma_buf[idx]
                    b_e = Beta_buf[idx]
                    added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                    y = (added - mean) * rstd
                    y = y * g + b
                    y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                    row_out[idx] = y_e

    @flyc.jit
    def launch_fused_add_layernorm(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        Output: fx.Tensor,
        ResidualOut: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = fused_add_layernorm_kernel(Input, ResidualIn, Gamma, Beta, Output, ResidualOut)
        launcher.launch(
            grid=(m_in, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_fused_add_layernorm


def _build_layernorm_quant_module(
    N: int,
    dtype_str: str,
    *,
    is_smooth: bool,
    quant_dtype_str: str = "i8",
    eps: float = EPS,
):
    elem_bits = 32 if dtype_str == "f32" else 16
    USE_VEC_N = elem_bits <= 16 and N % VEC_WIDTH == 0
    VEC_TILES = N // VEC_WIDTH if USE_VEC_N else 0
    NUM_VEC_ITERS = (VEC_TILES + BLOCK_THREADS - 1) // BLOCK_THREADS if USE_VEC_N else 0
    quant_dtype_max = _quant_dtype_max(quant_dtype_str)

    BlockReduce = fx.coop.BlockReduce[fx.Float32x2, BLOCK_THREADS]
    SharedStorage = BlockReduce.SharedStorage

    @flyc.kernel
    def layernorm_quant_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        XScale: fx.Tensor,
        YScale: fx.Tensor,
        Output: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        quant_dtype = _quant_dtype_to_elem_type(quant_dtype_str)

        fm_fast = arith.FastMathFlags.fast
        eps_c = eps
        n_float = float(N)
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_dtype_max = fx.Float32(quant_dtype_max)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        YScale_buf = fx.rocdl.make_buffer_tensor(YScale)

        def block_reduce_add2(val0, val1):
            with fx.fastmath(fm_fast):
                values = fx.Vector.from_elements([val0, val1], fx.Float32)
                total = BlockReduce(values, fx.ReductionOp.ADD, storage=lds)
            return total[0], total[1]

        def block_reduce_max(val):
            # Finish readers of the previous sum before reusing its shared slots.
            gpu.barrier()
            values = fx.full(2, val, fx.Float32)
            return BlockReduce(values, fx.ReductionOp.MAX, storage=lds)[0]

        # ==================================================================
        # Vector path for every complete f16/bf16 vec8 row.
        # ==================================================================
        if const_expr(USE_VEC_N):
            quant_half_width = VEC_WIDTH // 2
            abs_mask = full(VEC_WIDTH, fx.Uint32(0x7FFFFFFF), fx.Uint32)

            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Beta_buf = fx.rocdl.make_buffer_tensor(Beta)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            if const_expr(is_smooth):
                XScale_buf = fx.rocdl.make_buffer_tensor(XScale)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VEC_WIDTH, 1))
            beta_div = fx.logical_divide(Beta_buf, fx.make_layout(VEC_WIDTH, 1))
            out_div_q = fx.logical_divide(row_out, fx.make_layout(quant_half_width, 1))
            if const_expr(is_smooth):
                xscale_div = fx.logical_divide(XScale_buf, fx.make_layout(VEC_WIDTH, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            copy_atom_q = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 8)
            if const_expr(is_smooth):
                copy_atom_xs = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)

            thread_sum = c_zero_f
            thread_sumsq = c_zero_f
            norm_input_local = []

            # Pass 1: prepare normalization input and accumulate sum/sumsq.
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                is_valid = idx < VEC_TILES
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx_safe)
                norm_input_local.append(x_e)
                x_norm = x_e.to(fx.Float32)
                x2 = x_norm * x_norm
                red = x_norm.reduce(ReductionOp.ADD, fastmath=fm_fast)
                red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sum = thread_sum + is_valid.select(red, c_zero_f)
                thread_sumsq = thread_sumsq + is_valid.select(red2, c_zero_f)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean = sum_val / n_float
            var = sumsq_val / n_float - mean * mean
            var = (var < c_zero_f).select(c_zero_f, var)
            rstd = fmath.rsqrt(var + eps_c, fastmath=fm_fast)

            thread_row_max = c_zero_f
            y_local = []

            # Pass 2: affine (+ optional smooth scale), cache y, accumulate row max.
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                is_valid = idx < VEC_TILES
                idx_safe = is_valid.select(idx, 0)
                x = norm_input_local[tile_i].to(fx.Float32)
                g = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, idx_safe).to(fx.Float32)
                b = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, beta_div, idx_safe).to(fx.Float32)
                y = (x - mean) * rstd
                y = y * g + b
                if const_expr(is_smooth):
                    s = _load_vec(copy_atom_xs, VEC_WIDTH, elem_dtype, xscale_div, idx_safe).to(fx.Float32)
                    y = y * s
                y_local.append(y)
                y_abs = (y.bitcast(fx.Uint32) & abs_mask).bitcast(fx.Float32)
                tile_max = y_abs.reduce(ReductionOp.MAX)
                thread_row_max = fx.max(thread_row_max, is_valid.select(tile_max, c_zero_f))

            row_max = block_reduce_max(thread_row_max)
            scale = row_max / c_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)

            if tid == 0:
                YScale_buf[bid] = final_scale

            inv_scale = c_one_f / final_scale

            # Pass 3: quantize + store using per-row scale.
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                if idx < VEC_TILES:
                    q = y_local[tile_i] * inv_scale
                    q_i8 = q.to(quant_dtype)
                    q_lo = q_i8.shuffle(q_i8, [0, 1, 2, 3])
                    q_hi = q_i8.shuffle(q_i8, [4, 5, 6, 7])
                    out_idx = idx * 2
                    _store_vec(copy_atom_q, quant_half_width, quant_dtype, q_lo, out_div_q, out_idx)
                    _store_vec(copy_atom_q, quant_half_width, quant_dtype, q_hi, out_div_q, out_idx + 1)

        else:
            # ==============================================================
            # Generic path: scalar 3-pass implementation for arbitrary N
            # ==============================================================
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Beta_buf = fx.rocdl.make_buffer_tensor(Beta)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            if const_expr(is_smooth):
                XScale_buf = fx.rocdl.make_buffer_tensor(XScale)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            def _abs_scalar(val):
                is_neg = val < c_zero_f
                neg_val = c_zero_f - val
                return is_neg.select(neg_val, val)

            thread_sum = c_zero_f
            thread_sumsq = c_zero_f

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = row_in[idx_safe]
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                x2 = x * x
                thread_sum = thread_sum + is_valid.select(x, c_zero_f)
                thread_sumsq = thread_sumsq + is_valid.select(x2, c_zero_f)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean = sum_val / n_float
            var = sumsq_val / n_float - mean * mean
            var = (var < c_zero_f).select(c_zero_f, var)
            rstd = fmath.rsqrt(var + eps_c, fastmath=fm_fast)

            thread_row_max = c_zero_f
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = row_in[idx_safe]
                g_e = Gamma_buf[idx_safe]
                b_e = Beta_buf[idx_safe]
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                y = (x - mean) * rstd
                y = y * g + b
                if const_expr(is_smooth):
                    s_e = XScale_buf[idx_safe]
                    s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                    y = y * s
                y_abs = _abs_scalar(y)
                thread_row_max = fx.max(thread_row_max, is_valid.select(y_abs, c_zero_f))

            row_max = block_reduce_max(thread_row_max)
            scale = row_max / c_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)

            if tid == 0:
                YScale_buf[bid] = final_scale

            inv_scale = c_one_f / final_scale

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    x_e = row_in[idx]
                    g_e = Gamma_buf[idx]
                    b_e = Beta_buf[idx]
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                    y = (x - mean) * rstd
                    y = y * g + b
                    if const_expr(is_smooth):
                        s_e = XScale_buf[idx]
                        s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                        y = y * s
                    q = y * inv_scale
                    q_i8 = q.to(quant_dtype)
                    row_out[idx] = q_i8

    if is_smooth:

        @flyc.jit
        def launch_layernorm_smoothquant(
            Input: fx.Tensor,
            Gamma: fx.Tensor,
            Beta: fx.Tensor,
            XScale: fx.Tensor,
            Output: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            validate_norm_operand_dtypes(Input, dtype_str, Gamma=Gamma, Beta=Beta, XScale=XScale)
            launcher = layernorm_quant_kernel(Input, Gamma, Beta, XScale, YScale, Output)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_layernorm_smoothquant

    else:

        @flyc.jit
        def launch_layernorm_dynamicquant(
            Input: fx.Tensor,
            Gamma: fx.Tensor,
            Beta: fx.Tensor,
            Output: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            validate_norm_operand_dtypes(Input, dtype_str, Gamma=Gamma, Beta=Beta)
            launcher = layernorm_quant_kernel(Input, Gamma, Beta, Gamma, YScale, Output)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

    return launch_layernorm_dynamicquant


def _build_fused_add_layernorm_quant_block_module(
    N: int,
    dtype_str: str,
    *,
    is_smooth: bool,
    quant_dtype_str: str = "i8",
    eps: float = EPS,
    BT: int,
    nt: bool = False,
    wpe: int = 0,
):
    arch = get_rocm_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    elem_bits = 32 if dtype_str == "f32" else 16
    # Use full-width 128-bit vector accesses for fp32 too.  The former fp32
    # fallback read and wrote one scalar at a time and did not keep the
    # normalized row in registers; both costs dominate these short rows.
    VW = 4 if elem_bits == 32 else 8
    USE_VEC_N = N >= VW
    VEC_TILES = N // VW if USE_VEC_N else 0
    VEC_ELEMS = VEC_TILES * VW
    TAIL_ELEMS = N - VEC_ELEMS
    NUM_VEC_ITERS = (VEC_TILES + BT - 1) // BT if USE_VEC_N else 0
    quant_dtype_max = _quant_dtype_max(quant_dtype_str)

    BlockReduce = fx.coop.BlockReduce[fx.Float32x2, BT]
    SharedStorage = BlockReduce.SharedStorage

    block_kernel_kwargs = {
        "name": f"fused_add_layernorm_quant_block_{N}_{dtype_str}_{BT}_{'nt' if nt else 'ca'}",
        "known_block_size": [BT, 1, 1],
    }

    @flyc.kernel(**block_kernel_kwargs)
    def fused_add_layernorm_quant_kernel(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        XScale: fx.Tensor,
        YScale: fx.Tensor,
        Output: fx.Tensor,
        ResidualOut: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        quant_dtype = _quant_dtype_to_elem_type(quant_dtype_str)

        fm_fast = arith.FastMathFlags.fast
        eps_c = eps
        n_float = float(N)
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_dtype_max = fx.Float32(quant_dtype_max)

        lds_alloc = fx.SharedAllocator()
        lds_sum = lds_alloc.allocate(SharedStorage).peek()
        lds_max = lds_alloc.allocate(SharedStorage).peek()

        YScale_buf = fx.rocdl.make_buffer_tensor(YScale)

        def block_reduce_add2(val0, val1):
            with fx.fastmath(fm_fast):
                values = fx.Vector.from_elements([val0, val1], fx.Float32)
                total = BlockReduce(values, fx.ReductionOp.ADD, storage=lds_sum)
            return total[0], total[1]

        def block_reduce_max(val):
            # The max lives in its own shared slots, so it does not need a
            # block-wide barrier to wait for readers of the sum reduction.
            values = fx.full(2, val, fx.Float32)
            return BlockReduce(values, fx.ReductionOp.MAX, storage=lds_max)[0]

        # ==================================================================
        # Vector path: 128-bit loads/stores for fp16, bf16 and fp32, plus a
        # scalar tail for widths that are not a multiple of the vector width.
        # ==================================================================
        if const_expr(USE_VEC_N):
            quant_chunk_width = 4
            abs_mask = full(VW, fx.Uint32(0x7FFFFFFF), fx.Uint32)

            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            ResidualIn_buf = fx.rocdl.make_buffer_tensor(ResidualIn)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Beta_buf = fx.rocdl.make_buffer_tensor(Beta)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            ResidualOut_buf = fx.rocdl.make_buffer_tensor(ResidualOut)
            if const_expr(is_smooth):
                XScale_buf = fx.rocdl.make_buffer_tensor(XScale)

            row_in = fx.slice(Input_buf, (bid, None))
            row_residual_in = fx.slice(ResidualIn_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))
            row_residual_out = fx.slice(ResidualOut_buf, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VW, 1))
            residual_in_div = fx.logical_divide(row_residual_in, fx.make_layout(VW, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VW, 1))
            beta_div = fx.logical_divide(Beta_buf, fx.make_layout(VW, 1))
            out_div_q = fx.logical_divide(row_out, fx.make_layout(quant_chunk_width, 1))
            residual_out_div = fx.logical_divide(row_residual_out, fx.make_layout(VW, 1))
            if const_expr(is_smooth):
                xscale_div = fx.logical_divide(XScale_buf, fx.make_layout(VW, 1))

            input_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(2 if nt else 0), elem_bits)
            weight_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            store_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            copy_atom_q = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 8)

            thread_sum = c_zero_f
            thread_sumsq = c_zero_f
            norm_input_local = []
            tail_added = c_zero_f

            # Pass 1: add residual, write the rounded residual output, cache
            # the rounded value and accumulate sum / sumsq.
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BT
                is_valid = idx < VEC_TILES
                idx_safe = is_valid.select(idx, 0)
                x = _load_vec(input_copy_atom, VW, elem_dtype, in_div, idx_safe).to(fx.Float32)
                residual = _load_vec(input_copy_atom, VW, elem_dtype, residual_in_div, idx_safe).to(fx.Float32)
                added_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, x + residual)
                norm_input_local.append(added_e)
                x_norm = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                if idx < VEC_TILES:
                    _store_vec(store_copy_atom, VW, elem_dtype, added_e, residual_out_div, idx)
                x2 = x_norm * x_norm
                red = x_norm.reduce(ReductionOp.ADD, fastmath=fm_fast)
                red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sum = thread_sum + is_valid.select(red, c_zero_f)
                thread_sumsq = thread_sumsq + is_valid.select(red2, c_zero_f)

            if const_expr(TAIL_ELEMS > 0):
                if tid < TAIL_ELEMS:
                    tail_idx = tid + VEC_ELEMS
                    x_e = row_in[tail_idx]
                    r_e = row_residual_in[tail_idx]
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    residual = r_e if dtype_str == "f32" else r_e.to(fx.Float32)
                    tail_e = _to_elem_scalar(dtype_str, elem_dtype, x + residual)
                    row_residual_out[tail_idx] = tail_e
                    tail_added = tail_e if dtype_str == "f32" else tail_e.to(fx.Float32)
                    thread_sum = thread_sum + tail_added
                    thread_sumsq = thread_sumsq + tail_added * tail_added

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean = sum_val / n_float
            var = sumsq_val / n_float - mean * mean
            var = (var < c_zero_f).select(c_zero_f, var)
            rstd = fmath.rsqrt(var + eps_c, fastmath=fm_fast)

            thread_row_max = c_zero_f
            y_local = []
            tail_y = c_zero_f

            # Pass 2: normalize + affine (+ optional smooth scale), cache y
            # in registers, and accumulate this thread's maximum.
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BT
                is_valid = idx < VEC_TILES
                idx_safe = is_valid.select(idx, 0)
                x_e = norm_input_local[tile_i]
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                g = _load_vec(weight_copy_atom, VW, elem_dtype, gamma_div, idx_safe).to(fx.Float32)
                b = _load_vec(weight_copy_atom, VW, elem_dtype, beta_div, idx_safe).to(fx.Float32)
                y = (x - mean) * rstd
                y = y * g + b
                if const_expr(is_smooth):
                    s = _load_vec(weight_copy_atom, VW, elem_dtype, xscale_div, idx_safe).to(fx.Float32)
                    y = y * s
                y_local.append(y)
                y_abs = (y.bitcast(fx.Uint32) & abs_mask).bitcast(fx.Float32)
                tile_max = y_abs.reduce(ReductionOp.MAX)
                thread_row_max = fx.max(thread_row_max, is_valid.select(tile_max, c_zero_f))

            if const_expr(TAIL_ELEMS > 0):
                if tid < TAIL_ELEMS:
                    tail_idx = tid + VEC_ELEMS
                    g_e = Gamma_buf[tail_idx]
                    b_e = Beta_buf[tail_idx]
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                    tail_y = (tail_added - mean) * rstd
                    tail_y = tail_y * g + b
                    if const_expr(is_smooth):
                        s_e = XScale_buf[tail_idx]
                        s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                        tail_y = tail_y * s
                    tail_abs = (tail_y.bitcast(fx.Uint32) & fx.Uint32(0x7FFFFFFF)).bitcast(fx.Float32)
                    thread_row_max = fx.max(thread_row_max, tail_abs)

            row_max = block_reduce_max(thread_row_max)
            scale = row_max / c_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)

            if tid == 0:
                YScale_buf[bid] = final_scale

            inv_scale = c_one_f / final_scale

            # Pass 3: quantize + store using the per-row scale.
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BT
                if idx < VEC_TILES:
                    q = y_local[tile_i] * inv_scale
                    q_i8 = q.to(quant_dtype)
                    if const_expr(VW == 8):
                        q_lo = q_i8.shuffle(q_i8, [0, 1, 2, 3])
                        q_hi = q_i8.shuffle(q_i8, [4, 5, 6, 7])
                        out_idx = idx * 2
                        _store_vec(copy_atom_q, quant_chunk_width, quant_dtype, q_lo, out_div_q, out_idx)
                        _store_vec(copy_atom_q, quant_chunk_width, quant_dtype, q_hi, out_div_q, out_idx + 1)
                    else:
                        _store_vec(copy_atom_q, quant_chunk_width, quant_dtype, q_i8, out_div_q, idx)

            if const_expr(TAIL_ELEMS > 0):
                if tid < TAIL_ELEMS:
                    tail_idx = tid + VEC_ELEMS
                    q = tail_y * inv_scale
                    row_out[tail_idx] = q.to(quant_dtype)

        else:
            # ==============================================================
            # Scalar fallback for N below one vector-width (the interface
            # requires N >= 8, but keep the builder total for direct users).
            # ==============================================================
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            ResidualIn_buf = fx.rocdl.make_buffer_tensor(ResidualIn)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Beta_buf = fx.rocdl.make_buffer_tensor(Beta)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            ResidualOut_buf = fx.rocdl.make_buffer_tensor(ResidualOut)
            if const_expr(is_smooth):
                XScale_buf = fx.rocdl.make_buffer_tensor(XScale)

            row_in = fx.slice(Input_buf, (bid, None))
            row_residual_in = fx.slice(ResidualIn_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))
            row_residual_out = fx.slice(ResidualOut_buf, (bid, None))

            thread_sum = c_zero_f
            thread_sumsq = c_zero_f

            for base_idx_int in range_constexpr(0, N, BT):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = row_in[idx_safe]
                r_e = row_residual_in[idx_safe]
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                residual = r_e if dtype_str == "f32" else r_e.to(fx.Float32)
                added_e = _to_elem_scalar(dtype_str, elem_dtype, x + residual)
                if idx < N:
                    row_residual_out[idx] = added_e
                x = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                x2 = x * x
                thread_sum = thread_sum + is_valid.select(x, c_zero_f)
                thread_sumsq = thread_sumsq + is_valid.select(x2, c_zero_f)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean = sum_val / n_float
            var = sumsq_val / n_float - mean * mean
            var = (var < c_zero_f).select(c_zero_f, var)
            rstd = fmath.rsqrt(var + eps_c, fastmath=fm_fast)
            thread_row_max = c_zero_f
            for base_idx_int in range_constexpr(0, N, BT):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = row_residual_out[idx_safe]
                g_e = Gamma_buf[idx_safe]
                b_e = Beta_buf[idx_safe]
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                y = (x - mean) * rstd
                y = y * g + b
                if const_expr(is_smooth):
                    s_e = XScale_buf[idx_safe]
                    s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                    y = y * s
                y_abs = (y.bitcast(fx.Uint32) & fx.Uint32(0x7FFFFFFF)).bitcast(fx.Float32)
                thread_row_max = fx.max(thread_row_max, is_valid.select(y_abs, c_zero_f))

            row_max = block_reduce_max(thread_row_max)
            scale = row_max / c_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)

            if tid == 0:
                YScale_buf[bid] = final_scale

            inv_scale = c_one_f / final_scale

            for base_idx_int in range_constexpr(0, N, BT):
                idx = tid + base_idx_int
                if idx < N:
                    x_e = row_residual_out[idx]
                    g_e = Gamma_buf[idx]
                    b_e = Beta_buf[idx]
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                    y = (x - mean) * rstd
                    y = y * g + b
                    if const_expr(is_smooth):
                        s_e = XScale_buf[idx]
                        s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                        y = y * s
                    q = y * inv_scale
                    row_out[idx] = q.to(quant_dtype)

    if is_smooth:

        @flyc.jit
        def launch_fused_add_layernorm_smoothquant(
            Input: fx.Tensor,
            ResidualIn: fx.Tensor,
            Gamma: fx.Tensor,
            Beta: fx.Tensor,
            XScale: fx.Tensor,
            Output: fx.Tensor,
            ResidualOut: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            validate_norm_operand_dtypes(
                Input, dtype_str, ResidualIn=ResidualIn, Gamma=Gamma, Beta=Beta, XScale=XScale, ResidualOut=ResidualOut
            )
            launcher = fused_add_layernorm_quant_kernel(
                Input, ResidualIn, Gamma, Beta, XScale, YScale, Output, ResidualOut
            )
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BT, 1, 1),
                stream=stream,
                value_attrs={"rocdl.waves_per_eu": wpe} if wpe else None,
            )

        return launch_fused_add_layernorm_smoothquant

    else:

        @flyc.jit
        def launch_fused_add_layernorm_dynamicquant(
            Input: fx.Tensor,
            ResidualIn: fx.Tensor,
            Gamma: fx.Tensor,
            Beta: fx.Tensor,
            Output: fx.Tensor,
            ResidualOut: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            validate_norm_operand_dtypes(
                Input, dtype_str, ResidualIn=ResidualIn, Gamma=Gamma, Beta=Beta, ResidualOut=ResidualOut
            )
            launcher = fused_add_layernorm_quant_kernel(
                Input, ResidualIn, Gamma, Beta, Gamma, YScale, Output, ResidualOut
            )
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BT, 1, 1),
                stream=stream,
                value_attrs={"rocdl.waves_per_eu": wpe} if wpe else None,
            )

    return launch_fused_add_layernorm_dynamicquant

def _build_fused_add_layernorm_quant_warp_module(
    N: int,
    dtype_str: str,
    *,
    is_smooth: bool,
    quant_dtype_str: str = "i8",
    eps: float = EPS,
    TPR: int,
    ROWS: int,
    wpe: int = 0,
):
    """One warp per row, packed several rows into a block.

    Short rows do not have enough vector tiles to keep a 256-thread block
    useful; a warp-only reduction avoids both idle lanes and the shared-memory
    block reduction.  The dynamic M is handled by a row guard.
    """
    arch = get_rocm_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")
    elem_bits = 32 if dtype_str == "f32" else 16
    VW = 4 if elem_bits == 32 else 8
    VEC_TILES = N // VW
    VEC_ELEMS = VEC_TILES * VW
    TAIL_ELEMS = N - VEC_ELEMS
    BT = ROWS * TPR
    NUM_ITERS = (VEC_TILES + TPR - 1) // TPR
    quant_dtype_max = _quant_dtype_max(quant_dtype_str)

    warp_kernel_kwargs = {
        "name": f"fused_add_layernorm_quant_warp_{N}_{dtype_str}_{TPR}x{ROWS}",
        "known_block_size": [BT, 1, 1],
    }

    @flyc.kernel(**warp_kernel_kwargs)
    def fused_add_layernorm_quant_warp_kernel(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        XScale: fx.Tensor,
        YScale: fx.Tensor,
        Output: fx.Tensor,
        ResidualOut: fx.Tensor,
        MIn: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        lane = tid % TPR
        row = bid * ROWS + tid // TPR

        elem_dtype = dtype_to_elem_type(dtype_str)
        quant_dtype = _quant_dtype_to_elem_type(quant_dtype_str)
        fm_fast = arith.FastMathFlags.fast
        eps_c = eps
        n_float = float(N)
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_dtype_max = fx.Float32(quant_dtype_max)
        abs_mask = full(VW, fx.Uint32(0x7FFFFFFF), fx.Uint32)

        YScale_buf = fx.rocdl.make_buffer_tensor(YScale)

        if row < MIn:
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            ResidualIn_buf = fx.rocdl.make_buffer_tensor(ResidualIn)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Beta_buf = fx.rocdl.make_buffer_tensor(Beta)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            ResidualOut_buf = fx.rocdl.make_buffer_tensor(ResidualOut)
            if const_expr(is_smooth):
                XScale_buf = fx.rocdl.make_buffer_tensor(XScale)

            row_in = fx.slice(Input_buf, (row, None))
            row_residual_in = fx.slice(ResidualIn_buf, (row, None))
            row_out = fx.slice(Output_buf, (row, None))
            row_residual_out = fx.slice(ResidualOut_buf, (row, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VW, 1))
            residual_in_div = fx.logical_divide(row_residual_in, fx.make_layout(VW, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VW, 1))
            beta_div = fx.logical_divide(Beta_buf, fx.make_layout(VW, 1))
            out_div_q = fx.logical_divide(row_out, fx.make_layout(4, 1))
            residual_out_div = fx.logical_divide(row_residual_out, fx.make_layout(VW, 1))
            if const_expr(is_smooth):
                xscale_div = fx.logical_divide(XScale_buf, fx.make_layout(VW, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            copy_atom_q = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 8)

            thread_sum = c_zero_f
            thread_sumsq = c_zero_f
            norm_input_local = []
            tail_added = c_zero_f

            for tile_i in range_constexpr(NUM_ITERS):
                idx = lane + tile_i * TPR
                is_valid = idx < VEC_TILES
                idx_safe = is_valid.select(idx, 0)
                x = _load_vec(copy_atom, VW, elem_dtype, in_div, idx_safe).to(fx.Float32)
                residual = _load_vec(copy_atom, VW, elem_dtype, residual_in_div, idx_safe).to(fx.Float32)
                added_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, x + residual)
                norm_input_local.append(added_e)
                x_norm = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                if idx < VEC_TILES:
                    _store_vec(copy_atom, VW, elem_dtype, added_e, residual_out_div, idx)
                x2 = x_norm * x_norm
                red = x_norm.reduce(ReductionOp.ADD, fastmath=fm_fast)
                red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sum = thread_sum + is_valid.select(red, c_zero_f)
                thread_sumsq = thread_sumsq + is_valid.select(red2, c_zero_f)

            if const_expr(TAIL_ELEMS > 0):
                if lane < TAIL_ELEMS:
                    tail_idx = lane + VEC_ELEMS
                    x_e = row_in[tail_idx]
                    r_e = row_residual_in[tail_idx]
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    residual = r_e if dtype_str == "f32" else r_e.to(fx.Float32)
                    tail_e = _to_elem_scalar(dtype_str, elem_dtype, x + residual)
                    row_residual_out[tail_idx] = tail_e
                    tail_added = tail_e if dtype_str == "f32" else tail_e.to(fx.Float32)
                    thread_sum = thread_sum + tail_added
                    thread_sumsq = thread_sumsq + tail_added * tail_added

            with fx.fastmath(fm_fast):
                sum_val = fx.coop.warp_reduce(thread_sum, fx.ReductionOp.ADD, width=TPR)
                sumsq_val = fx.coop.warp_reduce(thread_sumsq, fx.ReductionOp.ADD, width=TPR)
            mean = sum_val / n_float
            var = sumsq_val / n_float - mean * mean
            var = (var < c_zero_f).select(c_zero_f, var)
            rstd = fmath.rsqrt(var + eps_c, fastmath=fm_fast)

            thread_row_max = c_zero_f
            y_local = []
            tail_y = c_zero_f
            for tile_i in range_constexpr(NUM_ITERS):
                idx = lane + tile_i * TPR
                is_valid = idx < VEC_TILES
                idx_safe = is_valid.select(idx, 0)
                x_e = norm_input_local[tile_i]
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                g = _load_vec(copy_atom, VW, elem_dtype, gamma_div, idx_safe).to(fx.Float32)
                b = _load_vec(copy_atom, VW, elem_dtype, beta_div, idx_safe).to(fx.Float32)
                y = (x - mean) * rstd
                y = y * g + b
                if const_expr(is_smooth):
                    s = _load_vec(copy_atom, VW, elem_dtype, xscale_div, idx_safe).to(fx.Float32)
                    y = y * s
                y_local.append(y)
                y_abs = (y.bitcast(fx.Uint32) & abs_mask).bitcast(fx.Float32)
                tile_max = y_abs.reduce(ReductionOp.MAX)
                thread_row_max = fx.max(thread_row_max, is_valid.select(tile_max, c_zero_f))

            if const_expr(TAIL_ELEMS > 0):
                if lane < TAIL_ELEMS:
                    tail_idx = lane + VEC_ELEMS
                    g_e = Gamma_buf[tail_idx]
                    b_e = Beta_buf[tail_idx]
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                    tail_y = (tail_added - mean) * rstd
                    tail_y = tail_y * g + b
                    if const_expr(is_smooth):
                        s_e = XScale_buf[tail_idx]
                        s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                        tail_y = tail_y * s
                    tail_abs = (tail_y.bitcast(fx.Uint32) & fx.Uint32(0x7FFFFFFF)).bitcast(fx.Float32)
                    thread_row_max = fx.max(thread_row_max, tail_abs)

            row_max = fx.coop.warp_reduce(thread_row_max, fx.ReductionOp.MAX, width=TPR)
            scale = row_max / c_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)
            if lane == 0:
                YScale_buf[row] = final_scale
            inv_scale = c_one_f / final_scale

            for tile_i in range_constexpr(NUM_ITERS):
                idx = lane + tile_i * TPR
                if idx < VEC_TILES:
                    q = y_local[tile_i] * inv_scale
                    q_i8 = q.to(quant_dtype)
                    if const_expr(VW == 8):
                        q_lo = q_i8.shuffle(q_i8, [0, 1, 2, 3])
                        q_hi = q_i8.shuffle(q_i8, [4, 5, 6, 7])
                        out_idx = idx * 2
                        _store_vec(copy_atom_q, 4, quant_dtype, q_lo, out_div_q, out_idx)
                        _store_vec(copy_atom_q, 4, quant_dtype, q_hi, out_div_q, out_idx + 1)
                    else:
                        _store_vec(copy_atom_q, 4, quant_dtype, q_i8, out_div_q, idx)

            if const_expr(TAIL_ELEMS > 0):
                if lane < TAIL_ELEMS:
                    tail_idx = lane + VEC_ELEMS
                    q = tail_y * inv_scale
                    row_out[tail_idx] = q.to(quant_dtype)

    if is_smooth:

        @flyc.jit
        def launch_fused_add_layernorm_smoothquant(
            Input: fx.Tensor,
            ResidualIn: fx.Tensor,
            Gamma: fx.Tensor,
            Beta: fx.Tensor,
            XScale: fx.Tensor,
            Output: fx.Tensor,
            ResidualOut: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            validate_norm_operand_dtypes(
                Input, dtype_str, ResidualIn=ResidualIn, Gamma=Gamma, Beta=Beta, XScale=XScale, ResidualOut=ResidualOut
            )
            launcher = fused_add_layernorm_quant_warp_kernel(
                Input, ResidualIn, Gamma, Beta, XScale, YScale, Output, ResidualOut, m_in
            )
            launcher.launch(
                grid=((m_in + ROWS - 1) // ROWS, 1, 1),
                block=(BT, 1, 1),
                stream=stream,
                value_attrs={"rocdl.waves_per_eu": wpe} if wpe else None,
            )

        return launch_fused_add_layernorm_smoothquant

    @flyc.jit
    def launch_fused_add_layernorm_dynamicquant(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        Output: fx.Tensor,
        ResidualOut: fx.Tensor,
        YScale: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        validate_norm_operand_dtypes(
            Input, dtype_str, ResidualIn=ResidualIn, Gamma=Gamma, Beta=Beta, ResidualOut=ResidualOut
        )
        launcher = fused_add_layernorm_quant_warp_kernel(
            Input, ResidualIn, Gamma, Beta, Gamma, YScale, Output, ResidualOut, m_in
        )
        launcher.launch(
            grid=((m_in + ROWS - 1) // ROWS, 1, 1),
            block=(BT, 1, 1),
            stream=stream,
            value_attrs={"rocdl.waves_per_eu": wpe} if wpe else None,
        )

    return launch_fused_add_layernorm_dynamicquant



def _block_bt_for_small_m(vec_tiles: int, elem_bits: int) -> int:
    """Pick a one-row block width for a latency-sized grid.

    Larger blocks expose more parallel memory operations for a short row, but
    past one full pass their extra lanes only take part in the reduction.  A
    padded-lane model captures that tradeoff; very wide rows keep 512 threads
    because their extra iterations dominate.
    """
    if vec_tiles <= 128:
        return 128
    if vec_tiles <= 256:
        return 256
    if vec_tiles <= 448:
        # Two balanced 384-thread passes keep the reduction small while still
        # exposing enough memory-level parallelism for short rows.
        return 384
    if elem_bits <= 16 and 2000 <= vec_tiles <= 2200:
        # Around 2K vectors a single 1024-lane pass is best from the smallest
        # grids onward, not only for throughput-sized grids.
        return 1024
    if elem_bits <= 16 and vec_tiles >= 1900:
        # Extremely wide 16-bit rows gain another wave of load parallelism
        # without the reduction becoming the critical path.
        return 768
    if elem_bits <= 16 and vec_tiles >= 1400:
        return 640
    if vec_tiles >= 896:
        return 512
    # Between one and two full 512-thread passes, keep the launched lane count
    # close to the real row width.
    best_bt = 512
    best_cost = None
    for bt in (256, 384, 512):
        padded = ((vec_tiles + bt - 1) // bt) * bt
        cost = (padded, -bt)
        if best_cost is None or cost < best_cost:
            best_cost, best_bt = cost, bt
    return best_bt


def _make_fused_add_layernorm_quant_dispatch(N: int, dtype_str: str, *, is_smooth: bool,
                                              quant_dtype_str: str, eps: float):
    """Build a launcher whose runtime M selects the best compiled geometry.

    N is fixed at build time, so only the row count needs a runtime branch.
    """
    elem_bits = 32 if dtype_str == "f32" else 16
    VW = 4 if elem_bits == 32 else 8
    vec_tiles = N // VW

    def warp_builder(tpr, rows, wpe=0):
        return lambda: _build_fused_add_layernorm_quant_warp_module(
            N, dtype_str, is_smooth=is_smooth, quant_dtype_str=quant_dtype_str,
            eps=eps, TPR=tpr, ROWS=rows, wpe=wpe,
        )

    def block_builder(bt, nt=False, wpe=0):
        return lambda: _build_fused_add_layernorm_quant_block_module(
            N, dtype_str, is_smooth=is_smooth, quant_dtype_str=quant_dtype_str,
            eps=eps, BT=bt, nt=nt, wpe=wpe,
        )

    variants = []
    thresholds = []
    if N <= 512:
        # Pack multiple short rows into each block.  The ideal number of
        # participating lanes depends on both the row width and how many rows
        # are available: small grids want the whole vector covered in one
        # pass, while large grids want more independent rows in flight.
        tail_elems = N - vec_tiles * VW
        lane_floor = max(vec_tiles, tail_elems, 1)
        base_tpr = 1 << max(0, (lane_floor - 1).bit_length())
        small_tpr = max(4, min(base_tpr, 64))
        small_rows = 6 if small_tpr == 64 else (32 if small_tpr <= 16 else 16)
        # At 33..128 rows there is enough parallelism to use a compact block
        # without the idle-lane and reduction cost of a wider wave group.
        lane_tpr = max(4, base_tpr)
        def warp_spec(tpr, rows):
            return ("warp", tpr, rows, 1 if tpr == 16 else 0)

        if base_tpr == 16:
            mid_spec = warp_spec(lane_tpr, 48)
        elif base_tpr <= 8:
            mid_spec = warp_spec(lane_tpr, 32)
        else:
            mid_spec = ("block", 64, 2)
        if vec_tiles <= 16:
            large_spec = warp_spec(lane_tpr, 8 if base_tpr <= 8 else 4)
        elif vec_tiles <= 48:
            large_spec = ("warp", 64, 2)
        else:
            large_spec = ("warp", 64, 4)

        band_specs = []
        for upper, spec in ((33, warp_spec(small_tpr, small_rows)), (129, mid_spec), (None, large_spec)):
            if band_specs and band_specs[-1][0] == spec:
                band_specs[-1][1] = upper
            else:
                band_specs.append([spec, upper])

        def make_warp_or_block(spec):
            if spec[0] == "warp":
                return warp_builder(spec[1], spec[2], wpe=(spec[3] if len(spec) > 3 else 0))
            return block_builder(spec[1], wpe=(spec[2] if len(spec) > 2 else 0))

        variants = [make_warp_or_block(spec) for spec, _ in band_specs]
        thresholds = [upper for _, upper in band_specs[:-1] if upper is not None]
    elif N <= 1024:
        # One pass over the row here is already a full block reduction; the
        # block kernel is both faster and more stable than a long serial wave.
        # Once M is large enough to fill the machine, two rows per 128-thread
        # block and a wave-only reduction beat the one-row block kernels.
        small_block_bt = 256 if elem_bits == 32 or vec_tiles <= 80 else 128
        small_block_wpe = 4 if elem_bits == 32 or small_block_bt == 128 else 0
        # For compact 16-bit rows the two-row warp kernel wins as soon as
        # there are enough rows to fill the machine; wider rows need the
        # one-row block reduction for longer.
        if elem_bits <= 16 and vec_tiles <= 120:
            medium_warp_start = 257
        elif elem_bits == 32:
            # fp32 medium rows keep the block kernel through M~1024; the
            # warp path loses badly for a few hundred rows.
            medium_warp_start = 1025
        else:
            medium_warp_start = 513
        variants = [block_builder(small_block_bt, wpe=small_block_wpe), warp_builder(64, 2)]
        thresholds = [medium_warp_start]
    else:
        small_bt = _block_bt_for_small_m(vec_tiles, elem_bits)
        # Throughput-sized grids favor a smaller block until the row no longer
        # fits in one or two 512-lane passes.
        if vec_tiles <= 384:
            large_bt = 128
        elif vec_tiles <= 768:
            large_bt = 256
        elif elem_bits <= 16 and 2000 <= vec_tiles <= 2200:
            # Two full 1024-lane passes are best right around 2K vectors.
            large_bt = 1024
        elif elem_bits <= 16 and vec_tiles >= 1330:
            # Three 768-lane passes beat four 512-lane passes here; just
            # below this point the 512-thread kernel remains faster.
            large_bt = 768
        elif elem_bits == 32 and 3800 <= vec_tiles < 4000:
            # Three 768-lane passes win just below the 4K-vector 1024 path.
            large_bt = 768
        elif elem_bits == 32 and 4000 <= vec_tiles <= 4200:
            # Around 4K vectors one full 1024-lane pass wins; beyond this
            # range the narrower blocks avoid cache-set aliasing cliffs.
            large_bt = 1024
        elif elem_bits == 32 and 4200 < vec_tiles <= 4576:
            # Immediately above the 4K-vector band, 768 lanes are best.
            large_bt = 768
        elif elem_bits == 32 and 5500 <= vec_tiles <= 6200:
            # A three-pass 768-lane block is consistently best in this band.
            large_bt = 768
        else:
            large_bt = 512
        # At this exact 16-bit width the 1024-lane kernel is already best
        # once there are more than ~256 rows; do not wait for the usual 513.
        small_m_cutoff = 257 if elem_bits <= 16 and 2000 <= vec_tiles <= 2200 else 513
        # Past roughly 9/8 of the last-level cache, streaming input reads
        # avoids allocating lines that the kernel will not reuse.
        stream_bytes_per_row = max(1, N * (3 * (elem_bits // 8) + 1))
        # The 16-bit paths touch three vector streams plus the byte output and
        # start to lose to streaming a little earlier than the fp32 path.
        nt_cache_bytes = 272 * 1024 * 1024
        nt_min_rows = nt_cache_bytes // stream_bytes_per_row + 1
        if dtype_str == "f32":
            small_wpe = 4
            large_wpe = 4
        else:
            # Slightly higher occupancy helps the 512-thread 16-bit variants;
            # narrower blocks and the warp path retain their default settings.
            small_wpe = 2 if small_bt == 512 else 0
            large_wpe = 2 if large_bt == 512 else 0
        if elem_bits <= 16 and vec_tiles <= 192:
            # For narrow 16-bit throughput rows, four concurrent rows per
            # block beat the serial one-row block reduction.
            variants = [
                block_builder(small_bt, wpe=small_wpe),
                warp_builder(64, 4),
                block_builder(large_bt, nt=True, wpe=large_wpe),
            ]
            thresholds = [513, max(514, nt_min_rows)]
        elif nt_min_rows <= 513:
            variants = [block_builder(small_bt, wpe=small_wpe),
                        block_builder(large_bt, nt=True, wpe=large_wpe)]
            thresholds = [513]
        elif large_bt == small_bt:
            variants = [block_builder(small_bt, wpe=small_wpe),
                        block_builder(large_bt, nt=True, wpe=large_wpe)]
            thresholds = [nt_min_rows]
        else:
            variants = [
                block_builder(small_bt, wpe=small_wpe),
                block_builder(large_bt, wpe=large_wpe),
                block_builder(large_bt, nt=True, wpe=large_wpe),
            ]
            thresholds = [small_m_cutoff, nt_min_rows]

    if len(variants) == 1:
        return variants[0]()

    # Materialise the launchers now; the dispatch body only selects one.
    launchers = [make() for make in variants]
    # Keep the sub-launchers as individual closure variables.  The JIT cache
    # key follows callable free variables, so a list would hide which compiled
    # geometry a dispatch was built for and let two N specializations share a
    # stale launcher.
    launch_first = launchers[0]
    launch_second = launchers[1]
    first_limit = fx.Int32(thresholds[0])

    if is_smooth:

        if len(launchers) == 2:

            @flyc.jit
            def _dispatch_launch(
                Input: fx.Tensor,
                ResidualIn: fx.Tensor,
                Gamma: fx.Tensor,
                Beta: fx.Tensor,
                XScale: fx.Tensor,
                Output: fx.Tensor,
                ResidualOut: fx.Tensor,
                YScale: fx.Tensor,
                m_in: fx.Int32,
                stream: fx.Stream = fx.Stream(None),
            ):
                @flyc.jit
                def dispatch_m():
                    if m_in < first_limit:
                        launch_first(Input, ResidualIn, Gamma, Beta, XScale, Output, ResidualOut, YScale, m_in, stream)
                    else:
                        launch_second(Input, ResidualIn, Gamma, Beta, XScale, Output, ResidualOut, YScale, m_in, stream)

                dispatch_m()

            return _dispatch_launch

        launch_third = launchers[2]
        second_limit = fx.Int32(thresholds[1])

        @flyc.jit
        def _dispatch_launch(
            Input: fx.Tensor,
            ResidualIn: fx.Tensor,
            Gamma: fx.Tensor,
            Beta: fx.Tensor,
            XScale: fx.Tensor,
            Output: fx.Tensor,
            ResidualOut: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            @flyc.jit
            def dispatch_m():
                if m_in < first_limit:
                    launch_first(Input, ResidualIn, Gamma, Beta, XScale, Output, ResidualOut, YScale, m_in, stream)
                else:
                    if m_in < second_limit:
                        launch_second(Input, ResidualIn, Gamma, Beta, XScale, Output, ResidualOut, YScale, m_in, stream)
                    else:
                        launch_third(Input, ResidualIn, Gamma, Beta, XScale, Output, ResidualOut, YScale, m_in, stream)

            dispatch_m()

        return _dispatch_launch

    if len(launchers) == 2:

        @flyc.jit
        def _dispatch_launch_dynamic(
            Input: fx.Tensor,
            ResidualIn: fx.Tensor,
            Gamma: fx.Tensor,
            Beta: fx.Tensor,
            Output: fx.Tensor,
            ResidualOut: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            @flyc.jit
            def dispatch_m():
                if m_in < first_limit:
                    launch_first(Input, ResidualIn, Gamma, Beta, Output, ResidualOut, YScale, m_in, stream)
                else:
                    launch_second(Input, ResidualIn, Gamma, Beta, Output, ResidualOut, YScale, m_in, stream)

            dispatch_m()

        return _dispatch_launch_dynamic

    launch_third = launchers[2]
    second_limit = fx.Int32(thresholds[1])

    @flyc.jit
    def _dispatch_launch_dynamic(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        Output: fx.Tensor,
        ResidualOut: fx.Tensor,
        YScale: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        @flyc.jit
        def dispatch_m():
            if m_in < first_limit:
                launch_first(Input, ResidualIn, Gamma, Beta, Output, ResidualOut, YScale, m_in, stream)
            else:
                if m_in < second_limit:
                    launch_second(Input, ResidualIn, Gamma, Beta, Output, ResidualOut, YScale, m_in, stream)
                else:
                    launch_third(Input, ResidualIn, Gamma, Beta, Output, ResidualOut, YScale, m_in, stream)

        dispatch_m()

    return _dispatch_launch_dynamic


def _build_fused_add_layernorm_quant_module(
    N: int,
    dtype_str: str,
    *,
    is_smooth: bool,
    quant_dtype_str: str = "i8",
    eps: float = EPS,
):
    return _make_fused_add_layernorm_quant_dispatch(
        N,
        dtype_str,
        is_smooth=is_smooth,
        quant_dtype_str=quant_dtype_str,
        eps=eps,
    )


def build_layernorm_dynamicquant_module(
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
    eps: float = EPS,
):
    return _build_layernorm_quant_module(
        N,
        dtype_str,
        is_smooth=False,
        quant_dtype_str=quant_dtype_str,
        eps=eps,
    )


def build_layernorm_smoothquant_module(
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
    eps: float = EPS,
):
    return _build_layernorm_quant_module(
        N,
        dtype_str,
        is_smooth=True,
        quant_dtype_str=quant_dtype_str,
        eps=eps,
    )


def build_fused_add_layernorm_dynamicquant_module(
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
    eps: float = EPS,
):
    return _build_fused_add_layernorm_quant_module(
        N,
        dtype_str,
        is_smooth=False,
        quant_dtype_str=quant_dtype_str,
        eps=eps,
    )


def build_fused_add_layernorm_smoothquant_module(
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
    eps: float = EPS,
):
    return _build_fused_add_layernorm_quant_module(
        N,
        dtype_str,
        is_smooth=True,
        quant_dtype_str=quant_dtype_str,
        eps=eps,
    )


# =====================================================================
# Python wrappers + autograd (quack-aligned). PR 3: plain layernorm.
# =====================================================================
if torch is not None:

    def _torch_dtype_to_str(dt) -> str:
        if dt == torch.float32:
            return "f32"
        if dt == torch.float16:
            return "f16"
        if dt == torch.bfloat16:
            return "bf16"
        raise ValueError(f"unsupported torch dtype: {dt}")

    # Compiled-fn caches. Keys include device: a compiled function is bound to
    # the device/context it was built on, so reusing it on another GPU faults.
    # eps is a compile-time kernel constant, so it is part of the fwd key too.
    _FWD_CACHE: dict = {}
    _BWD_CACHE: dict = {}

    def _get_fwd_compiled(x, weight, bias, out, mean, rstd, M, N, dtype_str, store_stats, eps, stream):
        key = (N, dtype_str, store_stats, float(eps), x.device)
        entry = _FWD_CACHE.get(key)
        if entry is None:
            launch_fn = build_layernorm_module(N, dtype_str, store_stats=store_stats, eps=eps)
            elem_dtype = dtype_to_elem_type(dtype_str)
            x_ptr = flyc.from_c_void_p(elem_dtype, x.data_ptr())
            weight_ptr = flyc.from_c_void_p(elem_dtype, weight.data_ptr())
            bias_ptr = flyc.from_c_void_p(elem_dtype, bias.data_ptr())
            out_ptr = flyc.from_c_void_p(elem_dtype, out.data_ptr())
            if store_stats:
                mean_ptr = flyc.from_c_void_p(fx.Float32, mean.data_ptr())
                rstd_ptr = flyc.from_c_void_p(fx.Float32, rstd.data_ptr())
                compiled = flyc.compile(
                    launch_fn,
                    x_ptr,
                    weight_ptr,
                    bias_ptr,
                    out_ptr,
                    mean_ptr,
                    rstd_ptr,
                    M,
                    stream,
                )
            else:
                compiled = flyc.compile(launch_fn, x_ptr, weight_ptr, bias_ptr, out_ptr, M, stream)
            _FWD_CACHE[key] = compiled
            entry = compiled
        return entry

    def layernorm_fwd(x, weight, bias, eps=EPS, store_stats=False):
        """Forward LayerNorm. Returns (out, mean, rstd). eps is baked into the kernel."""
        assert x.dim() == 2, "layernorm_fwd expects a 2D (M, N) input"
        assert (
            x.is_contiguous() and weight.is_contiguous() and bias.is_contiguous()
        ), "layernorm_fwd expects contiguous inputs"
        # The kernel reads x/weight/bias with a single elem dtype derived from x;
        # a mismatch would silently bit-reinterpret weight/bias bytes.
        assert (
            x.dtype == weight.dtype == bias.dtype
        ), f"x/weight/bias dtypes must match, got {x.dtype}/{weight.dtype}/{bias.dtype}"
        assert weight.device == x.device and bias.device == x.device, "x/weight/bias must be on the same device"
        M, N = x.shape
        assert weight.dim() == 1 and bias.dim() == 1, "weight/bias must be 1D affine vectors"
        assert weight.shape[0] == N and bias.shape[0] == N, "weight/bias length must equal x last dim (N)"
        out = torch.empty_like(x)
        mean = torch.empty((M,), device=x.device, dtype=torch.float32) if store_stats else None
        rstd = torch.empty((M,), device=x.device, dtype=torch.float32) if store_stats else None
        dtype_str = _torch_dtype_to_str(x.dtype)
        # Bind compile + launch to the tensors' device so the compiled kernel and
        # the stream belong to the right GPU/context (multi-GPU correctness).
        with torch.cuda.device(x.device):
            stream = torch.cuda.current_stream()
            compiled = _get_fwd_compiled(x, weight, bias, out, mean, rstd, M, N, dtype_str, store_stats, eps, stream)
            if store_stats:
                compiled(
                    x.data_ptr(),
                    weight.data_ptr(),
                    bias.data_ptr(),
                    out.data_ptr(),
                    mean.data_ptr(),
                    rstd.data_ptr(),
                    M,
                    stream,
                )
            else:
                compiled(x.data_ptr(), weight.data_ptr(), bias.data_ptr(), out.data_ptr(), M, stream)
        return out, mean, rstd

    def layernorm_bwd(x, weight, dout, mean, rstd, eps=EPS):
        """Backward LayerNorm. Returns (dx, dweight, dbias) cast to weight dtype.

        eps is not used directly here — it is already baked into `rstd`/`mean` by
        the forward — but is accepted so callers can pass it symmetrically.
        """
        assert x.dim() == 2, "layernorm_bwd expects a 2D (M, N) input"
        assert all(t.is_contiguous() for t in (x, weight, dout, mean, rstd)), "layernorm_bwd expects contiguous inputs"
        assert (
            x.dtype == weight.dtype == dout.dtype
        ), f"x/weight/dout dtypes must match, got {x.dtype}/{weight.dtype}/{dout.dtype}"
        M, N = x.shape
        assert dout.shape == x.shape, "dout shape must equal x shape"
        assert weight.dim() == 1 and weight.shape[0] == N, "weight must be a length-N 1D affine vector"
        assert mean.numel() == M and rstd.numel() == M, "mean/rstd length must equal x rows (M)"
        assert all(
            t.device == x.device for t in (weight, dout, mean, rstd)
        ), "x/weight/dout/mean/rstd must be on the same device"
        # mean/rstd are per-row fp32 stats saved by the forward; the kernel reads
        # them with an fp32 copy atom, so a non-fp32 dtype would be misread.
        assert mean.dtype == torch.float32 and rstd.dtype == torch.float32, "mean/rstd must be fp32"
        dtype_str = _torch_dtype_to_str(x.dtype)
        elem_dtype = dtype_to_elem_type(dtype_str)
        dx = torch.empty_like(x)
        dweight = torch.zeros((N,), device=x.device, dtype=torch.float32)
        dbias = torch.zeros((N,), device=x.device, dtype=torch.float32)
        key = (N, dtype_str, x.device)
        # Bind compile + launch to the tensors' device (multi-GPU correctness).
        with torch.cuda.device(x.device):
            stream = torch.cuda.current_stream()
            compiled = _BWD_CACHE.get(key)
            if compiled is None:
                launch_fn = build_layernorm_bwd_module(N, dtype_str)
                # flyc.compile executes the kernel once during tracing, which would
                # accumulate into DGamma/DBias; zero them AFTER compiling.
                x_ptr = flyc.from_c_void_p(elem_dtype, x.data_ptr())
                weight_ptr = flyc.from_c_void_p(elem_dtype, weight.data_ptr())
                dout_ptr = flyc.from_c_void_p(elem_dtype, dout.data_ptr())
                mean_ptr = flyc.from_c_void_p(fx.Float32, mean.data_ptr())
                rstd_ptr = flyc.from_c_void_p(fx.Float32, rstd.data_ptr())
                dx_ptr = flyc.from_c_void_p(elem_dtype, dx.data_ptr())
                dweight_ptr = flyc.from_c_void_p(fx.Float32, dweight.data_ptr())
                dbias_ptr = flyc.from_c_void_p(fx.Float32, dbias.data_ptr())
                compiled = flyc.compile(
                    launch_fn,
                    x_ptr,
                    weight_ptr,
                    dout_ptr,
                    mean_ptr,
                    rstd_ptr,
                    dx_ptr,
                    dweight_ptr,
                    dbias_ptr,
                    M,
                    stream,
                )
                _BWD_CACHE[key] = compiled
            dweight.zero_()
            dbias.zero_()
            compiled(
                x.data_ptr(),
                weight.data_ptr(),
                dout.data_ptr(),
                mean.data_ptr(),
                rstd.data_ptr(),
                dx.data_ptr(),
                dweight.data_ptr(),
                dbias.data_ptr(),
                M,
                stream,
            )
        return dx, dweight.to(weight.dtype), dbias.to(weight.dtype)

    class LayerNormFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias, eps):
            need_grad = x.requires_grad or weight.requires_grad or bias.requires_grad
            out, mean, rstd = layernorm_fwd(x, weight, bias, eps=eps, store_stats=need_grad)
            ctx.save_for_backward(x, weight, mean, rstd)
            ctx.eps = eps
            return out

        @staticmethod
        def backward(ctx, dout):
            x, weight, mean, rstd = ctx.saved_tensors
            dx, dw, db = layernorm_bwd(x, weight, dout.contiguous(), mean, rstd, eps=ctx.eps)
            return dx, dw, db, None

    def layernorm(x, weight, bias, eps=EPS):
        """Public entry: plain LayerNorm with autograd."""
        assert weight is not None and bias is not None, "layernorm requires explicit weight and bias"
        N = weight.shape[-1]
        assert x.shape[-1] == N, f"x last dim {x.shape[-1]} != weight length {N}"
        assert (
            x.dtype == weight.dtype == bias.dtype
        ), f"x/weight/bias dtypes must match, got {x.dtype}/{weight.dtype}/{bias.dtype}"
        # Raw-pointer kernels reconstruct dense rows and stride-1 affine
        # vectors, so materialize those layouts at the public boundary.
        x_flat = x.reshape(-1, N).contiguous()
        weight_flat = weight.contiguous()
        bias_flat = bias.contiguous()
        out_flat = LayerNormFunction.apply(x_flat, weight_flat, bias_flat, eps)
        return out_flat.reshape(x.shape)
