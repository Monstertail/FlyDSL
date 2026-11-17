# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""RMSNorm kernel builder using the @flyc.kernel API.

RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma

Two paths:
  - Vector path (f16/bf16, N >= 8): 128-bit vec8 copies plus a scalar tail.
  - Scalar path (f32 or N < 8): guarded elementwise copies.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp
from flydsl.runtime.device import get_rocm_arch
from kernels.common.kernels_common import dtype_to_elem_type

# Backward builders live in their own module (see review on #800); re-exported
# here so existing importers (tests, callers) keep working unchanged.
from kernels.norm.rmsnorm_bwd_kernel import (  # noqa: E402,F401
    build_fused_add_rmsnorm_bwd_module,
    build_fused_add_rmsnorm_bwd_two_stage_module,
    build_rmsnorm_bwd_module,
    build_rmsnorm_bwd_two_stage_module,
    is_rmsnorm_bwd_two_stage_vec_config,
)
from kernels.norm.rmsnorm_common import (
    BLOCK_THREADS,
    EPS,
    VEC_WIDTH,
    WARP_SIZE,
)
from kernels.norm.rmsnorm_common import load_vec as _load_vec
from kernels.norm.rmsnorm_common import load_weight_vec as _load_weight_vec
from kernels.norm.rmsnorm_common import resolve_rmsnorm_weight_dtype as _resolve_rmsnorm_weight_dtype
from kernels.norm.rmsnorm_common import store_vec as _store_vec
from kernels.norm.rmsnorm_common import to_elem_scalar as _to_elem_scalar
from kernels.norm.rmsnorm_common import to_elem_vec as _to_elem_vec
from kernels.norm.rmsnorm_common import validate_norm_operand_dtypes as _validate_norm_operand_dtypes
from kernels.norm.rmsnorm_common import weight_vec_width as _weight_vec_width

try:
    import torch
except ImportError:
    torch = None

KERNEL_NAME = "rmsnorm"

SMALL_N_THRESHOLD = 2048
FWD_MULTI_ROW_THRESHOLD = 8192
FWD_THROUGHPUT_MIN_ROWS = 8 * WARP_SIZE
FWD_THROUGHPUT_MAX_VEC_ITERS = 8
DEFAULT_BLOCK_THREADS = BLOCK_THREADS
WIDE_ROW_MIN_N = 8192
WIDE_ROW_BLOCK_THREADS = 512

# Both CDNA3 and CDNA4 share a 256MB last-level cache. Non-temporal streaming
# only pays off once input plus output stop fitting in it; below that it throws
# away reuse the cache would have served. The two policies measure as a tie at
# 1.125x LLC and streaming wins outright past it, uniformly across row widths.
LLC_BYTES = 256 * 1024 * 1024
NT_MIN_WORKING_SET = LLC_BYTES * 9 // 8

# Fused add + quant moves 2+2 bytes per element in and 2+1 out.
FUSED_ADD_QUANT_BYTES_PER_ELEM = 7


def default_block_threads(N: int, arch=None) -> int:
    """Wide rows hold fewer resident vectors per thread at 512 than at 256."""
    if arch is None:
        arch = get_rocm_arch()
    if str(arch).startswith("gfx95") and N >= WIDE_ROW_MIN_N:
        return WIDE_ROW_BLOCK_THREADS
    return DEFAULT_BLOCK_THREADS


def _quant_dtype_to_elem_type(dtype_str: str):
    if dtype_str in ("i8", "int8"):
        return fx.Int8
    raise ValueError(f"unsupported quant dtype: {dtype_str!r} (expected 'i8' or 'int8')")


def _quant_dtype_max(dtype_str: str) -> float:
    if dtype_str in ("i8", "int8"):
        return 127.0
    raise ValueError(f"unsupported quant dtype: {dtype_str!r} (expected 'i8' or 'int8')")


def build_rmsnorm_module(
    N: int,
    dtype_str: str,
    store_rstd: bool = False,
    eps: float = EPS,
    BLOCK_THREADS: int | None = None,
    weight_dtype_str: str | None = None,
    _enable_throughput_dispatch: bool = True,
    _force_nt: bool | None = None,
):
    weight_dtype_str = _resolve_rmsnorm_weight_dtype(dtype_str, weight_dtype_str)
    arch = get_rocm_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")
    if BLOCK_THREADS is None:
        BLOCK_THREADS = default_block_threads(N, arch)

    # BLOCK_THREADS controls storage, tiling, and launch geometry.
    elem_bits = 32 if dtype_str == "f32" else 16
    weight_elem_bits = 32 if weight_dtype_str == "f32" else 16
    USE_VEC_N = elem_bits <= 16 and N >= VEC_WIDTH
    VEC_TILES = N // VEC_WIDTH if USE_VEC_N else 0
    VEC_ELEMS = VEC_TILES * VEC_WIDTH
    TAIL_ELEMS = N - VEC_ELEMS
    NUM_VEC_ITERS = (VEC_TILES + BLOCK_THREADS - 1) // BLOCK_THREADS if USE_VEC_N else 0
    _kernel_kwargs = {} if BLOCK_THREADS <= 256 else {"known_block_size": [BLOCK_THREADS, 1, 1]}

    # Stream large BF16 rows non-temporally. x stays in registers across both
    # passes, so x and y are each touched once and neither gains from being
    # cached; letting x fills allocate in the memory-side cache only evicts y's
    # pending stores. gfx942 streams unconditionally; gfx95x defers the choice
    # to a runtime row-count dispatch below.
    BF16_ROWS = dtype_str == "bf16" and weight_dtype_str == "bf16"
    USE_GFX942_BF16_FAST_PATH = str(arch).startswith("gfx942") and BF16_ROWS and N >= 4096
    # Whether streaming wins is set by the working set, not the row width, so on
    # gfx95x the runtime dispatch below covers every row the plain kernel handles.
    use_nt_dispatch = str(arch).startswith("gfx95") and BF16_ROWS and N > SMALL_N_THRESHOLD and _force_nt is None
    USE_NT = USE_GFX942_BF16_FAST_PATH or bool(_force_nt)
    PRELOAD_GAMMA = USE_GFX942_BF16_FAST_PATH and VEC_TILES >= BLOCK_THREADS and VEC_TILES % BLOCK_THREADS == 0
    INPUT_CACHE_MODIFIER = 2 if USE_NT else 0
    # Streaming the stores too costs 6-8% on gfx95x; letting them settle in LLC
    # drains them to HBM in better-scheduled bursts. gfx942 keeps its own tuning.
    OUTPUT_CACHE_MODIFIER = (
        2 if USE_GFX942_BF16_FAST_PATH and VEC_TILES > WARP_SIZE * FWD_THROUGHPUT_MAX_VEC_ITERS else 0
    )

    one_wave_vec_iters = (VEC_TILES + WARP_SIZE - 1) // WARP_SIZE if USE_VEC_N else 0
    THROUGHPUT_BLOCK_THREADS = WARP_SIZE if 0 < one_wave_vec_iters <= FWD_THROUGHPUT_MAX_VEC_ITERS else BLOCK_THREADS
    use_throughput_dispatch = (
        _enable_throughput_dispatch
        and USE_GFX942_BF16_FAST_PATH
        and N > SMALL_N_THRESHOLD
        and BLOCK_THREADS == DEFAULT_BLOCK_THREADS
        and THROUGHPUT_BLOCK_THREADS < BLOCK_THREADS
    )
    if use_throughput_dispatch:
        # Keep M runtime-only: compile both geometries into one launcher rather
        # than selecting a different cached build for each row count.
        latency_launch = build_rmsnorm_module(
            N,
            dtype_str,
            store_rstd=store_rstd,
            eps=eps,
            BLOCK_THREADS=BLOCK_THREADS,
            weight_dtype_str=weight_dtype_str,
            _enable_throughput_dispatch=False,
        )
        throughput_launch = build_rmsnorm_module(
            N,
            dtype_str,
            store_rstd=store_rstd,
            eps=eps,
            BLOCK_THREADS=THROUGHPUT_BLOCK_THREADS,
            weight_dtype_str=weight_dtype_str,
            _enable_throughput_dispatch=False,
        )

        if store_rstd:

            @flyc.jit
            def launch_rmsnorm(
                Input: fx.Tensor,
                Gamma: fx.Tensor,
                Output: fx.Tensor,
                Rstd: fx.Tensor,
                m_in: fx.Int32,
                stream: fx.Stream = fx.Stream(None),
            ):
                def launch_throughput():
                    throughput_launch(Input, Gamma, Output, Rstd, m_in, stream)

                def launch_default():
                    latency_launch(Input, Gamma, Output, Rstd, m_in, stream)

                @flyc.jit
                def dispatch():
                    if m_in >= fx.Int32(FWD_THROUGHPUT_MIN_ROWS):
                        launch_throughput()
                    else:
                        launch_default()

                dispatch()

            return launch_rmsnorm

        @flyc.jit
        def launch_rmsnorm(
            Input: fx.Tensor,
            Gamma: fx.Tensor,
            Output: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            def launch_throughput():
                throughput_launch(Input, Gamma, Output, m_in, stream)

            def launch_default():
                latency_launch(Input, Gamma, Output, m_in, stream)

            @flyc.jit
            def dispatch():
                if m_in >= fx.Int32(FWD_THROUGHPUT_MIN_ROWS):
                    launch_throughput()
                else:
                    launch_default()

            dispatch()

        return launch_rmsnorm

    if use_nt_dispatch:
        # Keep M runtime-only: compile both cache policies into one launcher
        # rather than selecting a different cached build for each row count.
        _nt_kwargs = dict(
            store_rstd=store_rstd,
            eps=eps,
            BLOCK_THREADS=BLOCK_THREADS,
            weight_dtype_str=weight_dtype_str,
            _enable_throughput_dispatch=False,
        )
        cached_launch = build_rmsnorm_module(N, dtype_str, _force_nt=False, **_nt_kwargs)
        streaming_launch = build_rmsnorm_module(N, dtype_str, _force_nt=True, **_nt_kwargs)
        NT_MIN_ROWS = NT_MIN_WORKING_SET // (N * 4)

        if store_rstd:

            @flyc.jit
            def launch_rmsnorm(
                Input: fx.Tensor,
                Gamma: fx.Tensor,
                Output: fx.Tensor,
                Rstd: fx.Tensor,
                m_in: fx.Int32,
                stream: fx.Stream = fx.Stream(None),
            ):
                def launch_streaming():
                    streaming_launch(Input, Gamma, Output, Rstd, m_in, stream)

                def launch_cached():
                    cached_launch(Input, Gamma, Output, Rstd, m_in, stream)

                @flyc.jit
                def dispatch():
                    if m_in > fx.Int32(NT_MIN_ROWS):
                        launch_streaming()
                    else:
                        launch_cached()

                dispatch()

            return launch_rmsnorm

        @flyc.jit
        def launch_rmsnorm(
            Input: fx.Tensor,
            Gamma: fx.Tensor,
            Output: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            def launch_streaming():
                streaming_launch(Input, Gamma, Output, m_in, stream)

            def launch_cached():
                cached_launch(Input, Gamma, Output, m_in, stream)

            @flyc.jit
            def dispatch():
                if m_in > fx.Int32(NT_MIN_ROWS):
                    launch_streaming()
                else:
                    launch_cached()

            dispatch()

        return launch_rmsnorm

    BlockReduce = fx.coop.BlockReduce[fx.Float32, BLOCK_THREADS]
    SharedStorage = BlockReduce.SharedStorage

    @flyc.kernel(**_kernel_kwargs)
    def rmsnorm_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Rstd: fx.Tensor,
        Output: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        weight_elem_dtype = dtype_to_elem_type(weight_dtype_str)
        fm_fast = arith.FastMathFlags.fast
        eps_c = eps
        n_float = float(N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        if const_expr(store_rstd):
            Rstd_buf = fx.rocdl.make_buffer_tensor(Rstd)

        def block_reduce_add(val):
            with fx.fastmath(fm_fast):
                return BlockReduce(val, fx.ReductionOp.ADD, storage=lds)

        # Vector path for the f16/bf16 vec8 prefix plus an optional scalar tail.
        if const_expr(USE_VEC_N):
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(_weight_vec_width(weight_dtype_str), 1))

            copy_atom = fx.make_copy_atom(
                fx.rocdl.BufferCopy128b(INPUT_CACHE_MODIFIER),
                elem_bits,
            )
            output_copy_atom = fx.make_copy_atom(
                fx.rocdl.BufferCopy128b(OUTPUT_CACHE_MODIFIER),
                elem_bits,
            )
            gamma_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), weight_elem_bits)

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f
            in_local = []
            gamma_local = []

            # Pass 1: load + cache + sumsq
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                is_valid = idx < VEC_TILES
                idx_safe = is_valid.select(idx, 0)
                vec = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx_safe)
                in_local.append(vec)
                if const_expr(PRELOAD_GAMMA):
                    gamma_local.append(_load_vec(gamma_copy_atom, VEC_WIDTH, weight_elem_dtype, gamma_div, idx_safe))
                x = vec.to(fx.Float32)

                x2 = x * x
                red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + is_valid.select(red2, c_zero_f)

            if const_expr(TAIL_ELEMS > 0):
                if tid < TAIL_ELEMS:
                    tail_idx = tid + VEC_ELEMS
                    x_tail_e = row_in[tail_idx]
                    x_tail = x_tail_e.to(fx.Float32)
                    thread_sumsq = thread_sumsq + x_tail * x_tail

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            if const_expr(store_rstd):
                if tid == 0:
                    Rstd_buf[bid] = rrms

            # Pass 2: normalize + gamma + store (reuse cached input)
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                if idx < VEC_TILES:
                    if const_expr(PRELOAD_GAMMA):
                        g = gamma_local[tile_i].to(fx.Float32)
                    else:
                        g = _load_weight_vec(gamma_copy_atom, weight_dtype_str, weight_elem_dtype, gamma_div, idx)
                    x = in_local[tile_i].to(fx.Float32)

                    y = (x * rrms) * g
                    out_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, y)
                    _store_vec(output_copy_atom, VEC_WIDTH, elem_dtype, out_e, out_div, idx)

            if const_expr(TAIL_ELEMS > 0):
                if tid < TAIL_ELEMS:
                    tail_idx = tid + VEC_ELEMS
                    x_tail_e = row_in[tail_idx]
                    g_tail_e = Gamma_buf[tail_idx]
                    x_tail = x_tail_e.to(fx.Float32)
                    g_tail = g_tail_e if weight_dtype_str == "f32" else g_tail_e.to(fx.Float32)
                    y_tail = (x_tail * rrms) * g_tail
                    y_tail_e = _to_elem_scalar(dtype_str, elem_dtype, y_tail)
                    row_out[tail_idx] = y_tail_e

        else:
            # Scalar fallback for arbitrary N.
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = row_in[idx_safe]
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                x2 = x * x
                x2_safe = is_valid.select(x2, c_zero_f)
                thread_sumsq = thread_sumsq + x2_safe

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            if const_expr(store_rstd):
                if tid == 0:
                    Rstd_buf[bid] = rrms

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    x_e = row_in[idx]
                    g_e = Gamma_buf[idx]
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if weight_dtype_str == "f32" else g_e.to(fx.Float32)
                    norm = x * rrms
                    y = norm * g
                    y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                    row_out[idx] = y_e

    if N <= SMALL_N_THRESHOLD:
        return _build_rmsnorm_small_n_module(
            N,
            dtype_str,
            store_rstd,
            eps,
            weight_dtype_str,
            one_row_kernel=rmsnorm_kernel,
            one_row_block_threads=BLOCK_THREADS,
        )

    if store_rstd:

        @flyc.jit
        def launch_rmsnorm(
            Input: fx.Tensor,
            Gamma: fx.Tensor,
            Output: fx.Tensor,
            Rstd: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = rmsnorm_kernel(Input, Gamma, Rstd, Output)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_rmsnorm

    @flyc.jit
    def launch_rmsnorm(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # store_rstd=False path: the Rstd slot is an unused placeholder here, so
        # we pass Gamma to fill the argument (it is never dereferenced in-kernel).
        launcher = rmsnorm_kernel(Input, Gamma, Gamma, Output)
        launcher.launch(
            grid=(m_in, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm


@flyc.jit
def rmsnorm_direct(
    Input: fx.Tensor,
    Gamma: fx.Tensor,
    Output: fx.Tensor,
    m_in: fx.Int32,
    N: fx.Constexpr[int],
    dtype_str: fx.Constexpr[str],
    BLOCK_THREADS: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
    weight_dtype_str: fx.Constexpr[str] = "",
):
    """Specialize the existing RMSNorm factory through JIT Constexpr inputs."""
    resolved_weight_dtype_str = dtype_str if weight_dtype_str == "" else weight_dtype_str
    launch = build_rmsnorm_module(
        N,
        dtype_str,
        BLOCK_THREADS=BLOCK_THREADS,
        weight_dtype_str=resolved_weight_dtype_str,
        _enable_throughput_dispatch=False,
    )
    launch(Input, Gamma, Output, m_in, stream)


def _build_rmsnorm_small_n_module(
    N: int,
    dtype_str: str,
    store_rstd: bool = False,
    eps: float = EPS,
    weight_dtype_str: str | None = None,
    *,
    one_row_kernel,
    one_row_block_threads: int,
):
    weight_dtype_str = _resolve_rmsnorm_weight_dtype(dtype_str, weight_dtype_str)
    BLOCK_N = 1 << (N - 1).bit_length()
    BLOCK_M = max(min(16384 // BLOCK_N, 32), 8)
    THREADS_PER_ROW = min(WARP_SIZE, 1024 // BLOCK_M)
    BLOCK_THREADS_SPECIAL = BLOCK_M * THREADS_PER_ROW
    elem_bits = 32 if dtype_str == "f32" else 16
    weight_elem_bits = 32 if weight_dtype_str == "f32" else 16
    USE_VEC_SMALL_N = elem_bits <= 16 and N >= VEC_WIDTH
    VEC_TILES = N // VEC_WIDTH if USE_VEC_SMALL_N else 0
    VEC_ELEMS = VEC_TILES * VEC_WIDTH
    TAIL_ELEMS = N - VEC_ELEMS
    NUM_VEC_ROW_ITERS = (VEC_TILES + THREADS_PER_ROW - 1) // THREADS_PER_ROW if USE_VEC_SMALL_N else 0
    arch = get_rocm_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    @flyc.kernel(known_block_size=[BLOCK_THREADS_SPECIAL, 1, 1])
    def rmsnorm_large_m_small_n_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Rstd: fx.Tensor,
        Output: fx.Tensor,
        MIn: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        lane = tid % THREADS_PER_ROW
        row_local = tid // THREADS_PER_ROW
        row = bid * BLOCK_M + row_local

        if row < MIn:
            elem_dtype = dtype_to_elem_type(dtype_str)
            weight_elem_dtype = dtype_to_elem_type(weight_dtype_str)
            fm_fast = arith.FastMathFlags.fast
            eps_c = eps
            n_float = float(N)

            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            if const_expr(store_rstd):
                Rstd_buf = fx.rocdl.make_buffer_tensor(Rstd)

            row_in = fx.slice(Input_buf, (row, None))
            row_out = fx.slice(Output_buf, (row, None))

            def group_reduce_add(x):
                with fx.fastmath(fm_fast):
                    return fx.coop.warp_reduce(x, fx.ReductionOp.ADD, width=THREADS_PER_ROW)

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f

            if const_expr(USE_VEC_SMALL_N):
                in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
                gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(_weight_vec_width(weight_dtype_str), 1))
                out_div = fx.logical_divide(row_out, fx.make_layout(VEC_WIDTH, 1))

                copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
                gamma_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), weight_elem_bits)

                in_local = []
                for tile_i in range_constexpr(NUM_VEC_ROW_ITERS):
                    idx = lane + tile_i * THREADS_PER_ROW
                    is_valid = idx < VEC_TILES
                    idx_safe = is_valid.select(idx, 0)
                    vec = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx_safe)
                    in_local.append(vec)
                    x = vec.to(fx.Float32)
                    x2 = x * x
                    red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                    thread_sumsq = thread_sumsq + is_valid.select(red2, c_zero_f)

                if const_expr(TAIL_ELEMS > 0):
                    if lane < TAIL_ELEMS:
                        tail_idx = lane + VEC_ELEMS
                        x_tail_e = row_in[tail_idx]
                        x_tail = x_tail_e.to(fx.Float32)
                        thread_sumsq = thread_sumsq + x_tail * x_tail

                sum_sq = group_reduce_add(thread_sumsq)
                mean_sq = sum_sq / n_float
                ms_eps = mean_sq + eps_c
                rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

                if const_expr(store_rstd):
                    if lane == 0:
                        Rstd_buf[row] = rrms

                for tile_i in range_constexpr(NUM_VEC_ROW_ITERS):
                    idx = lane + tile_i * THREADS_PER_ROW
                    if idx < VEC_TILES:
                        g = _load_weight_vec(
                            gamma_copy_atom,
                            weight_dtype_str,
                            weight_elem_dtype,
                            gamma_div,
                            idx,
                        )
                        x = in_local[tile_i].to(fx.Float32)
                        y = (x * rrms) * g
                        y_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, y)
                        _store_vec(copy_atom, VEC_WIDTH, elem_dtype, y_e, out_div, idx)

                if const_expr(TAIL_ELEMS > 0):
                    if lane < TAIL_ELEMS:
                        tail_idx = lane + VEC_ELEMS
                        x_tail_e = row_in[tail_idx]
                        g_tail_e = Gamma_buf[tail_idx]
                        x_tail = x_tail_e.to(fx.Float32)
                        g_tail = g_tail_e if weight_dtype_str == "f32" else g_tail_e.to(fx.Float32)
                        y_tail = (x_tail * rrms) * g_tail
                        y_tail_e = _to_elem_scalar(dtype_str, elem_dtype, y_tail)
                        row_out[tail_idx] = y_tail_e
            else:

                for base_idx_int in range_constexpr(0, BLOCK_N, THREADS_PER_ROW):
                    idx = lane + base_idx_int
                    is_valid = idx < N
                    idx_safe = is_valid.select(idx, 0)
                    x_e = row_in[idx_safe]
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    x2 = x * x
                    thread_sumsq = thread_sumsq + is_valid.select(x2, c_zero_f)

                sum_sq = group_reduce_add(thread_sumsq)
                mean_sq = sum_sq / n_float
                ms_eps = mean_sq + eps_c
                rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

                if const_expr(store_rstd):
                    if lane == 0:
                        Rstd_buf[row] = rrms

                for base_idx_int in range_constexpr(0, BLOCK_N, THREADS_PER_ROW):
                    idx = lane + base_idx_int
                    if idx < N:
                        x_e = row_in[idx]
                        g_e = Gamma_buf[idx]
                        x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                        g = g_e if weight_dtype_str == "f32" else g_e.to(fx.Float32)
                        y = (x * rrms) * g
                        y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                        row_out[idx] = y_e

    if store_rstd:

        @flyc.jit
        def launch_rmsnorm_small_n(
            Input: fx.Tensor,
            Gamma: fx.Tensor,
            Output: fx.Tensor,
            Rstd: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            def launch_one_row():
                launcher = one_row_kernel(Input, Gamma, Rstd, Output)
                launcher.launch(
                    grid=(m_in, 1, 1),
                    block=(one_row_block_threads, 1, 1),
                    stream=stream,
                )

            def launch_multi_row():
                launcher = rmsnorm_large_m_small_n_kernel(Input, Gamma, Rstd, Output, m_in)
                launcher.launch(
                    grid=((m_in + BLOCK_M - 1) // BLOCK_M, 1, 1),
                    block=(BLOCK_THREADS_SPECIAL, 1, 1),
                    stream=stream,
                )

            @flyc.jit
            def dispatch():
                if m_in > fx.Int32(FWD_MULTI_ROW_THRESHOLD):
                    launch_multi_row()
                else:
                    launch_one_row()

            dispatch()

        return launch_rmsnorm_small_n

    @flyc.jit
    def launch_rmsnorm_small_n(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        def launch_one_row():
            # The Rstd slot is unused when store_rstd=False.
            launcher = one_row_kernel(Input, Gamma, Gamma, Output)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(one_row_block_threads, 1, 1),
                stream=stream,
            )

        def launch_multi_row():
            launcher = rmsnorm_large_m_small_n_kernel(Input, Gamma, Gamma, Output, m_in)
            launcher.launch(
                grid=((m_in + BLOCK_M - 1) // BLOCK_M, 1, 1),
                block=(BLOCK_THREADS_SPECIAL, 1, 1),
                stream=stream,
            )

        @flyc.jit
        def dispatch():
            if m_in > fx.Int32(FWD_MULTI_ROW_THRESHOLD):
                launch_multi_row()
            else:
                launch_one_row()

        dispatch()

    return launch_rmsnorm_small_n


def build_fused_add_rmsnorm_module(
    N: int,
    dtype_str: str,
    store_rstd: bool = False,
    eps: float = EPS,
    weight_dtype_str: str | None = None,
):
    weight_dtype_str = _resolve_rmsnorm_weight_dtype(dtype_str, weight_dtype_str)
    arch = get_rocm_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    elem_bits = 32 if dtype_str == "f32" else 16
    weight_elem_bits = 32 if weight_dtype_str == "f32" else 16
    USE_VEC_N = elem_bits <= 16 and N % VEC_WIDTH == 0
    VEC_TILES = N // VEC_WIDTH if USE_VEC_N else 0
    NUM_VEC_ITERS = (VEC_TILES + BLOCK_THREADS - 1) // BLOCK_THREADS if USE_VEC_N else 0

    BlockReduce = fx.coop.BlockReduce[fx.Float32x2, BLOCK_THREADS]
    SharedStorage = BlockReduce.SharedStorage

    @flyc.kernel
    def fused_add_rmsnorm_kernel(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
        ResidualOut: fx.Tensor,
        Rstd: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        weight_elem_dtype = dtype_to_elem_type(weight_dtype_str)
        fm_fast = arith.FastMathFlags.fast
        eps_c = eps
        n_float = float(N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        if const_expr(store_rstd):
            Rstd_buf = fx.rocdl.make_buffer_tensor(Rstd)

        def block_reduce_add(val):
            dummy = fx.Float32(0.0)
            r0, _ = block_reduce_add2(val, dummy)
            return r0

        def block_reduce_add2(val0, val1):
            with fx.fastmath(fm_fast):
                values = fx.Vector.from_elements([val0, val1], fx.Float32)
                total = BlockReduce(values, fx.ReductionOp.ADD, storage=lds)
            return total[0], total[1]

        # Vector path for every complete f16/bf16 vec8 row.
        if const_expr(USE_VEC_N):
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            ResidualIn_buf = fx.rocdl.make_buffer_tensor(ResidualIn)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            ResidualOut_buf = fx.rocdl.make_buffer_tensor(ResidualOut)

            row_in = fx.slice(Input_buf, (bid, None))
            row_residual_in = fx.slice(ResidualIn_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))
            row_residual_out = fx.slice(ResidualOut_buf, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            residual_in_div = fx.logical_divide(row_residual_in, fx.make_layout(VEC_WIDTH, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(VEC_WIDTH, 1))
            residual_out_div = fx.logical_divide(row_residual_out, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(_weight_vec_width(weight_dtype_str), 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            gamma_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), weight_elem_bits)

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f
            thread_dummy = c_zero_f
            add_local = []

            # Pass 1: add + cache + sumsq (also write residual_out)
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                is_valid = idx < VEC_TILES
                idx_safe = is_valid.select(idx, 0)
                x = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx_safe).to(fx.Float32)
                residual = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, residual_in_div, idx_safe).to(fx.Float32)
                added_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, x + residual)
                add_local.append(added_e)
                added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)

                added2 = added * added
                red2 = added2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + is_valid.select(red2, c_zero_f)

                if idx < VEC_TILES:
                    _store_vec(copy_atom, VEC_WIDTH, elem_dtype, added_e, residual_out_div, idx)

            _, sum_sq = block_reduce_add2(thread_dummy, thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            if const_expr(store_rstd):
                if tid == 0:
                    Rstd_buf[bid] = rrms

            # Pass 2: normalize + gamma + store (reuse cached added values)
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                if idx < VEC_TILES:
                    g = _load_weight_vec(gamma_copy_atom, weight_dtype_str, weight_elem_dtype, gamma_div, idx)
                    added = add_local[tile_i] if dtype_str == "f32" else add_local[tile_i].to(fx.Float32)
                    y = (added * rrms) * g
                    y_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, y)
                    _store_vec(copy_atom, VEC_WIDTH, elem_dtype, y_e, out_div, idx)

        else:
            # Scalar fallback for arbitrary N.
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            ResidualIn_buf = fx.rocdl.make_buffer_tensor(ResidualIn)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            ResidualOut_buf = fx.rocdl.make_buffer_tensor(ResidualOut)

            row_in = fx.slice(Input_buf, (bid, None))
            row_residual_in = fx.slice(ResidualIn_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))
            row_residual_out = fx.slice(ResidualOut_buf, (bid, None))

            c_zero_f = fx.Float32(0.0)
            thread_sumsq = c_zero_f

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = row_in[idx_safe]
                residual_e = row_residual_in[idx_safe]
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                residual = residual_e if dtype_str == "f32" else residual_e.to(fx.Float32)
                added_e = _to_elem_scalar(dtype_str, elem_dtype, x + residual)
                if idx < N:
                    row_residual_out[idx] = added_e
                added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                added2 = added * added
                thread_sumsq = thread_sumsq + is_valid.select(added2, c_zero_f)

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            if const_expr(store_rstd):
                if tid == 0:
                    Rstd_buf[bid] = rrms

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    g_e = Gamma_buf[idx]
                    added_e = row_residual_out[idx]
                    g = g_e if weight_dtype_str == "f32" else g_e.to(fx.Float32)
                    added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                    y = (added * rrms) * g
                    y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                    row_out[idx] = y_e

    if store_rstd:

        @flyc.jit
        def launch_fused_add_rmsnorm(
            Input: fx.Tensor,
            ResidualIn: fx.Tensor,
            Gamma: fx.Tensor,
            Output: fx.Tensor,
            ResidualOut: fx.Tensor,
            Rstd: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = fused_add_rmsnorm_kernel(Input, ResidualIn, Gamma, Output, ResidualOut, Rstd)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_fused_add_rmsnorm

    @flyc.jit
    def launch_fused_add_rmsnorm(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
        ResidualOut: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = fused_add_rmsnorm_kernel(Input, ResidualIn, Gamma, Output, ResidualOut, Gamma)
        launcher.launch(
            grid=(m_in, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_fused_add_rmsnorm


def _build_rmsnorm_quant_module(
    N: int,
    dtype_str: str,
    *,
    is_smooth: bool,
    quant_dtype_str: str = "i8",
    eps: float = EPS,
):
    elem_bits = 32 if dtype_str == "f32" else 16
    quant_dtype_max = _quant_dtype_max(quant_dtype_str)
    USE_VEC_N = elem_bits <= 16 and N % VEC_WIDTH == 0
    VEC_TILES = N // VEC_WIDTH if USE_VEC_N else 0
    NUM_VEC_ITERS = (VEC_TILES + BLOCK_THREADS - 1) // BLOCK_THREADS if USE_VEC_N else 0

    BlockReduce = fx.coop.BlockReduce[fx.Float32x2, BLOCK_THREADS]
    SharedStorage = BlockReduce.SharedStorage

    @flyc.kernel
    def rmsnorm_quant_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
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

        def block_reduce_add(val):
            dummy = fx.Float32(0.0)
            r0, _ = block_reduce_add2(val, dummy)
            return r0

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

        # Vector path for every complete f16/bf16 vec8 row.
        if const_expr(USE_VEC_N):
            quant_half_width = VEC_WIDTH // 2
            abs_mask = fx.full(VEC_WIDTH, 0x7FFFFFFF, fx.Uint32)
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            if const_expr(is_smooth):
                XScale_buf = fx.rocdl.make_buffer_tensor(XScale)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            out_div_q = fx.logical_divide(row_out, fx.make_layout(quant_half_width, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VEC_WIDTH, 1))
            if const_expr(is_smooth):
                xscale_div = fx.logical_divide(XScale_buf, fx.make_layout(VEC_WIDTH, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            if const_expr(is_smooth):
                copy_atom_xs = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            copy_atom_q = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 8)

            thread_sumsq = c_zero_f
            thread_dummy = c_zero_f
            in_local = []

            # Pass 1: load + cache + sumsq
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                is_valid = idx < VEC_TILES
                idx_safe = is_valid.select(idx, 0)
                vec = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx_safe)
                in_local.append(vec)
                x = vec.to(fx.Float32)
                x2 = x * x
                red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + is_valid.select(red2, c_zero_f)

            _, sum_sq = block_reduce_add2(thread_dummy, thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            thread_row_max = c_zero_f
            y_local = []

            # Pass 2: normalize + gamma (+ optional smooth scale), cache output, and accumulate row max
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                is_valid = idx < VEC_TILES
                idx_safe = is_valid.select(idx, 0)

                g = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, idx_safe).to(fx.Float32)
                x = in_local[tile_i].to(fx.Float32)
                y = (x * rrms) * g
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

            # Pass 3: quantize + store using per-row scale
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
            # Scalar fallback for arbitrary N.
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            if const_expr(is_smooth):
                XScale_buf = fx.rocdl.make_buffer_tensor(XScale)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            def _abs_scalar(val):
                is_neg = val < c_zero_f
                neg_val = c_zero_f - val
                return is_neg.select(neg_val, val)

            thread_sumsq = c_zero_f

            # Pass 1: accumulate sumsq
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = row_in[idx_safe]
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                x2 = x * x
                thread_sumsq = thread_sumsq + is_valid.select(x2, c_zero_f)

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            thread_row_max = c_zero_f
            # Pass 2: normalize, apply gamma (+ optional smooth scale), and accumulate row max
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = row_in[idx_safe]
                g_e = Gamma_buf[idx_safe]
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                y = (x * rrms) * g
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

            # Pass 3: quantize + store using per-row scale
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    x_e = row_in[idx]
                    g_e = Gamma_buf[idx]
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    y = (x * rrms) * g
                    if const_expr(is_smooth):
                        s_e = XScale_buf[idx]
                        s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                        y = y * s
                    q = y * inv_scale
                    q_i8 = q.to(quant_dtype)
                    row_out[idx] = q_i8

    if is_smooth:

        @flyc.jit
        def launch_rmsnorm_smoothquant(
            Input: fx.Tensor,
            Gamma: fx.Tensor,
            XScale: fx.Tensor,
            Output: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            _validate_norm_operand_dtypes(Input, dtype_str, Gamma=Gamma, XScale=XScale)
            launcher = rmsnorm_quant_kernel(Input, Gamma, XScale, YScale, Output)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_rmsnorm_smoothquant

    else:

        @flyc.jit
        def launch_rmsnorm_dynamicquant(
            Input: fx.Tensor,
            Gamma: fx.Tensor,
            Output: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            _validate_norm_operand_dtypes(Input, dtype_str, Gamma=Gamma)
            launcher = rmsnorm_quant_kernel(Input, Gamma, Gamma, YScale, Output)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_rmsnorm_dynamicquant


def build_rmsnorm_dynamicquant_module(
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
    eps: float = EPS,
):
    return _build_rmsnorm_quant_module(
        N,
        dtype_str,
        is_smooth=False,
        quant_dtype_str=quant_dtype_str,
        eps=eps,
    )


def build_rmsnorm_smoothquant_module(
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
    eps: float = EPS,
):
    return _build_rmsnorm_quant_module(
        N,
        dtype_str,
        is_smooth=True,
        quant_dtype_str=quant_dtype_str,
        eps=eps,
    )


def _build_fused_add_rmsnorm_quant_module(
    N: int,
    dtype_str: str,
    *,
    is_smooth: bool,
    quant_dtype_str: str = "i8",
    eps: float = EPS,
):
    arch = get_rocm_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    elem_bits = 32 if dtype_str == "f32" else 16
    quant_dtype_max = _quant_dtype_max(quant_dtype_str)
    USE_VEC_N = elem_bits <= 16 and N % VEC_WIDTH == 0
    VEC_TILES = N // VEC_WIDTH if USE_VEC_N else 0
    NUM_VEC_ITERS = (VEC_TILES + BLOCK_THREADS - 1) // BLOCK_THREADS if USE_VEC_N else 0

    BlockReduce = fx.coop.BlockReduce[fx.Float32x2, BLOCK_THREADS]
    SharedStorage = BlockReduce.SharedStorage

    @flyc.kernel
    def fused_add_rmsnorm_quant_kernel(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
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

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        YScale_buf = fx.rocdl.make_buffer_tensor(YScale)

        def block_reduce_add(val):
            dummy = fx.Float32(0.0)
            r0, _ = block_reduce_add2(val, dummy)
            return r0

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

        # Vector path for every complete f16/bf16 vec8 row.
        if const_expr(USE_VEC_N):
            quant_half_width = VEC_WIDTH // 2
            abs_mask = fx.full(VEC_WIDTH, 0x7FFFFFFF, fx.Uint32)
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            ResidualIn_buf = fx.rocdl.make_buffer_tensor(ResidualIn)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            ResidualOut_buf = fx.rocdl.make_buffer_tensor(ResidualOut)
            if const_expr(is_smooth):
                XScale_buf = fx.rocdl.make_buffer_tensor(XScale)

            row_in = fx.slice(Input_buf, (bid, None))
            row_residual_in = fx.slice(ResidualIn_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))
            row_residual_out = fx.slice(ResidualOut_buf, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            residual_in_div = fx.logical_divide(row_residual_in, fx.make_layout(VEC_WIDTH, 1))
            out_div_q = fx.logical_divide(row_out, fx.make_layout(quant_half_width, 1))
            residual_out_div = fx.logical_divide(row_residual_out, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VEC_WIDTH, 1))
            if const_expr(is_smooth):
                xscale_div = fx.logical_divide(XScale_buf, fx.make_layout(VEC_WIDTH, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            if const_expr(is_smooth):
                copy_atom_xs = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            copy_atom_q = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 8)

            thread_sumsq = c_zero_f
            thread_dummy = c_zero_f
            add_local = []

            # Pass 1: add + cache + sumsq (also write residual_out)
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                is_valid = idx < VEC_TILES
                idx_safe = is_valid.select(idx, 0)
                x = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx_safe).to(fx.Float32)
                residual = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, residual_in_div, idx_safe).to(fx.Float32)
                added_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, x + residual)
                add_local.append(added_e)
                added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                added2 = added * added
                red2 = added2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + is_valid.select(red2, c_zero_f)
                if idx < VEC_TILES:
                    _store_vec(copy_atom, VEC_WIDTH, elem_dtype, added_e, residual_out_div, idx)

            _, sum_sq = block_reduce_add2(thread_dummy, thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            thread_row_max = c_zero_f
            y_local = []

            # Pass 2: normalize + gamma (+ optional smooth scale), cache output, and accumulate row max
            for tile_i in range_constexpr(NUM_VEC_ITERS):
                idx = tid + tile_i * BLOCK_THREADS
                is_valid = idx < VEC_TILES
                idx_safe = is_valid.select(idx, 0)
                g = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, idx_safe).to(fx.Float32)
                added = add_local[tile_i] if dtype_str == "f32" else add_local[tile_i].to(fx.Float32)
                y = (added * rrms) * g
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

            # Pass 3: quantize + store using per-row scale
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
            # Scalar fallback for arbitrary N.
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            ResidualIn_buf = fx.rocdl.make_buffer_tensor(ResidualIn)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            ResidualOut_buf = fx.rocdl.make_buffer_tensor(ResidualOut)
            if const_expr(is_smooth):
                XScale_buf = fx.rocdl.make_buffer_tensor(XScale)

            row_in = fx.slice(Input_buf, (bid, None))
            row_residual_in = fx.slice(ResidualIn_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))
            row_residual_out = fx.slice(ResidualOut_buf, (bid, None))

            def _abs_scalar(val):
                is_neg = val < c_zero_f
                neg_val = c_zero_f - val
                return is_neg.select(neg_val, val)

            thread_sumsq = c_zero_f

            # Pass 1: add, write residual_out, and accumulate sumsq
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = row_in[idx_safe]
                residual_e = row_residual_in[idx_safe]
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                residual = residual_e if dtype_str == "f32" else residual_e.to(fx.Float32)
                added_e = _to_elem_scalar(dtype_str, elem_dtype, x + residual)
                if idx < N:
                    row_residual_out[idx] = added_e
                added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                added2 = added * added
                thread_sumsq = thread_sumsq + is_valid.select(added2, c_zero_f)

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = sum_sq / n_float
            ms_eps = mean_sq + eps_c
            rrms = fmath.rsqrt(ms_eps, fastmath=fm_fast)

            thread_row_max = c_zero_f
            # Pass 2: normalize, apply gamma (+ optional smooth scale), and accumulate row max
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                g_e = Gamma_buf[idx_safe]
                added_e = row_residual_out[idx_safe]
                g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                y = (added * rrms) * g
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

            # Pass 3: quantize + store using per-row scale
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    g_e = Gamma_buf[idx]
                    added_e = row_residual_out[idx]
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                    y = (added * rrms) * g
                    if const_expr(is_smooth):
                        s_e = XScale_buf[idx]
                        s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                        y = y * s
                    q = y * inv_scale
                    q_i8 = q.to(quant_dtype)
                    row_out[idx] = q_i8

    if is_smooth:

        @flyc.jit
        def launch_fused_add_rmsnorm_smoothquant(
            Input: fx.Tensor,
            ResidualIn: fx.Tensor,
            Gamma: fx.Tensor,
            XScale: fx.Tensor,
            Output: fx.Tensor,
            ResidualOut: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            _validate_norm_operand_dtypes(
                Input, dtype_str, ResidualIn=ResidualIn, Gamma=Gamma, XScale=XScale, ResidualOut=ResidualOut
            )
            launcher = fused_add_rmsnorm_quant_kernel(Input, ResidualIn, Gamma, XScale, YScale, Output, ResidualOut)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_fused_add_rmsnorm_smoothquant

    else:

        @flyc.jit
        def launch_fused_add_rmsnorm_dynamicquant(
            Input: fx.Tensor,
            ResidualIn: fx.Tensor,
            Gamma: fx.Tensor,
            Output: fx.Tensor,
            ResidualOut: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            _validate_norm_operand_dtypes(Input, dtype_str, ResidualIn=ResidualIn, Gamma=Gamma, ResidualOut=ResidualOut)
            launcher = fused_add_rmsnorm_quant_kernel(Input, ResidualIn, Gamma, Gamma, YScale, Output, ResidualOut)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_fused_add_rmsnorm_dynamicquant


def _fused_add_quant_geometry(
    dtype_str: str,
    N: int,
    threads_per_row: int | None = None,
    groups: int | None = None,
) -> tuple[int, int, int, int, int]:
    """(threads_per_row, groups, threads, vec_tiles, iters) for the fused add + quant kernel.

    One *group* of ``threads_per_row`` lanes owns one row and keeps the whole row in
    registers. ``threads_per_row`` rounds up to a power of two, so a row that is a whole
    number of waves instead fills its block exactly: no lane is idle in the last pass,
    which measures 3-8% faster on the row-per-block shapes. Rows wider than ten waves
    get half a row per lane when both passes then fill exactly.

    Narrow rows pack several groups into one block: a block of 1024 threads costs the
    same to dispatch as a block of 64, and a kernel whose grid fits in the smallest
    dispatch step is measurably faster than the same work spread over more blocks. From
    two waves per row up, packing rows only makes each block longer and measures slower
    at every row count, so those keep one row per block.
    """
    elem_bits = 32 if dtype_str == "f32" else 16
    vec = 128 // elem_bits
    vec_tiles = N // vec
    if threads_per_row is None:
        threads_per_row = 8
        while threads_per_row < vec_tiles and threads_per_row < 512:
            threads_per_row *= 2
        # A row that is a whole number of waves (up to ten) should fill its block
        # exactly instead of rounding up to a power of two: the last iteration then
        # has no idle lanes and the row needs one pass rather than two.
        if 0 < vec_tiles <= 640 and vec_tiles % WARP_SIZE == 0:
            threads_per_row = vec_tiles
        elif (
            threads_per_row < vec_tiles
            and vec_tiles % 2 == 0
            and (vec_tiles // 2) % WARP_SIZE == 0
            and vec_tiles // 2 <= 512
        ):
            # Same argument one step out: a row of 14 or 12 waves is not worth
            # rounding up to 512 lanes only to leave a quarter of them idle in the
            # second pass, so give each lane half a row and keep both passes full.
            threads_per_row = vec_tiles // 2
    if groups is None:
        # Narrow rows pack several groups into one block: a block of 1024 threads
        # costs the same to dispatch as a block of 64, and a kernel whose grid fits
        # in the smallest dispatch step is measurably faster than the same work
        # spread over more blocks. From two waves per row up, packing rows into one
        # block only makes each block longer and measures slower at every row count.
        target = 1024 if vec_tiles <= 64 else 512
        groups = max(1, target // threads_per_row) if threads_per_row <= WARP_SIZE else 1
    threads = threads_per_row * groups
    iters = (vec_tiles + threads_per_row - 1) // threads_per_row
    return threads_per_row, groups, threads, vec_tiles, iters


def _tile_index(idx, limit: int, full: bool):
    """The tile index itself when the tile is complete, else one valid for idle lanes."""
    return idx if full else (idx < limit).select(idx, 0)


def _bf16x2_unpack(words, vec: int):
    """A ``vector<vec x bf16>`` in ``vec/2`` words as ``vec`` fp32 lanes.

    Rounding ``x + residual`` to bf16 and reading it back as fp32 is what the
    max/quantize pass needs. Reading it back through ``fpext`` costs two
    instructions per lane (round again, then shift); the value that was just
    packed is already in a register, so taking it apart with one shift or mask
    per lane halves that, and the packed word is shared with the residual store.
    """
    out = []
    high_half_mask = fx.Uint32(0xFFFF0000)
    for k in range_constexpr(vec // 2):
        word = words[k]
        out.append((word << 16).bitcast(fx.Float32))
        out.append((word & high_half_mask).bitcast(fx.Float32))
    return fx.Vector.from_elements(out, fx.Float32)


def _make_fused_add_quant_launch(
    N: int,
    dtype_str: str,
    *,
    quant_dtype_str: str,
    eps: float,
    threads_per_row: int,
    groups: int,
    load_cache: int,
    store_cache: int,
    bit_unpack: bool = True,
):
    """Single-pass fused residual add + RMSNorm + per-row int8 quantization.

    A group of ``threads_per_row`` lanes owns one row: its ``x``, ``residual`` and
    ``gamma`` vectors are loaded up front, ``a = x + residual`` is rounded once and
    kept in registers, and both row statistics (``sum(a^2)`` and ``max|a * gamma|``)
    come out of one reduction. Since ``y = a * rsqrt(mean(a^2) + eps) * gamma`` is
    scaled by the same ``rrms`` everywhere, ``max|y| = rrms * max|a * gamma|`` and the
    quantized output is ``trunc(a * gamma * 127 / max|a * gamma|)`` -- the normalization
    cancels, so the int8 pass never waits on a second reduction of ``y``.
    """
    arch = get_rocm_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")
    BIT_UNPACK = bit_unpack and dtype_str == "bf16" and USE_HW_CVT_PK_BF16_F32

    quant_dtype = _quant_dtype_to_elem_type(quant_dtype_str)
    quant_dtype_max = _quant_dtype_max(quant_dtype_str)
    elem_bits = 32 if dtype_str == "f32" else 16
    VEC = 128 // elem_bits
    TP, GROUPS = threads_per_row, groups
    _, _, THREADS, VEC_TILES, ITERS = _fused_add_quant_geometry(dtype_str, N, TP, GROUPS)
    TAIL = N - VEC_TILES * VEC
    TAIL_BASE = VEC_TILES * VEC
    NUM_WARPS = THREADS // WARP_SIZE
    WARPS_PER_GROUP = max(TP // WARP_SIZE, 1)
    SUB_WARP = TP < WARP_SIZE
    SHARED_SLOTS = max(NUM_WARPS, 1)
    _kernel_kwargs = {} if THREADS <= 256 else {"known_block_size": [THREADS, 1, 1]}

    @fx.struct
    class SharedStats:
        sums: fx.Array[fx.Float32, SHARED_SLOTS, 16]
        maxs: fx.Array[fx.Float32, SHARED_SLOTS, 16]

    def _added_f32(added_e):
        """``a = x + residual`` rounded to ``dtype_str``, as fp32 lanes."""
        if const_expr(dtype_str == "f32"):
            return added_e
        if const_expr(BIT_UNPACK):
            return _bf16x2_unpack(added_e.bitcast(fx.Uint32), VEC)
        return added_e.to(fx.Float32)

    def _tree_sum(vals):
        while len(vals) > 1:
            nxt = [vals[i] + vals[i + 1] for i in range(0, len(vals) - 1, 2)]
            if len(vals) % 2:
                nxt.append(vals[-1])
            vals = nxt
        return vals[0]

    def _tree_max(vals):
        while len(vals) > 1:
            nxt = [fx.max(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
            if len(vals) % 2:
                nxt.append(vals[-1])
            vals = nxt
        return vals[0]

    @flyc.kernel(**_kernel_kwargs)
    def fused_add_rmsnorm_dynamicquant_kernel(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
        ResidualOut: fx.Tensor,
        YScale: fx.Tensor,
        m_in: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        fm_fast = arith.FastMathFlags.fast
        eps_c = eps
        n_float = float(N)
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_dtype_max = fx.Float32(quant_dtype_max)

        grp = tid // TP
        gl = tid % TP
        row = bid * GROUPS + grp
        if const_expr(GROUPS > 1):
            row_valid = row < m_in
            row = row_valid.select(row, 0)

        lds = fx.SharedAllocator().allocate(SharedStats).peek()

        Input_buf = fx.rocdl.make_buffer_tensor(Input)
        ResidualIn_buf = fx.rocdl.make_buffer_tensor(ResidualIn)
        Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
        Output_buf = fx.rocdl.make_buffer_tensor(Output)
        ResidualOut_buf = fx.rocdl.make_buffer_tensor(ResidualOut)
        YScale_buf = fx.rocdl.make_buffer_tensor(YScale)

        row_in = fx.slice(Input_buf, (row, None))
        row_residual_in = fx.slice(ResidualIn_buf, (row, None))
        row_out = fx.slice(Output_buf, (row, None))
        row_residual_out = fx.slice(ResidualOut_buf, (row, None))

        in_div = fx.logical_divide(row_in, fx.make_layout(VEC, 1))
        residual_in_div = fx.logical_divide(row_residual_in, fx.make_layout(VEC, 1))
        residual_out_div = fx.logical_divide(row_residual_out, fx.make_layout(VEC, 1))
        out_div = fx.logical_divide(row_out, fx.make_layout(VEC, 1))
        gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VEC, 1))

        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(load_cache), elem_bits)
        copy_atom_st = fx.make_copy_atom(fx.rocdl.BufferCopy128b(store_cache), elem_bits)
        copy_atom_q = fx.make_copy_atom(
            fx.rocdl.BufferCopy64b(store_cache) if VEC > 4 else fx.rocdl.BufferCopy32b(store_cache), 8
        )

        abs_mask = fx.full(VEC, 0x7FFFFFFF, fx.Uint32)

        # Every lane loads first: the row's three operand vectors are independent, so
        # one round trip covers them and the reduction sees them all at once.
        x_all = []
        residual_all = []
        gamma_all = []
        for tile_i in range_constexpr(ITERS):
            idx_safe = _tile_index(gl + tile_i * TP, VEC_TILES, (tile_i + 1) * TP <= VEC_TILES)
            x_all.append(_load_vec(copy_atom, VEC, elem_dtype, in_div, idx_safe))
            residual_all.append(_load_vec(copy_atom, VEC, elem_dtype, residual_in_div, idx_safe))
            gamma_all.append(_load_vec(copy_atom, VEC, elem_dtype, gamma_div, idx_safe))

        thread_sumsq = c_zero_f
        thread_row_max = c_zero_f
        ag_local = []

        for tile_i in range_constexpr(ITERS):
            idx = gl + tile_i * TP
            is_full = (tile_i + 1) * TP <= VEC_TILES
            x = x_all[tile_i].to(fx.Float32)
            residual = residual_all[tile_i].to(fx.Float32)
            added_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, x + residual)
            if const_expr(is_full):
                _store_vec(copy_atom_st, VEC, elem_dtype, added_e, residual_out_div, idx)
            else:
                if idx < VEC_TILES:
                    _store_vec(copy_atom_st, VEC, elem_dtype, added_e, residual_out_div, idx)
            added = _added_f32(added_e)
            ag = added * gamma_all[tile_i].to(fx.Float32)
            ag_local.append(ag)
            sumsq = (added * added).reduce(ReductionOp.ADD, fastmath=fm_fast)
            ag_abs = (ag.bitcast(fx.Uint32) & abs_mask).bitcast(fx.Float32)
            tile_max = ag_abs.reduce(ReductionOp.MAX)
            if const_expr(is_full):
                thread_sumsq = thread_sumsq + sumsq
                thread_row_max = fx.max(thread_row_max, tile_max)
            else:
                is_valid = idx < VEC_TILES
                thread_sumsq = thread_sumsq + is_valid.select(sumsq, c_zero_f)
                thread_row_max = fx.max(thread_row_max, is_valid.select(tile_max, c_zero_f))

        ag_tail = c_zero_f
        if const_expr(TAIL > 0):
            tail_idx = TAIL_BASE + gl
            tail_valid = gl < TAIL
            tail_safe = tail_valid.select(tail_idx, 0)
            x_t = row_in[tail_safe]
            residual_t = row_residual_in[tail_safe]
            x_t = x_t if dtype_str == "f32" else x_t.to(fx.Float32)
            residual_t = residual_t if dtype_str == "f32" else residual_t.to(fx.Float32)
            added_t = _to_elem_scalar(dtype_str, elem_dtype, x_t + residual_t)
            gamma_t = Gamma_buf[tail_safe]
            gamma_t = gamma_t if dtype_str == "f32" else gamma_t.to(fx.Float32)
            if tail_valid:
                row_residual_out[tail_idx] = added_t
            added_tf = added_t if dtype_str == "f32" else added_t.to(fx.Float32)
            ag_tail = tail_valid.select(added_tf * gamma_t, c_zero_f)
            thread_sumsq = thread_sumsq + tail_valid.select(added_tf * added_tf, c_zero_f)
            ag_abs_t = (ag_tail.bitcast(fx.Uint32) & 0x7FFFFFFF).bitcast(fx.Float32)
            thread_row_max = fx.max(thread_row_max, ag_abs_t)

        if const_expr(SUB_WARP):
            with fx.fastmath(fm_fast):
                total_sum = fx.coop.warp_reduce(thread_sumsq, fx.ReductionOp.ADD, width=TP)
            total_max = fx.coop.warp_reduce(thread_row_max, fx.ReductionOp.MAX, width=TP)
        else:
            with fx.fastmath(fm_fast):
                sum_warp = fx.coop.warp_reduce(thread_sumsq, fx.ReductionOp.ADD)
            max_warp = fx.coop.warp_reduce(thread_row_max, fx.ReductionOp.MAX)
            if const_expr(WARPS_PER_GROUP == 1):
                total_sum = sum_warp
                total_max = max_warp
            else:
                warp_id = tid // WARP_SIZE
                if gl % WARP_SIZE == 0:
                    lds.sums[warp_id] = sum_warp
                    lds.maxs[warp_id] = max_warp
                gpu.barrier()
                base = grp * WARPS_PER_GROUP
                total_sum = _tree_sum([lds.sums[base + i] for i in range_constexpr(WARPS_PER_GROUP)])
                total_max = _tree_max([lds.maxs[base + i] for i in range_constexpr(WARPS_PER_GROUP)])

        with fx.fastmath(fm_fast):
            mean_sq = total_sum / n_float
            rrms = fmath.rsqrt(mean_sq + eps_c, fastmath=fm_fast)
            row_max = total_max * rrms
            scale = row_max / c_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)
            # y / yscale == a * gamma * 127 / max|a * gamma|: the rrms factor cancels.
            inv_scale = (total_max == c_zero_f).select(c_zero_f, c_dtype_max / total_max)

        if gl == 0:
            if const_expr(GROUPS > 1):
                if row_valid:
                    YScale_buf[row] = final_scale
            else:
                YScale_buf[row] = final_scale

        for tile_i in range_constexpr(ITERS):
            idx = gl + tile_i * TP
            q_i8 = (ag_local[tile_i] * inv_scale).to(quant_dtype)
            if const_expr((tile_i + 1) * TP <= VEC_TILES):
                _store_vec(copy_atom_q, VEC, quant_dtype, q_i8, out_div, idx)
            else:
                if idx < VEC_TILES:
                    _store_vec(copy_atom_q, VEC, quant_dtype, q_i8, out_div, idx)

        if const_expr(TAIL > 0):
            if gl < TAIL:
                row_out[TAIL_BASE + gl] = (ag_tail * inv_scale).to(quant_dtype)

    @flyc.jit
    def launch_fused_add_rmsnorm_dynamicquant(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
        ResidualOut: fx.Tensor,
        YScale: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = fused_add_rmsnorm_dynamicquant_kernel(Input, ResidualIn, Gamma, Output, ResidualOut, YScale, m_in)
        launcher.launch(
            grid=((fx.Index(m_in) + (GROUPS - 1)) // GROUPS, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    return launch_fused_add_rmsnorm_dynamicquant


def build_fused_add_rmsnorm_dynamicquant_module(
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
    eps: float = EPS,
):
    """Fused residual add + RMSNorm + per-row int8 dynamic quantization.

    Both cache policies are compiled into one launcher and picked by the row count at
    run time: a call whose operands fit the last-level cache keeps them, and only a
    call that streams past it goes non-temporal. Below 1.125x the cache, streaming
    throws away reuse the cache would have served.

    Inside the cached branch the two outputs still go non-temporal for 16-bit rows: no
    call reads them back, and keeping them out of L2 leaves the cache to the operands,
    which measures a few percent faster on the row-per-block shapes. f32 rows move
    eight bytes per element of operand instead of four and lose that few percent, so
    they keep the default write policy.
    """
    threads_per_row, groups, _, _, _ = _fused_add_quant_geometry(dtype_str, N)
    cached_store_cache = 2 if dtype_str in ("f16", "bf16") else 0
    cached_launch = _make_fused_add_quant_launch(
        N,
        dtype_str,
        quant_dtype_str=quant_dtype_str,
        eps=eps,
        threads_per_row=threads_per_row,
        groups=groups,
        load_cache=0,
        store_cache=cached_store_cache,
    )
    streaming_launch = _make_fused_add_quant_launch(
        N,
        dtype_str,
        quant_dtype_str=quant_dtype_str,
        eps=eps,
        threads_per_row=threads_per_row,
        groups=groups,
        load_cache=2,
        store_cache=2,
    )
    nt_min_rows = max(NT_MIN_WORKING_SET // (N * FUSED_ADD_QUANT_BYTES_PER_ELEM), 0)

    @flyc.jit
    def launch_fused_add_rmsnorm_dynamicquant(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
        ResidualOut: fx.Tensor,
        YScale: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        _validate_norm_operand_dtypes(Input, dtype_str, ResidualIn=ResidualIn, Gamma=Gamma, ResidualOut=ResidualOut)

        def launch_cached():
            cached_launch(Input, ResidualIn, Gamma, Output, ResidualOut, YScale, m_in, stream)

        def launch_streaming():
            streaming_launch(Input, ResidualIn, Gamma, Output, ResidualOut, YScale, m_in, stream)

        @flyc.jit
        def dispatch():
            if m_in > fx.Int32(nt_min_rows):
                launch_streaming()
            else:
                launch_cached()

        dispatch()

    return launch_fused_add_rmsnorm_dynamicquant


def build_fused_add_rmsnorm_smoothquant_module(
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
    eps: float = EPS,
):
    return _build_fused_add_rmsnorm_quant_module(
        N,
        dtype_str,
        is_smooth=True,
        quant_dtype_str=quant_dtype_str,
        eps=eps,
    )


# Python wrappers and autograd.
if torch is not None:
    from kernels.common.tensor_shim import _run_compiled
    from kernels.norm.rmsnorm_common import torch_dtype_to_str as _torch_dtype_to_str

    # Each cache retains one JitFunction launcher per specialization and device;
    # _run_compiled owns the launcher's CompiledFunction in ``._cf``. eps is a
    # compile-time forward constant, so it remains part of those cache keys.
    _FWD_CACHE: dict = {}
    _BWD_CACHE: dict = {}

    # A cached FlyDSL callable accepts the raw cudaStream_t directly.  Avoid
    # constructing a Python Stream wrapper or entering a no-op device guard on
    # the hot same-device path, while retaining the guard when tensors belong
    # to a non-current device.
    _get_current_raw_stream = getattr(
        torch._C,
        "_cuda_getCurrentRawStream",
        lambda device_index: torch.cuda.current_stream(device_index).cuda_stream,
    )

    def _run_compiled_on_device(launcher, device, args):
        device_index = device.index
        if torch.cuda.current_device() == device_index:
            _run_compiled(launcher, *args, _get_current_raw_stream(device_index))
        else:
            with torch.cuda.device(device_index):
                _run_compiled(launcher, *args, _get_current_raw_stream(device_index))

    # The staged path needs enough rows to amortize its second launch.  Its
    # 512-thread vec8 kernel needs fewer persistent programs than the scalar
    # fallback, which trades more programs for latency hiding.  Keep this
    # selector as the single source of truth shared by plain and fused wrappers.
    _BWD_TWO_STAGE_MIN_ROWS = 512
    _BWD_TWO_STAGE_MAX_N = 8192
    _BWD_CU_COUNT_CACHE: dict = {}

    def _get_rmsnorm_bwd_cu_count(device):
        num_cus = _BWD_CU_COUNT_CACHE.get(device)
        if num_cus is None:
            num_cus = torch.cuda.get_device_properties(device).multi_processor_count
            _BWD_CU_COUNT_CACHE[device] = num_cus
        return num_cus

    def _select_rmsnorm_bwd_config(M, N, dtype_str, device):
        if M >= _BWD_TWO_STAGE_MIN_ROWS and N <= _BWD_TWO_STAGE_MAX_N:
            num_cus = _get_rmsnorm_bwd_cu_count(device)
            if is_rmsnorm_bwd_two_stage_vec_config(N, dtype_str):
                num_programs = num_cus if M < 2048 else (3 * num_cus) // 2
            else:
                num_programs = num_cus if M < 1024 else 2 * num_cus
            return ("two_stage", min(M, num_programs))
        return ("atomic", None)

    def _get_fwd_launcher(
        N,
        dtype_str,
        weight_dtype_str,
        store_rstd,
        eps,
        device,
    ):
        key = (N, dtype_str, weight_dtype_str, store_rstd, float(eps), device)
        launcher = _FWD_CACHE.get(key)
        if launcher is None:
            with torch.cuda.device(device):
                launcher = build_rmsnorm_module(
                    N,
                    dtype_str,
                    store_rstd=store_rstd,
                    eps=eps,
                    weight_dtype_str=weight_dtype_str,
                )
            _FWD_CACHE[key] = launcher
        return launcher

    def rmsnorm_fwd(x, weight, eps=EPS, store_rstd=False):
        """Forward RMSNorm. Returns (out, rstd). eps is baked into the kernel."""
        assert x.dim() == 2, "rmsnorm_fwd expects a 2D (M, N) input"
        assert x.is_contiguous() and weight.is_contiguous(), "rmsnorm_fwd expects contiguous inputs"
        assert weight.device == x.device, "rmsnorm_fwd: weight and x must be on the same device"
        device = x.device
        M, N = x.shape
        out = torch.empty_like(x)
        rstd = torch.empty((M,), device=device, dtype=torch.float32) if store_rstd else None
        dtype_str = _torch_dtype_to_str(x.dtype)
        weight_dtype_str = _resolve_rmsnorm_weight_dtype(dtype_str, _torch_dtype_to_str(weight.dtype))
        launcher = _get_fwd_launcher(
            N,
            dtype_str,
            weight_dtype_str,
            store_rstd,
            eps,
            device,
        )
        args = (x, weight, out, rstd, M) if store_rstd else (x, weight, out, M)
        _run_compiled_on_device(launcher, device, args)
        return out, rstd

    def rmsnorm_bwd(x, weight, dout, rstd, eps=EPS):
        """Backward RMSNorm. Returns (dx, dw) with dw in weight dtype.

        eps is not used directly here — it is already baked into `rstd` by the
        forward — but is accepted so callers can pass it symmetrically.
        """
        device = x.device
        dtype = x.dtype
        assert x.dim() == 2, "rmsnorm_bwd expects a 2D (M, N) input"
        assert x.is_contiguous() and dout.is_contiguous(), "rmsnorm_bwd expects contiguous inputs"
        assert weight.is_contiguous(), "rmsnorm_bwd: weight must be contiguous"
        assert weight.device == device == dout.device == rstd.device, "rmsnorm_bwd: inputs must share a device"
        assert dout.dtype == dtype, "rmsnorm_bwd: dout dtype must match x"
        M, N = x.shape
        dtype_str = _torch_dtype_to_str(dtype)
        weight_dtype_str = _resolve_rmsnorm_weight_dtype(dtype_str, _torch_dtype_to_str(weight.dtype))
        dx = torch.empty_like(x)
        path, num_programs = _select_rmsnorm_bwd_config(M, N, dtype_str, device)
        if path == "two_stage":
            dweight = torch.empty_like(weight)
            partial = torch.empty((num_programs * N,), device=device, dtype=torch.float32)
            key = (path, N, dtype_str, weight_dtype_str, num_programs, device)
            launcher = _BWD_CACHE.get(key)
            if launcher is None:
                # The builder has an arch-dependent vec conversion choice, and
                # compilation/code-object loading are device-context bound.
                with torch.cuda.device(device):
                    device_index = device.index
                    launcher = build_rmsnorm_bwd_two_stage_module(
                        N,
                        dtype_str,
                        num_programs,
                        weight_dtype_str=weight_dtype_str,
                    )
                    _run_compiled(
                        launcher,
                        x,
                        weight,
                        dout,
                        rstd,
                        dx,
                        dweight,
                        partial,
                        M,
                        _get_current_raw_stream(device_index),
                    )
                _BWD_CACHE[key] = launcher
            else:
                _run_compiled_on_device(launcher, device, (x, weight, dout, rstd, dx, dweight, partial, M))
            return dx, dweight

        dweight = torch.empty((N,), device=device, dtype=torch.float32)
        key = (path, N, dtype_str, weight_dtype_str, num_programs, device)
        launcher = _BWD_CACHE.get(key)
        dweight.zero_()
        if launcher is None:
            with torch.cuda.device(device):
                device_index = device.index
                launcher = build_rmsnorm_bwd_module(N, dtype_str, weight_dtype_str=weight_dtype_str)
                _run_compiled(
                    launcher,
                    x,
                    weight,
                    dout,
                    rstd,
                    dx,
                    dweight,
                    M,
                    _get_current_raw_stream(device_index),
                )
            _BWD_CACHE[key] = launcher
        else:
            _run_compiled_on_device(launcher, device, (x, weight, dout, rstd, dx, dweight, M))
        return dx, dweight.to(weight.dtype)

    class RMSNormFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, eps):
            need_grad = x.requires_grad or weight.requires_grad
            out, rstd = rmsnorm_fwd(x, weight, eps=eps, store_rstd=need_grad)
            ctx.save_for_backward(x, weight, rstd)
            ctx.eps = eps
            return out

        @staticmethod
        def backward(ctx, dout):
            x, weight, rstd = ctx.saved_tensors
            dx, dw = rmsnorm_bwd(x, weight, dout.contiguous(), rstd, eps=ctx.eps)
            return dx, dw, None

    def rmsnorm(x, weight=None, eps=EPS):
        """Public entry: plain RMSNorm with autograd (weight required in PR 1)."""
        assert weight is not None, "PR 1 rmsnorm requires an explicit weight"
        N = weight.shape[-1]
        assert x.shape[-1] == N, f"x last dim {x.shape[-1]} != weight length {N}"
        # reshape() can return a non-contiguous view (e.g. from a strided slice);
        # the kernel indexes rows by raw stride, so force contiguity here rather
        # than relying only on the fwd assert (which vanishes under python -O).
        x_flat = x.reshape(-1, N).contiguous()
        out_flat = RMSNormFunction.apply(x_flat, weight, eps)
        return out_flat.reshape(x.shape)

    # Fused-add / prenorm wrappers and autograd.
    _FUSED_ADD_FWD_CACHE: dict = {}
    _FUSED_ADD_BWD_CACHE: dict = {}

    def _get_fused_add_fwd_launcher(
        N,
        dtype_str,
        weight_dtype_str,
        store_rstd,
        eps,
        device,
    ):
        key = (N, dtype_str, weight_dtype_str, store_rstd, float(eps), device)
        launcher = _FUSED_ADD_FWD_CACHE.get(key)
        if launcher is None:
            with torch.cuda.device(device):
                launcher = build_fused_add_rmsnorm_module(
                    N,
                    dtype_str,
                    store_rstd=store_rstd,
                    eps=eps,
                    weight_dtype_str=weight_dtype_str,
                )
            _FUSED_ADD_FWD_CACHE[key] = launcher
        return launcher

    def fused_add_rmsnorm_fwd(x, residual, weight, eps=EPS, store_rstd=False):
        """Forward fused-add RMSNorm. Returns (out, residual_out, rstd).

        residual_out = x + residual ; out = residual_out * rstd * weight.
        eps is baked into the kernel.
        """
        assert x.dim() == 2, "fused_add_rmsnorm_fwd expects a 2D (M, N) input"
        assert (
            x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()
        ), "fused_add_rmsnorm_fwd expects contiguous inputs"
        assert x.shape == residual.shape, "x and residual must have the same shape"
        assert x.dtype == residual.dtype, f"x/residual dtypes must match, got {x.dtype}/{residual.dtype}"
        # Only x.device gates compile/stream/cache; all operands must co-reside.
        assert (
            x.device == residual.device == weight.device
        ), f"x/residual/weight must be on the same device, got {x.device}/{residual.device}/{weight.device}"
        M, N = x.shape
        out = torch.empty_like(x)
        residual_out = torch.empty_like(x)
        device = x.device
        rstd = torch.empty((M,), device=device, dtype=torch.float32) if store_rstd else None
        dtype_str = _torch_dtype_to_str(x.dtype)
        weight_dtype_str = _resolve_rmsnorm_weight_dtype(dtype_str, _torch_dtype_to_str(weight.dtype))
        launcher = _get_fused_add_fwd_launcher(
            N,
            dtype_str,
            weight_dtype_str,
            store_rstd,
            eps,
            device,
        )
        args = (
            (x, residual, weight, out, residual_out, rstd, M)
            if store_rstd
            else (x, residual, weight, out, residual_out, M)
        )
        _run_compiled_on_device(launcher, device, args)
        return out, residual_out, rstd

    def fused_add_rmsnorm_bwd(added, weight, dout, rstd, dresidual_out=None, eps=EPS):
        """Backward fused-add RMSNorm. Returns (dx, dresidual, dw).

        `added` is the residual_out saved by the forward. Because added = x +
        residual_in, dx == dresidual unconditionally, so the kernel computes it
        once and this returns dx aliased as dresidual (no second buffer/store).
        dresidual_out is the downstream grad flowing into residual_out; when None
        it is treated as zero (a zero tensor is passed to the branch-free
        kernel). eps is already baked into `rstd`.
        """
        device = added.device
        dtype = added.dtype
        assert added.dim() == 2, "fused_add_rmsnorm_bwd expects a 2D (M, N) input"
        assert added.is_contiguous() and dout.is_contiguous(), "fused_add_rmsnorm_bwd expects contiguous inputs"
        assert dtype == dout.dtype, f"added/dout dtypes must match, got {dtype}/{dout.dtype}"
        assert (
            device == weight.device == dout.device == rstd.device
        ), "fused_add_rmsnorm_bwd expects all tensors on the same device"
        if dresidual_out is not None:
            assert dresidual_out.is_contiguous(), "fused_add_rmsnorm_bwd expects contiguous dresidual_out"
            assert dresidual_out.dtype == dtype, "dresidual_out dtype must match added"
            assert dresidual_out.device == device, "dresidual_out must be on the same device as added"
        M, N = added.shape
        dtype_str = _torch_dtype_to_str(dtype)
        weight_dtype_str = _resolve_rmsnorm_weight_dtype(dtype_str, _torch_dtype_to_str(weight.dtype))
        if dresidual_out is None:
            dresidual_out = torch.zeros_like(added)
        dx = torch.empty_like(added)
        path, num_programs = _select_rmsnorm_bwd_config(M, N, dtype_str, device)
        if path == "two_stage":
            dweight = torch.empty_like(weight)
            partial = torch.empty((num_programs * N,), device=device, dtype=torch.float32)
            key = (path, N, dtype_str, weight_dtype_str, num_programs, device)
            launcher = _FUSED_ADD_BWD_CACHE.get(key)
            if launcher is None:
                with torch.cuda.device(device):
                    device_index = device.index
                    launcher = build_fused_add_rmsnorm_bwd_two_stage_module(
                        N,
                        dtype_str,
                        num_programs,
                        weight_dtype_str=weight_dtype_str,
                    )
                    _run_compiled(
                        launcher,
                        added,
                        weight,
                        dout,
                        dresidual_out,
                        rstd,
                        dx,
                        dweight,
                        partial,
                        M,
                        _get_current_raw_stream(device_index),
                    )
                _FUSED_ADD_BWD_CACHE[key] = launcher
            else:
                _run_compiled_on_device(
                    launcher,
                    device,
                    (added, weight, dout, dresidual_out, rstd, dx, dweight, partial, M),
                )
            return dx, dx, dweight

        dweight = torch.empty((N,), device=device, dtype=torch.float32)
        key = (path, N, dtype_str, weight_dtype_str, num_programs, device)
        launcher = _FUSED_ADD_BWD_CACHE.get(key)
        dweight.zero_()
        if launcher is None:
            with torch.cuda.device(device):
                device_index = device.index
                launcher = build_fused_add_rmsnorm_bwd_module(
                    N,
                    dtype_str,
                    weight_dtype_str=weight_dtype_str,
                )
                _run_compiled(
                    launcher,
                    added,
                    weight,
                    dout,
                    dresidual_out,
                    rstd,
                    dx,
                    dweight,
                    M,
                    _get_current_raw_stream(device_index),
                )
            _FUSED_ADD_BWD_CACHE[key] = launcher
        else:
            _run_compiled_on_device(
                launcher,
                device,
                (added, weight, dout, dresidual_out, rstd, dx, dweight, M),
            )
        # dx == dresidual by construction; return dx as both (aliased).
        return dx, dx, dweight.to(weight.dtype)

    class FusedAddRMSNormFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, residual, weight, eps, prenorm):
            need_grad = x.requires_grad or residual.requires_grad or weight.requires_grad
            out, residual_out, rstd = fused_add_rmsnorm_fwd(x, residual, weight, eps=eps, store_rstd=need_grad)
            ctx.save_for_backward(residual_out, weight, rstd)
            ctx.eps = eps
            ctx.prenorm = prenorm
            if prenorm:
                return out, residual_out
            return out

        @staticmethod
        def backward(ctx, dout, *args):
            added, weight, rstd = ctx.saved_tensors
            dresidual_out = args[0].contiguous() if ctx.prenorm else None
            dx, dresidual, dw = fused_add_rmsnorm_bwd(
                added, weight, dout.contiguous(), rstd, dresidual_out=dresidual_out, eps=ctx.eps
            )
            return dx, dresidual, dw, None, None

    def fused_add_rmsnorm(x, residual, weight, eps=EPS, prenorm=True):
        """Public entry: fused-add (prenorm) RMSNorm with autograd.

        residual_out = x + residual ; out = rmsnorm(residual_out) * weight.
        prenorm=True (training-relevant) returns (out, residual_out).
        """
        assert weight is not None, "fused_add_rmsnorm requires an explicit weight"
        N = weight.shape[-1]
        assert x.shape[-1] == N, f"x last dim {x.shape[-1]} != weight length {N}"
        assert x.shape == residual.shape, "x and residual must have the same shape"
        # reshape() can return a non-contiguous view; force contiguity so the
        # kernel's raw-stride row indexing stays correct even under python -O.
        x_flat = x.reshape(-1, N).contiguous()
        residual_flat = residual.reshape(-1, N).contiguous()
        if prenorm:
            out_flat, residual_out_flat = FusedAddRMSNormFunction.apply(x_flat, residual_flat, weight, eps, True)
            return out_flat.reshape(x.shape), residual_out_flat.reshape(x.shape)
        out_flat = FusedAddRMSNormFunction.apply(x_flat, residual_flat, weight, eps, False)
        return out_flat.reshape(x.shape)
