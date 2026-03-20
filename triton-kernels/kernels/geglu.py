"""
GeGLU Activation - Fused Triton Kernel

GeGLU(a, b) = GELU(a) * b

用于 GPT-J / Falcon / GPT-NeoX FFN.
与 SwiGLU 共享同一套框架，仅激活函数不同。

GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))  [tanh approx]

核心优化：同 SwiGLU
1. Forward: AutoTune on BLOCK_SIZE + num_warps, tiled column loop
2. Backward: in-place 写入 saved tensors, 无 AutoTune
3. fp32 计算 GELU 保精度
"""

import torch
import triton
import triton.language as tl
import math

# Triton tanh 兼容不同版本
try:
    from triton.language.extra.libdevice import tanh as _tanh
except ImportError:
    try:
        from triton.language.extra.cuda.libdevice import tanh as _tanh
    except ImportError:
        from triton.language.math import tanh as _tanh

# sqrt(2/pi) and tanh-GELU coefficient
_SQRT_2_OVER_PI = tl.constexpr(math.sqrt(2.0 / math.pi))
_COEFF = tl.constexpr(0.044715)


# ============================================================================
# AutoTune 配置 (forward only)
# ============================================================================

def _get_geglu_configs():
    configs = []
    for BLOCK in [1024, 2048, 4096, 8192]:
        for nw in [4, 8, 16]:
            configs.append(triton.Config({'BLOCK_SIZE': BLOCK}, num_warps=nw))
    return configs


# ============================================================================
# Forward Kernel
# ============================================================================

@triton.autotune(configs=_get_geglu_configs(), key=['n_cols'])
@triton.jit
def _geglu_fwd_kernel(
    a_ptr, b_ptr, c_ptr,
    stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_off = row * stride

    for col_start in range(0, n_cols, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        a = tl.load(a_ptr + row_off + cols, mask=mask, other=0).to(tl.float32)
        b = tl.load(b_ptr + row_off + cols, mask=mask, other=0)
        # gelu_tanh(a) = 0.5 * a * (1 + tanh(sqrt(2/pi) * (a + 0.044715 * a^3)))
        inner = _SQRT_2_OVER_PI * (a + _COEFF * a * a * a)
        gelu_a = 0.5 * a * (1.0 + _tanh(inner))
        c = gelu_a.to(b.dtype) * b
        tl.store(c_ptr + row_off + cols, c, mask=mask)


# ============================================================================
# Backward Kernel (no AutoTune — in-place write to a, b)
# ============================================================================

def _calc_settings(n_cols):
    BLOCK = triton.next_power_of_2(n_cols)
    if BLOCK > 8192:
        BLOCK = 8192
    nw = 4
    if BLOCK >= 2048:
        nw = 8
    if BLOCK >= 8192:
        nw = 16
    return BLOCK, nw


@triton.jit
def _geglu_bwd_kernel(
    a_ptr, b_ptr, dc_ptr,
    stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward: overwrites a with da, b with db (in-place)."""
    row = tl.program_id(0)
    row_off = row * stride

    for col_start in range(0, n_cols, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        a  = tl.load(a_ptr  + row_off + cols, mask=mask, other=0).to(tl.float32)
        b  = tl.load(b_ptr  + row_off + cols, mask=mask, other=0).to(tl.float32)
        dc = tl.load(dc_ptr + row_off + cols, mask=mask, other=0).to(tl.float32)

        # Recompute gelu and its derivative
        inner = _SQRT_2_OVER_PI * (a + _COEFF * a * a * a)
        tanh_inner = _tanh(inner)
        gelu_a = 0.5 * a * (1.0 + tanh_inner)

        # gelu'(a) = 0.5*(1+tanh) + 0.5*a*(1-tanh^2)*sqrt(2/pi)*(1+3*0.044715*a^2)
        dtanh = 1.0 - tanh_inner * tanh_inner
        dgelu = 0.5 * (1.0 + tanh_inner) + 0.5 * a * dtanh * _SQRT_2_OVER_PI * (1.0 + 3.0 * _COEFF * a * a)

        # db = dc * gelu(a)
        db = dc * gelu_a
        # da = dc * b * gelu'(a)
        da = dc * b * dgelu

        tl.store(a_ptr + row_off + cols, da, mask=mask)
        tl.store(b_ptr + row_off + cols, db, mask=mask)


# ============================================================================
# Python wrappers
# ============================================================================

def geglu_forward(a, b):
    assert a.shape == b.shape, "a and b must have the same shape"
    a, b = a.contiguous(), b.contiguous()
    shape = a.shape
    a_2d = a.reshape(-1, shape[-1])
    b_2d = b.reshape(-1, shape[-1])
    n_rows, n_cols = a_2d.shape
    c_2d = torch.empty_like(b_2d)
    _geglu_fwd_kernel[(n_rows,)](
        a_2d, b_2d, c_2d,
        a_2d.stride(0), n_cols,
    )
    return c_2d.reshape(shape)


def geglu_backward(a, b, dc):
    dc = dc.contiguous()
    shape = a.shape
    a_2d  = a.reshape(-1, shape[-1])
    b_2d  = b.reshape(-1, shape[-1])
    dc_2d = dc.reshape(-1, shape[-1])
    n_rows, n_cols = a_2d.shape
    BLOCK, nw = _calc_settings(n_cols)
    _geglu_bwd_kernel[(n_rows,)](
        a_2d, b_2d, dc_2d,
        a_2d.stride(0), n_cols,
        BLOCK_SIZE=BLOCK, num_warps=nw,
    )
    return a.reshape(shape), b.reshape(shape)


# ============================================================================
# PyTorch Autograd Function
# ============================================================================

class HildaGeGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        c = geglu_forward(a, b)
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        da, db = geglu_backward(a, b, dc)
        return da, db


def hilda_geglu(a, b):
    """Functional interface: GeGLU(a, b) = GELU(a) * b"""
    return HildaGeGLUFunction.apply(a, b)


# ============================================================================
# nn.Module
# ============================================================================

class GeGLU(torch.nn.Module):
    """Drop-in GeGLU activation: forward(a, b) -> GELU(a) * b"""
    def forward(self, a, b):
        return hilda_geglu(a, b)
