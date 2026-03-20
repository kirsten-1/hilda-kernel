"""
SwiGLU Activation - Fused Triton Kernel

SwiGLU(a, b) = SiLU(a) * b = (a * sigmoid(a)) * b

用于 LLaMA / Mistral FFN:
  gate = x @ W_gate   # → a
  up   = x @ W_up     # → b
  out  = SwiGLU(a, b) # fused silu + mul

核心优化：
1. Forward: AutoTune on BLOCK_SIZE + num_warps, tiled column loop
2. Backward: in-place 写入 saved tensors (省 2 个大 tensor 分配), 无 AutoTune
3. fp32 计算 sigmoid 保精度, 结果 cast 回原 dtype
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# AutoTune 配置 (forward only)
# ============================================================================

def _get_swiglu_configs():
    configs = []
    for BLOCK in [1024, 2048, 4096, 8192]:
        for nw in [4, 8, 16]:
            configs.append(triton.Config({'BLOCK_SIZE': BLOCK}, num_warps=nw))
    return configs


# ============================================================================
# Forward Kernel
# ============================================================================

@triton.autotune(configs=_get_swiglu_configs(), key=['n_cols'])
@triton.jit
def _swiglu_fwd_kernel(
    a_ptr, b_ptr, c_ptr,
    stride,    # row stride (elements)
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
        # silu(a) = a * sigmoid(a), computed in fp32
        silu_a = a * tl.sigmoid(a)
        c = silu_a.to(b.dtype) * b
        tl.store(c_ptr + row_off + cols, c, mask=mask)

# ============================================================================
# Backward Kernel (no AutoTune — in-place write to a, b)
# ============================================================================

def _calc_settings(n_cols):
    BLOCK = triton.next_power_of_2(n_cols)
    if BLOCK > 65536:
        BLOCK = 65536
    nw = 4
    if BLOCK >= 2048:
        nw = 8
    if BLOCK >= 8192:
        nw = 16
    if BLOCK >= 32768:
        nw = 32
    return BLOCK, nw


@triton.jit
def _swiglu_bwd_kernel(
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

        sig_a  = tl.sigmoid(a)
        silu_a = a * sig_a

        # db = dc * silu(a)
        db = dc * silu_a
        # da = dc * b * silu'(a)
        # silu'(a) = sig(a) + a * sig(a) * (1 - sig(a)) = sig(a) * (1 + a*(1-sig(a)))
        da = dc * b * (sig_a + a * sig_a * (1.0 - sig_a))

        tl.store(a_ptr + row_off + cols, da, mask=mask)
        tl.store(b_ptr + row_off + cols, db, mask=mask)


# ============================================================================
# Python wrappers
# ============================================================================

def swiglu_forward(a, b):
    assert a.shape == b.shape, "a and b must have the same shape"
    a, b = a.contiguous(), b.contiguous()
    shape = a.shape
    a_2d = a.reshape(-1, shape[-1])
    b_2d = b.reshape(-1, shape[-1])
    n_rows, n_cols = a_2d.shape
    c_2d = torch.empty_like(b_2d)
    _swiglu_fwd_kernel[(n_rows,)](
        a_2d, b_2d, c_2d,
        a_2d.stride(0), n_cols,
    )
    return c_2d.reshape(shape)


def swiglu_backward(a, b, dc):
    dc = dc.contiguous()
    shape = a.shape
    a_2d  = a.reshape(-1, shape[-1])
    b_2d  = b.reshape(-1, shape[-1])
    dc_2d = dc.reshape(-1, shape[-1])
    n_rows, n_cols = a_2d.shape
    BLOCK, nw = _calc_settings(n_cols)
    _swiglu_bwd_kernel[(n_rows,)](
        a_2d, b_2d, dc_2d,
        a_2d.stride(0), n_cols,
        BLOCK_SIZE=BLOCK, num_warps=nw,
    )
    # a_2d and b_2d now contain da and db (in-place)
    return a.reshape(shape), b.reshape(shape)


# ============================================================================
# PyTorch Autograd Function
# ============================================================================

class HildaSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        c = swiglu_forward(a, b)
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        da, db = swiglu_backward(a, b, dc)
        return da, db


def hilda_swiglu(a, b):
    """Functional interface: SwiGLU(a, b) = SiLU(a) * b"""
    return HildaSwiGLUFunction.apply(a, b)


# ============================================================================
# nn.Module
# ============================================================================

class SwiGLU(torch.nn.Module):
    """Drop-in SwiGLU activation: forward(a, b) -> SiLU(a) * b"""
    def forward(self, a, b):
        return hilda_swiglu(a, b)
