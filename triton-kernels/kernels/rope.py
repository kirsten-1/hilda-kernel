"""
RoPE (Rotary Position Embedding) - Per-Head Kernel

核心优化：直接操作原始 (bsz, n_head, seq_len, hd) layout，
彻底消除 Liger 中的 transpose + contiguous 拷贝开销。

RoPE 的关键性质：
  backward = forward with -sin（旋转矩阵的逆 = 转置 = 旋转角取反）
  dx1 = dy1*cos + dy2*sin = forward(dy, cos, -sin) 第一半
  dx2 = dy2*cos - dy1*sin = forward(dy, cos, -sin) 第二半

所以 forward/backward 共用同一个 kernel，backward 只需传 -sin。

内存流量对比（LLaMA-3 8B GQA, bsz=4, seq=2048）：
  Liger:  83MB(contiguous copy) + 83MB(read) + 83MB(write) = 249MB per pass
  Ours:   83MB(read) + 83MB(write)                         = 166MB per pass  ← 节省 33%
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Per-Head Kernel（forward 和 backward 共用，backward 传 neg_sin）
# ============================================================================

@triton.jit
def _rope_fwd_kernel(
    x_in_ptr,      # input  (bsz, n_heads, seq_len, hd)
    x_out_ptr,     # output (same shape; may equal x_in_ptr for in-place)
    x_bs_stride,   # stride(0)
    x_h_stride,    # stride(1)
    x_s_stride,    # stride(2)
    cos_ptr,
    cos_bs_stride,
    cos_s_stride,
    sin_ptr,       # pass -sin for backward pass
    n_heads,
    seq_len,
    cos_bs: tl.constexpr,
    hd: tl.constexpr,
    pad_hd: tl.constexpr,
):
    """
    Forward:  y1 = x1*cos - x2*sin,   y2 = x2*cos + x1*sin
    Backward: dx1 = dy1*cos + dy2*sin, dx2 = dy2*cos - dy1*sin
              ← same formula with sin replaced by -sin

    每个 program 处理一个 (batch, head, seq_pos) 的 hd 个元素。
    x[b, h, s, :] 在内存中连续（stride(3)=1），coalesced access。
    """
    pid = tl.program_id(0).to(tl.int64)

    b   = pid // (n_heads * seq_len)
    rem = pid %  (n_heads * seq_len)
    h   = rem // seq_len
    s   = rem %  seq_len

    row_off   = b * x_bs_stride + h * x_h_stride + s * x_s_stride
    x_in_row  = x_in_ptr  + row_off
    x_out_row = x_out_ptr + row_off

    cos_offset = tl.where(
        cos_bs == 1,
        s * cos_s_stride,
        b * cos_bs_stride + s * cos_s_stride,
    )

    cs_idx  = tl.arange(0, pad_hd // 2)
    cs_mask = cs_idx < hd // 2

    cos_row = tl.load(cos_ptr + cos_offset + cs_idx, mask=cs_mask, other=0.0)
    sin_row = tl.load(sin_ptr + cos_offset + cs_idx, mask=cs_mask, other=0.0)

    x1 = tl.load(x_in_row + cs_idx,           mask=cs_mask, other=0.0).to(cos_row.dtype)
    x2 = tl.load(x_in_row + hd // 2 + cs_idx, mask=cs_mask, other=0.0).to(cos_row.dtype)

    y1 = x1 * cos_row - x2 * sin_row
    y2 = x2 * cos_row + x1 * sin_row

    tl.store(x_out_row + cs_idx,           y1, mask=cs_mask)
    tl.store(x_out_row + hd // 2 + cs_idx, y2, mask=cs_mask)


# ============================================================================
# 工具函数
# ============================================================================

def _rope_num_warps(hd: int) -> int:
    half = triton.next_power_of_2(hd // 2)
    return max(1, min(half // 32, 32))


def _launch(x_in, x_out, cos, sin, n_heads, nw):
    """Launch rope kernel for one tensor (Q or K)."""
    bsz, _, seq_len, hd = x_in.shape
    pad_hd = triton.next_power_of_2(hd)
    cos_bs = cos.shape[0]
    _rope_fwd_kernel[(bsz * n_heads * seq_len,)](
        x_in, x_out,
        x_in.stride(0), x_in.stride(1), x_in.stride(2),
        cos, cos.stride(0), cos.stride(1), sin,
        n_heads, seq_len, cos_bs, hd, pad_hd,
        num_warps=nw,
    )


# ============================================================================
# Python Wrappers
# ============================================================================

def rope_forward(q, k, cos, sin):
    """
    RoPE Forward（分配新输出张量，避免 autograd leaf 原地修改问题）

    Args:
        q:   (bsz, n_qh, seq_len, hd)
        k:   (bsz, n_kh, seq_len, hd)
        cos: (1, seq_len, hd//2) or (bsz, seq_len, hd//2)
        sin: same shape as cos

    Returns:
        q_out, k_out (new tensors), cos, sin
    """
    bsz, n_qh, seq_len, hd = q.shape
    n_kh = k.shape[1]
    nw   = _rope_num_warps(hd)

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    _launch(q, q_out, cos, sin, n_qh, nw)
    _launch(k, k_out, cos, sin, n_kh, nw)

    return q_out, k_out, cos, sin


def rope_backward(dq, dk, cos, sin):
    """
    RoPE Backward（共用 forward kernel，传 -sin 实现转置旋转）

    原理：RoPE 是正交变换，其逆等于转置，等价于旋转角取反（即 sin 取反）。
    dx = R(-θ) * dy = forward(dy, cos, -sin)

    Args:
        dq: (bsz, n_qh, seq_len, hd)
        dk: (bsz, n_kh, seq_len, hd)
        cos, sin: 与 forward 相同

    Returns:
        dq_out, dk_out (new tensors)
    """
    # PyTorch may pass zero-stride broadcast tensors for gradient of sum()
    # Triton requires contiguous memory; materialize before kernel launch.
    if not dq.is_contiguous():
        dq = dq.contiguous()
    if not dk.is_contiguous():
        dk = dk.contiguous()

    bsz, n_qh, seq_len, hd = dq.shape
    n_kh    = dk.shape[1]
    nw      = _rope_num_warps(hd)
    neg_sin = -sin  # 取反 sin 实现 backward（= R^T = R(-θ)）

    dq_out = torch.empty_like(dq)
    dk_out = torch.empty_like(dk)

    _launch(dq, dq_out, cos, neg_sin, n_qh, nw)
    _launch(dk, dk_out, cos, neg_sin, n_kh, nw)

    return dq_out, dk_out


# ============================================================================
# PyTorch Autograd Function
# ============================================================================

class HildaRopeFunction(torch.autograd.Function):
    """
    RoPE with per-head kernel（无 transpose+contiguous 开销）

    Interface 与 Liger 兼容：
        q, k = HildaRopeFunction.apply(q, k, cos, sin)
    """

    @staticmethod
    def forward(ctx, q, k, cos, sin):
        q_out, k_out, cos, sin = rope_forward(q, k, cos, sin)
        ctx.save_for_backward(cos, sin)
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        dq_out, dk_out = rope_backward(dq, dk, cos, sin)
        return dq_out, dk_out, None, None


def hilda_rope(q, k, cos, sin):
    """Functional interface for RoPE"""
    return HildaRopeFunction.apply(q, k, cos, sin)


# ============================================================================
# nn.Module
# ============================================================================

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(max_position_embeddings, device)

    def _set_cos_sin_cache(self, seq_len, device):
        t     = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, :, :self.dim // 2])
        self.register_buffer("sin_cached", emb.sin()[None, :, :self.dim // 2])

    def forward(self, x, seq_len=None):
        return self.cos_cached[:, :seq_len], self.sin_cached[:, :seq_len]

    def apply_rotary(self, q, k, cos, sin):
        return hilda_rope(q, k, cos, sin)
