"""
RoPE (Rotary Position Embedding) with AutoTune

Implements the HuggingFace Llama/Mistral variant of RoPE.

Improvements over Liger-Kernel:
1. AutoTune num_warps / num_stages - Liger 硬编码 num_warps=4
2. 分离 forward/backward kernel - 比 constexpr 分支更清晰，各自独立调优
3. 保持 in-place 执行 + Q/K coarsening（同一 kernel 处理 Q 和 K，节省 launch overhead）
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# AutoTune 配置
# ============================================================================

def get_rope_configs():
    """
    RoPE 的 AutoTune 配置

    注意：RoPE 的 tile 形状由模型架构决定（n_qh, n_kh, hd），
    不适合搜索 BLOCK_SIZE。只搜索 num_warps 和 num_stages。
    """
    configs = []
    for num_warps in [2, 4, 8, 16, 32]:
        for num_stages in [1, 2, 3]:
            configs.append(triton.Config({}, num_warps=num_warps, num_stages=num_stages))
    return configs


def get_rope_num_warps(n_qh, n_kh, hd):
    """
    启发式选择 num_warps

    RoPE 是 in-place kernel，不能用 AutoTune（AutoTune 多次调用 kernel
    会反复旋转同一份数据，导致结果错误）。
    根据 tile 大小选择合理的 num_warps。
    """
    total_elements = (n_qh + n_kh) * hd
    if total_elements <= 1024:
        return 4
    elif total_elements <= 4096:
        return 8
    elif total_elements <= 16384:
        return 16
    else:
        return 32


# ============================================================================
# Forward Kernel
# ============================================================================

@triton.jit
def _rope_forward_kernel(
    q_ptr, q_row_stride,
    k_ptr, k_row_stride,
    cos_ptr, cos_row_stride,
    sin_ptr, sin_row_stride,
    sl,
    cos_bs: tl.constexpr,  # 1 或 bsz：cos/sin 是否 batch 维广播
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
):
    """
    RoPE Forward Kernel

    数学：
      y1 = x1 * cos - x2 * sin
      y2 = x2 * cos + x1 * sin
    其中 x1 = x[..., :hd//2], x2 = x[..., hd//2:]

    优化：
    1. In-place：直接写回原始 q/k 内存
    2. Q/K coarsening：一个 program 同时处理 Q 和 K 的一个 (batch, seq) 位置
    3. constexpr shape：所有 tile 维度在编译时确定，无运行时边界判断开销
    """
    pid = tl.program_id(0).to(tl.int64)

    # 各 program 负责 q/k 中的一个 (batch, seq) 位置
    q_ptr = q_ptr + pid * q_row_stride
    k_ptr = k_ptr + pid * k_row_stride

    # 定位 cos/sin 行
    batch_idx = pid // sl
    seq_idx   = pid % sl
    cos_offset = tl.where(
        cos_bs == 1,
        seq_idx * cos_row_stride,
        batch_idx * sl * cos_row_stride + seq_idx * cos_row_stride,
    )
    cos_ptr = cos_ptr + cos_offset
    sin_ptr = sin_ptr + cos_offset

    # 加载 cos/sin（只需要左半部分 hd//2）
    cs_idx = tl.arange(0, pad_hd // 2)
    cs_mask = cs_idx < hd // 2
    cos_row = tl.load(cos_ptr + cs_idx, mask=cs_mask, other=0.0)
    sin_row = tl.load(sin_ptr + cs_idx, mask=cs_mask, other=0.0)

    # ---- Process Q ----
    # Tile shape: [pad_n_qh, hd//2]，同时加载左右两半
    qh_idx = tl.arange(0, pad_n_qh)
    q_mask = (qh_idx[:, None] < n_qh) & (cs_idx[None, :] < hd // 2)
    q1_off = qh_idx[:, None] * hd + cs_idx[None, :]   # 左半部分
    q2_off = q1_off + hd // 2                          # 右半部分

    q1 = tl.load(q_ptr + q1_off, mask=q_mask, other=0.0).to(cos_row.dtype)
    q2 = tl.load(q_ptr + q2_off, mask=q_mask, other=0.0).to(cos_row.dtype)

    # In-place 写回
    tl.store(q_ptr + q1_off, q1 * cos_row[None, :] - q2 * sin_row[None, :], mask=q_mask)
    tl.store(q_ptr + q2_off, q2 * cos_row[None, :] + q1 * sin_row[None, :], mask=q_mask)

    # ---- Process K ----
    kh_idx = tl.arange(0, pad_n_kh)
    k_mask = (kh_idx[:, None] < n_kh) & (cs_idx[None, :] < hd // 2)
    k1_off = kh_idx[:, None] * hd + cs_idx[None, :]
    k2_off = k1_off + hd // 2

    k1 = tl.load(k_ptr + k1_off, mask=k_mask, other=0.0).to(cos_row.dtype)
    k2 = tl.load(k_ptr + k2_off, mask=k_mask, other=0.0).to(cos_row.dtype)

    tl.store(k_ptr + k1_off, k1 * cos_row[None, :] - k2 * sin_row[None, :], mask=k_mask)
    tl.store(k_ptr + k2_off, k2 * cos_row[None, :] + k1 * sin_row[None, :], mask=k_mask)


# ============================================================================
# Backward Kernel
# ============================================================================

@triton.jit
def _rope_backward_kernel(
    dq_ptr, dq_row_stride,
    dk_ptr, dk_row_stride,
    cos_ptr, cos_row_stride,
    sin_ptr, sin_row_stride,
    sl,
    cos_bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
):
    """
    RoPE Backward Kernel

    Forward: y1 = x1*cos - x2*sin, y2 = x2*cos + x1*sin
    Backward（对 x 求梯度）:
      dx1 = dy1*cos + dy2*sin
      dx2 = dy2*cos - dy1*sin
    即用 -sin 做一次 forward RoPE。
    """
    pid = tl.program_id(0).to(tl.int64)
    dq_ptr = dq_ptr + pid * dq_row_stride
    dk_ptr = dk_ptr + pid * dk_row_stride

    batch_idx = pid // sl
    seq_idx   = pid % sl
    cos_offset = tl.where(
        cos_bs == 1,
        seq_idx * cos_row_stride,
        batch_idx * sl * cos_row_stride + seq_idx * cos_row_stride,
    )
    cos_ptr = cos_ptr + cos_offset
    sin_ptr = sin_ptr + cos_offset

    cs_idx = tl.arange(0, pad_hd // 2)
    cs_mask = cs_idx < hd // 2
    cos_row = tl.load(cos_ptr + cs_idx, mask=cs_mask, other=0.0)
    sin_row = tl.load(sin_ptr + cs_idx, mask=cs_mask, other=0.0)

    # ---- dQ ----
    qh_idx = tl.arange(0, pad_n_qh)
    q_mask = (qh_idx[:, None] < n_qh) & (cs_idx[None, :] < hd // 2)
    q1_off = qh_idx[:, None] * hd + cs_idx[None, :]
    q2_off = q1_off + hd // 2

    dq1 = tl.load(dq_ptr + q1_off, mask=q_mask, other=0.0).to(cos_row.dtype)
    dq2 = tl.load(dq_ptr + q2_off, mask=q_mask, other=0.0).to(cos_row.dtype)

    tl.store(dq_ptr + q1_off, dq1 * cos_row[None, :] + dq2 * sin_row[None, :], mask=q_mask)
    tl.store(dq_ptr + q2_off, dq2 * cos_row[None, :] - dq1 * sin_row[None, :], mask=q_mask)

    # ---- dK ----
    kh_idx = tl.arange(0, pad_n_kh)
    k_mask = (kh_idx[:, None] < n_kh) & (cs_idx[None, :] < hd // 2)
    k1_off = kh_idx[:, None] * hd + cs_idx[None, :]
    k2_off = k1_off + hd // 2

    dk1 = tl.load(dk_ptr + k1_off, mask=k_mask, other=0.0).to(cos_row.dtype)
    dk2 = tl.load(dk_ptr + k2_off, mask=k_mask, other=0.0).to(cos_row.dtype)

    tl.store(dk_ptr + k1_off, dk1 * cos_row[None, :] + dk2 * sin_row[None, :], mask=k_mask)
    tl.store(dk_ptr + k2_off, dk2 * cos_row[None, :] - dk1 * sin_row[None, :], mask=k_mask)


# ============================================================================
# Python Wrappers
# ============================================================================

def rope_forward(q, k, cos, sin):
    """
    RoPE Forward

    Args:
        q:   (bsz, n_qh, seq_len, hd) - HuggingFace 格式
        k:   (bsz, n_kh, seq_len, hd)
        cos: (1, seq_len, hd//2) 或 (bsz, seq_len, hd//2)
        sin: same shape as cos

    Returns:
        q, k (in-place modified), cos, sin
    """
    # 转置到 (bsz, seq_len, n_h, hd)，使行步长 = n_h * hd（物理连续）
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    bsz, seq_len, n_qh, hd = q.shape
    n_kh = k.shape[2]
    pad_n_qh = triton.next_power_of_2(n_qh)
    pad_n_kh = triton.next_power_of_2(n_kh)
    pad_hd   = triton.next_power_of_2(hd)

    q   = q.contiguous()
    k   = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    cos_bs = cos.shape[0]
    num_warps = get_rope_num_warps(n_qh, n_kh, hd)

    _rope_forward_kernel[(bsz * seq_len,)](
        q, q.stride(1),
        k, k.stride(1),
        cos, cos.stride(-2),
        sin, sin.stride(-2),
        seq_len, cos_bs,
        n_qh, n_kh, hd,
        pad_n_qh, pad_n_kh, pad_hd,
        num_warps=num_warps,
    )

    return q.transpose(1, 2), k.transpose(1, 2), cos, sin


def rope_backward(dq, dk, cos, sin):
    """
    RoPE Backward

    Args:
        dq: (bsz, n_qh, seq_len, hd)
        dk: (bsz, n_kh, seq_len, hd)
        cos, sin: 与 forward 相同

    Returns:
        dq, dk (in-place modified)
    """
    dq = dq.transpose(1, 2)
    dk = dk.transpose(1, 2)

    bsz, seq_len, n_qh, hd = dq.shape
    n_kh = dk.shape[2]
    pad_n_qh = triton.next_power_of_2(n_qh)
    pad_n_kh = triton.next_power_of_2(n_kh)
    pad_hd   = triton.next_power_of_2(hd)

    dq = dq.contiguous()
    dk = dk.contiguous()

    cos_bs = cos.shape[0]
    num_warps = get_rope_num_warps(n_qh, n_kh, hd)

    _rope_backward_kernel[(bsz * seq_len,)](
        dq, dq.stride(1),
        dk, dk.stride(1),
        cos, cos.stride(-2),
        sin, sin.stride(-2),
        seq_len, cos_bs,
        n_qh, n_kh, hd,
        pad_n_qh, pad_n_kh, pad_hd,
        num_warps=num_warps,
    )

    return dq.transpose(1, 2), dk.transpose(1, 2)


# ============================================================================
# PyTorch Autograd Function
# ============================================================================

class HildaRopeFunction(torch.autograd.Function):
    """
    RoPE with AutoTune

    Interface 与 Liger 兼容：
        q, k = HildaRopeFunction.apply(q, k, cos, sin)
    """

    @staticmethod
    def forward(ctx, q, k, cos, sin):
        """
        q: (bsz, n_qh, seq_len, hd)
        k: (bsz, n_kh, seq_len, hd)
        cos: (1, seq_len, hd//2) or (bsz, seq_len, hd//2)
        sin: same as cos
        """
        q, k, cos, sin = rope_forward(q, k, cos, sin)
        ctx.save_for_backward(cos, sin)
        return q, k

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        dq, dk = rope_backward(dq, dk, cos, sin)
        return dq, dk, None, None


def hilda_rope(q, k, cos, sin):
    """Functional interface for RoPE"""
    return HildaRopeFunction.apply(q, k, cos, sin)


# ============================================================================
# nn.Module（用于直接替换 HuggingFace 模型中的 RoPE）
# ============================================================================

class RotaryEmbedding(torch.nn.Module):
    """
    Drop-in replacement for HuggingFace LlamaRotaryEmbedding

    用法：
        rope = RotaryEmbedding(head_dim, max_seq_len)
        cos, sin = rope(x, seq_len)
        q, k = rope.apply(q, k, cos, sin)
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(max_position_embeddings, device)

    def _set_cos_sin_cache(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, :, :self.dim // 2])  # (1, seq, dim//2)
        self.register_buffer("sin_cached", emb.sin()[None, :, :self.dim // 2])

    def forward(self, x, seq_len=None):
        return self.cos_cached[:, :seq_len], self.sin_cached[:, :seq_len]

    def apply_rotary(self, q, k, cos, sin):
        return hilda_rope(q, k, cos, sin)
