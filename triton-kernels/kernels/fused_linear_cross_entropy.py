"""
FusedLinearCrossEntropy - Fused lm_head + CrossEntropy

避免实例化完整 (BT, V) logits tensor，V=128K 时省 ~2GB 显存。

策略：将 BT tokens 分成 chunks，每个 chunk:
1. logits_chunk = input_chunk @ weight.T    → (chunk_size, V)
2. CE forward on logits_chunk (in-place → scaled dlogits)
3. grad_input_chunk = dlogits @ weight      → (chunk_size, H)
4. grad_weight += dlogits.T @ input_chunk   → (V, H)
5. logits_chunk 释放

v2 优化:
- matmul 全程 compute dtype (fp16/bf16)，不转 fp32
- scale 折入 kernel，省掉单独的 * scale + fp32 buffer
- grad_weight 在 compute dtype 累加
- BLOCK_V=32768, num_warps=32 (匹配 Liger)
"""

import torch
import triton
import triton.language as tl

MAX_FUSED_SIZE = 65536 // 2  # 32768


# ============================================================================
# CE forward kernel: in-place write scaled dlogits (no AutoTune!)
# ============================================================================

@triton.jit
def _ce_fwd_inplace_kernel(
    logits_ptr, logits_row_stride,
    targets_ptr, loss_ptr,
    n_cols,
    ignore_index: tl.constexpr,
    scale,
    BLOCK_V: tl.constexpr,
    COMPUTE_GRAD: tl.constexpr,
):
    """2-pass CE forward + write (softmax - onehot) * scale in-place."""
    row = tl.program_id(0)
    row_base = row * logits_row_stride
    target = tl.load(targets_ptr + row).to(tl.int64)

    # Pass 1: online logsumexp + extract target logit
    m = float('-inf')
    d = 0.0
    target_logit = 0.0

    for col_start in range(0, n_cols, BLOCK_V):
        cols = col_start + tl.arange(0, BLOCK_V)
        mask = cols < n_cols
        x = tl.load(logits_ptr + row_base + cols,
                     mask=mask, other=-float('inf')).to(tl.float32)
        tile_max = tl.max(x, axis=0)
        m_new = tl.maximum(m, tile_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new
        target_logit += tl.sum(tl.where(cols == target, x, 0.0))

    lse = m + tl.log(d)
    is_valid = (target != ignore_index)
    loss = tl.where(is_valid, lse - target_logit, 0.0)
    tl.store(loss_ptr + row, loss)

    # Pass 2: write dlogits = (softmax - one_hot) * scale in-place
    if COMPUTE_GRAD:
        for col_start in range(0, n_cols, BLOCK_V):
            cols = col_start + tl.arange(0, BLOCK_V)
            mask = cols < n_cols
            x = tl.load(logits_ptr + row_base + cols,
                         mask=mask, other=-float('inf')).to(tl.float32)
            softmax = tl.exp(x - lse)
            one_hot = tl.where(cols == target, 1.0, 0.0)
            dlogit = tl.where(is_valid, (softmax - one_hot) * scale, 0.0)
            tl.store(logits_ptr + row_base + cols, dlogit, mask=mask)


# ============================================================================
# Element-wise multiply kernel for backward (scalar * tensor, in-place)
# ============================================================================

@triton.jit
def _element_mul_kernel(
    x_ptr, x_stride, scalar_ptr, n_cols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    s = tl.load(scalar_ptr)
    row_base = row * x_stride
    for col_start in range(0, n_cols, BLOCK):
        cols = col_start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + row_base + cols, mask=mask)
        tl.store(x_ptr + row_base + cols, x * s, mask=mask)


# ============================================================================
# Python wrapper
# ============================================================================

def fused_linear_cross_entropy_forward(
    hidden_states, weight, targets,
    ignore_index=-100, reduction='mean',
    compute_grad=True,
):
    BT, H = hidden_states.shape
    V = weight.shape[0]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    inc_factor = triton.cdiv(V, H)
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))
    num_chunks = triton.cdiv(BT, chunk_size)

    loss_1d = torch.zeros(BT, dtype=torch.float32, device=hidden_states.device)
    grad_input = torch.zeros_like(hidden_states) if compute_grad else None
    grad_weight = torch.zeros_like(weight) if compute_grad else None

    # Count valid tokens
    n_valid = int((targets != ignore_index).sum().item())

    # Scale factor folded into kernel
    if reduction == 'mean':
        scale = 1.0 / max(n_valid, 1)
    else:
        scale = 1.0

    nw = 32

    for chunk_id in range(num_chunks):
        start = chunk_id * chunk_size
        end = min(start + chunk_size, BT)
        cs = end - start

        input_chunk = hidden_states[start:end]
        target_chunk = targets[start:end]

        # 1. Linear in compute dtype (fp16/bf16 cuBLAS)
        logits_chunk = input_chunk @ weight.t()

        # 2. CE forward + in-place scaled dlogits
        logits_chunk = logits_chunk.contiguous()
        _ce_fwd_inplace_kernel[(cs,)](
            logits_chunk, logits_chunk.stride(0),
            target_chunk, loss_1d[start:end],
            V, ignore_index, scale,
            BLOCK_V=BLOCK_SIZE, num_warps=nw,
            COMPUTE_GRAD=compute_grad,
        )

        if compute_grad:
            # logits_chunk is now scaled dlogits in fp32 from kernel
            # Cast back to compute dtype for matmul
            dlogits = logits_chunk.to(hidden_states.dtype)

            # 3. grad_input in compute dtype
            grad_input[start:end] = dlogits @ weight

            # 4. grad_weight: mm in compute dtype, accumulate
            grad_weight += dlogits.t() @ input_chunk

    # Final loss
    if reduction == 'mean':
        loss = loss_1d.sum() / max(n_valid, 1)
    elif reduction == 'sum':
        loss = loss_1d.sum()
    else:
        loss = loss_1d

    return loss, grad_input, grad_weight, n_valid


# ============================================================================
# PyTorch Autograd Function
# ============================================================================

class HildaFusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, targets, ignore_index, reduction):
        compute_grad = hidden_states.requires_grad or weight.requires_grad
        loss, grad_input, grad_weight, n_valid = fused_linear_cross_entropy_forward(
            hidden_states, weight, targets, ignore_index, reduction,
            compute_grad=compute_grad,
        )
        if compute_grad:
            ctx.save_for_backward(grad_input, grad_weight)
        ctx.compute_grad = compute_grad
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.compute_grad:
            return None, None, None, None, None
        grad_input, grad_weight = ctx.saved_tensors
        # If grad_output != 1.0, scale in-place via Triton kernel
        if not torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
            BT, H = grad_input.shape
            V = grad_weight.shape[0]
            BLOCK = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))
            _element_mul_kernel[(BT,)](
                grad_input, grad_input.stride(0), grad_output, H,
                BLOCK=BLOCK, num_warps=32,
            )
            BLOCK_W = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))
            _element_mul_kernel[(V,)](
                grad_weight, grad_weight.stride(0), grad_output, H,
                BLOCK=BLOCK_W, num_warps=32,
            )
        return grad_input, grad_weight, None, None, None


def hilda_fused_linear_cross_entropy(
    hidden_states, weight, targets,
    ignore_index=-100, reduction='mean',
):
    """Functional interface: fused lm_head + CE loss."""
    return HildaFusedLinearCrossEntropyFunction.apply(
        hidden_states, weight, targets, ignore_index, reduction,
    )


# ============================================================================
# nn.Module
# ============================================================================

class FusedLinearCrossEntropyLoss(torch.nn.Module):
    """
    Fused linear projection + cross-entropy loss.
    Avoids materializing full (BT, V) logits.
    """
    def __init__(self, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, hidden_states, weight, targets):
        return hilda_fused_linear_cross_entropy(
            hidden_states, weight, targets,
            self.ignore_index, self.reduction,
        )
