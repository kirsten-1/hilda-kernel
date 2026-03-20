"""
CrossEntropy Loss - Tiled Fused Triton Kernel

核心优化：
1. Inference 继续走单次读取 logits 的 tiled forward kernel
2. 训练路径改成在 forward 中直接预计算 dlogits，backward 只做必要缩放
3. vocab 维度按 tile 处理，支持任意 V（含 LLaMA-3 V=128K）
4. 对常见 no-ignore 训练场景做专门 fast path
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# AutoTune 配置
# ============================================================================

def _get_ce_configs():
    configs = []
    for block_v in [512, 1024, 2048, 4096, 8192, 16384]:
        for num_warps in [4, 8, 16, 32]:
            configs.append(triton.Config({'BLOCK_V': block_v}, num_warps=num_warps))
    return configs


# ============================================================================
# Inference Forward Kernel
# ============================================================================

@triton.autotune(configs=_get_ce_configs(), key=['n_cols'])
@triton.jit
def _ce_forward_kernel(
    logits_ptr, logits_row_stride,
    targets_ptr, lse_ptr, loss_ptr,
    n_cols, ignore_index,
    BLOCK_V: tl.constexpr,
):
    """
    Single-pass inference forward: online logsumexp + extract target_logit.
    One program per token row, O(BLOCK_V) registers regardless of vocab size.
    """
    row = tl.program_id(0).to(tl.int64)
    row_base = row * logits_row_stride
    target = tl.load(targets_ptr + row).to(tl.int64)

    m = float('-inf')
    d = 0.0
    target_logit = 0.0

    for col_start in range(0, n_cols, BLOCK_V):
        cols = col_start + tl.arange(0, BLOCK_V)
        mask = cols < n_cols
        x = tl.load(logits_ptr + row_base + cols, mask=mask, other=-float('inf')).to(tl.float32)
        tile_max = tl.max(x, axis=0)
        m_new = tl.maximum(m, tile_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new
        target_logit += tl.sum(tl.where(cols == target, x, 0.0))

    lse = m + tl.log(d)
    tl.store(lse_ptr + row, lse)
    is_valid = target != ignore_index
    tl.store(loss_ptr + row, tl.where(is_valid, lse - target_logit, 0.0))


# ============================================================================
# Training Forward Kernels
# ============================================================================

@triton.autotune(configs=_get_ce_configs(), key=['n_cols'])
@triton.jit
def _ce_train_forward_kernel_no_ignore(
    logits_ptr, logits_row_stride,
    targets_ptr, loss_ptr, dlogits_ptr,
    n_cols, loss_scale,
    BLOCK_V: tl.constexpr,
):
    """
    Training fast path for the common no-ignore case.
    Computes loss and dlogits together so backward becomes almost free.
    """
    row = tl.program_id(0).to(tl.int64)
    row_base = row * logits_row_stride
    target = tl.load(targets_ptr + row).to(tl.int64)
    target_logit = tl.load(logits_ptr + row_base + target).to(tl.float32)

    m = float('-inf')
    d = 0.0

    for col_start in range(0, n_cols, BLOCK_V):
        cols = col_start + tl.arange(0, BLOCK_V)
        mask = cols < n_cols
        x = tl.load(logits_ptr + row_base + cols, mask=mask, other=-float('inf')).to(tl.float32)
        tile_max = tl.max(x, axis=0)
        m_new = tl.maximum(m, tile_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new

    lse = m + tl.log(d)
    tl.store(loss_ptr + row, (lse - target_logit) * loss_scale)

    for col_start in range(0, n_cols, BLOCK_V):
        cols = col_start + tl.arange(0, BLOCK_V)
        mask = cols < n_cols
        x = tl.load(logits_ptr + row_base + cols, mask=mask, other=-float('inf'))
        softmax = tl.exp(x.to(tl.float32) - lse)
        one_hot = tl.where(cols == target, 1.0, 0.0)
        grad = (softmax - one_hot) * loss_scale
        tl.store(dlogits_ptr + row_base + cols, grad.to(x.dtype), mask=mask)


@triton.autotune(configs=_get_ce_configs(), key=['n_cols'])
@triton.jit
def _ce_train_forward_kernel_ignore(
    logits_ptr, logits_row_stride,
    targets_ptr, loss_ptr, dlogits_ptr,
    n_valid_ptr, n_cols, ignore_index,
    SCALE_BY_N_VALID: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """
    Training path with ignore_index support.
    Ignored rows are zeroed here so backward can directly reuse saved dlogits.
    """
    row = tl.program_id(0).to(tl.int64)
    row_base = row * logits_row_stride
    target = tl.load(targets_ptr + row).to(tl.int64)

    if target == ignore_index:
        tl.store(loss_ptr + row, 0.0)
        for col_start in range(0, n_cols, BLOCK_V):
            cols = col_start + tl.arange(0, BLOCK_V)
            mask = cols < n_cols
            tl.store(dlogits_ptr + row_base + cols, 0.0, mask=mask)
        return

    loss_scale = 1.0
    if SCALE_BY_N_VALID:
        loss_scale = 1.0 / tl.maximum(tl.load(n_valid_ptr).to(tl.float32), 1.0)

    target_logit = tl.load(logits_ptr + row_base + target).to(tl.float32)
    m = float('-inf')
    d = 0.0

    for col_start in range(0, n_cols, BLOCK_V):
        cols = col_start + tl.arange(0, BLOCK_V)
        mask = cols < n_cols
        x = tl.load(logits_ptr + row_base + cols, mask=mask, other=-float('inf')).to(tl.float32)
        tile_max = tl.max(x, axis=0)
        m_new = tl.maximum(m, tile_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new

    lse = m + tl.log(d)
    tl.store(loss_ptr + row, (lse - target_logit) * loss_scale)

    for col_start in range(0, n_cols, BLOCK_V):
        cols = col_start + tl.arange(0, BLOCK_V)
        mask = cols < n_cols
        x = tl.load(logits_ptr + row_base + cols, mask=mask, other=-float('inf'))
        softmax = tl.exp(x.to(tl.float32) - lse)
        one_hot = tl.where(cols == target, 1.0, 0.0)
        grad = (softmax - one_hot) * loss_scale
        tl.store(dlogits_ptr + row_base + cols, grad.to(x.dtype), mask=mask)


# ============================================================================
# Gradient Scaling Kernel
# ============================================================================

@triton.autotune(configs=_get_ce_configs(), key=['n_cols'])
@triton.jit
def _ce_scale_grad_kernel(
    out_ptr, in_ptr, row_stride,
    grad_ptr, n_cols,
    PER_ROW_GRAD: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    row_base = row * row_stride
    scale = tl.load(grad_ptr + row) if PER_ROW_GRAD else tl.load(grad_ptr)

    for col_start in range(0, n_cols, BLOCK_V):
        cols = col_start + tl.arange(0, BLOCK_V)
        mask = cols < n_cols
        x = tl.load(in_ptr + row_base + cols, mask=mask, other=0.0)
        tl.store(out_ptr + row_base + cols, (x.to(tl.float32) * scale).to(x.dtype), mask=mask)


# ============================================================================
# Python Wrappers
# ============================================================================

def _n_valid_tensor(targets, ignore_index):
    if ignore_index == -1:
        return torch.full((), targets.numel(), device=targets.device, dtype=torch.int32)
    return (targets != ignore_index).sum(dtype=torch.int32)


def _fast_scalar_is_one(x):
    return x.ndim == 0 and torch.equal(x, torch.ones_like(x))


def cross_entropy_forward(logits, targets, ignore_index=-100, reduction='mean', needs_grad=False):
    """
    Fused CrossEntropy forward.
    Returns (loss, aux, n_valid), where aux is lse for inference path and precomputed dlogits for training path.
    """
    logits = logits.contiguous()
    targets = targets.contiguous()
    n_rows, n_cols = logits.shape
    per_token_loss = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
    n_valid = _n_valid_tensor(targets, ignore_index)

    if needs_grad:
        dlogits = torch.empty_like(logits)
        if ignore_index == -1:
            loss_scale = 1.0 / max(n_rows, 1) if reduction == 'mean' else 1.0
            _ce_train_forward_kernel_no_ignore[(n_rows,)](
                logits, logits.stride(0),
                targets, per_token_loss, dlogits,
                n_cols, loss_scale,
            )
        else:
            _ce_train_forward_kernel_ignore[(n_rows,)](
                logits, logits.stride(0),
                targets, per_token_loss, dlogits,
                n_valid, n_cols, ignore_index,
                SCALE_BY_N_VALID=(reduction == 'mean'),
            )

        if reduction == 'none':
            return per_token_loss, dlogits, n_valid
        return per_token_loss.sum(), dlogits, n_valid

    lse = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
    _ce_forward_kernel[(n_rows,)](
        logits, logits.stride(0),
        targets, lse, per_token_loss,
        n_cols, ignore_index,
    )

    if reduction == 'none':
        return per_token_loss, lse, n_valid
    if reduction == 'mean':
        return per_token_loss.sum() / n_valid.clamp_min(1).to(per_token_loss.dtype), lse, n_valid
    return per_token_loss.sum(), lse, n_valid


def cross_entropy_backward(dloss, saved_grad, reduction='mean'):
    """Backward for the training path. saved_grad already matches the loss reduction."""
    if reduction != 'none' and _fast_scalar_is_one(dloss):
        return saved_grad

    dlogits = torch.empty_like(saved_grad)
    n_rows, n_cols = saved_grad.shape
    _ce_scale_grad_kernel[(n_rows,)](
        dlogits, saved_grad, saved_grad.stride(0),
        dloss, n_cols,
        PER_ROW_GRAD=(dloss.ndim > 0),
    )
    return dlogits


# ============================================================================
# PyTorch Autograd Function
# ============================================================================

class HildaCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, ignore_index, reduction):
        needs_grad = logits.requires_grad
        loss, aux, _ = cross_entropy_forward(
            logits, targets, ignore_index, reduction, needs_grad=needs_grad
        )
        ctx.reduction = reduction
        ctx.has_saved_grad = needs_grad
        if needs_grad:
            ctx.save_for_backward(aux)
        return loss

    @staticmethod
    def backward(ctx, dloss):
        if not ctx.has_saved_grad:
            return None, None, None, None
        (saved_grad,) = ctx.saved_tensors
        dlogits = cross_entropy_backward(dloss, saved_grad, ctx.reduction)
        return dlogits, None, None, None


def hilda_cross_entropy(logits, targets, ignore_index=-100, reduction='mean'):
    """Functional interface for CrossEntropy."""
    return HildaCrossEntropyFunction.apply(logits, targets, ignore_index, reduction)


# ============================================================================
# nn.Module
# ============================================================================

class CrossEntropyLoss(torch.nn.Module):
    """
    Drop-in replacement for torch.nn.CrossEntropyLoss.
    Expects (N, V) logits and (N,) targets (flatten bsz*seq before passing).
    """
    def __init__(self, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        return hilda_cross_entropy(logits, targets, self.ignore_index, self.reduction)
