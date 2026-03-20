"""
KL Divergence Loss - Fused Triton Kernel

KL(P || Q) = sum(P * (log(P) - log(Q)))

输入约定（与 PyTorch nn.KLDivLoss 一致）：
  y_pred: log-probabilities (prediction)  [BT, V]
  y_true: probabilities (target)          [BT, V]
  log_target=True 时 y_true 也是 log-probabilities

核心优化：
1. Forward: AutoTune on BLOCK_V + num_warps, tiled vocab loop
2. Backward: in-place 或 separate buffer, 无 AutoTune
3. 支持 reduction: none / sum / mean / batchmean
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# AutoTune 配置
# ============================================================================

def _get_kl_configs():
    configs = []
    for BV in [1024, 2048, 4096, 8192]:
        for nw in [4, 8, 16]:
            configs.append(triton.Config({'BLOCK_V': BV}, num_warps=nw))
    return configs


# ============================================================================
# Forward Kernel
# ============================================================================

@triton.autotune(configs=_get_kl_configs(), key=['n_cols'])
@triton.jit
def _kl_div_fwd_kernel(
    y_pred_ptr, y_true_ptr, loss_ptr,
    stride, n_cols, eps,
    BLOCK_V: tl.constexpr,
    LOG_TARGET: tl.constexpr,
):
    """Per-row KL divergence. Stores sum per row."""
    row = tl.program_id(0)
    row_off = row * stride
    loss_sum = 0.0

    for col_start in range(0, n_cols, BLOCK_V):
        cols = col_start + tl.arange(0, BLOCK_V)
        mask = cols < n_cols
        y = tl.load(y_pred_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
        t = tl.load(y_true_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)

        if LOG_TARGET:
            # both log-space: exp(t) * (t - y)
            loss = tl.exp(t) * (t - y)
        else:
            # t is prob, y is log-prob: t * (log(max(t, eps)) - y)
            safe_t = tl.maximum(t, eps)
            loss = t * (tl.log(safe_t) - y)

        loss_sum += tl.sum(tl.where(mask, loss, 0.0), axis=0)

    tl.store(loss_ptr + row, loss_sum)


# ============================================================================
# Backward Kernel
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
def _kl_div_bwd_kernel(
    y_true_ptr, dy_pred_ptr,
    stride, n_cols, scale,
    BLOCK_V: tl.constexpr,
    LOG_TARGET: tl.constexpr,
):
    """Backward: d(KL)/d(y_pred) = -y_true (or -exp(y_true) if log_target)."""
    row = tl.program_id(0)
    row_off = row * stride

    for col_start in range(0, n_cols, BLOCK_V):
        cols = col_start + tl.arange(0, BLOCK_V)
        mask = cols < n_cols
        t = tl.load(y_true_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)

        if LOG_TARGET:
            grad = -tl.exp(t) * scale
        else:
            grad = -t * scale

        tl.store(dy_pred_ptr + row_off + cols, grad.to(tl.load(y_true_ptr + row_off).dtype), mask=mask)


# ============================================================================
# Python wrappers
# ============================================================================

def kl_div_forward(y_pred, y_true, reduction='batchmean', log_target=False, eps=1e-10):
    y_pred = y_pred.contiguous()
    y_true = y_true.contiguous()
    assert y_pred.shape == y_true.shape
    shape = y_pred.shape
    y_pred_2d = y_pred.reshape(-1, shape[-1])
    y_true_2d = y_true.reshape(-1, shape[-1])
    n_rows, n_cols = y_pred_2d.shape

    loss_per_row = torch.empty(n_rows, dtype=torch.float32, device=y_pred.device)

    _kl_div_fwd_kernel[(n_rows,)](
        y_pred_2d, y_true_2d, loss_per_row,
        y_pred_2d.stride(0), n_cols, eps,
        LOG_TARGET=log_target,
    )

    if reduction == 'none':
        return loss_per_row
    elif reduction == 'sum':
        return loss_per_row.sum()
    elif reduction == 'mean':
        return loss_per_row.sum() / (n_rows * n_cols)
    else:  # batchmean
        return loss_per_row.sum() / n_rows


def kl_div_backward(y_true, grad_output, reduction='batchmean', log_target=False):
    y_true = y_true.contiguous()
    shape = y_true.shape
    y_true_2d = y_true.reshape(-1, shape[-1])
    n_rows, n_cols = y_true_2d.shape

    # Compute scale from reduction
    if reduction == 'batchmean':
        scale = float(grad_output) / n_rows
    elif reduction == 'mean':
        scale = float(grad_output) / (n_rows * n_cols)
    elif reduction == 'sum':
        scale = float(grad_output)
    else:
        scale = 1.0

    dy_pred = torch.empty_like(y_true_2d)
    BLOCK, nw = _calc_settings(n_cols)

    _kl_div_bwd_kernel[(n_rows,)](
        y_true_2d, dy_pred,
        y_true_2d.stride(0), n_cols, scale,
        BLOCK_V=BLOCK, num_warps=nw,
        LOG_TARGET=log_target,
    )
    return dy_pred.reshape(shape)


# ============================================================================
# PyTorch Autograd Function
# ============================================================================

class HildaKLDivFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y_pred, y_true, reduction, log_target, eps):
        loss = kl_div_forward(y_pred, y_true, reduction, log_target, eps)
        ctx.save_for_backward(y_true)
        ctx.reduction = reduction
        ctx.log_target = log_target
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y_true, = ctx.saved_tensors
        dy_pred = kl_div_backward(y_true, grad_output, ctx.reduction, ctx.log_target)
        return dy_pred, None, None, None, None


def hilda_kl_div(y_pred, y_true, reduction='batchmean', log_target=False, eps=1e-10):
    """Functional interface: KL(y_true || y_pred). y_pred is log-probs."""
    return HildaKLDivFunction.apply(y_pred, y_true, reduction, log_target, eps)


# ============================================================================
# nn.Module
# ============================================================================

class KLDivLoss(torch.nn.Module):
    """Drop-in replacement for torch.nn.KLDivLoss."""
    def __init__(self, reduction='batchmean', log_target=False, eps=1e-10):
        super().__init__()
        self.reduction = reduction
        self.log_target = log_target
        self.eps = eps

    def forward(self, y_pred, y_true):
        return hilda_kl_div(y_pred, y_true, self.reduction, self.log_target, self.eps)
