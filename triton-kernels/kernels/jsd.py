"""
Jensen-Shannon Divergence (JSD) - Fused Triton Kernel

JSD(β)(P || Q) = β * KL(P || M) + (1-β) * KL(Q || M)
where M = β * P + (1-β) * Q

输入约定:
  X: student log-probs (log Q)  [BT, V]
  Y: teacher log-probs (log P)  [BT, V]
  β=0 → forward KL(P||Q), β=1 → reverse KL(Q||P), β=0.5 → symmetric JSD

核心优化:
1. Forward: single-pass loss + gradient computation (no separate backward kernel)
2. Gradient stored alongside loss, backward is just scalar multiply
3. Tiled vocab loop with BLOCK_V
"""

import torch
import triton
import triton.language as tl

MAX_FUSED_SIZE = 65536


# ============================================================================
# Forward + gradient kernel (single pass)
# ============================================================================

@triton.jit
def _jsd_kernel(
    X_ptr, Y_ptr, loss_ptr, dX_ptr,
    x_stride, y_stride,
    n_cols,
    n_non_ignore,
    ignore_index: tl.constexpr,
    label_ptr,
    BETA: tl.constexpr,
    BLOCK_V: tl.constexpr,
    HAS_LABEL: tl.constexpr,
):
    """Compute JSD loss and dX per row. Stores per-row loss sum."""
    row = tl.program_id(0)
    x_base = row * x_stride
    y_base = row * y_stride

    # Check ignore_index
    if HAS_LABEL:
        label = tl.load(label_ptr + row)
        if label == ignore_index:
            tl.store(loss_ptr + row, 0.0)
            # Zero out dX
            for col_start in range(0, n_cols, BLOCK_V):
                cols = col_start + tl.arange(0, BLOCK_V)
                mask = cols < n_cols
                tl.store(dX_ptr + x_base + cols, 0.0, mask=mask)
            return

    loss_sum = 0.0

    for col_start in range(0, n_cols, BLOCK_V):
        cols = col_start + tl.arange(0, BLOCK_V)
        mask = cols < n_cols

        x = tl.load(X_ptr + x_base + cols, mask=mask, other=0.0).to(tl.float32)  # log Q
        y = tl.load(Y_ptr + y_base + cols, mask=mask, other=0.0).to(tl.float32)  # log P

        if BETA == 0.0:
            # Forward KL: KL(P||Q) = sum(P * (Y - X))
            p = tl.exp(y)
            loss = p * (y - x)
            dx = -p / n_non_ignore
        elif BETA == 1.0:
            # Reverse KL: KL(Q||P) = sum(Q * (X - Y))
            q = tl.exp(x)
            loss = q * (x - y)
            dx = q * (x - y + 1.0) / n_non_ignore
        else:
            # General JSD: β*KL(P||M) + (1-β)*KL(Q||M)
            q = tl.exp(x)
            p = tl.exp(y)
            m = BETA * p + (1.0 - BETA) * q
            log_m = tl.log(m)
            loss = BETA * p * (y - log_m) + (1.0 - BETA) * q * (x - log_m)
            dx = (1.0 - BETA) * q * (x - log_m) / n_non_ignore

        loss = tl.where(mask, loss, 0.0)
        loss_sum += tl.sum(loss, axis=0)
        tl.store(dX_ptr + x_base + cols, dx, mask=mask)

    tl.store(loss_ptr + row, loss_sum / n_non_ignore)


# ============================================================================
# Python wrappers
# ============================================================================

def jsd_forward(X, Y, beta=0.5, labels=None, ignore_index=-100):
    """
    Args:
        X: student log-probs [BT, V]
        Y: teacher log-probs [BT, V]
        beta: interpolation weight (0=fwd KL, 0.5=JSD, 1=rev KL)
        labels: optional [BT] for ignore_index masking
    Returns:
        loss: scalar
        dX: [BT, V] gradient w.r.t. X (pre-computed)
    """
    X = X.contiguous()
    Y = Y.contiguous()
    assert X.shape == Y.shape
    n_rows, n_cols = X.shape

    has_label = labels is not None
    if has_label:
        n_non_ignore = max(int((labels != ignore_index).sum().item()), 1)
    else:
        n_non_ignore = n_rows

    loss_per_row = torch.empty(n_rows, dtype=torch.float32, device=X.device)
    dX = torch.empty_like(X)

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    nw = 32

    _jsd_kernel[(n_rows,)](
        X, Y, loss_per_row, dX,
        X.stride(0), Y.stride(0),
        n_cols, n_non_ignore, ignore_index,
        labels,
        BETA=beta,
        BLOCK_V=BLOCK_SIZE, num_warps=nw,
        HAS_LABEL=has_label,
    )

    return loss_per_row.sum(), dX


# ============================================================================
# Autograd Function
# ============================================================================

class HildaJSDFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, Y, beta, labels, ignore_index):
        loss, dX = jsd_forward(X, Y, beta, labels, ignore_index)
        ctx.save_for_backward(dX)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        dX, = ctx.saved_tensors
        return dX * grad_output, None, None, None, None


def hilda_jsd(X, Y, beta=0.5, labels=None, ignore_index=-100):
    """Functional interface: JSD(β)(teacher || student)."""
    return HildaJSDFunction.apply(X, Y, beta, labels, ignore_index)


# ============================================================================
# nn.Module
# ============================================================================

class JSDLoss(torch.nn.Module):
    """Jensen-Shannon Divergence loss."""
    def __init__(self, beta=0.5, ignore_index=-100):
        super().__init__()
        self.beta = beta
        self.ignore_index = ignore_index

    def forward(self, X, Y, labels=None):
        return hilda_jsd(X, Y, self.beta, labels, self.ignore_index)
