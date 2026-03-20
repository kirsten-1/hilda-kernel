"""
FusedLinearJSD - Fused lm_head + JSD Loss

Fuses linear projection + JSD loss, avoids materializing full (BT, V) logits.
Same chunking strategy as FusedLinearCE.

Student: hidden_states @ student_weight.T → log_softmax → X
Teacher: teacher_hidden @ teacher_weight.T → log_softmax → Y (detached)
Loss: JSD(β)(P || Q) where P=exp(Y), Q=exp(X)

Gradient flows only through student path.
"""

import torch
import triton
import triton.language as tl
from .jsd import _jsd_kernel

MAX_FUSED_SIZE = 65536


# ============================================================================
# Element-wise multiply kernel (for backward scaling)
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
# Log-softmax kernel (tiled, writes in-place)
# ============================================================================

@triton.jit
def _log_softmax_kernel(
    X_ptr, x_stride, n_cols, temperature,
    BLOCK_V: tl.constexpr,
):
    """In-place: X[row] = log_softmax(X[row] / temperature)."""
    row = tl.program_id(0)
    row_base = row * x_stride

    # Pass 1: online logsumexp
    m = float('-inf')
    d = 0.0
    for col_start in range(0, n_cols, BLOCK_V):
        cols = col_start + tl.arange(0, BLOCK_V)
        mask = cols < n_cols
        x = tl.load(X_ptr + row_base + cols, mask=mask, other=-float('inf')).to(tl.float32) / temperature
        tile_max = tl.max(x, axis=0)
        m_new = tl.maximum(m, tile_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new
    lse = m + tl.log(d)

    # Pass 2: write log_softmax in-place
    for col_start in range(0, n_cols, BLOCK_V):
        cols = col_start + tl.arange(0, BLOCK_V)
        mask = cols < n_cols
        x = tl.load(X_ptr + row_base + cols, mask=mask, other=-float('inf')).to(tl.float32) / temperature
        log_sm = x - lse
        tl.store(X_ptr + row_base + cols, log_sm, mask=mask)


# ============================================================================
# Softmax-grad kernel: dlogits = (dX - softmax * sum(dX)) / temperature
# ============================================================================

@triton.jit
def _softmax_grad_kernel(
    dX_ptr, logprob_ptr, dlogits_ptr,
    stride, n_cols, temperature,
    BLOCK_V: tl.constexpr,
):
    """Compute dlogits from dX and log-probs: dlogits_j = (dX_j - p_j * sum(dX)) / T."""
    row = tl.program_id(0)
    row_base = row * stride

    # Pass 1: compute sum(dX * p) = sum(dX * exp(logprob))
    dot_sum = 0.0
    for col_start in range(0, n_cols, BLOCK_V):
        cols = col_start + tl.arange(0, BLOCK_V)
        mask = cols < n_cols
        dx = tl.load(dX_ptr + row_base + cols, mask=mask, other=0.0).to(tl.float32)
        lp = tl.load(logprob_ptr + row_base + cols, mask=mask, other=-float('inf')).to(tl.float32)
        dot_sum += tl.sum(dx * tl.exp(lp), axis=0)

    # Pass 2: dlogits = (dX - p * dot_sum) / temperature
    for col_start in range(0, n_cols, BLOCK_V):
        cols = col_start + tl.arange(0, BLOCK_V)
        mask = cols < n_cols
        dx = tl.load(dX_ptr + row_base + cols, mask=mask, other=0.0).to(tl.float32)
        lp = tl.load(logprob_ptr + row_base + cols, mask=mask, other=-float('inf')).to(tl.float32)
        p = tl.exp(lp)
        dlogit = (dx - p * dot_sum) / temperature
        tl.store(dlogits_ptr + row_base + cols, dlogit, mask=mask)


# ============================================================================
# Python wrapper
# ============================================================================

def fused_linear_jsd_forward(
    student_input, student_weight,
    teacher_input, teacher_weight,
    labels=None, beta=0.5, temperature=1.0, ignore_index=-100,
):
    """
    Args:
        student_input: (BT, H)
        student_weight: (V, H)
        teacher_input: (BT, H)
        teacher_weight: (V, H)
        labels: (BT,) optional for ignore_index
    Returns:
        loss, grad_student_input, grad_student_weight
    """
    BT, H = student_input.shape
    V = student_weight.shape[0]
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    inc_factor = triton.cdiv(V, H)
    chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))
    num_chunks = triton.cdiv(BT, chunk_size)

    has_label = labels is not None
    if has_label:
        n_non_ignore = max(int((labels != ignore_index).sum().item()), 1)
    else:
        n_non_ignore = BT

    loss_acc = torch.tensor(0.0, dtype=torch.float32, device=student_input.device)
    grad_student_input = torch.zeros_like(student_input)
    grad_student_weight = torch.zeros_like(student_weight)

    nw = 32

    for chunk_id in range(num_chunks):
        start = chunk_id * chunk_size
        end = min(start + chunk_size, BT)
        cs = end - start

        s_chunk = student_input[start:end]
        t_chunk = teacher_input[start:end]
        label_chunk = labels[start:end] if has_label else None

        # 1. Compute logits in compute dtype
        s_logits = (s_chunk @ student_weight.t()).contiguous()  # (cs, V)
        t_logits = (t_chunk @ teacher_weight.t()).contiguous()  # (cs, V)

        # 2. In-place log-softmax with temperature
        # We need fp32 copies for log-softmax
        s_logprobs = s_logits.float().contiguous()
        t_logprobs = t_logits.float().contiguous()

        _log_softmax_kernel[(cs,)](
            s_logprobs, s_logprobs.stride(0), V, temperature,
            BLOCK_V=BLOCK_SIZE, num_warps=nw,
        )
        _log_softmax_kernel[(cs,)](
            t_logprobs, t_logprobs.stride(0), V, temperature,
            BLOCK_V=BLOCK_SIZE, num_warps=nw,
        )

        # 3. JSD kernel: loss + dX (gradient w.r.t. student log-probs)
        loss_per_row = torch.empty(cs, dtype=torch.float32, device=student_input.device)
        dX = torch.empty(cs, V, dtype=torch.float32, device=student_input.device)

        _jsd_kernel[(cs,)](
            s_logprobs, t_logprobs, loss_per_row, dX,
            s_logprobs.stride(0), t_logprobs.stride(0),
            V, n_non_ignore, ignore_index,
            label_chunk,
            BETA=beta,
            BLOCK_V=BLOCK_SIZE, num_warps=nw,
            HAS_LABEL=has_label,
        )

        loss_acc += loss_per_row.sum()

        # 4. Backprop through log-softmax: dlogits from dX
        dlogits = torch.empty_like(dX)
        _softmax_grad_kernel[(cs,)](
            dX, s_logprobs, dlogits,
            s_logprobs.stride(0), V, temperature,
            BLOCK_V=BLOCK_SIZE, num_warps=nw,
        )

        # 5. Backprop through linear: grad_input, grad_weight
        dlogits_compute = dlogits.to(student_input.dtype)
        grad_student_input[start:end] = dlogits_compute @ student_weight
        grad_student_weight += dlogits_compute.t() @ s_chunk

    return loss_acc, grad_student_input, grad_student_weight


# ============================================================================
# Autograd Function
# ============================================================================

class HildaFusedLinearJSDFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student_input, student_weight,
                teacher_input, teacher_weight,
                labels, beta, temperature, ignore_index):
        loss, grad_input, grad_weight = fused_linear_jsd_forward(
            student_input, student_weight,
            teacher_input, teacher_weight,
            labels, beta, temperature, ignore_index,
        )
        ctx.save_for_backward(grad_input, grad_weight)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight = ctx.saved_tensors
        if not torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
            H = grad_input.shape[1]
            BT = grad_input.shape[0]
            V = grad_weight.shape[0]
            BLOCK = min(MAX_FUSED_SIZE, triton.next_power_of_2(H))
            _element_mul_kernel[(BT,)](
                grad_input, grad_input.stride(0), grad_output, H,
                BLOCK=BLOCK, num_warps=32,
            )
            _element_mul_kernel[(V,)](
                grad_weight, grad_weight.stride(0), grad_output, H,
                BLOCK=BLOCK, num_warps=32,
            )
        return grad_input, grad_weight, None, None, None, None, None, None


def hilda_fused_linear_jsd(
    student_input, student_weight,
    teacher_input, teacher_weight,
    labels=None, beta=0.5, temperature=1.0, ignore_index=-100,
):
    """Functional interface: fused linear + JSD loss."""
    return HildaFusedLinearJSDFunction.apply(
        student_input, student_weight,
        teacher_input, teacher_weight,
        labels, beta, temperature, ignore_index,
    )


# ============================================================================
# nn.Module
# ============================================================================

class FusedLinearJSDLoss(torch.nn.Module):
    """Fused linear projection + JSD loss for knowledge distillation."""
    def __init__(self, beta=0.5, temperature=1.0, ignore_index=-100):
        super().__init__()
        self.beta = beta
        self.temperature = temperature
        self.ignore_index = ignore_index

    def forward(self, student_input, student_weight,
                teacher_input, teacher_weight, labels=None):
        return hilda_fused_linear_jsd(
            student_input, student_weight,
            teacher_input, teacher_weight,
            labels, self.beta, self.temperature, self.ignore_index,
        )
