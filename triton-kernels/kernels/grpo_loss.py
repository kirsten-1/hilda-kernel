"""
GRPO Loss (Group Relative Policy Optimization) - Fused Triton Kernel

PPO-variant loss for RLHF training:
  ρ = exp(logp_new - logp_old)           # importance ratio
  ρ_clip = clamp(ρ, 1-ε_low, 1+ε_high)
  loss = -min(ρ*A, ρ_clip*A)             # clipped PPO objective
  loss += β * KL(ref || new)             # optional KL penalty

核心优化:
1. Forward: tiled online logsumexp → selective log-softmax per token
2. PPO clipping + KL penalty fused in single kernel
3. Backward: softmax Jacobian via stored LSE, no recomputation
4. 支持 token-level importance sampling (standard GRPO)
"""

import torch
import triton
import triton.language as tl


BLOCK_N_DEFAULT = 4096


# ============================================================================
# Selective log-softmax kernel (no grad needed)
# ============================================================================

@triton.jit
def _selective_log_softmax_kernel(
    LOGITS, INPUT_IDS, LOG_P, MASK,
    TEMPERATURE,
    stride_ids_b,
    L: tl.constexpr, N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_b = tl.program_id(0).cast(tl.int64)
    off_l = tl.program_id(1).cast(tl.int64)

    LOGITS += off_b * (L + 1) * N + off_l * N
    INPUT_IDS += off_b * stride_ids_b + off_l
    LOG_P += off_b * L + off_l

    if MASK is not None:
        MASK += off_b * stride_ids_b + off_l
        if tl.load(MASK) == 0:
            return

    # Online logsumexp
    m_i = float('-inf')
    l_i = 0.0
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        logits = tl.load(LOGITS + cols, mask=cols < N, other=float('-inf')).to(tl.float32) / TEMPERATURE
        new_m = tl.maximum(m_i, tl.max(logits))
        l_i = l_i * tl.exp(m_i - new_m) + tl.sum(tl.exp(logits - new_m))
        m_i = new_m
    lse = m_i + tl.log(l_i)

    idx = tl.load(INPUT_IDS)
    x = tl.load(LOGITS + idx).to(tl.float32) / TEMPERATURE
    tl.store(LOG_P, x - lse)


@torch.no_grad()
def fused_selective_log_softmax(logits, input_ids, temperature=1.0, mask=None):
    """Compute log-softmax only at target token positions."""
    assert logits.is_contiguous()
    B, L_plus_1, N = logits.shape
    L = L_plus_1 - 1
    input_ids = input_ids[:, -L:]
    if mask is not None:
        mask = mask[:, -L:]
    log_p = torch.zeros(B, L, dtype=torch.float32, device=logits.device)
    _selective_log_softmax_kernel[(B, L)](
        logits, input_ids, log_p, mask, temperature,
        input_ids.stride(0), L, N,
        BLOCK_N=2048, num_stages=4, num_warps=1,
    )
    return log_p


# ============================================================================
# GRPO forward kernel (token-level importance sampling)
# ============================================================================

@triton.jit
def _grpo_loss_fwd_kernel(
    LOGITS, OLD_LOGP, REF_LOGP, INPUT_IDS, COMPLETION_MASK, ADVANTAGES,
    LOSS, LSE, KL, IS_CLIPPED,
    TEMPERATURE,
    BETA: tl.constexpr,
    EPS_LOW, EPS_HIGH,
    HAS_OLD_LOGP: tl.constexpr,
    HAS_MASK: tl.constexpr,
    L: tl.constexpr, N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_b = tl.program_id(0).cast(tl.int64)
    off_l = tl.program_id(1).cast(tl.int64)

    if HAS_MASK:
        COMPLETION_MASK += off_b * L + off_l
        if tl.load(COMPLETION_MASK) == 0:
            return

    LOGITS += off_b * (L + 1) * N + off_l * N
    INPUT_IDS += off_b * L + off_l
    ADVANTAGES += off_b
    LOSS += off_b * L + off_l
    LSE += off_b * L + off_l
    IS_CLIPPED += off_b * L + off_l

    # Online logsumexp
    m_i = float('-inf')
    l_i = 0.0
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        logits = tl.load(LOGITS + cols, mask=cols < N, other=float('-inf')).to(tl.float32) / TEMPERATURE
        new_m = tl.maximum(m_i, tl.max(logits))
        l_i = l_i * tl.exp(m_i - new_m) + tl.sum(tl.exp(logits - new_m))
        m_i = new_m
    lse = m_i + tl.log(l_i)

    idx = tl.load(INPUT_IDS)
    x = tl.load(LOGITS + idx).to(tl.float32) / TEMPERATURE
    logp = x - lse

    # Importance ratio
    if HAS_OLD_LOGP:
        OLD_LOGP += off_b * L + off_l
        old_logp = tl.load(OLD_LOGP).to(tl.float32)
    else:
        old_logp = logp
    coef_1 = tl.exp(logp - old_logp)
    advantage = tl.load(ADVANTAGES).to(tl.float32)

    # PPO clipping
    coef_2 = tl.clamp(coef_1, 1 - EPS_LOW, 1 + EPS_HIGH)
    is_low_clipped = (coef_1 < 1 - EPS_LOW) & (advantage < 0)
    is_high_clipped = (coef_1 > 1 + EPS_HIGH) & (advantage > 0)
    is_clipped = is_low_clipped | is_high_clipped

    per_token_loss1 = coef_1 * advantage
    per_token_loss2 = coef_2 * advantage
    per_token_loss = -tl.minimum(per_token_loss1, per_token_loss2)

    # Optional KL penalty
    if BETA != 0.0:
        REF_LOGP += off_b * L + off_l
        KL += off_b * L + off_l
        ref_logp = tl.load(REF_LOGP).to(tl.float32)
        kl = tl.exp(ref_logp - logp) - (ref_logp - logp) - 1
        per_token_loss += BETA * kl
        tl.store(KL, kl)

    tl.store(LOSS, per_token_loss)
    tl.store(LSE, lse)
    tl.store(IS_CLIPPED, is_clipped)


# ============================================================================
# GRPO backward kernel
# ============================================================================

@triton.jit
def _grpo_loss_bwd_kernel(
    DLOSS, DLOGITS, LOGITS, OLD_LOGP, REF_LOGP,
    INPUT_IDS, ADVANTAGES, COMPLETION_MASK, LSE,
    TEMPERATURE,
    BETA: tl.constexpr,
    EPS_LOW, EPS_HIGH,
    HAS_OLD_LOGP: tl.constexpr,
    HAS_MASK: tl.constexpr,
    loss_stride0, loss_stride1,
    L: tl.constexpr, N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_b = tl.program_id(0).cast(tl.int64)
    off_l = tl.program_id(1).cast(tl.int64)

    DLOGITS += off_b * (L + 1) * N + off_l * N
    if HAS_MASK:
        COMPLETION_MASK += off_b * L + off_l
        if tl.load(COMPLETION_MASK) == 0:
            for start in range(0, N, BLOCK_N):
                cols = tl.arange(0, BLOCK_N) + start
                tl.store(DLOGITS + cols, 0.0, mask=cols < N)
            return

    LOGITS += off_b * (L + 1) * N + off_l * N
    DLOSS += off_b * loss_stride0 + off_l * loss_stride1
    INPUT_IDS += off_b * L + off_l
    ADVANTAGES += off_b
    LSE += off_b * L + off_l

    dloss = tl.load(DLOSS).to(tl.float32)
    lse = tl.load(LSE).to(tl.float32)

    idx = tl.load(INPUT_IDS)
    x = tl.load(LOGITS + idx).to(tl.float32) / TEMPERATURE
    logp = x - lse

    if HAS_OLD_LOGP:
        OLD_LOGP += off_b * L + off_l
        old_logp = tl.load(OLD_LOGP).to(tl.float32)
    else:
        old_logp = logp
    coef_1 = tl.exp(logp - old_logp)
    advantage = tl.load(ADVANTAGES).to(tl.float32)

    # Gradient: d(loss)/d(logp)
    coef_2 = tl.clamp(coef_1, 1 - EPS_LOW, 1 + EPS_HIGH)
    per_token_loss1 = coef_1 * advantage
    per_token_loss2 = coef_2 * advantage
    is_unclipped = per_token_loss2 >= per_token_loss1
    dlogp = -coef_1 * advantage * is_unclipped

    # KL gradient
    if BETA != 0.0:
        REF_LOGP += off_b * L + off_l
        ref_logp = tl.load(REF_LOGP).to(tl.float32)
        dlogp += BETA * (1 - tl.exp(ref_logp - logp))

    dlogp = dlogp * dloss / TEMPERATURE
    tl.debug_barrier()

    # Write dlogits using softmax Jacobian: (δ_ij - p_j) * dlogp
    for start_n in tl.range(0, N, BLOCK_N):
        cols = start_n + tl.arange(0, BLOCK_N)
        logits = tl.load(LOGITS + cols, mask=cols < N, other=-float('inf')).to(tl.float32) / TEMPERATURE
        probs = tl.exp(logits - lse)
        dlogits = tl.where(cols == idx, 1 - probs, -probs) * dlogp
        tl.store(DLOGITS + cols, dlogits, mask=cols < N)


# ============================================================================
# Python wrapper
# ============================================================================

def grpo_loss_forward(
    logits, completion_ids, advantages,
    old_logp=None, ref_logp=None, completion_mask=None,
    temperature=1.0, beta=0.0, eps_low=0.2, eps_high=0.2,
):
    """
    Args:
        logits: (B, L+1, V) model logits
        completion_ids: (B, L) token IDs
        advantages: (B,) per-sequence advantage
        old_logp: (B, L) or None — old policy log-probs
        ref_logp: (B, L) or None — reference log-probs (required if β≠0)
        completion_mask: (B, L) or None
    Returns:
        loss: scalar (reduced)
        kl_mean: scalar or None
        clip_ratio: scalar
        lse: (B, L) for backward
    """
    assert logits.is_contiguous()
    B, L_plus_1, N = logits.shape
    L = L_plus_1 - 1

    mask = completion_mask.float() if completion_mask is not None else torch.ones(B, L, device=logits.device)

    loss = torch.zeros(B, L, device=logits.device, dtype=torch.float32)
    lse = torch.zeros_like(loss)
    is_clipped = torch.zeros_like(loss)
    kl = torch.zeros_like(loss)  # always allocate (Triton can't handle None ptr)

    has_old_logp = old_logp is not None
    has_mask = completion_mask is not None

    # Triton can't handle None pointers — pass dummy tensors
    dummy = torch.empty(1, device=logits.device)
    _old_logp = old_logp if has_old_logp else dummy
    _ref_logp = ref_logp if ref_logp is not None else dummy
    _mask = completion_mask if has_mask else dummy

    kwargs = {'BLOCK_N': 2048, 'num_stages': 2, 'num_warps': 1}
    _grpo_loss_fwd_kernel[(B, L)](
        logits, _old_logp, _ref_logp, completion_ids, _mask, advantages,
        loss, lse, kl, is_clipped,
        temperature, beta, eps_low, eps_high,
        HAS_OLD_LOGP=has_old_logp, HAS_MASK=has_mask,
        L=L, N=N, **kwargs,
    )

    # Reduce: per-sequence average, then batch mean (standard GRPO)
    mask_sum = mask.sum().clamp(min=1.0)
    kl_mean = (kl * mask).sum() / mask_sum if beta != 0.0 else None
    clip_ratio = (is_clipped.float() * mask).sum() / mask_sum

    seq_lens = mask.sum(-1).clamp(min=1.0)
    reduced_loss = ((loss * mask).sum(-1) / seq_lens).mean()

    return reduced_loss, kl_mean, clip_ratio, lse, mask


def grpo_loss_backward(
    grad_output, logits, old_logp, ref_logp, completion_ids,
    advantages, completion_mask, lse, mask,
    temperature, beta, eps_low, eps_high,
):
    B, L_plus_1, N = logits.shape
    L = L_plus_1 - 1

    # Compute per-token dloss scaling (GRPO reduction)
    seq_lens = mask.sum(-1, keepdim=True).clamp(min=1.0)
    dloss = grad_output * mask / (seq_lens * B)

    has_old_logp = old_logp is not None
    has_mask = completion_mask is not None
    dummy = torch.empty(1, device=logits.device)
    _old_logp = old_logp if has_old_logp else dummy
    _ref_logp = ref_logp if ref_logp is not None else dummy
    _mask = completion_mask if has_mask else dummy

    dlogits = torch.empty_like(logits)
    kwargs = {'BLOCK_N': 4096, 'num_stages': 1, 'num_warps': 16}
    _grpo_loss_bwd_kernel[(B, L)](
        dloss, dlogits, logits, _old_logp, _ref_logp,
        completion_ids, advantages, _mask, lse,
        temperature, beta, eps_low, eps_high,
        HAS_OLD_LOGP=has_old_logp, HAS_MASK=has_mask,
        loss_stride0=dloss.stride(0), loss_stride1=dloss.stride(1),
        L=L, N=N, **kwargs,
    )
    dlogits[:, -1, :] = 0
    return dlogits


# ============================================================================
# Autograd Function
# ============================================================================

class HildaGRPOLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, logits, completion_ids, advantages,
        old_logp, ref_logp, completion_mask,
        temperature, beta, eps_low, eps_high,
    ):
        loss, kl_mean, clip_ratio, lse, mask = grpo_loss_forward(
            logits, completion_ids, advantages,
            old_logp, ref_logp, completion_mask,
            temperature, beta, eps_low, eps_high,
        )
        ctx.save_for_backward(logits, old_logp, ref_logp, completion_ids, advantages, completion_mask, lse, mask)
        ctx.params = (temperature, beta, eps_low, eps_high)
        return loss, kl_mean, clip_ratio

    @staticmethod
    def backward(ctx, grad_loss, grad_kl, grad_clip):
        logits, old_logp, ref_logp, completion_ids, advantages, completion_mask, lse, mask = ctx.saved_tensors
        temperature, beta, eps_low, eps_high = ctx.params
        dlogits = grpo_loss_backward(
            grad_loss, logits, old_logp, ref_logp, completion_ids,
            advantages, completion_mask, lse, mask,
            temperature, beta, eps_low, eps_high,
        )
        return dlogits, None, None, None, None, None, None, None, None, None


def hilda_grpo_loss(
    logits, completion_ids, advantages,
    old_logp=None, ref_logp=None, completion_mask=None,
    temperature=1.0, beta=0.0, eps_low=0.2, eps_high=0.2,
):
    """Functional interface: GRPO loss."""
    return HildaGRPOLossFunction.apply(
        logits, completion_ids, advantages,
        old_logp, ref_logp, completion_mask,
        temperature, beta, eps_low, eps_high,
    )


# ============================================================================
# nn.Module
# ============================================================================

class GRPOLoss(torch.nn.Module):
    """GRPO Loss for RLHF training."""
    def __init__(self, temperature=1.0, beta=0.0, eps_low=0.2, eps_high=0.2):
        super().__init__()
        self.temperature = temperature
        self.beta = beta
        self.eps_low = eps_low
        self.eps_high = eps_high

    def forward(self, logits, completion_ids, advantages,
                old_logp=None, ref_logp=None, completion_mask=None):
        return hilda_grpo_loss(
            logits, completion_ids, advantages,
            old_logp, ref_logp, completion_mask,
            self.temperature, self.beta, self.eps_low, self.eps_high,
        )
