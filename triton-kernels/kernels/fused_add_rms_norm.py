"""
FusedAddRMSNorm - Fused Residual Add + RMSNorm

S = X + R
Y = RMSNorm(S) * W = (S / sqrt(mean(S^2) + eps)) * W

标准 Transformer 残差连接：每层执行 residual add + norm，
分开做需要 3 次 HBM 读写（读 X,R → 写 S → 读 S → 写 Y），
融合后只需 2 次（读 X,R → 写 S,Y）。

返回 (Y, S)：Y 给当前层用，S 作为 residual 传给下一层。
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# AutoTune 配置
# ============================================================================

def _get_fused_configs():
    configs = []
    for BLOCK in [1024, 2048, 4096, 8192]:
        for nw in [4, 8, 16]:
            configs.append(triton.Config({'BLOCK_SIZE': BLOCK}, num_warps=nw))
    return configs


def _prune_configs(configs, named_args, **kwargs):
    n_cols = named_args['n_cols']
    return [c for c in configs if c.kwargs['BLOCK_SIZE'] >= n_cols]


# ============================================================================
# Forward Kernel
# ============================================================================

@triton.autotune(
    configs=_get_fused_configs(),
    key=['n_cols'],
    prune_configs_by={'early_config_prune': _prune_configs},
)
@triton.jit
def _fused_add_rms_norm_fwd_kernel(
    Y_ptr, S_ptr,
    X_ptr, R_ptr, W_ptr,
    RSTD_ptr,
    stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_off = row * stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # Load X and R, compute S = X + R
    x = tl.load(X_ptr + row_off + cols, mask=mask, other=0.0)
    r = tl.load(R_ptr + row_off + cols, mask=mask, other=0.0)
    s = x + r

    # Store S (updated residual for next layer)
    tl.store(S_ptr + row_off + cols, s, mask=mask)

    # RMSNorm in fp32
    s_f32 = s.to(tl.float32)
    ms = tl.sum(s_f32 * s_f32, axis=0) / n_cols
    rstd = tl.rsqrt(ms + eps)
    tl.store(RSTD_ptr + row, rstd)

    # Y = norm(S) * W
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    y = (s_f32 * rstd).to(s.dtype) * w
    tl.store(Y_ptr + row_off + cols, y, mask=mask)


# ============================================================================
# Backward Kernel (atomic dW — same adaptive strategy as RMSNorm)
# ============================================================================

@triton.jit
def _fused_add_rms_norm_bwd_atomic(
    dY_ptr, dS_out_ptr,
    dX_ptr,
    S_ptr, W_ptr, RSTD_ptr,
    dW_ptr,
    stride, n_cols,
    HAS_DS_OUT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward: one program per row, atomic dW accumulation."""
    row = tl.program_id(0)
    row_off = row * stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    dy = tl.load(dY_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
    s  = tl.load(S_ptr  + row_off + cols, mask=mask, other=0.0).to(tl.float32)
    w  = tl.load(W_ptr  + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_ptr + row)

    # dX = d(RMSNorm)/dS = rstd * (dy*w) - rstd^3 * (1/n) * sum(dy*w*s) * s
    m = dy * w
    sum_m_s = tl.sum(m * s, axis=0)
    dx = rstd * m - rstd * rstd * rstd * (sum_m_s / n_cols) * s

    # Add upstream residual gradient if present
    if HAS_DS_OUT:
        ds_out = tl.load(dS_out_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
        dx = dx + ds_out

    tl.store(dX_ptr + row_off + cols, dx.to(tl.load(dY_ptr + row_off).dtype), mask=mask)

    # dW accumulation
    dw_contrib = dy * (s * rstd)
    tl.atomic_add(dW_ptr + cols, dw_contrib, mask=mask)


@triton.autotune(
    configs=_get_fused_configs(),
    key=['n_cols'],
    prune_configs_by={'early_config_prune': _prune_configs},
)
@triton.jit
def _fused_add_rms_norm_bwd_partial(
    dY_ptr, dS_out_ptr,
    dX_ptr,
    S_ptr, W_ptr, RSTD_ptr,
    dW_ptr, dW_row_stride,
    stride, n_rows, n_cols,
    rows_per_program,
    HAS_DS_OUT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward: partial-sum dW, each program handles multiple rows."""
    pid = tl.program_id(0).to(tl.int64)
    row_start = pid * rows_per_program
    row_end = min(row_start + rows_per_program, n_rows)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    dw_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    for row in range(row_start, row_end):
        row_off = row * stride
        dy = tl.load(dY_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
        s  = tl.load(S_ptr  + row_off + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD_ptr + row)

        m = dy * w
        sum_m_s = tl.sum(m * s, axis=0)
        dx = rstd * m - rstd * rstd * rstd * (sum_m_s / n_cols) * s

        if HAS_DS_OUT:
            ds_out = tl.load(dS_out_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
            dx = dx + ds_out

        tl.store(dX_ptr + row_off + cols, dx.to(tl.load(dY_ptr + row_off).dtype), mask=mask)
        dw_acc += dy * (s * rstd)

    tl.store(dW_ptr + pid * dW_row_stride + cols, dw_acc, mask=mask)


# ============================================================================
# Python wrappers
# ============================================================================

def fused_add_rms_norm_forward(X, R, W, eps=1e-6):
    shape = X.shape
    n_cols = shape[-1]
    X_2d = X.reshape(-1, n_cols).contiguous()
    R_2d = R.reshape(-1, n_cols).contiguous()
    W = W.contiguous()
    n_rows = X_2d.shape[0]

    Y    = torch.empty_like(X_2d)
    S    = torch.empty_like(X_2d)
    RSTD = torch.empty(n_rows, dtype=torch.float32, device=X.device)

    _fused_add_rms_norm_fwd_kernel[(n_rows,)](
        Y, S, X_2d, R_2d, W, RSTD,
        X_2d.stride(0), n_cols, eps,
    )
    return Y.view(shape), S.view(shape), RSTD


def fused_add_rms_norm_backward(dY, dS_out, S, W, RSTD, eps=1e-6):
    shape = dY.shape
    n_cols = shape[-1]
    dY_2d = dY.reshape(-1, n_cols).contiguous()
    S_2d  = S.reshape(-1, n_cols).contiguous()
    n_rows = dY_2d.shape[0]

    has_ds = dS_out is not None
    dS_out_2d = dS_out.reshape(-1, n_cols).contiguous() if has_ds else dY_2d  # dummy

    dX = torch.empty_like(dY_2d)

    sm_count = torch.cuda.get_device_properties(dY.device).multi_processor_count
    use_atomic = n_rows <= sm_count * 8

    if use_atomic:
        BLOCK = triton.next_power_of_2(n_cols)
        nw = min(max(BLOCK // 32, 1), 32)
        dW = torch.zeros(n_cols, dtype=torch.float32, device=dY.device)
        _fused_add_rms_norm_bwd_atomic[(n_rows,)](
            dY_2d, dS_out_2d, dX,
            S_2d, W, RSTD, dW,
            dY_2d.stride(0), n_cols,
            HAS_DS_OUT=has_ds,
            BLOCK_SIZE=BLOCK, num_warps=nw,
        )
    else:
        num_programs = sm_count
        rows_per_program = (n_rows + num_programs - 1) // num_programs
        dW_partial = torch.empty((num_programs, n_cols), dtype=torch.float32, device=dY.device)
        _fused_add_rms_norm_bwd_partial[(num_programs,)](
            dY_2d, dS_out_2d, dX,
            S_2d, W, RSTD,
            dW_partial, dW_partial.stride(0),
            dY_2d.stride(0), n_rows, n_cols, rows_per_program,
            HAS_DS_OUT=has_ds,
        )
        dW = dW_partial.sum(dim=0)

    # dX = dR (same gradient for both inputs of the add)
    dX_out = dX.view(shape)
    return dX_out, dX_out, dW.to(W.dtype)


# ============================================================================
# PyTorch Autograd Function
# ============================================================================

class HildaFusedAddRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, R, W, eps):
        Y, S, RSTD = fused_add_rms_norm_forward(X, R, W, eps)
        ctx.save_for_backward(S, W, RSTD)
        ctx.eps = eps
        return Y, S

    @staticmethod
    def backward(ctx, dY, dS_out):
        S, W, RSTD = ctx.saved_tensors
        dX, dR, dW = fused_add_rms_norm_backward(dY, dS_out, S, W, RSTD, ctx.eps)
        return dX, dR, dW, None


def hilda_fused_add_rms_norm(X, R, W, eps=1e-6):
    """Functional interface: returns (Y, S) where S=X+R, Y=RMSNorm(S)*W"""
    return HildaFusedAddRMSNormFunction.apply(X, R, W, eps)


# ============================================================================
# nn.Module
# ============================================================================

class FusedAddRMSNorm(torch.nn.Module):
    """
    Fused residual add + RMSNorm.
    forward(X, R) -> (Y, S) where S = X + R, Y = RMSNorm(S) * W
    """
    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_dim))

    def forward(self, X, R):
        return hilda_fused_add_rms_norm(X, R, self.weight, self.eps)
