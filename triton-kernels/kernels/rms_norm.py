"""
RMSNorm with AutoTune - 优化版本

相比 Liger Kernel 的改进：
1. 使用 triton.autotune 自动选择最优配置
2. 添加 num_stages 支持软件流水线
3. 针对 4090 等消费级 GPU 优化配置空间
"""

import torch
import triton
import triton.language as tl

# ============================================================================
# AutoTune 配置
# ============================================================================

def get_autotune_configs():
    """
    生成 AutoTune 配置列表

    关键参数：
    - BLOCK_SIZE: 每个 block 处理的元素数量
    - num_warps: warp 数量（每个 warp 32 threads）
    - num_stages: 软件流水线阶段数（用于隐藏内存延迟）
    """
    configs = []

    # 小 hidden_dim (256-1024): 用较小的 block 和 warps
    for block_size in [256, 512, 1024]:
        for num_warps in [2, 4, 8]:
            for num_stages in [2, 3, 4]:
                configs.append(
                    triton.Config(
                        {'BLOCK_SIZE': block_size},
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )

    # 中等 hidden_dim (2048-4096): LLM 常见配置
    for block_size in [2048, 4096]:
        for num_warps in [4, 8, 16]:
            for num_stages in [2, 3]:
                configs.append(
                    triton.Config(
                        {'BLOCK_SIZE': block_size},
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )

    # 大 hidden_dim (8192+): 大模型配置
    for block_size in [8192, 16384]:
        for num_warps in [8, 16, 32]:
            for num_stages in [2, 3]:
                configs.append(
                    triton.Config(
                        {'BLOCK_SIZE': block_size},
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )

    return configs


def _prune_configs(configs, named_args, **kwargs):
    """剪枝：只保留 BLOCK_SIZE >= n_cols 的配置"""
    n_cols = named_args['n_cols']
    return [c for c in configs if c.kwargs['BLOCK_SIZE'] >= n_cols]


# ============================================================================
# Forward Kernel
# ============================================================================

@triton.autotune(
    configs=get_autotune_configs(),
    key=['n_cols', 'n_rows'],
    prune_configs_by={'early_config_prune': _prune_configs},
)
@triton.jit
def _rms_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    RSTD_ptr,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMSNorm Forward Kernel

    y_i = (x_i / RMS(x)) * w_i
    RMS(x) = sqrt(mean(x^2) + eps)

    优化点：
    1. 只保存 rstd (1/RMS) 用于 backward，不保存 normalized tensor
    2. 使用 FP32 计算 rstd 保证数值稳定性
    """
    # 获取当前处理的行
    row_idx = tl.program_id(0).to(tl.int64)

    # 计算指针偏移
    X_row_ptr = X_ptr + row_idx * X_row_stride
    Y_row_ptr = Y_ptr + row_idx * Y_row_stride

    # 列偏移（提取到循环外，避免重复计算）
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # 加载输入
    X_row = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0)
    X_row_dtype = X_row.dtype

    # 加载权重
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)

    # 转换为 FP32 计算（数值稳定性）
    X_row_f32 = X_row.to(tl.float32)

    # 计算 RMS：sqrt(mean(x^2) + eps)
    # rstd = 1 / RMS = rsqrt(mean(x^2) + eps)
    mean_square = tl.sum(X_row_f32 * X_row_f32, axis=0) / n_cols
    rstd = tl.rsqrt(mean_square + eps)

    # 保存 rstd 用于 backward（关键优化：只保存这个标量）
    tl.store(RSTD_ptr + row_idx, rstd)

    # 计算输出：(x / RMS) * w = x * rstd * w
    X_norm = X_row_f32 * rstd

    # 转回原始精度后乘权重（Llama 风格）
    Y_row = X_norm.to(X_row_dtype) * W_row

    # 存储输出
    tl.store(Y_row_ptr + col_offsets, Y_row, mask=mask)


# ============================================================================
# Backward Kernel - Atomic 版本（小 batch 友好）
# ============================================================================

# 注意：不使用 @triton.autotune，因为 AutoTune 会多次调用 kernel，
# 导致 atomic_add 累加多次，使 dW 结果错误。
# 小 batch 场景优化重点是减少 kernel launch 开销，而非调参，手动计算配置即可。
@triton.jit
def _rms_norm_backward_kernel_atomic(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    RSTD_ptr,
    dW_ptr,   # [n_cols] fp32 - 直接 atomic_add，无需 partial sum
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMSNorm Backward - Atomic 版本

    每个 program 处理一行，dW 用 atomic_add 直接累加到全局缓冲。
    消除了 partial sum 分配和 Python 端 reduce，对小 batch 更友好。
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY_row = tl.load(dY_ptr + row_idx * dY_row_stride + col_offsets, mask=mask, other=0.0)
    X_row  = tl.load(X_ptr  + row_idx * X_row_stride  + col_offsets, mask=mask, other=0.0)
    W_row  = tl.load(W_ptr  + col_offsets, mask=mask, other=0.0)
    rstd   = tl.load(RSTD_ptr + row_idx)

    X_f32  = X_row.to(tl.float32)
    dY_f32 = dY_row.to(tl.float32)
    W_f32  = W_row.to(tl.float32)

    # dX
    m = dY_f32 * W_f32
    sum_m_x = tl.sum(m * X_f32, axis=0)
    dX_row = rstd * m - rstd * rstd * rstd * (sum_m_x / n_cols) * X_f32
    tl.store(dX_ptr + row_idx * dX_row_stride + col_offsets, dX_row.to(dY_row.dtype), mask=mask)

    # dW：直接原子累加，省去 partial sum 分配 + reduce kernel
    dW_contrib = dY_f32 * (X_f32 * rstd)
    tl.atomic_add(dW_ptr + col_offsets, dW_contrib, mask=mask)


# ============================================================================
# Backward Kernel - Partial Sum 版本（大 batch 友好）
# ============================================================================

@triton.autotune(
    configs=get_autotune_configs(),
    key=['n_cols', 'n_rows'],
    prune_configs_by={'early_config_prune': _prune_configs},
)
@triton.jit
def _rms_norm_backward_kernel_partial(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    RSTD_ptr,
    dW_ptr,          # [num_programs, n_cols] partial dW
    dW_row_stride,
    n_rows,
    n_cols,
    rows_per_program,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMSNorm Backward - Partial Sum 版本

    每个 program 处理多行，在寄存器内累积 dW，最后一次性写出。
    大 batch 时并行度足够，避免 atomic 竞争，整体更快。
    """
    program_id = tl.program_id(0).to(tl.int64)
    row_start = program_id * rows_per_program
    row_end = min(row_start + rows_per_program, n_rows)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dW_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    W_row  = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)

    for row_idx in range(row_start, row_end):
        dY_row = tl.load(dY_ptr + row_idx * dY_row_stride + col_offsets, mask=mask, other=0.0)
        X_row  = tl.load(X_ptr  + row_idx * X_row_stride  + col_offsets, mask=mask, other=0.0)
        rstd   = tl.load(RSTD_ptr + row_idx)

        X_f32  = X_row.to(tl.float32)
        dY_f32 = dY_row.to(tl.float32)
        W_f32  = W_row.to(tl.float32)

        m = dY_f32 * W_f32
        sum_m_x = tl.sum(m * X_f32, axis=0)
        dX_row = rstd * m - rstd * rstd * rstd * (sum_m_x / n_cols) * X_f32
        tl.store(dX_ptr + row_idx * dX_row_stride + col_offsets, dX_row.to(X_row.dtype), mask=mask)

        dW_acc += dY_f32 * (X_f32 * rstd)

    tl.store(dW_ptr + program_id * dW_row_stride + col_offsets, dW_acc, mask=mask)


# ============================================================================
# Python Wrapper
# ============================================================================

def rms_norm_forward(X, W, eps=1e-6):
    """
    RMSNorm Forward Pass

    Args:
        X: Input tensor, shape (..., hidden_dim)
        W: Weight tensor, shape (hidden_dim,)
        eps: Epsilon for numerical stability

    Returns:
        Y: Output tensor, same shape as X
        X_flat: Flattened input for backward
        RSTD: Reciprocal of RMS for each row
    """
    # 保存原始形状
    shape = X.shape
    hidden_dim = shape[-1]

    # Flatten 到 2D
    X_flat = X.view(-1, hidden_dim)
    n_rows, n_cols = X_flat.shape

    # 确保连续
    X_flat = X_flat.contiguous()
    W = W.contiguous()

    # 分配输出
    Y = torch.empty_like(X_flat)
    RSTD = torch.empty(n_rows, dtype=torch.float32, device=X.device)

    # 检查维度兼容性
    assert X_flat.shape[1] == W.shape[0], f"Hidden dim mismatch: {X_flat.shape[1]} vs {W.shape[0]}"

    # 启动 kernel
    grid = (n_rows,)
    _rms_norm_forward_kernel[grid](
        Y, Y.stride(0),
        X_flat, X_flat.stride(0),
        W,
        RSTD,
        n_rows,
        n_cols,
        eps,
    )

    return Y.view(*shape), X_flat, RSTD


def rms_norm_backward(dY, X, W, RSTD):
    """
    RMSNorm Backward Pass

    自适应策略：
    - 小 batch (n_rows <= sm_count * 8)：atomic 版本，省去 partial sum 分配 + reduce
    - 大 batch (n_rows >  sm_count * 8)：partial sum 版本，避免 atomic 竞争
    """
    shape = dY.shape
    hidden_dim = shape[-1]

    dY_flat = dY.view(-1, hidden_dim).contiguous()
    n_rows, n_cols = dY_flat.shape

    dX = torch.empty_like(dY_flat)

    sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
    use_atomic = n_rows <= sm_count * 8

    if use_atomic:
        # Atomic 版本：一个 program 处理一行，dW 直接原子累加
        # 手动计算 BLOCK_SIZE（不用 AutoTune，防止多次调用导致 dW 重复累加）
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = min(max(BLOCK_SIZE // 32, 1), 32)
        dW = torch.zeros(n_cols, dtype=torch.float32, device=X.device)
        _rms_norm_backward_kernel_atomic[(n_rows,)](
            dY_flat, dY_flat.stride(0),
            dX, dX.stride(0),
            X, X.stride(0),
            W, RSTD,
            dW,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    else:
        # Partial sum 版本：sm_count 个 program 各处理多行，最后 Python reduce
        num_programs = sm_count
        rows_per_program = (n_rows + num_programs - 1) // num_programs
        dW_partial = torch.empty((num_programs, n_cols), dtype=torch.float32, device=X.device)
        _rms_norm_backward_kernel_partial[(num_programs,)](
            dY_flat, dY_flat.stride(0),
            dX, dX.stride(0),
            X, X.stride(0),
            W, RSTD,
            dW_partial, dW_partial.stride(0),
            n_rows, n_cols, rows_per_program,
        )
        dW = dW_partial.sum(dim=0)

    return dX.view(*shape), dW.to(W.dtype)


# ============================================================================
# PyTorch Autograd Function
# ============================================================================

class HildaRMSNormFunction(torch.autograd.Function):
    """
    RMSNorm with AutoTune

    特点：
    1. 自动选择最优 kernel 配置
    2. 使用 recomputation 节省内存
    3. 支持 FP16/BF16 混合精度
    """

    @staticmethod
    def forward(ctx, X, W, eps=1e-6):
        Y, X_flat, RSTD = rms_norm_forward(X, W, eps)
        ctx.save_for_backward(X_flat, W, RSTD)
        return Y

    @staticmethod
    def backward(ctx, dY):
        X_flat, W, RSTD = ctx.saved_tensors
        dX, dW = rms_norm_backward(dY, X_flat, W, RSTD)
        return dX, dW, None


def hilda_rms_norm(X, W, eps=1e-6):
    """Functional interface for RMSNorm"""
    return HildaRMSNormFunction.apply(X, W, eps)


# ============================================================================
# nn.Module Wrapper
# ============================================================================

class RMSNorm(torch.nn.Module):
    """
    RMSNorm Module with AutoTune

    用法：
        norm = RMSNorm(hidden_dim=4096)
        output = norm(input)
    """

    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x):
        return hilda_rms_norm(x, self.weight, self.eps)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    # 简单测试
    torch.manual_seed(42)
    device = "cuda"

    # 测试配置
    batch_size = 4
    seq_len = 512
    hidden_dim = 4096

    print(f"Testing RMSNorm with AutoTune")
    print(f"Shape: ({batch_size}, {seq_len}, {hidden_dim})")
    print("-" * 50)

    # 创建输入
    X = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.float16, requires_grad=True)
    W = torch.ones(hidden_dim, device=device, dtype=torch.float16, requires_grad=True)

    # 使用我们的实现
    Y = hilda_rms_norm(X, W)

    # 使用 PyTorch 参考实现
    def pytorch_rms_norm(x, w, eps=1e-6):
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
        return (x.float() / rms * w.float()).to(x.dtype)

    Y_ref = pytorch_rms_norm(X, W)

    # 比较结果
    max_diff = (Y - Y_ref).abs().max().item()
    print(f"Forward max diff: {max_diff:.2e}")

    # 测试 backward
    loss = Y.sum()
    loss.backward()

    loss_ref = Y_ref.sum()
    X_ref = X.detach().clone().requires_grad_(True)
    W_ref = W.detach().clone().requires_grad_(True)
    Y_ref2 = pytorch_rms_norm(X_ref, W_ref)
    loss_ref2 = Y_ref2.sum()
    loss_ref2.backward()

    dX_diff = (X.grad - X_ref.grad).abs().max().item()
    dW_diff = (W.grad - W_ref.grad).abs().max().item()
    print(f"dX max diff: {dX_diff:.2e}")
    print(f"dW max diff: {dW_diff:.2e}")

    print("-" * 50)
    print("Test passed!" if max_diff < 1e-2 and dX_diff < 1e-2 else "Test FAILED!")
