"""
RMSNorm Benchmark: 我们的 AutoTune 版本 vs Liger Kernel vs PyTorch

测试不同配置下的性能：
1. 不同 hidden_dim (1024, 2048, 4096, 8192)
2. 不同 batch_size × seq_len 组合
3. 不同精度 (FP16, BF16, FP32)
"""

import torch
import triton
import time
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.rms_norm import hilda_rms_norm, RMSNorm

# 尝试导入 Liger Kernel 进行对比
try:
    # 修复 Liger Kernel 对 torch.distributed.tensor 的依赖
    import torch.distributed
    if not hasattr(torch.distributed, 'tensor'):
        import types
        torch.distributed.tensor = types.ModuleType('torch.distributed.tensor')
        torch.distributed.tensor.DTensor = type('DTensor', (), {})
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction as OriginalLigerRMSNorm
    HAS_LIGER = True
    print("Liger Kernel found, will compare against it.")
except ImportError:
    HAS_LIGER = False
    print("Liger Kernel not found, will only compare against PyTorch.")


def pytorch_rms_norm(x, weight, eps=1e-6):
    """PyTorch 原生实现（作为 baseline）"""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return x_normed * weight


class PyTorchRMSNorm(torch.nn.Module):
    """PyTorch RMSNorm Module"""

    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x):
        return pytorch_rms_norm(x, self.weight, self.eps)


def benchmark_fn(fn, *args, warmup=10, rep=100):
    """Benchmark 一个函数"""
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(rep):
        fn(*args)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / rep * 1000  # ms


def benchmark_forward(batch_size, seq_len, hidden_dim, dtype, device="cuda"):
    """Benchmark forward pass"""
    # 创建输入
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype, requires_grad=False)
    weight = torch.ones(hidden_dim, device=device, dtype=dtype)

    results = {}

    # 1. PyTorch baseline
    def pytorch_fn():
        return pytorch_rms_norm(x, weight)

    results['pytorch'] = benchmark_fn(pytorch_fn)

    # 2. 我们的 AutoTune 版本
    def ours_fn():
        return hilda_rms_norm(x, weight)

    results['ours_autotune'] = benchmark_fn(ours_fn)

    # 3. Liger Kernel（如果可用）
    if HAS_LIGER:
        def liger_fn():
            return OriginalLigerRMSNorm.apply(x, weight, 1e-6)

        results['liger_original'] = benchmark_fn(liger_fn)

    return results


def benchmark_forward_backward(batch_size, seq_len, hidden_dim, dtype, device="cuda"):
    """Benchmark forward + backward pass"""
    results = {}

    # 1. PyTorch baseline
    def pytorch_fn():
        x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype, requires_grad=True)
        weight = torch.ones(hidden_dim, device=device, dtype=dtype, requires_grad=True)
        y = pytorch_rms_norm(x, weight)
        loss = y.sum()
        loss.backward()
        return y

    results['pytorch'] = benchmark_fn(pytorch_fn, warmup=5, rep=50)

    # 2. 我们的 AutoTune 版本
    def ours_fn():
        x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype, requires_grad=True)
        weight = torch.ones(hidden_dim, device=device, dtype=dtype, requires_grad=True)
        y = hilda_rms_norm(x, weight)
        loss = y.sum()
        loss.backward()
        return y

    results['ours_autotune'] = benchmark_fn(ours_fn, warmup=5, rep=50)

    # 3. Liger Kernel（如果可用）
    if HAS_LIGER:
        def liger_fn():
            x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype, requires_grad=True)
            weight = torch.ones(hidden_dim, device=device, dtype=dtype, requires_grad=True)
            y = OriginalLigerRMSNorm.apply(x, weight, 1e-6)
            loss = y.sum()
            loss.backward()
            return y

        results['liger_original'] = benchmark_fn(liger_fn, warmup=5, rep=50)

    return results


def print_results(title, results, baseline_key='pytorch'):
    """打印结果表格"""
    print(f"\n{title}")
    print("=" * 60)

    baseline = results.get(baseline_key, 1.0)

    for name, time_ms in results.items():
        speedup = baseline / time_ms
        marker = " (baseline)" if name == baseline_key else ""
        print(f"  {name:20s}: {time_ms:8.3f} ms  ({speedup:5.2f}x){marker}")


def main():
    print("=" * 70)
    print("RMSNorm Benchmark: AutoTune vs Liger vs PyTorch")
    print("=" * 70)

    # 获取 GPU 信息
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"\nGPU: {gpu_name}")
    print(f"CUDA Version: {torch.version.cuda}")

    # 测试配置
    configs = [
        # (batch_size, seq_len, hidden_dim, dtype)
        (4, 512, 4096, torch.float16),    # 小规模
        (8, 1024, 4096, torch.float16),   # 中规模
        (4, 2048, 4096, torch.float16),   # 长序列
        (8, 2048, 4096, torch.float16),   # 大规模
        (4, 1024, 8192, torch.float16),   # 大 hidden_dim
        (4, 1024, 4096, torch.bfloat16),  # BF16
    ]

    print("\n" + "=" * 70)
    print("Forward Pass Benchmark")
    print("=" * 70)

    for batch_size, seq_len, hidden_dim, dtype in configs:
        title = f"[B={batch_size}, S={seq_len}, H={hidden_dim}, {dtype}]"
        results = benchmark_forward(batch_size, seq_len, hidden_dim, dtype, device)
        print_results(title, results)

    print("\n" + "=" * 70)
    print("Forward + Backward Benchmark")
    print("=" * 70)

    for batch_size, seq_len, hidden_dim, dtype in configs[:4]:  # 只测试部分配置
        title = f"[B={batch_size}, S={seq_len}, H={hidden_dim}, {dtype}]"
        results = benchmark_forward_backward(batch_size, seq_len, hidden_dim, dtype, device)
        print_results(title, results)

    # AutoTune 配置选择信息
    print("\n" + "=" * 70)
    print("AutoTune Selected Configs")
    print("=" * 70)
    print("\nNote: First run triggers AutoTune, subsequent runs use cached config.")
    print("Check ~/.triton/cache for cached configurations.")


def quick_test():
    """快速正确性测试"""
    print("Running quick correctness test...")

    device = "cuda"
    dtype = torch.float16

    for hidden_dim in [1024, 4096, 8192]:
        x = torch.randn(2, 128, hidden_dim, device=device, dtype=dtype, requires_grad=True)
        w = torch.ones(hidden_dim, device=device, dtype=dtype, requires_grad=True)

        # 我们的实现
        y_ours = hilda_rms_norm(x, w)

        # PyTorch 参考
        y_ref = pytorch_rms_norm(x.detach(), w.detach())

        # 检查 forward
        max_diff = (y_ours - y_ref).abs().max().item()
        status = "PASS" if max_diff < 1e-2 else "FAIL"
        print(f"  hidden_dim={hidden_dim}: forward max_diff={max_diff:.2e} [{status}]")

        # 检查 backward
        y_ours.sum().backward()
        x_ref = x.detach().clone().requires_grad_(True)
        w_ref = w.detach().clone().requires_grad_(True)
        pytorch_rms_norm(x_ref, w_ref).sum().backward()

        dx_diff = (x.grad - x_ref.grad).abs().max().item()
        dw_diff = (w.grad - w_ref.grad).abs().max().item()
        # dW 是多 program partial sum 累加，大 hidden_dim 下 fp16 误差更大，用 5e-2 阈值
        status = "PASS" if dx_diff < 1e-2 and dw_diff < 5e-2 else "FAIL"
        print(f"  hidden_dim={hidden_dim}: backward dx_diff={dx_diff:.2e}, dw_diff={dw_diff:.2e} [{status}]")

        # 重置 grad
        x.grad = None
        w.grad = None

    print("Quick test done!\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run quick correctness test only")
    args = parser.parse_args()

    if args.quick:
        quick_test()
    else:
        quick_test()
        main()
