"""
SwiGLU Benchmark: Ours vs Liger vs PyTorch

测试配置覆盖 LLM 常见 FFN intermediate dim:
- LLaMA-7B:  intermediate=11008
- LLaMA-3-8B / Mistral-7B: intermediate=14336
- LLaMA-70B: intermediate=28672

N = bsz * seq_len (flattened token count)
"""

import sys
import os
import torch
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.swiglu import hilda_swiglu

# 尝试导入 Liger
try:
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    HAS_LIGER = True
    print("Liger SwiGLU found.")
except ImportError:
    HAS_LIGER = False
    print("Liger SwiGLU not found.")


# ============================================================================
# Benchmark 工具
# ============================================================================

def benchmark_fn(fn, warmup=10, rep=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / rep * 1000  # ms


def pytorch_swiglu(a, b):
    return torch.nn.functional.silu(a) * b


# ============================================================================
# 正确性测试
# ============================================================================

def quick_test():
    print("Running correctness test...")
    device = 'cuda'

    configs = [
        (512,  11008,  torch.float16,  "LLaMA-7B  (D=11008, fp16)"),
        (512,  14336,  torch.float16,  "LLaMA-3   (D=14336, fp16)"),
        (512,  14336,  torch.bfloat16, "Mistral    (D=14336, bf16)"),
        (512,  28672,  torch.float16,  "LLaMA-70B (D=28672, fp16)"),
        (512,   4096,  torch.float32,  "Small      (D=4096,  fp32)"),
    ]

    for N, D, dtype, name in configs:
        a = torch.randn(N, D, device=device, dtype=dtype)
        b = torch.randn(N, D, device=device, dtype=dtype)

        ref  = pytorch_swiglu(a, b)
        ours = hilda_swiglu(a, b)

        diff = (ours.float() - ref.float()).abs().max().item()
        tol  = 1e-2 if dtype != torch.float32 else 1e-5
        status = "PASS" if diff < tol else "FAIL"
        print(f"  {name}: diff={diff:.2e} [{status}]")

    print("Correctness test done!\n")


def backward_test():
    print("Running backward correctness test...")
    device = 'cuda'

    configs = [
        (512, 11008, torch.float16,  "LLaMA-7B  (D=11008, fp16)"),
        (512, 14336, torch.float16,  "LLaMA-3   (D=14336, fp16)"),
        (512, 14336, torch.float32,  "LLaMA-3   (D=14336, fp32)"),
    ]

    for N, D, dtype, name in configs:
        a = torch.randn(N, D, device=device, dtype=dtype)
        b = torch.randn(N, D, device=device, dtype=dtype)

        # Reference
        a_ref = a.clone().requires_grad_(True)
        b_ref = b.clone().requires_grad_(True)
        out_ref = pytorch_swiglu(a_ref, b_ref)
        out_ref.sum().backward()

        # Ours
        a_ours = a.clone().requires_grad_(True)
        b_ours = b.clone().requires_grad_(True)
        out_ours = hilda_swiglu(a_ours, b_ours)
        out_ours.sum().backward()

        da_diff = (a_ours.grad.float() - a_ref.grad.float()).abs().max().item()
        db_diff = (b_ours.grad.float() - b_ref.grad.float()).abs().max().item()
        tol = 1e-1 if dtype != torch.float32 else 1e-4
        status_a = "PASS" if da_diff < tol else "FAIL"
        status_b = "PASS" if db_diff < tol else "FAIL"
        print(f"  {name}: da_diff={da_diff:.2e} [{status_a}], "
              f"db_diff={db_diff:.2e} [{status_b}]")

    print("Backward test done!\n")


# ============================================================================
# 性能测试
# ============================================================================

def bench_forward(N, D, dtype, device='cuda'):
    results = {}
    a = torch.randn(N, D, device=device, dtype=dtype)
    b = torch.randn(N, D, device=device, dtype=dtype)

    def pt_fn():
        return pytorch_swiglu(a, b)
    results['pytorch'] = benchmark_fn(pt_fn)

    def ours_fn():
        return hilda_swiglu(a, b)
    results['ours'] = benchmark_fn(ours_fn)

    if HAS_LIGER:
        def liger_fn():
            return LigerSiLUMulFunction.apply(a, b)
        try:
            results['liger'] = benchmark_fn(liger_fn)
        except Exception as e:
            print(f"  Liger error: {e}")
            results['liger'] = float('nan')

    return results


def bench_fwd_bwd(N, D, dtype, device='cuda'):
    results = {}

    def pt_fn():
        a = torch.randn(N, D, device=device, dtype=dtype, requires_grad=True)
        b = torch.randn(N, D, device=device, dtype=dtype, requires_grad=True)
        out = pytorch_swiglu(a, b)
        out.sum().backward()
    results['pytorch'] = benchmark_fn(pt_fn, warmup=5, rep=50)

    def ours_fn():
        a = torch.randn(N, D, device=device, dtype=dtype, requires_grad=True)
        b = torch.randn(N, D, device=device, dtype=dtype, requires_grad=True)
        out = hilda_swiglu(a, b)
        out.sum().backward()
    results['ours'] = benchmark_fn(ours_fn, warmup=5, rep=50)

    if HAS_LIGER:
        def liger_fn():
            a = torch.randn(N, D, device=device, dtype=dtype, requires_grad=True)
            b = torch.randn(N, D, device=device, dtype=dtype, requires_grad=True)
            out = LigerSiLUMulFunction.apply(a, b)
            out.sum().backward()
        try:
            results['liger'] = benchmark_fn(liger_fn, warmup=5, rep=50)
        except Exception as e:
            results['liger'] = float('nan')

    return results


def print_results(title, results, baseline='pytorch'):
    print(f"\n{title}")
    print("=" * 60)
    base = results.get(baseline, 1.0)
    for name, ms in results.items():
        speedup = base / ms if ms == ms else float('nan')
        tag = " (baseline)" if name == baseline else ""
        print(f"  {name:20s}: {ms:8.3f} ms  ({speedup:5.2f}x){tag}")


def main():
    print("=" * 70)
    print("SwiGLU Benchmark: Ours vs Liger vs PyTorch")
    print("=" * 70)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    # bsz*seq_len → N tokens
    configs = [
        # (N,     D,      dtype,          name)
        (8192,  11008,  torch.float16,  "LLaMA-7B   [D=11008, fp16]"),
        (8192,  14336,  torch.float16,  "LLaMA-3-8B [D=14336, fp16]"),
        (8192,  28672,  torch.float16,  "LLaMA-70B  [D=28672, fp16]"),
        (8192,  14336,  torch.bfloat16, "Mistral-7B  [D=14336, bf16]"),
        (2048,  14336,  torch.float16,  "Short seq   [D=14336, fp16, N=2K]"),
    ]

    print("\n" + "=" * 70)
    print("Forward Pass")
    print("=" * 70)
    for N, D, dtype, name in configs:
        results = bench_forward(N, D, dtype)
        print_results(name, results)

    print("\n" + "=" * 70)
    print("Forward + Backward")
    print("=" * 70)
    for N, D, dtype, name in configs:
        results = bench_fwd_bwd(N, D, dtype)
        print_results(name, results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        quick_test()
        backward_test()
    else:
        quick_test()
        backward_test()
        main()
