"""
FusedAddRMSNorm Benchmark: Ours vs Liger vs PyTorch

S = X + R
Y = RMSNorm(S) * W
"""

import sys
import os
import torch
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.fused_add_rms_norm import hilda_fused_add_rms_norm

try:
    from liger_kernel.ops.fused_add_rms_norm import LigerFusedAddRMSNormFunction
    HAS_LIGER = True
    print("Liger FusedAddRMSNorm found.")
except ImportError:
    HAS_LIGER = False
    print("Liger FusedAddRMSNorm not found.")


def benchmark_fn(fn, warmup=10, rep=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / rep * 1000


def pytorch_fused_add_rms_norm(X, R, W, eps=1e-6):
    S = X + R
    rms = torch.sqrt(torch.mean(S.float() ** 2, dim=-1, keepdim=True) + eps)
    Y = (S.float() / rms).to(S.dtype) * W
    return Y, S


# ============================================================================
# 正确性测试
# ============================================================================

def quick_test():
    print("Running correctness test...")
    device = 'cuda'

    configs = [
        (4, 512, 4096,  torch.float16,  "LLaMA-7B  (D=4096, fp16)"),
        (4, 512, 4096,  torch.bfloat16, "LLaMA-7B  (D=4096, bf16)"),
        (4, 512, 5120,  torch.float16,  "LLaMA-13B (D=5120, fp16)"),
        (4, 512, 8192,  torch.float16,  "LLaMA-70B (D=8192, fp16)"),
        (2, 256, 4096,  torch.float32,  "Small      (D=4096, fp32)"),
    ]

    for B, T, D, dtype, name in configs:
        X = torch.randn(B, T, D, device=device, dtype=dtype)
        R = torch.randn(B, T, D, device=device, dtype=dtype)
        W = torch.ones(D, device=device, dtype=dtype)

        Y_ref, S_ref = pytorch_fused_add_rms_norm(X, R, W)
        Y_ours, S_ours = hilda_fused_add_rms_norm(X, R, W)

        s_diff = (S_ours.float() - S_ref.float()).abs().max().item()
        y_diff = (Y_ours.float() - Y_ref.float()).abs().max().item()
        tol = 2e-2 if dtype != torch.float32 else 1e-5
        status = "PASS" if y_diff < tol and s_diff < tol else "FAIL"
        print(f"  {name}: S_diff={s_diff:.2e}, Y_diff={y_diff:.2e} [{status}]")

    print("Correctness test done!\n")


def backward_test():
    print("Running backward correctness test...")
    device = 'cuda'

    configs = [
        (4, 512, 4096, torch.float16,  "D=4096, fp16"),
        (4, 512, 4096, torch.float32,  "D=4096, fp32"),
    ]

    for B, T, D, dtype, name in configs:
        X = torch.randn(B, T, D, device=device, dtype=dtype)
        R = torch.randn(B, T, D, device=device, dtype=dtype)
        W = torch.ones(D, device=device, dtype=dtype)

        # Reference
        X_ref = X.clone().requires_grad_(True)
        R_ref = R.clone().requires_grad_(True)
        W_ref = W.clone().requires_grad_(True)
        Y_ref, S_ref = pytorch_fused_add_rms_norm(X_ref, R_ref, W_ref)
        (Y_ref.sum() + S_ref.sum()).backward()

        # Ours
        X_ours = X.clone().requires_grad_(True)
        R_ours = R.clone().requires_grad_(True)
        W_ours = W.clone().requires_grad_(True)
        Y_ours, S_ours = hilda_fused_add_rms_norm(X_ours, R_ours, W_ours)
        (Y_ours.sum() + S_ours.sum()).backward()

        dx_diff = (X_ours.grad.float() - X_ref.grad.float()).abs().max().item()
        dr_diff = (R_ours.grad.float() - R_ref.grad.float()).abs().max().item()
        dw_diff = (W_ours.grad.float() - W_ref.grad.float()).abs().max().item()
        tol_dx = 5e-2 if dtype != torch.float32 else 1e-4
        # dW accumulates across all rows; fp32 atomic vs fp16 ref → larger diff expected
        tol_dw = 5e-1 if dtype != torch.float32 else 1e-3
        status = "PASS" if dx_diff < tol_dx and dr_diff < tol_dx and dw_diff < tol_dw else "FAIL"
        print(f"  {name}: dX={dx_diff:.2e}, dR={dr_diff:.2e}, dW={dw_diff:.2e} [{status}]")

    print("Backward test done!\n")


# ============================================================================
# 性能测试
# ============================================================================

def bench_forward(B, T, D, dtype, device='cuda'):
    results = {}
    N = B * T
    X = torch.randn(B, T, D, device=device, dtype=dtype)
    R = torch.randn(B, T, D, device=device, dtype=dtype)
    W = torch.ones(D, device=device, dtype=dtype)

    def pt_fn():
        return pytorch_fused_add_rms_norm(X, R, W)
    results['pytorch'] = benchmark_fn(pt_fn)

    def ours_fn():
        return hilda_fused_add_rms_norm(X, R, W)
    results['ours'] = benchmark_fn(ours_fn)

    if HAS_LIGER:
        def liger_fn():
            return LigerFusedAddRMSNormFunction.apply(X, R, W, 1e-6, 0.0, "llama", False)
        try:
            results['liger'] = benchmark_fn(liger_fn)
        except Exception as e:
            print(f"  Liger error: {e}")
            results['liger'] = float('nan')

    return results


def bench_fwd_bwd(B, T, D, dtype, device='cuda'):
    results = {}
    W = torch.ones(D, device=device, dtype=dtype)

    def pt_fn():
        X = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
        R = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
        Y, S = pytorch_fused_add_rms_norm(X, R, W)
        (Y.sum() + S.sum()).backward()
    results['pytorch'] = benchmark_fn(pt_fn, warmup=5, rep=50)

    def ours_fn():
        X = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
        R = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
        Y, S = hilda_fused_add_rms_norm(X, R, W)
        (Y.sum() + S.sum()).backward()
    results['ours'] = benchmark_fn(ours_fn, warmup=5, rep=50)

    if HAS_LIGER:
        W_liger = W.clone()
        def liger_fn():
            X = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
            R = torch.randn(B, T, D, device=device, dtype=dtype, requires_grad=True)
            Y, S = LigerFusedAddRMSNormFunction.apply(X, R, W_liger, 1e-6, 0.0, "llama", False)
            (Y.sum() + S.sum()).backward()
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
    print("FusedAddRMSNorm Benchmark: Ours vs Liger vs PyTorch")
    print("=" * 70)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    configs = [
        (4, 2048, 4096,  torch.float16,  "LLaMA-7B   [D=4096,  fp16]"),
        (4, 2048, 5120,  torch.float16,  "LLaMA-13B  [D=5120,  fp16]"),
        (4, 2048, 8192,  torch.float16,  "LLaMA-70B  [D=8192,  fp16]"),
        (4, 2048, 4096,  torch.bfloat16, "LLaMA-7B   [D=4096,  bf16]"),
        (2, 1024, 4096,  torch.float16,  "Short       [D=4096,  fp16, N=2K]"),
    ]

    print("\n" + "=" * 70)
    print("Forward Pass")
    print("=" * 70)
    for B, T, D, dtype, name in configs:
        results = bench_forward(B, T, D, dtype)
        print_results(name, results)

    print("\n" + "=" * 70)
    print("Forward + Backward")
    print("=" * 70)
    for B, T, D, dtype, name in configs:
        results = bench_fwd_bwd(B, T, D, dtype)
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
