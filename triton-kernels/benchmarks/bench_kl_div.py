"""
KL Divergence Benchmark: Ours vs Liger vs PyTorch
"""

import sys
import os
import torch
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.kl_div import hilda_kl_div

try:
    from liger_kernel.ops.kl_div import LigerKLDivLossFunction
    HAS_LIGER = True
    print("Liger KLDiv found.")
except ImportError:
    HAS_LIGER = False
    print("Liger KLDiv not found.")


def benchmark_fn(fn, warmup=10, rep=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / rep * 1000


def quick_test():
    print("Running correctness test...")
    device = 'cuda'

    configs = [
        (512,  32000, torch.float16,  "V=32K,  fp16"),
        (512, 128256, torch.float16,  "V=128K, fp16"),
        (512,  32000, torch.float32,  "V=32K,  fp32"),
    ]

    for N, V, dtype, name in configs:
        y_pred = torch.randn(N, V, device=device, dtype=dtype).log_softmax(dim=-1)
        y_true = torch.randn(N, V, device=device, dtype=dtype).softmax(dim=-1)

        ref  = torch.nn.functional.kl_div(y_pred, y_true, reduction='batchmean')
        ours = hilda_kl_div(y_pred, y_true, reduction='batchmean')

        diff = abs(ours.item() - ref.item())
        tol = 1e-1 if dtype != torch.float32 else 1e-4
        status = "PASS" if diff < tol else "FAIL"
        print(f"  {name}: ref={ref.item():.6f}, ours={ours.item():.6f}, diff={diff:.2e} [{status}]")

    print("Correctness test done!\n")


def backward_test():
    print("Running backward correctness test...")
    device = 'cuda'

    for N, V, dtype, name in [(512, 32000, torch.float16, "fp16"), (512, 32000, torch.float32, "fp32")]:
        y_pred = torch.randn(N, V, device=device, dtype=dtype).log_softmax(dim=-1)
        y_true = torch.randn(N, V, device=device, dtype=dtype).softmax(dim=-1)

        yp_ref = y_pred.clone().requires_grad_(True)
        loss_ref = torch.nn.functional.kl_div(yp_ref, y_true, reduction='batchmean')
        loss_ref.backward()

        yp_ours = y_pred.clone().requires_grad_(True)
        loss_ours = hilda_kl_div(yp_ours, y_true, reduction='batchmean')
        loss_ours.backward()

        grad_diff = (yp_ours.grad.float() - yp_ref.grad.float()).abs().max().item()
        tol = 1e-2 if dtype != torch.float32 else 1e-5
        status = "PASS" if grad_diff < tol else "FAIL"
        print(f"  {name}: grad_diff={grad_diff:.2e} [{status}]")

    print("Backward test done!\n")


def bench_forward(N, V, dtype, device='cuda'):
    results = {}
    y_pred = torch.randn(N, V, device=device, dtype=dtype).log_softmax(dim=-1)
    y_true = torch.randn(N, V, device=device, dtype=dtype).softmax(dim=-1)

    def pt_fn():
        return torch.nn.functional.kl_div(y_pred, y_true, reduction='batchmean')
    results['pytorch'] = benchmark_fn(pt_fn)

    def ours_fn():
        return hilda_kl_div(y_pred, y_true, reduction='batchmean')
    results['ours'] = benchmark_fn(ours_fn)

    if HAS_LIGER:
        def liger_fn():
            return LigerKLDivLossFunction.apply(y_pred, y_true, 'batchmean', False, 1e-10)
        try:
            results['liger'] = benchmark_fn(liger_fn)
        except Exception as e:
            print(f"  Liger error: {e}")
            results['liger'] = float('nan')

    return results


def bench_fwd_bwd(N, V, dtype, device='cuda'):
    results = {}

    def pt_fn():
        yp = torch.randn(N, V, device=device, dtype=dtype).log_softmax(dim=-1).requires_grad_(True)
        yt = torch.randn(N, V, device=device, dtype=dtype).softmax(dim=-1)
        loss = torch.nn.functional.kl_div(yp, yt, reduction='batchmean')
        loss.backward()
    results['pytorch'] = benchmark_fn(pt_fn, warmup=5, rep=50)

    def ours_fn():
        yp = torch.randn(N, V, device=device, dtype=dtype).log_softmax(dim=-1).requires_grad_(True)
        yt = torch.randn(N, V, device=device, dtype=dtype).softmax(dim=-1)
        loss = hilda_kl_div(yp, yt, reduction='batchmean')
        loss.backward()
    results['ours'] = benchmark_fn(ours_fn, warmup=5, rep=50)

    if HAS_LIGER:
        def liger_fn():
            yp = torch.randn(N, V, device=device, dtype=dtype).log_softmax(dim=-1).requires_grad_(True)
            yt = torch.randn(N, V, device=device, dtype=dtype).softmax(dim=-1)
            loss = LigerKLDivLossFunction.apply(yp, yt, 'batchmean', False, 1e-10)
            loss.backward()
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
    print("KL Divergence Benchmark: Ours vs Liger vs PyTorch")
    print("=" * 70)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    configs = [
        (8192,  32000, torch.float16,  "V=32K  [fp16]"),
        (8192, 128256, torch.float16,  "V=128K [fp16]"),
        (8192,  32000, torch.bfloat16, "V=32K  [bf16]"),
        (2048,  32000, torch.float16,  "V=32K  [fp16, N=2K]"),
    ]

    print("\n" + "=" * 70)
    print("Forward Pass")
    print("=" * 70)
    for N, V, dtype, name in configs:
        results = bench_forward(N, V, dtype)
        print_results(name, results)

    print("\n" + "=" * 70)
    print("Forward + Backward")
    print("=" * 70)
    for N, V, dtype, name in configs:
        results = bench_fwd_bwd(N, V, dtype)
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
