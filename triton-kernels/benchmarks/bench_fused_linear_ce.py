"""
FusedLinearCrossEntropy Benchmark: Ours vs Liger vs PyTorch

Fuses lm_head linear + CE loss, avoids materializing (BT, V) logits.
"""

import sys
import os
import torch
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.fused_linear_cross_entropy import hilda_fused_linear_cross_entropy

try:
    from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
    HAS_LIGER = True
    print("Liger FusedLinearCE found.")
except ImportError:
    HAS_LIGER = False
    print("Liger FusedLinearCE not found.")


def benchmark_fn(fn, warmup=10, rep=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / rep * 1000


def pytorch_linear_ce(hidden, weight, targets):
    logits = hidden @ weight.T
    return torch.nn.functional.cross_entropy(logits, targets, reduction='mean')


def quick_test():
    print("Running correctness test...")
    device = 'cuda'

    configs = [
        (512, 4096, 32000,  torch.float16,  "H=4096, V=32K,  fp16"),
        (512, 4096, 128256, torch.float16,  "H=4096, V=128K, fp16"),
        (256, 4096, 32000,  torch.float32,  "H=4096, V=32K,  fp32"),
    ]

    for N, H, V, dtype, name in configs:
        hidden  = torch.randn(N, H, device=device, dtype=dtype)
        weight  = torch.randn(V, H, device=device, dtype=dtype) * 0.02
        targets = torch.randint(0, V, (N,), device=device)

        ref  = pytorch_linear_ce(hidden, weight, targets)
        ours = hilda_fused_linear_cross_entropy(hidden, weight, targets)

        diff = abs(ours.item() - ref.item())
        tol = 1e-1 if dtype != torch.float32 else 1e-3
        status = "PASS" if diff < tol else "FAIL"
        print(f"  {name}: ref={ref.item():.4f}, ours={ours.item():.4f}, diff={diff:.2e} [{status}]")

    print("Correctness test done!\n")


def backward_test():
    print("Running backward correctness test...")
    device = 'cuda'

    for N, H, V, dtype, name in [(256, 4096, 32000, torch.float16, "fp16"), (256, 4096, 32000, torch.float32, "fp32")]:
        hidden  = torch.randn(N, H, device=device, dtype=dtype)
        weight  = torch.randn(V, H, device=device, dtype=dtype) * 0.02
        targets = torch.randint(0, V, (N,), device=device)

        h_ref = hidden.clone().requires_grad_(True)
        w_ref = weight.clone().requires_grad_(True)
        loss_ref = pytorch_linear_ce(h_ref, w_ref, targets)
        loss_ref.backward()

        h_ours = hidden.clone().requires_grad_(True)
        w_ours = weight.clone().requires_grad_(True)
        loss_ours = hilda_fused_linear_cross_entropy(h_ours, w_ours, targets)
        loss_ours.backward()

        dh_diff = (h_ours.grad.float() - h_ref.grad.float()).abs().max().item()
        dw_diff = (w_ours.grad.float() - w_ref.grad.float()).abs().max().item()
        tol_h = 5e-2 if dtype != torch.float32 else 1e-4
        tol_w = 5e-1 if dtype != torch.float32 else 1e-2
        status = "PASS" if dh_diff < tol_h and dw_diff < tol_w else "FAIL"
        print(f"  {name}: dH={dh_diff:.2e}, dW={dw_diff:.2e} [{status}]")

    print("Backward test done!\n")


def bench_forward(N, H, V, dtype, device='cuda'):
    results = {}
    hidden  = torch.randn(N, H, device=device, dtype=dtype)
    weight  = torch.randn(V, H, device=device, dtype=dtype) * 0.02
    targets = torch.randint(0, V, (N,), device=device)

    def pt_fn():
        return pytorch_linear_ce(hidden, weight, targets)
    results['pytorch'] = benchmark_fn(pt_fn)

    def ours_fn():
        return hilda_fused_linear_cross_entropy(hidden, weight, targets)
    results['ours'] = benchmark_fn(ours_fn)

    if HAS_LIGER:
        def liger_fn():
            return LigerFusedLinearCrossEntropyFunction.apply(
                hidden, weight, targets, None, None, -100, 0.0, 0.0, 'mean', None, False
            )[0]
        try:
            results['liger'] = benchmark_fn(liger_fn)
        except Exception as e:
            print(f"  Liger error: {e}")
            results['liger'] = float('nan')

    return results


def bench_fwd_bwd(N, H, V, dtype, device='cuda'):
    results = {}
    targets = torch.randint(0, V, (N,), device=device)

    def pt_fn():
        h = torch.randn(N, H, device=device, dtype=dtype, requires_grad=True)
        w = torch.randn(V, H, device=device, dtype=dtype, requires_grad=True)
        loss = pytorch_linear_ce(h, w, targets)
        loss.backward()
    results['pytorch'] = benchmark_fn(pt_fn, warmup=5, rep=50)

    def ours_fn():
        h = torch.randn(N, H, device=device, dtype=dtype, requires_grad=True)
        w = torch.randn(V, H, device=device, dtype=dtype, requires_grad=True)
        loss = hilda_fused_linear_cross_entropy(h, w, targets)
        loss.backward()
    results['ours'] = benchmark_fn(ours_fn, warmup=5, rep=50)

    if HAS_LIGER:
        def liger_fn():
            h = torch.randn(N, H, device=device, dtype=dtype, requires_grad=True)
            w = torch.randn(V, H, device=device, dtype=dtype, requires_grad=True)
            loss = LigerFusedLinearCrossEntropyFunction.apply(
                h, w, targets, None, None, -100, 0.0, 0.0, 'mean', None, False
            )[0]
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


# ============================================================================
# 显存对比 — 核心价值
# ============================================================================

def measure_peak_memory(fn, device='cuda'):
    """Run fn and return peak GPU memory allocated (MB)."""
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    fn()
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 1024 / 1024  # MB


def bench_memory(N, H, V, dtype, device='cuda'):
    """Compare peak memory: PyTorch (full logits) vs Ours (chunked) vs Liger."""
    results = {}
    targets = torch.randint(0, V, (N,), device=device)

    # Theoretical logits size
    elem_bytes = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    logits_mb = N * V * elem_bytes / 1024 / 1024

    # PyTorch: materializes full (N, V) logits
    def pt_fn():
        h = torch.randn(N, H, device=device, dtype=dtype, requires_grad=True)
        w = torch.randn(V, H, device=device, dtype=dtype, requires_grad=True)
        loss = pytorch_linear_ce(h, w, targets)
        loss.backward()
    # warmup
    pt_fn()
    results['pytorch'] = measure_peak_memory(pt_fn, device)

    # Ours: chunked, avoids full logits
    def ours_fn():
        h = torch.randn(N, H, device=device, dtype=dtype, requires_grad=True)
        w = torch.randn(V, H, device=device, dtype=dtype, requires_grad=True)
        loss = hilda_fused_linear_cross_entropy(h, w, targets)
        loss.backward()
    ours_fn()
    results['ours'] = measure_peak_memory(ours_fn, device)

    if HAS_LIGER:
        def liger_fn():
            h = torch.randn(N, H, device=device, dtype=dtype, requires_grad=True)
            w = torch.randn(V, H, device=device, dtype=dtype, requires_grad=True)
            loss = LigerFusedLinearCrossEntropyFunction.apply(
                h, w, targets, None, None, -100, 0.0, 0.0, 'mean', None, False
            )[0]
            loss.backward()
        try:
            liger_fn()
            results['liger'] = measure_peak_memory(liger_fn, device)
        except Exception:
            results['liger'] = float('nan')

    return results, logits_mb


def print_memory_results(title, results, logits_mb):
    print(f"\n{title}")
    print(f"  (full logits tensor = {logits_mb:.0f} MB)")
    print("=" * 60)
    base = results.get('pytorch', 1.0)
    for name, mb in results.items():
        saved = base - mb
        pct = saved / base * 100 if base > 0 else 0
        tag = " (baseline)" if name == 'pytorch' else f"  saved {saved:+.0f} MB ({pct:+.1f}%)"
        print(f"  {name:20s}: {mb:8.0f} MB{tag}")


def main():
    print("=" * 70)
    print("FusedLinearCrossEntropy Benchmark: Ours vs Liger vs PyTorch")
    print("=" * 70)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    configs = [
        (8192, 4096,  32000,  torch.float16,  "LLaMA-7B  [V=32K,  fp16]"),
        (8192, 4096, 128256,  torch.float16,  "LLaMA-3   [V=128K, fp16]"),
        (8192, 4096,  32000,  torch.bfloat16, "LLaMA-7B  [V=32K,  bf16]"),
        (2048, 4096,  32000,  torch.float16,  "Short      [V=32K,  fp16, N=2K]"),
    ]

    print("\n" + "=" * 70)
    print("Forward Pass (includes linear + CE)")
    print("=" * 70)
    for N, H, V, dtype, name in configs:
        results = bench_forward(N, H, V, dtype)
        print_results(name, results)

    print("\n" + "=" * 70)
    print("Forward + Backward")
    print("=" * 70)
    for N, H, V, dtype, name in configs:
        results = bench_fwd_bwd(N, H, V, dtype)
        print_results(name, results)

    # ── 显存对比 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Peak Memory (Fwd+Bwd) — the core value of fusion")
    print("=" * 70)
    mem_configs = [
        (8192, 4096,  32000,  torch.float16,  "LLaMA-7B  [V=32K,  fp16]"),
        (8192, 4096, 128256,  torch.float16,  "LLaMA-3   [V=128K, fp16]"),
        (4096, 4096, 128256,  torch.float16,  "LLaMA-3   [V=128K, fp16, N=4K]"),
    ]
    for N, H, V, dtype, name in mem_configs:
        results, logits_mb = bench_memory(N, H, V, dtype)
        print_memory_results(name, results, logits_mb)


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
