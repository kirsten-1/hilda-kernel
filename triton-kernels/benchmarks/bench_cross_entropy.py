"""
CrossEntropy Benchmark: Ours vs Liger vs PyTorch

两组对比：
  [A] 无 ignore token（三方公平对比：Ours / Liger / PyTorch）
  [B] 含 ignore token（Ours / PyTorch；Liger 不支持负 ignore_index）

测试配置覆盖：
- LLaMA-2 7B: V=32000,  N=bsz*seq_len
- LLaMA-3 8B: V=128256, N=bsz*seq_len
- Mistral 7B: V=32000,  N=bsz*seq_len (same as LLaMA-2)
"""

import sys
import os
import torch
import triton
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.cross_entropy import hilda_cross_entropy

# 尝试导入 Liger
try:
    from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
    HAS_LIGER = True
    print("Liger CrossEntropy found.")
except ImportError:
    HAS_LIGER = False
    print("Liger CrossEntropy not found.")


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


def make_inputs(N, V, dtype, device='cuda', ignore_frac=0.0):
    """Create logits + targets. ignore_frac>0 marks that fraction as -100."""
    logits  = torch.randn(N, V, device=device, dtype=dtype)
    targets = torch.randint(0, V, (N,), device=device)
    if ignore_frac > 0:
        ignore_mask = torch.rand(N, device=device) < ignore_frac
        targets[ignore_mask] = -100
    return logits, targets


# ============================================================================
# 正确性测试
# ============================================================================

def quick_test():
    print("Running correctness test...")
    device = 'cuda'
    dtype  = torch.float32

    configs = [
        (512,  32000, "LLaMA-2 style (V=32K)"),
        (512, 128256, "LLaMA-3 style (V=128K)"),
        (512,   4096, "Small vocab (V=4K)"),
    ]

    for N, V, name in configs:
        logits, targets = make_inputs(N, V, dtype, device, ignore_frac=0.1)

        loss_ref  = torch.nn.functional.cross_entropy(
            logits, targets, ignore_index=-100, reduction='mean'
        )
        # 推理版
        loss_ours = hilda_cross_entropy(logits.clone(), targets, ignore_index=-100, reduction='mean')
        diff   = abs(loss_ours.item() - loss_ref.item())
        status = "PASS" if diff < 1e-4 else "FAIL"
        print(f"  {name} [inference]: loss_ref={loss_ref.item():.6f}, loss_ours={loss_ours.item():.6f}, "
              f"diff={diff:.2e} [{status}]")


    print("Correctness test done!\n")


def backward_test():
    print("Running backward correctness test...")
    device = 'cuda'
    dtype  = torch.float32

    N, V = 512, 32000
    logits, targets = make_inputs(N, V, dtype, device, ignore_frac=0.1)

    # Reference
    logits_ref = logits.clone().requires_grad_(True)
    loss_ref   = torch.nn.functional.cross_entropy(
        logits_ref, targets, ignore_index=-100, reduction='mean'
    )
    loss_ref.backward()

    # Ours (inference)
    logits_ours = logits.clone().requires_grad_(True)
    loss_ours   = hilda_cross_entropy(logits_ours, targets, ignore_index=-100, reduction='mean')
    loss_ours.backward()

    dlogits_diff = (logits_ours.grad - logits_ref.grad).abs().max().item()
    status = "PASS" if dlogits_diff < 1e-4 else "FAIL"
    print(f"  [inference] backward dlogits_diff={dlogits_diff:.2e} [{status}]")

    # Test ignore_index zeroing: ignored tokens must have zero gradient
    ignored_mask = (targets == -100)
    if ignored_mask.any():
        max_ignored_grad = logits_ours.grad[ignored_mask].abs().max().item()
        print(f"  ignored token gradient max: {max_ignored_grad:.2e} "
              f"[{'PASS' if max_ignored_grad < 1e-6 else 'FAIL'}]")

    print("Backward test done!\n")


# ============================================================================
# 性能测试 — [A] 无 ignore（三方公平对比）
# ============================================================================

def bench_forward_no_ignore(N, V, dtype, device='cuda'):
    """All tokens valid — fair 3-way comparison."""
    results = {}
    logits, targets = make_inputs(N, V, dtype, device, ignore_frac=0.0)

    def pt_fn():
        return torch.nn.functional.cross_entropy(logits, targets, reduction='mean')
    results['pytorch'] = benchmark_fn(pt_fn)

    def ours_fn():
        return hilda_cross_entropy(logits, targets, ignore_index=-1, reduction='mean')
    results['ours'] = benchmark_fn(ours_fn)

    if HAS_LIGER:
        # Liger signature: apply(logits, targets, weight, ignore_index, lse_square_scale,
        #                        label_smoothing, reduction, softcap, return_z_loss,
        #                        return_token_accuracy)
        def liger_fn():
            # Liger returns (loss, z_loss, token_accuracy); take [0]
            return LigerCrossEntropyFunction.apply(
                logits, targets, None, -1, 0.0, 0.0, 'mean', None, False, False
            )[0]
        try:
            results['liger'] = benchmark_fn(liger_fn)
        except Exception as e:
            print(f"  Liger error: {e}")
            results['liger'] = float('nan')

    return results


def bench_fwd_bwd_no_ignore(N, V, dtype, device='cuda'):
    """All tokens valid — fair 3-way comparison."""
    results = {}

    def pt_fn():
        logits  = torch.randn(N, V, device=device, dtype=dtype, requires_grad=True)
        targets = torch.randint(0, V, (N,), device=device)
        loss = torch.nn.functional.cross_entropy(logits, targets, reduction='mean')
        loss.backward()
    results['pytorch'] = benchmark_fn(pt_fn, warmup=5, rep=50)

    def ours_fn():
        logits  = torch.randn(N, V, device=device, dtype=dtype, requires_grad=True)
        targets = torch.randint(0, V, (N,), device=device)
        loss = hilda_cross_entropy(logits, targets, ignore_index=-1, reduction='mean')
        loss.backward()
    results['ours'] = benchmark_fn(ours_fn, warmup=5, rep=50)

    if HAS_LIGER:
        def liger_fn():
            logits  = torch.randn(N, V, device=device, dtype=dtype, requires_grad=True)
            targets = torch.randint(0, V, (N,), device=device)
            # Liger returns (loss, z_loss, token_accuracy); take [0]
            loss = LigerCrossEntropyFunction.apply(
                logits, targets, None, -1, 0.0, 0.0, 'mean', None, False, False
            )[0]
            loss.backward()
        try:
            results['liger'] = benchmark_fn(liger_fn, warmup=5, rep=50)
        except Exception as e:
            results['liger'] = float('nan')

    return results


# ============================================================================
# 性能测试 — [B] 含 ignore（Ours vs PyTorch，~10% padding）
# ============================================================================

def bench_forward_with_ignore(N, V, dtype, device='cuda'):
    """~10% ignored tokens — Ours vs PyTorch only (Liger doesn't support negative ignore_index)."""
    results = {}
    logits, targets = make_inputs(N, V, dtype, device, ignore_frac=0.1)

    def pt_fn():
        return torch.nn.functional.cross_entropy(
            logits, targets, ignore_index=-100, reduction='mean'
        )
    results['pytorch'] = benchmark_fn(pt_fn)

    def ours_fn():
        return hilda_cross_entropy(logits, targets, ignore_index=-100, reduction='mean')
    results['ours'] = benchmark_fn(ours_fn)

    return results


def bench_fwd_bwd_with_ignore(N, V, dtype, device='cuda'):
    results = {}

    def pt_fn():
        logits  = torch.randn(N, V, device=device, dtype=dtype, requires_grad=True)
        targets = torch.randint(0, V, (N,), device=device)
        targets[torch.rand(N, device=device) < 0.1] = -100
        loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-100)
        loss.backward()
    results['pytorch'] = benchmark_fn(pt_fn, warmup=5, rep=50)

    def ours_fn():
        logits  = torch.randn(N, V, device=device, dtype=dtype, requires_grad=True)
        targets = torch.randint(0, V, (N,), device=device)
        targets[torch.rand(N, device=device) < 0.1] = -100
        loss = hilda_cross_entropy(logits, targets, ignore_index=-100)
        loss.backward()
    results['ours'] = benchmark_fn(ours_fn, warmup=5, rep=50)

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
    print("CrossEntropy Benchmark: Ours vs Liger vs PyTorch")
    print("=" * 70)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    # bsz=4, seq_len=2048 → N=8192 (typical LLM training)
    configs = [
        # (N,     V,       dtype,          name)
        (8192,  32000,  torch.float16, "LLaMA-2/Mistral [V=32K,  fp16]"),
        (8192, 128256,  torch.float16, "LLaMA-3 8B      [V=128K, fp16]"),
        (8192,  32000,  torch.bfloat16,"LLaMA-2/Mistral [V=32K,  bf16]"),
        (8192, 128256,  torch.bfloat16,"LLaMA-3 8B      [V=128K, bf16]"),
        (2048,  32000,  torch.float16, "Short seq       [V=32K,  fp16, N=2K]"),
    ]

    # ── [A] 无 ignore：三方公平对比 ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[A] Forward Pass  (no ignore_index — fair 3-way comparison)")
    print("=" * 70)
    for N, V, dtype, name in configs:
        results = bench_forward_no_ignore(N, V, dtype)
        print_results(name, results)

    print("\n" + "=" * 70)
    print("[A] Forward + Backward  (no ignore_index — fair 3-way comparison)")
    print("=" * 70)
    for N, V, dtype, name in configs:
        results = bench_fwd_bwd_no_ignore(N, V, dtype)
        print_results(name, results)

    # ── [B] 含 ignore：Ours vs PyTorch ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("[B] Forward Pass  (with ~10% ignore tokens — Ours vs PyTorch)")
    print("=" * 70)
    for N, V, dtype, name in configs:
        results = bench_forward_with_ignore(N, V, dtype)
        print_results(name, results)

    print("\n" + "=" * 70)
    print("[B] Forward + Backward  (with ~10% ignore tokens — Ours vs PyTorch)")
    print("=" * 70)
    for N, V, dtype, name in configs:
        results = bench_fwd_bwd_with_ignore(N, V, dtype)
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
