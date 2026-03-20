"""
RoPE Benchmark: AutoTune 版本 vs Liger vs PyTorch

测试配置覆盖：
- LLaMA-2 7B:  n_qh=32, n_kh=32, hd=128 (MHA)
- LLaMA-3 8B:  n_qh=32, n_kh=8,  hd=128 (GQA)
- LLaMA-3 70B: n_qh=64, n_kh=8,  hd=128 (GQA)
- Mistral 7B:  n_qh=32, n_kh=8,  hd=128 (GQA)
"""

import sys
import os
import torch
import triton
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.rope import hilda_rope, rope_forward, rope_backward

# 尝试导入 Liger
try:
    import torch.distributed
    if not hasattr(torch.distributed, 'tensor'):
        import types
        torch.distributed.tensor = types.ModuleType('torch.distributed.tensor')
        torch.distributed.tensor.DTensor = type('DTensor', (), {})
    from liger_kernel.ops.rope import LigerRopeFunction
    HAS_LIGER = True
    print("Liger Kernel found.")
except ImportError:
    HAS_LIGER = False
    print("Liger Kernel not found.")


# ============================================================================
# PyTorch 参考实现
# ============================================================================

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def pytorch_rope(q, k, cos, sin):
    """标准 HuggingFace RoPE 实现"""
    # cos/sin: (1, seq, hd//2) → expand to (1, seq, hd)
    cos_full = torch.cat([cos, cos], dim=-1)  # (1, seq, hd)
    sin_full = torch.cat([sin, sin], dim=-1)

    # q, k: (bsz, n_h, seq, hd)
    cos_4d = cos_full.unsqueeze(1)  # (1, 1, seq, hd)
    sin_4d = sin_full.unsqueeze(1)

    q_embed = (q * cos_4d) + (rotate_half(q) * sin_4d)
    k_embed = (k * cos_4d) + (rotate_half(k) * sin_4d)
    return q_embed, k_embed


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


def make_inputs(bsz, seq_len, n_qh, n_kh, hd, dtype, device='cuda'):
    q   = torch.randn(bsz, n_qh, seq_len, hd, device=device, dtype=dtype)
    k   = torch.randn(bsz, n_kh, seq_len, hd, device=device, dtype=dtype)
    cos = torch.randn(1, seq_len, hd // 2, device=device, dtype=dtype)
    sin = torch.randn(1, seq_len, hd // 2, device=device, dtype=dtype)
    return q, k, cos, sin


# ============================================================================
# 正确性测试
# ============================================================================

def quick_test():
    print("Running quick correctness test...")
    device = 'cuda'
    dtype  = torch.float32  # 用 fp32 方便精度对比

    configs = [
        (2, 512,  32, 8,  128, "LLaMA-3 8B style (GQA)"),
        (2, 512,  32, 32, 128, "LLaMA-2 7B style (MHA)"),
        (2, 512,  64, 8,  128, "LLaMA-3 70B style (GQA)"),
    ]

    for bsz, seq_len, n_qh, n_kh, hd, name in configs:
        q, k, cos, sin = make_inputs(bsz, seq_len, n_qh, n_kh, hd, dtype, device)

        # 参考：PyTorch
        q_ref = q.clone()
        k_ref = k.clone()
        q_out_ref, k_out_ref = pytorch_rope(q_ref, k_ref, cos, sin)

        # 我们的实现（in-place，需要 clone）
        q_ours = q.clone()
        k_ours = k.clone()
        q_out_ours, k_out_ours = hilda_rope(q_ours, k_ours, cos, sin)

        q_diff = (q_out_ours - q_out_ref).abs().max().item()
        k_diff = (k_out_ours - k_out_ref).abs().max().item()
        status = "PASS" if q_diff < 1e-4 and k_diff < 1e-4 else "FAIL"
        print(f"  {name}: q_diff={q_diff:.2e}, k_diff={k_diff:.2e} [{status}]")

    print("Quick test done!\n")


# ============================================================================
# Backward 正确性测试
# ============================================================================

def backward_test():
    print("Running backward correctness test...")
    device = 'cuda'
    dtype  = torch.float32

    bsz, seq_len, n_qh, n_kh, hd = 2, 256, 32, 8, 128
    q, k, cos, sin = make_inputs(bsz, seq_len, n_qh, n_kh, hd, dtype, device)

    # PyTorch backward
    q_ref = q.clone().requires_grad_(True)
    k_ref = k.clone().requires_grad_(True)
    q_out, k_out = pytorch_rope(q_ref, k_ref, cos, sin)
    (q_out.sum() + k_out.sum()).backward()

    # Ours backward
    q_ours = q.clone().requires_grad_(True)
    k_ours = k.clone().requires_grad_(True)
    q_o, k_o = hilda_rope(q_ours, k_ours, cos, sin)
    (q_o.sum() + k_o.sum()).backward()

    dq_diff = (q_ours.grad - q_ref.grad).abs().max().item()
    dk_diff = (k_ours.grad - k_ref.grad).abs().max().item()
    status = "PASS" if dq_diff < 1e-4 and dk_diff < 1e-4 else "FAIL"
    print(f"  backward: dq_diff={dq_diff:.2e}, dk_diff={dk_diff:.2e} [{status}]")
    print()


# ============================================================================
# 性能测试
# ============================================================================

def bench_forward(bsz, seq_len, n_qh, n_kh, hd, dtype, device='cuda'):
    results = {}

    # PyTorch
    q, k, cos, sin = make_inputs(bsz, seq_len, n_qh, n_kh, hd, dtype, device)
    def pt_fn():
        return pytorch_rope(q.clone(), k.clone(), cos, sin)
    results['pytorch'] = benchmark_fn(pt_fn)

    # Ours
    q, k, cos, sin = make_inputs(bsz, seq_len, n_qh, n_kh, hd, dtype, device)
    def ours_fn():
        return hilda_rope(q.clone(), k.clone(), cos, sin)
    results['ours_autotune'] = benchmark_fn(ours_fn)

    # Liger
    if HAS_LIGER:
        q, k, cos, sin = make_inputs(bsz, seq_len, n_qh, n_kh, hd, dtype, device)
        def liger_fn():
            return LigerRopeFunction.apply(q.clone(), k.clone(), cos, sin)
        try:
            results['liger'] = benchmark_fn(liger_fn)
        except Exception as e:
            results['liger'] = float('nan')
            print(f"  Liger forward error: {e}")

    return results


def bench_fwd_bwd(bsz, seq_len, n_qh, n_kh, hd, dtype, device='cuda'):
    results = {}

    # PyTorch
    def pt_fn():
        q = torch.randn(bsz, n_qh, seq_len, hd, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(bsz, n_kh, seq_len, hd, device=device, dtype=dtype, requires_grad=True)
        cos = torch.randn(1, seq_len, hd // 2, device=device, dtype=dtype)
        sin = torch.randn(1, seq_len, hd // 2, device=device, dtype=dtype)
        q_o, k_o = pytorch_rope(q, k, cos, sin)
        (q_o.sum() + k_o.sum()).backward()
    results['pytorch'] = benchmark_fn(pt_fn, warmup=5, rep=50)

    # Ours
    def ours_fn():
        q = torch.randn(bsz, n_qh, seq_len, hd, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(bsz, n_kh, seq_len, hd, device=device, dtype=dtype, requires_grad=True)
        cos = torch.randn(1, seq_len, hd // 2, device=device, dtype=dtype)
        sin = torch.randn(1, seq_len, hd // 2, device=device, dtype=dtype)
        q_o, k_o = hilda_rope(q, k, cos, sin)
        (q_o.sum() + k_o.sum()).backward()
    results['ours_autotune'] = benchmark_fn(ours_fn, warmup=5, rep=50)

    # Liger
    if HAS_LIGER:
        def liger_fn():
            q = torch.randn(bsz, n_qh, seq_len, hd, device=device, dtype=dtype, requires_grad=True)
            k = torch.randn(bsz, n_kh, seq_len, hd, device=device, dtype=dtype, requires_grad=True)
            cos = torch.randn(1, seq_len, hd // 2, device=device, dtype=dtype)
            sin = torch.randn(1, seq_len, hd // 2, device=device, dtype=dtype)
            q_o, k_o = LigerRopeFunction.apply(q, k, cos, sin)
            (q_o.sum() + k_o.sum()).backward()
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
        speedup = base / ms if ms == ms else float('nan')  # nan check
        tag = " (baseline)" if name == baseline else ""
        print(f"  {name:20s}: {ms:8.3f} ms  ({speedup:5.2f}x){tag}")


def main():
    print("=" * 70)
    print("RoPE Benchmark: AutoTune vs Liger vs PyTorch")
    print("=" * 70)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    # 典型 LLM 配置
    configs = [
        # (bsz, seq_len, n_qh, n_kh, hd, name)
        (4, 2048, 32, 8,  128, torch.float16, "LLaMA-3 8B  [GQA, fp16]"),
        (4, 2048, 32, 32, 128, torch.float16, "LLaMA-2 7B  [MHA, fp16]"),
        (4, 2048, 64, 8,  128, torch.float16, "LLaMA-3 70B [GQA, fp16]"),
        (8, 2048, 32, 8,  128, torch.float16, "LLaMA-3 8B  [GQA, fp16, bs=8]"),
        (4, 2048, 32, 8,  128, torch.bfloat16,"LLaMA-3 8B  [GQA, bf16]"),
        (4, 512,  32, 8,  128, torch.float16, "LLaMA-3 8B  [GQA, fp16, short]"),
    ]

    print("\n" + "=" * 70)
    print("Forward Pass")
    print("=" * 70)
    for bsz, sl, n_qh, n_kh, hd, dtype, name in configs:
        results = bench_forward(bsz, sl, n_qh, n_kh, hd, dtype)
        print_results(name, results)

    print("\n" + "=" * 70)
    print("Forward + Backward")
    print("=" * 70)
    for bsz, sl, n_qh, n_kh, hd, dtype, name in configs:
        results = bench_fwd_bwd(bsz, sl, n_qh, n_kh, hd, dtype)
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
