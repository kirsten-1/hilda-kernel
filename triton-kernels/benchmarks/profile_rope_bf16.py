"""
RoPE BF16 backward 性能诊断：扫描不同 num_warps 组合
（新版 per-head kernel，已无 transpose+contiguous 拷贝）
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import triton
from kernels.rope import _rope_kernel_per_head, _rope_num_warps

device = 'cuda'

def benchmark_fn(fn, warmup=10, rep=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / rep * 1e6  # μs

def run_rope_kernel(q, k, cos, sin, n_qh, n_kh, hd, num_warps, backward=False):
    bsz, _, seq_len, _ = q.shape
    pad_hd  = triton.next_power_of_2(hd)
    cos_bs  = cos.shape[0]

    _rope_kernel_per_head[(bsz * n_qh * seq_len,)](
        q, q.stride(0), q.stride(1), q.stride(2),
        cos, cos.stride(0), cos.stride(1), sin,
        n_qh, seq_len, cos_bs, hd, pad_hd,
        IS_BACKWARD=backward, num_warps=num_warps,
    )
    _rope_kernel_per_head[(bsz * n_kh * seq_len,)](
        k, k.stride(0), k.stride(1), k.stride(2),
        cos, cos.stride(0), cos.stride(1), sin,
        n_kh, seq_len, cos_bs, hd, pad_hd,
        IS_BACKWARD=backward, num_warps=num_warps,
    )

# ---- Config ----
bsz, seq_len, n_qh, n_kh, hd = 4, 2048, 32, 8, 128
dtype = torch.bfloat16

q   = torch.randn(bsz, n_qh, seq_len, hd, device=device, dtype=dtype)
k   = torch.randn(bsz, n_kh, seq_len, hd, device=device, dtype=dtype)
cos = torch.randn(1, seq_len, hd // 2, device=device, dtype=dtype)
sin = torch.randn(1, seq_len, hd // 2, device=device, dtype=dtype)

default_nw = _rope_num_warps(hd)
print(f"LLaMA-3 8B GQA, BF16, bsz={bsz}, seq={seq_len} (per-head kernel, no transpose)")
print(f"{'num_warps':>10} {'fwd(μs)':>10} {'bwd(μs)':>10}")
print("-" * 35)

for num_warps in [1, 2, 4, 8, 16, 32]:
    fwd_fn = lambda nw=num_warps: run_rope_kernel(q.clone(), k.clone(), cos, sin, n_qh, n_kh, hd, nw, backward=False)
    bwd_fn = lambda nw=num_warps: run_rope_kernel(q.clone(), k.clone(), cos, sin, n_qh, n_kh, hd, nw, backward=True)
    fwd_us = benchmark_fn(fwd_fn)
    bwd_us = benchmark_fn(bwd_fn)
    marker = " ◄" if num_warps == default_nw else ""
    print(f"{num_warps:>10} {fwd_us:>10.1f} {bwd_us:>10.1f}{marker}")

print(f"\nNote: ◄ = current heuristic (num_warps={default_nw})")
print("No num_stages sweep: per-head kernel is memory-bandwidth bound, num_stages has no effect")
