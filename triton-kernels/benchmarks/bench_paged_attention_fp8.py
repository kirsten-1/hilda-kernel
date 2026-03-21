"""
Benchmark fused FP8 paged decode attention against FlashAttention baselines.
"""

import os
import sys
import time

import torch
from flash_attn import flash_attn_with_kvcache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.paged_attention_fp8 import hilda_paged_attention_fp8_decode, reference_paged_attention_decode


def benchmark_fn(fn, warmup=20, rep=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / rep * 1000


def make_inputs(batch, num_heads, num_kv_heads, context_len, block_size, dtype=torch.bfloat16, device="cuda"):
    head_dim = 128
    num_blocks_per_seq = (context_len + block_size - 1) // block_size
    total_blocks = batch * num_blocks_per_seq
    q = torch.randn(batch, num_heads, head_dim, device=device, dtype=dtype)
    k_cache_bf16 = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_cache_bf16 = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    k_cache_fp8 = k_cache_bf16.to(torch.float8_e4m3fn)
    v_cache_fp8 = v_cache_bf16.to(torch.float8_e4m3fn)
    block_tables = torch.arange(total_blocks, device=device, dtype=torch.int32).view(batch, num_blocks_per_seq).contiguous()
    context_lens = torch.full((batch,), context_len, device=device, dtype=torch.int32)
    return q, k_cache_bf16, v_cache_bf16, k_cache_fp8, v_cache_fp8, block_tables, context_lens


def quick_test():
    print("Running correctness test...")
    configs = [
        (1, 16, 8, 512, 256, "B=1, ctx=512"),
        (8, 16, 8, 1024, 256, "B=8, ctx=1024"),
        (16, 16, 8, 4096, 256, "B=16, ctx=4096"),
    ]
    for batch, num_heads, num_kv_heads, context_len, block_size, name in configs:
        q, _, _, k_cache_fp8, v_cache_fp8, block_tables, context_lens = make_inputs(
            batch, num_heads, num_kv_heads, context_len, block_size
        )
        scale = q.shape[-1] ** -0.5
        ref = reference_paged_attention_decode(q, k_cache_fp8, v_cache_fp8, block_tables, context_lens, scale)
        flash_ref = flash_attn_with_kvcache(
            q.unsqueeze(1),
            k_cache_fp8.to(torch.bfloat16),
            v_cache_fp8.to(torch.bfloat16),
            cache_seqlens=context_lens,
            block_table=block_tables,
            softmax_scale=scale,
            causal=True,
        )
        ours = hilda_paged_attention_fp8_decode(q, k_cache_fp8, v_cache_fp8, block_tables, context_lens, scale)
        diff_ref = (ours.float() - ref.float()).abs().max().item()
        diff_flash = (ours.float() - flash_ref.squeeze(1).float()).abs().max().item()
        status = "PASS" if diff_flash < 1e-1 else "FAIL"
        print(f"  {name}: diff_vs_ref={diff_ref:.3e}, diff_vs_flash={diff_flash:.3e} [{status}]")
    print("Correctness test done!\n")


def bench_config(batch, num_heads, num_kv_heads, context_len, block_size):
    q, k_cache_bf16, v_cache_bf16, k_cache_fp8, v_cache_fp8, block_tables, context_lens = make_inputs(
        batch, num_heads, num_kv_heads, context_len, block_size
    )
    scale = q.shape[-1] ** -0.5

    results = {}

    def flash_bf16():
        return flash_attn_with_kvcache(
            q.unsqueeze(1),
            k_cache_bf16,
            v_cache_bf16,
            cache_seqlens=context_lens,
            block_table=block_tables,
            softmax_scale=scale,
            causal=True,
        )

    def flash_fp8_dequant():
        return flash_attn_with_kvcache(
            q.unsqueeze(1),
            k_cache_fp8.to(torch.bfloat16),
            v_cache_fp8.to(torch.bfloat16),
            cache_seqlens=context_lens,
            block_table=block_tables,
            softmax_scale=scale,
            causal=True,
        )

    def fused_fp8():
        return hilda_paged_attention_fp8_decode(q, k_cache_fp8, v_cache_fp8, block_tables, context_lens, scale)

    results["flash_bf16_cache"] = benchmark_fn(flash_bf16)
    results["flash_fp8_dequant"] = benchmark_fn(flash_fp8_dequant)
    results["hilda_fp8_fused"] = benchmark_fn(fused_fp8)
    return results


def print_results(title, batch, results, baseline="flash_bf16_cache"):
    print(f"\n{title}")
    print("=" * 72)
    base = results[baseline]
    for name, ms in results.items():
        speedup = base / ms
        toks = batch / (ms / 1000.0)
        tag = " (baseline)" if name == baseline else ""
        print(f"  {name:18s}: {ms:8.3f} ms  ({speedup:5.2f}x)  {toks:9.1f} tok/s{tag}")


def main():
    print("=" * 72)
    print("FP8 Paged Decode Attention Benchmark")
    print("=" * 72)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    configs = [
        (8, 16, 8, 1024, 256),
        (16, 16, 8, 4096, 256),
        (32, 16, 8, 4096, 256),
    ]
    for batch, num_heads, num_kv_heads, context_len, block_size in configs:
        results = bench_config(batch, num_heads, num_kv_heads, context_len, block_size)
        title = f"batch={batch}, ctx={context_len}, heads={num_heads}/{num_kv_heads}, block={block_size}"
        print_results(title, batch, results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    quick_test()
    if not args.quick:
        main()
