"""
GRPO Loss Benchmark: Ours vs Liger vs PyTorch
"""

import sys, os, torch, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.grpo_loss import hilda_grpo_loss, fused_selective_log_softmax

try:
    from liger_kernel.ops.grpo_loss import GrpoLossFunction as LigerGRPOFunction
    HAS_LIGER = True
    print("Liger GRPO found.")
except ImportError:
    HAS_LIGER = False
    print("Liger GRPO not found.")


def benchmark_fn(fn, warmup=10, rep=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / rep * 1000


def pytorch_grpo_loss(logits, completion_ids, advantages, old_logp=None,
                      ref_logp=None, completion_mask=None, temperature=1.0,
                      beta=0.0, eps_low=0.2, eps_high=0.2):
    """Reference GRPO using PyTorch."""
    B, L_plus_1, V = logits.shape
    L = L_plus_1 - 1
    # Selective log-softmax
    logits_shifted = logits[:, :-1, :] / temperature
    log_probs = logits_shifted.log_softmax(dim=-1)
    ids = completion_ids[:, -L:].unsqueeze(-1)
    logp = log_probs.gather(-1, ids).squeeze(-1)  # (B, L)

    if old_logp is None:
        old_logp = logp.detach()
    ratio = torch.exp(logp - old_logp)
    ratio_clipped = torch.clamp(ratio, 1 - eps_low, 1 + eps_high)
    adv = advantages.unsqueeze(-1)  # (B, 1)
    loss1 = ratio * adv
    loss2 = ratio_clipped * adv
    per_token_loss = -torch.min(loss1, loss2)

    if beta != 0.0 and ref_logp is not None:
        kl = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1
        per_token_loss = per_token_loss + beta * kl

    mask = completion_mask.float() if completion_mask is not None else torch.ones_like(per_token_loss)
    seq_lens = mask.sum(-1).clamp(min=1.0)
    return ((per_token_loss * mask).sum(-1) / seq_lens).mean()


def quick_test():
    print("Running correctness test...")
    device = 'cuda'

    configs = [
        (4, 128, 32000,  torch.float16,  0.0, "B=4, L=128, V=32K, β=0"),
        (4, 128, 32000,  torch.float16,  0.1, "B=4, L=128, V=32K, β=0.1"),
        (4, 128, 32000,  torch.float32,  0.0, "B=4, L=128, V=32K, β=0, fp32"),
        (2, 256, 128256, torch.float16,  0.0, "B=2, L=256, V=128K, β=0"),
    ]

    for B, L, V, dtype, beta, name in configs:
        logits = torch.randn(B, L+1, V, device=device, dtype=dtype) * 0.1
        completion_ids = torch.randint(0, V, (B, L), device=device)
        advantages = torch.randn(B, device=device, dtype=torch.float32)
        old_logp = torch.randn(B, L, device=device, dtype=torch.float32) * 0.01
        ref_logp = torch.randn(B, L, device=device, dtype=torch.float32) * 0.01 if beta > 0 else None
        mask = torch.ones(B, L, device=device, dtype=torch.int32)
        mask[:, -16:] = 0  # mask last 16 tokens

        ref = pytorch_grpo_loss(logits.float(), completion_ids, advantages, old_logp,
                                ref_logp, mask, beta=beta)
        loss_ours, kl_ours, clip_ours = hilda_grpo_loss(
            logits, completion_ids, advantages, old_logp, ref_logp, mask, beta=beta)

        diff = abs(loss_ours.item() - ref.item())
        tol = 5e-1 if dtype != torch.float32 else 1e-3
        status = "PASS" if diff < tol else "FAIL"
        print(f"  {name}: ref={ref.item():.6f}, ours={loss_ours.item():.6f}, diff={diff:.2e} [{status}]")

    print("Correctness test done!\n")


def backward_test():
    print("Running backward correctness test...")
    device = 'cuda'

    for B, L, V, dtype, name in [(4, 128, 32000, torch.float16, "fp16"), (4, 128, 32000, torch.float32, "fp32")]:
        logits = torch.randn(B, L+1, V, device=device, dtype=dtype) * 0.1
        completion_ids = torch.randint(0, V, (B, L), device=device)
        advantages = torch.randn(B, device=device, dtype=torch.float32)
        old_logp = torch.randn(B, L, device=device, dtype=torch.float32) * 0.01

        l_ref = logits.clone().float().requires_grad_(True)
        loss_ref = pytorch_grpo_loss(l_ref, completion_ids, advantages, old_logp)
        loss_ref.backward()

        l_ours = logits.clone().requires_grad_(True)
        loss_ours, _, _ = hilda_grpo_loss(l_ours, completion_ids, advantages, old_logp)
        loss_ours.backward()

        grad_diff = (l_ours.grad.float() - l_ref.grad.float()).abs().max().item()
        tol = 1e-1 if dtype != torch.float32 else 1e-4
        status = "PASS" if grad_diff < tol else "FAIL"
        print(f"  {name}: grad_diff={grad_diff:.2e} [{status}]")

    print("Backward test done!\n")


def bench_forward(B, L, V, dtype, beta=0.0, device='cuda'):
    results = {}
    logits = torch.randn(B, L+1, V, device=device, dtype=dtype) * 0.1
    completion_ids = torch.randint(0, V, (B, L), device=device)
    advantages = torch.randn(B, device=device, dtype=torch.float32)
    old_logp = torch.randn(B, L, device=device, dtype=torch.float32) * 0.01
    ref_logp = torch.randn(B, L, device=device, dtype=torch.float32) * 0.01 if beta > 0 else None

    def pt_fn():
        return pytorch_grpo_loss(logits.float(), completion_ids, advantages, old_logp, ref_logp, beta=beta)
    results['pytorch'] = benchmark_fn(pt_fn)

    def ours_fn():
        return hilda_grpo_loss(logits, completion_ids, advantages, old_logp, ref_logp, beta=beta)
    results['ours'] = benchmark_fn(ours_fn)

    if HAS_LIGER:
        def liger_fn():
            return LigerGRPOFunction.apply(
                logits, old_logp, ref_logp, completion_ids, advantages, None,
                1.0, beta, 0.2, 0.2, False, "grpo",
            )
        try:
            results['liger'] = benchmark_fn(liger_fn)
        except Exception as e:
            print(f"  Liger error: {e}")
            results['liger'] = float('nan')

    return results


def bench_fwd_bwd(B, L, V, dtype, beta=0.0, device='cuda'):
    results = {}
    completion_ids = torch.randint(0, V, (B, L), device=device)
    advantages = torch.randn(B, device=device, dtype=torch.float32)

    def pt_fn():
        l = torch.randn(B, L+1, V, device=device, dtype=dtype, requires_grad=True) * 0.1
        old = torch.randn(B, L, device=device, dtype=torch.float32) * 0.01
        loss = pytorch_grpo_loss(l.float(), completion_ids, advantages, old, beta=beta)
        loss.backward()
    results['pytorch'] = benchmark_fn(pt_fn, warmup=5, rep=50)

    def ours_fn():
        l = torch.randn(B, L+1, V, device=device, dtype=dtype, requires_grad=True) * 0.1
        old = torch.randn(B, L, device=device, dtype=torch.float32) * 0.01
        loss, _, _ = hilda_grpo_loss(l, completion_ids, advantages, old, beta=beta)
        loss.backward()
    results['ours'] = benchmark_fn(ours_fn, warmup=5, rep=50)

    if HAS_LIGER:
        def liger_fn():
            l = torch.randn(B, L+1, V, device=device, dtype=dtype, requires_grad=True) * 0.1
            old = torch.randn(B, L, device=device, dtype=torch.float32) * 0.01
            ref = None
            loss, _, _ = LigerGRPOFunction.apply(
                l, old, ref, completion_ids, advantages, None,
                1.0, beta, 0.2, 0.2, False, "grpo",
            )
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
    print("GRPO Loss Benchmark: Ours vs Liger vs PyTorch")
    print("=" * 70)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    configs = [
        (8, 512,  32000,  torch.float16,  0.0, "B=8, L=512, V=32K, β=0"),
        (8, 512,  32000,  torch.float16,  0.1, "B=8, L=512, V=32K, β=0.1"),
        (4, 1024, 32000,  torch.float16,  0.0, "B=4, L=1K, V=32K, β=0"),
        (4, 512, 128256,  torch.float16,  0.0, "B=4, L=512, V=128K, β=0"),
    ]

    print("\n" + "=" * 70)
    print("Forward Pass")
    print("=" * 70)
    for B, L, V, dtype, beta, name in configs:
        results = bench_forward(B, L, V, dtype, beta)
        print_results(name, results)

    print("\n" + "=" * 70)
    print("Forward + Backward")
    print("=" * 70)
    for B, L, V, dtype, beta, name in configs:
        results = bench_fwd_bwd(B, L, V, dtype, beta)
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
