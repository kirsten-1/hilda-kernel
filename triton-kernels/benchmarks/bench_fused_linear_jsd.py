"""
FusedLinearJSD Benchmark: Ours vs Liger vs PyTorch
"""

import sys, os, torch, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.fused_linear_jsd import hilda_fused_linear_jsd

try:
    from liger_kernel.ops.fused_linear_jsd import LigerFusedLinearJSDFunction
    HAS_LIGER = True
    print("Liger FusedLinearJSD found.")
except ImportError:
    HAS_LIGER = False
    print("Liger FusedLinearJSD not found.")


def benchmark_fn(fn, warmup=10, rep=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / rep * 1000


def pytorch_linear_jsd(s_input, s_weight, t_input, t_weight, beta=0.5, temperature=1.0):
    """Reference: linear + log_softmax + JSD."""
    s_logits = (s_input @ s_weight.t()) / temperature
    t_logits = (t_input @ t_weight.t()) / temperature
    X = s_logits.log_softmax(dim=-1)
    Y = t_logits.log_softmax(dim=-1)
    Q = X.exp()
    P = Y.exp()
    M = beta * P + (1 - beta) * Q
    log_M = M.log()
    loss = beta * (P * (Y - log_M)).sum(-1) + (1 - beta) * (Q * (X - log_M)).sum(-1)
    return loss.mean()


def quick_test():
    print("Running correctness test...")
    device = 'cuda'

    configs = [
        (512, 4096, 32000,  torch.float16,  "H=4096, V=32K,  fp16"),
        (256, 4096, 128256, torch.float16,  "H=4096, V=128K, fp16"),
        (256, 4096, 32000,  torch.float32,  "H=4096, V=32K,  fp32"),
    ]

    for N, H, V, dtype, name in configs:
        s_input  = torch.randn(N, H, device=device, dtype=dtype) * 0.02
        s_weight = torch.randn(V, H, device=device, dtype=dtype) * 0.02
        t_input  = torch.randn(N, H, device=device, dtype=dtype) * 0.02
        t_weight = torch.randn(V, H, device=device, dtype=dtype) * 0.02

        ref  = pytorch_linear_jsd(s_input.float(), s_weight.float(), t_input.float(), t_weight.float())
        ours = hilda_fused_linear_jsd(s_input, s_weight, t_input, t_weight)

        diff = abs(ours.item() - ref.item())
        tol = 5e-1 if dtype != torch.float32 else 1e-3
        status = "PASS" if diff < tol else "FAIL"
        print(f"  {name}: ref={ref.item():.6f}, ours={ours.item():.6f}, diff={diff:.2e} [{status}]")

    print("Correctness test done!\n")


def backward_test():
    print("Running backward correctness test...")
    device = 'cuda'

    for N, H, V, dtype, name in [(256, 4096, 32000, torch.float16, "fp16"), (256, 4096, 32000, torch.float32, "fp32")]:
        s_input  = torch.randn(N, H, device=device, dtype=dtype) * 0.02
        s_weight = torch.randn(V, H, device=device, dtype=dtype) * 0.02
        t_input  = torch.randn(N, H, device=device, dtype=dtype) * 0.02
        t_weight = torch.randn(V, H, device=device, dtype=dtype) * 0.02

        si_ref = s_input.clone().float().requires_grad_(True)
        sw_ref = s_weight.clone().float().requires_grad_(True)
        loss_ref = pytorch_linear_jsd(si_ref, sw_ref, t_input.float(), t_weight.float())
        loss_ref.backward()

        si_ours = s_input.clone().requires_grad_(True)
        sw_ours = s_weight.clone().requires_grad_(True)
        loss_ours = hilda_fused_linear_jsd(si_ours, sw_ours, t_input, t_weight)
        loss_ours.backward()

        dh_diff = (si_ours.grad.float() - si_ref.grad.float()).abs().max().item()
        dw_diff = (sw_ours.grad.float() - sw_ref.grad.float()).abs().max().item()
        tol_h = 5e-2 if dtype != torch.float32 else 1e-4
        tol_w = 5e-1 if dtype != torch.float32 else 1e-2
        status = "PASS" if dh_diff < tol_h and dw_diff < tol_w else "FAIL"
        print(f"  {name}: dH={dh_diff:.2e}, dW={dw_diff:.2e} [{status}]")

    print("Backward test done!\n")


def bench_fwd_bwd(N, H, V, dtype, device='cuda'):
    results = {}

    def pt_fn():
        si = torch.randn(N, H, device=device, dtype=dtype, requires_grad=True) * 0.02
        sw = torch.randn(V, H, device=device, dtype=dtype, requires_grad=True) * 0.02
        ti = torch.randn(N, H, device=device, dtype=dtype) * 0.02
        tw = torch.randn(V, H, device=device, dtype=dtype) * 0.02
        loss = pytorch_linear_jsd(si.float(), sw.float(), ti.float(), tw.float())
        loss.backward()
    results['pytorch'] = benchmark_fn(pt_fn, warmup=5, rep=50)

    def ours_fn():
        si = torch.randn(N, H, device=device, dtype=dtype, requires_grad=True) * 0.02
        sw = torch.randn(V, H, device=device, dtype=dtype, requires_grad=True) * 0.02
        ti = torch.randn(N, H, device=device, dtype=dtype) * 0.02
        tw = torch.randn(V, H, device=device, dtype=dtype) * 0.02
        loss = hilda_fused_linear_jsd(si, sw, ti, tw)
        loss.backward()
    results['ours'] = benchmark_fn(ours_fn, warmup=5, rep=50)

    if HAS_LIGER:
        def liger_fn():
            si = torch.randn(N, H, device=device, dtype=dtype, requires_grad=True) * 0.02
            sw = torch.randn(V, H, device=device, dtype=dtype, requires_grad=True) * 0.02
            ti = torch.randn(N, H, device=device, dtype=dtype) * 0.02
            tw = torch.randn(V, H, device=device, dtype=dtype) * 0.02
            loss = LigerFusedLinearJSDFunction.apply(si, sw, ti, tw, None, 0.5, 1.0, -100)
            loss.backward()
        try:
            results['liger'] = benchmark_fn(liger_fn, warmup=5, rep=50)
        except Exception as e:
            results['liger'] = float('nan')

    return results


def measure_peak_memory(fn, device='cuda'):
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    fn()
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 1024 / 1024


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
    print("FusedLinearJSD Benchmark: Ours vs Liger vs PyTorch")
    print("=" * 70)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    configs = [
        (4096, 4096,  32000,  torch.float16,  "LLaMA-7B  [V=32K,  fp16, N=4K]"),
        (2048, 4096, 128256,  torch.float16,  "LLaMA-3   [V=128K, fp16, N=2K]"),
        (4096, 4096,  32000,  torch.bfloat16, "LLaMA-7B  [V=32K,  bf16, N=4K]"),
    ]

    print("\n" + "=" * 70)
    print("Forward + Backward")
    print("=" * 70)
    for N, H, V, dtype, name in configs:
        results = bench_fwd_bwd(N, H, V, dtype)
        print_results(name, results)

    # Memory
    print("\n" + "=" * 70)
    print("Peak Memory (Fwd+Bwd)")
    print("=" * 70)
    for N, H, V, dtype, name in configs[:2]:
        print(f"\n{name}")
        print("=" * 60)

        def pt_fn():
            si = torch.randn(N, H, device='cuda', dtype=dtype, requires_grad=True)
            sw = torch.randn(V, H, device='cuda', dtype=dtype, requires_grad=True)
            ti = torch.randn(N, H, device='cuda', dtype=dtype)
            tw = torch.randn(V, H, device='cuda', dtype=dtype)
            loss = pytorch_linear_jsd(si.float(), sw.float(), ti.float(), tw.float())
            loss.backward()
        pt_fn()
        pt_mem = measure_peak_memory(pt_fn)

        def ours_fn():
            si = torch.randn(N, H, device='cuda', dtype=dtype, requires_grad=True)
            sw = torch.randn(V, H, device='cuda', dtype=dtype, requires_grad=True)
            ti = torch.randn(N, H, device='cuda', dtype=dtype)
            tw = torch.randn(V, H, device='cuda', dtype=dtype)
            loss = hilda_fused_linear_jsd(si, sw, ti, tw)
            loss.backward()
        ours_fn()
        ours_mem = measure_peak_memory(ours_fn)

        saved = pt_mem - ours_mem
        pct = saved / pt_mem * 100
        print(f"  pytorch: {pt_mem:.0f} MB")
        print(f"  ours:    {ours_mem:.0f} MB  (saved {saved:+.0f} MB, {pct:+.1f}%)")


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
