"""
JSD Benchmark: Ours vs Liger vs PyTorch
"""

import sys, os, torch, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.jsd import hilda_jsd

try:
    from liger_kernel.ops.jsd import LigerJSD
    HAS_LIGER = True
    print("Liger JSD found.")
except ImportError:
    HAS_LIGER = False
    print("Liger JSD not found.")


def benchmark_fn(fn, warmup=10, rep=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / rep * 1000


def pytorch_jsd(X, Y, beta=0.5):
    """Reference JSD using PyTorch. β=0 → fwd KL(P||Q), β=1 → rev KL(Q||P)."""
    Q = X.exp()
    P = Y.exp()
    if beta == 0.0:
        # Forward KL: sum(P * (log P - log Q))
        loss = (P * (Y - X)).sum(-1)
    elif beta == 1.0:
        # Reverse KL: sum(Q * (log Q - log P))
        loss = (Q * (X - Y)).sum(-1)
    else:
        M = beta * P + (1 - beta) * Q
        log_M = M.log()
        loss = beta * (P * (Y - log_M)).sum(-1) + (1 - beta) * (Q * (X - log_M)).sum(-1)
    return loss.mean()


def quick_test():
    print("Running correctness test...")
    device = 'cuda'

    configs = [
        (512, 32000,  torch.float16,  0.5, "V=32K,  fp16, β=0.5"),
        (512, 128256, torch.float16,  0.5, "V=128K, fp16, β=0.5"),
        (512, 32000,  torch.float32,  0.5, "V=32K,  fp32, β=0.5"),
        (512, 32000,  torch.float16,  0.0, "V=32K,  fp16, β=0 (fwd KL)"),
        (512, 32000,  torch.float16,  1.0, "V=32K,  fp16, β=1 (rev KL)"),
    ]

    for N, V, dtype, beta, name in configs:
        X = torch.randn(N, V, device=device, dtype=dtype).log_softmax(dim=-1)
        Y = torch.randn(N, V, device=device, dtype=dtype).log_softmax(dim=-1)

        ref  = pytorch_jsd(X.float(), Y.float(), beta)
        ours = hilda_jsd(X, Y, beta=beta)

        diff = abs(ours.item() - ref.item())
        tol = 1e-1 if dtype != torch.float32 else 1e-4
        status = "PASS" if diff < tol else "FAIL"
        print(f"  {name}: ref={ref.item():.6f}, ours={ours.item():.6f}, diff={diff:.2e} [{status}]")

    print("Correctness test done!\n")


def backward_test():
    print("Running backward correctness test...")
    device = 'cuda'

    for N, V, dtype, name in [(512, 32000, torch.float16, "fp16"), (512, 32000, torch.float32, "fp32")]:
        X = torch.randn(N, V, device=device, dtype=dtype).log_softmax(dim=-1)
        Y = torch.randn(N, V, device=device, dtype=dtype).log_softmax(dim=-1)

        x_ref = X.clone().float().requires_grad_(True)
        loss_ref = pytorch_jsd(x_ref, Y.float(), 0.5)
        loss_ref.backward()

        x_ours = X.clone().requires_grad_(True)
        loss_ours = hilda_jsd(x_ours, Y, beta=0.5)
        loss_ours.backward()

        grad_diff = (x_ours.grad.float() - x_ref.grad.float()).abs().max().item()
        tol = 5e-2 if dtype != torch.float32 else 1e-4
        status = "PASS" if grad_diff < tol else "FAIL"
        print(f"  {name}: grad_diff={grad_diff:.2e} [{status}]")

    print("Backward test done!\n")


def bench_forward(N, V, dtype, device='cuda'):
    results = {}
    X = torch.randn(N, V, device=device, dtype=dtype).log_softmax(dim=-1)
    Y = torch.randn(N, V, device=device, dtype=dtype).log_softmax(dim=-1)

    def pt_fn():
        return pytorch_jsd(X.float(), Y.float(), 0.5)
    results['pytorch'] = benchmark_fn(pt_fn)

    def ours_fn():
        return hilda_jsd(X, Y, beta=0.5)
    results['ours'] = benchmark_fn(ours_fn)

    if HAS_LIGER:
        liger_jsd = LigerJSD(beta=0.5)
        def liger_fn():
            return liger_jsd(X, Y)
        try:
            results['liger'] = benchmark_fn(liger_fn)
        except Exception as e:
            print(f"  Liger error: {e}")
            results['liger'] = float('nan')

    return results


def bench_fwd_bwd(N, V, dtype, device='cuda'):
    results = {}

    def pt_fn():
        x = torch.randn(N, V, device=device, dtype=dtype).log_softmax(dim=-1).requires_grad_(True)
        y = torch.randn(N, V, device=device, dtype=dtype).log_softmax(dim=-1)
        loss = pytorch_jsd(x.float(), y.float(), 0.5)
        loss.backward()
    results['pytorch'] = benchmark_fn(pt_fn, warmup=5, rep=50)

    def ours_fn():
        x = torch.randn(N, V, device=device, dtype=dtype).log_softmax(dim=-1).requires_grad_(True)
        y = torch.randn(N, V, device=device, dtype=dtype).log_softmax(dim=-1)
        loss = hilda_jsd(x, y, beta=0.5)
        loss.backward()
    results['ours'] = benchmark_fn(ours_fn, warmup=5, rep=50)

    if HAS_LIGER:
        liger_jsd = LigerJSD(beta=0.5)
        def liger_fn():
            x = torch.randn(N, V, device=device, dtype=dtype).log_softmax(dim=-1).requires_grad_(True)
            y = torch.randn(N, V, device=device, dtype=dtype).log_softmax(dim=-1)
            loss = liger_jsd(x, y)
            loss.backward()
        try:
            results['liger'] = benchmark_fn(liger_fn, warmup=5, rep=50)
        except Exception:
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
    print("JSD Benchmark: Ours vs Liger vs PyTorch")
    print("=" * 70)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    configs = [
        (8192,  32000, torch.float16,  "V=32K  [fp16]"),
        (4096, 128256, torch.float16,  "V=128K [fp16, N=4K]"),
        (8192,  32000, torch.bfloat16, "V=32K  [bf16]"),
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
