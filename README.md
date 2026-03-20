# Hilda Kernel

High-performance Triton kernel library for LLM training. Drop-in replacements for common operators with full forward + backward support.

Benchmarked on **NVIDIA RTX 5090** (CUDA 12.8). All kernels include correctness tests against PyTorch reference implementations.

## Operators

### Normalization & Residual

| Operator | Description | vs PyTorch (Fwd+Bwd) | vs Liger (Fwd+Bwd) |
|---|---|---|---|
| **RMSNorm** | Root mean square normalization | 1.18x - 8.21x | 1.10x - 1.50x |
| **FusedAddRMSNorm** | Residual add + RMSNorm fused | 4.15x - 5.66x | 1.02x - 1.32x |

### Positional Encoding

| Operator | Description | vs PyTorch (Fwd) | vs Liger (Fwd) |
|---|---|---|---|
| **RoPE** | Rotary position embedding | 1.27x - 1.34x faster | 1.27x - 1.34x faster |

### Activation Functions

| Operator | Description | vs PyTorch (Fwd+Bwd) | vs Liger (Fwd+Bwd) |
|---|---|---|---|
| **SwiGLU** | SiLU-gated linear unit | 1.26x - 1.68x | ~1.00x (matched) |
| **GeGLU** | GELU-gated linear unit | 1.25x - 1.68x | 1.00x - 1.76x |

### Loss Functions

| Operator | Description | vs PyTorch (Fwd+Bwd) | vs Liger (Fwd+Bwd) |
|---|---|---|---|
| **CrossEntropy** | Cross-entropy with ignore_index | 1.66x - 2.55x | 1.00x - 1.66x |
| **KLDiv** | KL divergence | 1.71x - 1.84x | 1.18x - 1.20x |
| **JSD** | Jensen-Shannon divergence | 9.18x - 11.02x | N/A |
| **GRPO Loss** | PPO-clip loss for RLHF | 6.10x - 6.61x | N/A |

### Fused Operators

| Operator | Description | vs PyTorch (Fwd+Bwd) | Memory Saved |
|---|---|---|---|
| **FusedLinearCrossEntropy** | lm_head + CE fused | 0.91x | 47.7% - 54.7% |
| **FusedLinearJSD** | lm_head + JSD fused | 1.10x - 2.86x | 73.7% - 78.3% |

### Custom Operators

| Operator | Description | vs PyTorch (Fwd+Bwd) |
|---|---|---|
| **AttnRes** | Multi-block attention residual fusion | 20.55x - 24.01x |

## Installation

```bash
pip install torch triton
git clone https://github.com/your-username/hilda-kernel.git
cd hilda-kernel
```

## Quick Start

```python
import torch
from triton_kernels.kernels import (
    RMSNorm, CrossEntropyLoss, GRPOLoss,
    hilda_cross_entropy, hilda_rms_norm, hilda_grpo_loss,
)

# nn.Module interface
norm = RMSNorm(hidden_size=4096).cuda()
output = norm(x)

# Functional interface
loss = hilda_cross_entropy(logits, targets)

# GRPO for RLHF
loss, kl, clip_ratio = hilda_grpo_loss(
    logits, completion_ids, advantages,
    old_logp=old_logp, ref_logp=ref_logp,
    completion_mask=mask, beta=0.1,
)
```

### Full API

```python
from triton_kernels.kernels import (
    # Normalization
    hilda_rms_norm, RMSNorm,
    hilda_fused_add_rms_norm, FusedAddRMSNorm,
    # Positional encoding
    hilda_rope, RotaryEmbedding,
    # Activations
    hilda_swiglu, SwiGLU,
    hilda_geglu, GeGLU,
    # Losses
    hilda_cross_entropy, CrossEntropyLoss,
    hilda_kl_div, KLDivLoss,
    hilda_jsd, JSDLoss,
    hilda_grpo_loss, GRPOLoss,
    # Fused operators
    hilda_fused_linear_cross_entropy, FusedLinearCrossEntropyLoss,
    hilda_fused_linear_jsd, FusedLinearJSDLoss,
    # Custom
    hilda_attn_res, AttnRes,
    # Utilities
    fused_selective_log_softmax,
)
```

## Benchmarks

Run correctness tests:

```bash
python benchmarks/bench_rms_norm.py --quick
python benchmarks/bench_cross_entropy.py --quick
python benchmarks/bench_grpo.py --quick
```

Run full benchmarks (requires GPU):

```bash
python benchmarks/bench_rms_norm.py
python benchmarks/bench_rope.py
python benchmarks/bench_cross_entropy.py
python benchmarks/bench_swiglu.py
python benchmarks/bench_geglu.py
python benchmarks/bench_fused_add_rms_norm.py
python benchmarks/bench_kl_div.py
python benchmarks/bench_jsd.py
python benchmarks/bench_fused_linear_ce.py
python benchmarks/bench_fused_linear_jsd.py
python benchmarks/bench_grpo.py
python benchmarks/bench_attn_res.py
```

## Key Optimization Techniques

1. **AutoTune** — Runtime search over `BLOCK_SIZE x num_warps x num_stages` configurations, adapted per input shape and hardware.

2. **Tiled Online Logsumexp** — Single-pass numerically stable softmax via streaming max + sum, with fixed-size tile buffers (O(BLOCK_V) registers regardless of vocab size).

3. **Fused Forward-Backward** — Loss kernels compute gradients in the same kernel as the forward pass (CrossEntropy, KLDiv, JSD), eliminating a separate backward kernel launch.

4. **Recomputation** — Only store scalar statistics (e.g. `rstd`) in forward; reload inputs and recompute in backward, saving O(hidden_dim) memory per token.

5. **Chunked Matmul** — FusedLinear operators split the `(BT, V)` logit space into small chunks, never materializing the full logit tensor (73-78% memory savings for FusedLinearJSD).

6. **Adaptive Backward Paths** — Small-batch backward uses atomic accumulation; large-batch uses partial-sum + reduce, selected automatically based on SM count.

7. **Constexpr Branching** — Optional features (mask, KL penalty, old_logp) use `tl.constexpr` flags so unused code paths are eliminated at compile time.

## Project Structure

```
triton-kernels/
├── kernels/
│   ├── __init__.py
│   ├── rms_norm.py
│   ├── fused_add_rms_norm.py
│   ├── rope.py
│   ├── swiglu.py
│   ├── geglu.py
│   ├── cross_entropy.py
│   ├── kl_div.py
│   ├── jsd.py
│   ├── grpo_loss.py
│   ├── fused_linear_cross_entropy.py
│   ├── fused_linear_jsd.py
│   └── attn_res.py
├── benchmarks/
│   ├── bench_rms_norm.py
│   ├── bench_rope.py
│   ├── bench_cross_entropy.py
│   ├── bench_swiglu.py
│   ├── bench_geglu.py
│   ├── bench_fused_add_rms_norm.py
│   ├── bench_kl_div.py
│   ├── bench_jsd.py
│   ├── bench_grpo.py
│   ├── bench_fused_linear_ce.py
│   ├── bench_fused_linear_jsd.py
│   └── bench_attn_res.py
└── README.md
```

## Test Environment

- GPU: NVIDIA GeForce RTX 5090
- CUDA: 12.8
- Framework: PyTorch + Triton

## License

MIT
