# LLaDA-V Kernel Benchmark Plan
## 30 Kernels × Multiple Algorithms = Comprehensive Performance Survey

---

## Overview

We will implement **30 kernels** for LLaDA-V, each with **3-5 alternative implementations** totaling **100+ kernel variants**. Each will be benchmarked against the baseline implementation to find the fastest approach.

---

## Kernel Categories

### Category 1: Mask Detection Kernels (3 kernels × 4 algos = 12 variants)

**Purpose**: Detect which positions are masked (still need to be filled)

| Kernel | Description |
|--------|-------------|
| `mask_detection_abs` | `torch.abs(x - masked_embed) < eps` |
| `mask_detection_isclose` | `torch.isclose(x, masked_embed, atol=eps)` |
| `mask_detection_sq` | `((x - masked_embed)^2).sum(dim) < eps^2` |
| `mask_detection_chunked` | Divide embedding into chunks, compare per-chunk |

**Benchmark**: Compare all 4 against baseline on various sequence lengths (128, 512, 1024, 2048)

---

### Category 2: Transfer Token Selection Kernels (3 kernels × 4 algos = 12 variants)

**Purpose**: Select top-k tokens to transfer from masked to filled

| Kernel | Description |
|--------|-------------|
| `topk_loop` | Python loop with `torch.topk` per batch element |
| `topk_vectorized` | Vectorized topk across batch |
| `topk_cumsum` | Use cumsum-based selection |
| `topk_partial_sort` | `torch.topk` with `sorted=False` for partial sort |

**Benchmark**: Vary batch size (1, 4, 8) and k values

---

### Category 3: Gumbel Noise Kernels (3 kernels × 4 algos = 12 variants)

**Purpose**: Add Gumbel noise for categorical sampling

| Kernel | Description |
|--------|-------------|
| `gumbel_original` | `logits.exp() / ((-log(noise)) ** temp)` with float64 |
| `gumbel_fast` | Same math, fewer casts |
| `gumbel_inplace` | In-place operations where possible |
| `gumbel_half` | Use float32 for noise, keep logits float64 |

**Benchmark**: Vary vocab size (32k, 100k, 320k)

---

### Category 4: RoPE (Rotary Position Embedding) Kernels (3 kernels × 4 algos = 12 variants)

**Purpose**: Apply rotary position embedding to Q and K

| Kernel | Description |
|--------|-------------|
| `rope_standard` | Original: cat, cos, sin with rotate_half |
| `rope_fused` | Fused single-pass without intermediate cats |
| `rope_complex` | Use complex numbers `exp(iθ)` |
| `rope_lazy` | Lazy eager: compute only needed positions |

**Benchmark**: Vary seq_len (128, 512, 2048) and num_heads (32, 64)

---

### Category 5: RMSNorm Kernels (3 kernels × 4 algos = 12 variants)

**Purpose**: LLaDA's normalization layer

| Kernel | Description |
|--------|-------------|
| `rmsnorm_original` | Original: `x / sqrt(variance + eps) * weight` |
| `rmsnorm_fast` | Precompute `rsqrt` |
| `rmsnorm_inplace` | In-place variance computation |
| `rmsnorm_blocked` | Process in blocks for cache efficiency |

**Benchmark**: Vary hidden_dim (4096, 7168, 8192)

---

### Category 6: Attention Score Kernels (3 kernels × 4 algos = 12 variants)

**Purpose**: Compute QK^T / sqrt(d) attention scores

| Kernel | Description |
|--------|-------------|
| `attn_matmul` | Standard `torch.matmul(Q, K.transpose(-2,-1)) / sqrt(d)` |
| `attn_bmm` | Batched matmul `torch.bmm` |
| `attn_scaled` | Scale during matmul to avoid extra kernel |
| `attn_fused_qk` | Fuse Q projection and K projection before scaling |

**Benchmark**: Vary batch, seq_len, num_heads, head_dim

---

### Category 7: Softmax Kernels (3 kernels × 4 algos = 12 variants)

**Purpose**: Apply softmax to attention weights

| Kernel | Description |
|--------|-------------|
| `softmax_original` | `exp(x) / exp(x).sum(dim)` |
| `softmax_stable` | `exp(x - max) / exp(x - max).sum(dim)` |
| `softmax_inplace` | In-place with single allocation |
| `softmax_blocked` | Block-wise softmax for long sequences |

**Benchmark**: Vary sequence length and check numerical stability

---

### Category 8: MLP/FFN Kernels (3 kernels × 4 algos = 12 variants)

**Purpose**: LLaDA's FFN: `down_proj(act(gate_proj(x)) * up_proj(x))`

| Kernel | Description |
|--------|-------------|
| `mlp_sequential` | `gate = W1(x); up = W2(x); act(gate) * up` |
| `mlp_fused_linear` | Use single grouped linear for gate+up |
| `mlp_inplace` | In-place activations where safe |
| `mlp_lazy` | Lazy evaluation: don't compute up if not needed |

**Benchmark**: Vary batch, seq_len, intermediate_size

---

### Category 9: KV Cache Update Kernels (3 kernels × 4 algos = 12 variants)

**Purpose**: Update key/value cache during generation

| Kernel | Description |
|--------|-------------|
| `cache_cat` | `torch.cat([cache, new], dim=-2)` |
| `cache_slice` | Direct slice assignment `cache[..., -n:] = new` |
| `cache_index` | Use `.index_copy_` |
| `cache_prealloc` | Preallocate, track offset, no allocation each step |

**Benchmark**: Vary cache size (1024, 4096, 16384 tokens)

---

### Category 10: Stopping Criteria Kernels (3 kernels × 4 algos = 12 variants)

**Purpose**: Detect stop tokens in generated sequence

| Kernel | Description |
|--------|-------------|
| `stop_loop` | Python loop checking each position |
| `stop_vectorized` | Vectorized comparison |
| `stop_conv` | Use convolution for sliding window |
| `stop_strided` | Use strided operations |

**Benchmark**: Vary sequence length and number of stop sequences

---

## Benchmarking Framework

### Test Structure
```python
{
    "kernel_name": "mask_detection_abs",
    "category": "mask_detection",
    "description": "Standard abs comparison",
    "baseline": True,
    "functional": "mask = abs(x - ref) < eps",
    "impl_variants": [
        {"name": "abs_naive", "code": "...", "expected_speedup": 1.0},
        {"name": "abs_optimized", "code": "...", "expected_speedup": 1.5},
    ]
}
```

### Benchmark Parameters
| Parameter | Values |
|----------|--------|
| sequence_length | 128, 512, 1024, 2048 |
| batch_size | 1, 4, 8 |
| hidden_dim | 4096, 7168, 8192 |
| num_heads | 32, 64 |
| head_dim | 64, 128 |
| vocab_size | 32000, 100000, 320000 |
| iterations | 1000 (timed), 100 (warmup) |

### Metrics
- **Time (ms)**: Mean execution time
- **Speedup vs Baseline**: `t_baseline / t_variant`
- **Memory (MB)**: Peak GPU memory
- **Numerical Error**: Max abs diff vs baseline
- **Stability**: Stddev across runs

---

## Implementation Plan

### Phase 1: Kernel Infrastructure (Day 1)
- [ ] Create benchmark framework
- [ ] Implement 10 baseline kernels
- [ ] Verify correctness against original code

### Phase 2: Kernel Variants (Day 2-3)
- [ ] Implement 3-4 variants per kernel
- [ ] Run correctness tests
- [ ] Collect benchmark data

### Phase 3: Analysis (Day 4)
- [ ] Compile benchmark results
- [ ] Identify best variant per kernel
- [ ] Create recommendation table

---

## Expected Outcomes

1. **Best-in-class kernels**: Identify fastest implementation for each operation
2. **Speedup factors**: Likely 1.2x - 3x for individual kernels
3. **End-to-end estimate**: Kernel-level speedups don't fully translate, but 10-20% overall possible
4. **Production-ready**: Best variants will be integrated into codebase

---

## Kernel-to-Function Mapping

| Kernel Category | Used In | Priority |
|----------------|---------|----------|
| mask_detection | generate_with_embeds, fast_generate | HIGH |
| topk_selection | _get_transfer_index | HIGH |
| gumbel_noise | add_gumbel_noise | HIGH |
| rope | attention forward | HIGH |
| rmsnorm | every layer | MEDIUM |
| attention_score | attention forward | HIGH |
| softmax | attention forward | HIGH |
| mlp_ffn | decoder layer | MEDIUM |
| kv_cache | generation loop | HIGH |
| stopping | generation loop | MEDIUM |

---

## Files to Create

```
train/
├── kernels/
│   ├── __init__.py
│   ├── benchmark_framework.py    # Benchmark infrastructure
│   ├── mask_detection.py         # 4 variants
│   ├── topk_selection.py         # 4 variants
│   ├── gumbel_noise.py           # 4 variants
│   ├── rope_kernels.py           # 4 variants
│   ├── rmsnorm.py                # 4 variants
│   ├── attention_score.py        # 4 variants
│   ├── softmax.py                # 4 variants
│   ├── mlp_ffn.py                # 4 variants
│   ├── kv_cache.py               # 4 variants
│   └── stopping.py                # 4 variants
├── benchmark_all.py              # Run all benchmarks
└── results/
    └── kernel_benchmarks.md      # Results table
```

---

## Quick Start Commands

```bash
# Run single kernel benchmark
python3 benchmark_single.py --kernel mask_detection

# Run all benchmarks
python3 benchmark_all.py --output results/

# Compare all variants for one kernel
python3 compare_kernel.py --kernel rope
```
