# LLaDA-V Kernel Benchmark Results

**Device**: NVIDIA RTX 3090 (24GB)
**Date**: 2026-03-31
**Test**: 10 kernel categories, 20+ variants, ~100 benchmark runs

---

## Summary Table

| Kernel | Config | Baseline (ms) | Optimized (ms) | Speedup | Status |
|--------|--------|--------------|-----------------|---------|--------|
| attention_score (fused_scale) | B=1, H=64, L=1024 | 1.1293 | 0.6016 | **1.88x** | ✓ WINNER |
| attention_score (fused_scale) | B=4, H=32, L=512 | 0.5794 | 0.3316 | **1.75x** | ✓ WINNER |
| get_num_transfer | B=4, L=512, steps=32 | 0.6983 | 0.4323 | **1.62x** | ✓ WINNER |
| get_num_transfer | B=1, L=512, steps=32 | 0.7006 | 0.4340 | **1.61x** | ✓ WINNER |
| repeat_kv (repeat_interleave) | B=1, L=1024 | 0.1063 | 0.0673 | **1.58x** | ✓ WINNER |
| repeat_kv (repeat_interleave) | B=4, L=512 | 0.0965 | 0.0648 | **1.49x** | ✓ WINNER |
| rmsnorm (fast rsqrt) | B=1, L=512, D=4096 | 0.2105 | 0.1669 | **1.26x** | ✓ WINNER |
| attention_score (fused_scale) | B=1, H=32, L=512 | 0.1625 | 0.1313 | **1.24x** | ✓ WINNER |
| rope (fused) | B=1, H=32, L=512 | 0.4395 | 0.4160 | 1.06x | ~ SAME |
| rope (fused) | B=4, H=32, L=512 | 0.4457 | 0.4344 | 1.03x | ~ SAME |
| rmsnorm (fast rsqrt) | B=1, L=1024, D=8192 | 0.2850 | 0.2832 | 1.01x | ~ SAME |
| softmax (inplace) | all configs | ~1.00x | ~1.00x | 1.00x | ~ SAME |
| kv_cache (slice) | B=4, L=2048 | 0.1628 | 0.1621 | 1.00x | ~ SAME |
| gumbel_noise (half) | all configs | ~1.00x | ~0.99x | 0.99x | ~ SAME |
| topk_selection (vectorized) | B=1 | 0.2397 | 0.2923 | 0.82x | ✗ SLOWER |
| kv_cache (slice) | B=1 | 0.0420 | 0.1211 | 0.35x | ✗ SLOWER |

---

## Top 5 Optimizations

| Rank | Kernel | Config | Speedup | Improvement |
|------|--------|--------|---------|-------------|
| 1 | **attention_score (fused_scale)** | B=1, H=64, L=1024 | **1.88x** | 0.53ms saved |
| 2 | **attention_score (fused_scale)** | B=4, H=32, L=512 | **1.75x** | 0.25ms saved |
| 3 | **get_num_transfer** | B=4 | **1.62x** | 0.27ms saved |
| 4 | **get_num_transfer** | B=1 | **1.61x** | 0.27ms saved |
| 5 | **repeat_kv** | B=1 | **1.58x** | 0.04ms saved |

---

## Top Winners Summary

### ✓ Winners (9 benchmarks showing speedup > 5%)

1. **attention_score (fused_scale)** - Best optimization! ✓ BIT-IDENTICAL
   - Pre-multiply Q by scale before matmul instead of multiplying after
   - 1.24x - 1.88x speedup depending on config
   - Verified: produces bit-identical results

2. **get_num_transfer_tokens** ✓ BIT-IDENTICAL
   - Avoid `.clone()` by using `add()` instead of `+=`
   - Consistent 1.61-1.62x speedup
   - Verified: produces bit-identical results

3. **repeat_kv** (repeat_interleave) ✓ BIT-IDENTICAL
   - `repeat_interleave` instead of `expand + reshape`
   - 1.49x - 1.58x speedup
   - Verified: produces bit-identical results

4. **rmsnorm (fast)** ~ CLOSE (not bit-identical)
   - Precompute `rsqrt` instead of `1 / sqrt()`
   - 1.26x speedup for single batch
   - Note: slight FP differences but mathematically equivalent

### ~ Neutral (8 benchmarks within 5%)

- rope, softmax, conversation_mask: Essentially same performance
- gumbel_half: No significant gain

### ✗ Not Recommended (2 benchmarks)

- **topk_vectorized**: 0.82x (slower) - Python loop overhead
- **kv_cache_slice**: 0.35x for small cache (faster for large caches)

---

## Overall Impact

**Kernel-level speedup**: ~1.05x average across all kernels

**Note**: These are microbenchmarks. Actual end-to-end speedup will be lower since:
1. Kernels are part of larger operations
2. Memory bandwidth is often the bottleneck
3. Flash attention already handles most attention kernels

---

## Recommended Optimizations to Integrate

| Priority | Kernel | Action | Expected Gain |
|----------|--------|--------|---------------|
| HIGH | attention_score | Fuse scale into matmul | 1.5-2x on attention |
| HIGH | repeat_kv | Use repeat_interleave | Already in codebase ✓ |
| HIGH | get_num_transfer | Avoid clone, use add | Already in codebase ✓ |
| MEDIUM | rmsnorm | Precompute rsqrt | Small but free |
| LOW | gumbel_half | Try float32 noise | No gain, skip |

---

## Files Created

```
train/kernels/
├── __init__.py
├── master_benchmark.py     # Main benchmark runner
├── mask_detection.py       # 5 variants
├── topk_selection.py       # 4 variants
├── gumbel_noise.py         # 5 variants
├── rope_kernels.py         # 5 variants
├── rmsnorm.py              # 5 variants
├── attention_score.py      # 5 variants
├── softmax.py              # 4 variants
├── mlp_ffn.py              # 5 variants
├── kv_cache.py             # 5 variants
└── stopping.py             # 5 variants
```

---

## Running Benchmarks

```bash
# Run master benchmark
python3 train/kernels/master_benchmark.py

# Run individual kernel benchmarks
python3 train/kernels/attention_score.py
python3 train/kernels/rope_kernels.py
# etc.
```

---

## Round 2: 100 Additional Kernels (2026-03-31)

**Test**: `train/kernels_v2/more_kernels.py`
**Categories**: 10 categories × 10 variants = 100 kernel variants

### Summary

| Kernel | Config | Baseline (ms) | Best (ms) | Speedup | Notes |
|--------|--------|--------------|-----------|---------|-------|
| confidence_gather | B=4, L=512 | 2.61 | 0.13 | **19.82x** | ⚠️ Baseline does softmax first |
| confidence_gather | B=1, L=1024 | 1.35 | 0.14 | **9.83x** | ⚠️ Baseline does softmax first |
| logits_postproc | B=1, L=1024 | 1.35 | 0.28 | **4.80x** | ⚠️ Baseline does softmax first |
| logits_postproc | B=1, L=512 | 1.33 | 0.30 | **4.46x** | ⚠️ Baseline does softmax first |
| confidence_gather | B=1, L=512 | 0.70 | 0.17 | **4.04x** | ⚠️ Baseline does softmax first |
| transfer_mask | B=4, L=512 | 0.26 | 0.06 | **3.97x** | ✓ Legitimate |
| topk_variants | B=1, L=512, K=32 | 0.26 | 0.07 | **3.60x** | ✓ Legitimate |
| topk_variants | B=4, L=512, K=32 | 0.25 | 0.07 | **3.58x** | ✓ Legitimate |
| topk_variants | B=1, L=512, K=64 | 0.25 | 0.07 | **3.55x** | ✓ Legitimate |
| stop_detect | B=1, L=1024 | 0.09 | 0.04 | **2.03x** | ✓ Legitimate |

**Note**: ⚠️ indicates comparisons where baseline does extra unnecessary work (e.g., softmax before argmax). Since `argmax(softmax(x)) = argmax(x)`, these are valid optimizations but speedup numbers are inflated.

### Legitimate Winners (>5% speedup, fair comparison)

| Rank | Kernel | Speedup | Reason |
|------|--------|---------|--------|
| 1 | transfer_mask (scatter) | **3.97x** | scatter_add vs Python loop |
| 2 | topk_variants (topk) | **3.55-3.60x** | torch.topk vs argsort+slice |
| 3 | stop_detect (vec_first) | **2.03x** | vectorized vs nested Python loops |
| 4 | embed_lookup (view_flat) | **1.11x** | view+reshape vs embedding lookup |
| 5 | cfg_logits (addcmul) | **1.05x** | fused multiply-add |

### What Didn't Work

| Kernel | Result | Reason |
|--------|--------|--------|
| confidence_gather adv_idx | ~1.0x | Advanced indexing overhead |
| transfer_mask vec_topk | ~0.95x | Python loop over batch |
| residual operations | ~1.0x | PyTorch already optimized |
| cached_ops | ~1.0x | Negligible overhead |
| mem_layout | ~0.98x | No real improvement |

### Overall Round 2

- **Total benchmarks**: 27
- **Winners (>5% speedup, fair comparison)**: ~5
- **Neutral**: ~20
- **Slower**: ~2

**Conclusion**: Round 2 found fewer genuine wins than Round 1. The `topk`, `scatter`-based transfer mask, and vectorized stop detection are legitimate improvements. Most other "gains" were comparing against artificially slow baselines.
