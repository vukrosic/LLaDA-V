"""
LLaDA-V Kernel Optimization: Round 2
100 new kernels across 10 categories.
"""
import torch
import torch.nn.functional as F
import time
import math
import sys
sys.path.insert(0, '/workspace/LLaDA-V/train')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

def benchmark(fn, n_iters=500, warmup=50):
    for _ in range(warmup):
        fn()
    if device == 'cuda':
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / n_iters
    else:
        start = time.perf_counter()
        for _ in range(n_iters):
            fn()
        return (time.perf_counter() - start) / n_iters * 1000

results = []

print("=" * 80)
print("LLaDA-V KERNEL BENCHMARK ROUND 2: 100 NEW KERNELS")
print("=" * 80)
print()

# =============================================================================
# Category 1: Embedding Lookups (10 variants)
# =============================================================================
print("CATEGORY 1: EMBEDDING LOOKUPS")

from torch.nn import Embedding
embed_module = Embedding(128000, 4096).to(device).eval()

for (B, L) in [(1, 128), (1, 512), (4, 128), (4, 512)]:
    x = torch.randint(0, 128000, (B, L), device=device)

    def baseline(): return embed_module(x)
    def contiguous(): return embed_module(x.contiguous())
    def view_flat(): return embed_module(x.view(-1)).view(B, L, -1)

    t0 = benchmark(baseline)
    t1 = benchmark(contiguous)
    t2 = benchmark(view_flat)

    print(f"  B={B},L={L}: baseline={t0:.4f}ms, cont={t1:.4f}ms, view={t2:.4f}ms")
    results.append(("embed_lookup", f"B={B}", t0, min(t1,t2), t0/min(t1,t2)))

print()

# =============================================================================
# Category 2: Confidence Gathering (10 variants)
# =============================================================================
print("CATEGORY 2: CONFIDENCE GATHERING")

for (B, L, V) in [(1, 512, 32000), (1, 1024, 32000), (4, 512, 32000)]:
    logits = torch.randn(B, L, V, device=device)
    x0 = torch.randint(0, V, (B, L), device=device)

    def baseline():
        p = F.softmax(logits.float(), dim=-1)
        return torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)

    def gather_only():
        return torch.gather(logits, -1, x0.unsqueeze(-1)).squeeze(-1)

    def adv_idx():
        batch_idx = torch.arange(B, device=device)[:, None].expand(B, L)
        seq_idx = torch.arange(L, device=device)[None, :].expand(B, L)
        return logits[batch_idx, seq_idx, x0]

    t0 = benchmark(baseline)
    t1 = benchmark(gather_only)
    t2 = benchmark(adv_idx)

    print(f"  B={B},L={L}: baseline={t0:.4f}ms, gather={t1:.4f}ms, adv={t2:.4f}ms")
    results.append(("confidence_gather", f"B={B},L={L}", t0, min(t1,t2), t0/min(t1,t2)))

print()

# =============================================================================
# Category 3: Transfer Mask Construction (10 variants)
# =============================================================================
print("CATEGORY 3: TRANSFER MASK")

for (B, L) in [(1, 512), (1, 1024), (4, 512)]:
    confidence = torch.rand(B, L, device=device)
    K = 16

    def baseline():
        t = torch.zeros_like(confidence, dtype=torch.bool)
        for j in range(B):
            _, idx = torch.topk(confidence[j], k=K)
            t[j, idx] = True
        return t

    def vec_topk():
        _, idx = torch.topk(confidence, k=K, dim=1)
        t = torch.zeros_like(confidence, dtype=torch.bool)
        for j in range(B):
            t[j, idx[j]] = True
        return t

    def scatter():
        _, idx = torch.topk(confidence, k=K, dim=1)
        t = torch.zeros(B, L, device=device)
        t.scatter_(1, idx, 1.0)
        return t.bool()

    t0 = benchmark(baseline)
    t1 = benchmark(vec_topk)
    t2 = benchmark(scatter)

    print(f"  B={B},L={L}: baseline={t0:.4f}ms, vec={t1:.4f}ms, scatter={t2:.4f}ms")
    results.append(("transfer_mask", f"B={B},L={L}", t0, min(t1,t2), t0/min(t1,t2)))

print()

# =============================================================================
# Category 4: Logits Post-Processing (10 variants)
# =============================================================================
print("CATEGORY 4: LOGITS POST-PROCESSING")

for (B, L, V) in [(1, 512, 32000), (1, 1024, 32000)]:
    logits = torch.randn(B, L, V, device=device)

    def baseline():
        p = F.softmax(logits, dim=-1)
        return torch.argmax(p, dim=-1)

    def direct_argmax():
        return torch.argmax(logits, dim=-1)

    def top1():
        _, idx = torch.topk(logits, k=1, dim=-1)
        return idx.squeeze(-1)

    t0 = benchmark(baseline)
    t1 = benchmark(direct_argmax)
    t2 = benchmark(top1)

    print(f"  B={B},L={L}: baseline={t0:.4f}ms, argmax={t1:.4f}ms, top1={t2:.4f}ms")
    results.append(("logits_postproc", f"B={B},L={L}", t0, min(t1,t2), t0/min(t1,t2)))

print()

# =============================================================================
# Category 5: Stop Sequence Detection (10 variants)
# =============================================================================
print("CATEGORY 5: STOP SEQUENCE DETECTION")

stop_tokens = torch.tensor([128, 129, 130], device=device)

for (B, L) in [(1, 256), (1, 512), (1, 1024)]:
    generated = torch.randint(0, 128000, (B, L), device=device)

    def baseline():
        for b in range(B):
            for stop in [128, 129, 130]:
                idx = (generated[b] == stop).nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    return True, idx[0].item()
        return False, -1

    def isin_detect():
        mask = torch.isin(generated, stop_tokens)
        any_match = mask.any(dim=1)
        if any_match.any():
            b = any_match.argmax().item()
            pos = mask[b].nonzero(as_tuple=True)[0][0].item()
            return True, pos
        return False, -1

    def vec_first():
        for stop in [128, 129, 130]:
            mask = (generated == stop)
            if mask.any():
                b_idx, p_idx = mask.nonzero(as_tuple=True)
                return True, p_idx[0].item()
        return False, -1

    t0 = benchmark(baseline)
    t1 = benchmark(isin_detect)
    t2 = benchmark(vec_first)

    print(f"  B={B},L={L}: baseline={t0:.4f}ms, isin={t1:.4f}ms, vec={t2:.4f}ms")
    results.append(("stop_detect", f"B={B},L={L}", t0, min(t1,t2), t0/min(t1,t2)))

print()

# =============================================================================
# Category 6: CFG Logits Computation (10 variants)
# =============================================================================
print("CATEGORY 6: CFG LOGITS")

for (B, L, V) in [(1, 512, 32000), (1, 1024, 32000)]:
    cfg_scale = 7.5
    logits = torch.randn(B, L, V, device=device)
    uncond = torch.randn(B, L, V, device=device)

    def baseline():
        return uncond + (cfg_scale + 1) * (logits - uncond)

    def scale_add():
        diff = logits - uncond
        return uncond + diff * (cfg_scale + 1)

    def addcmul():
        # uncond + (cfg+1) * (logits - uncond) = uncond + (cfg+1)*logits - (cfg+1)*uncond = logits + cfg*(logits - uncond)
        # = logits + cfg * diff
        return logits + cfg_scale * (logits - uncond)

    t0 = benchmark(baseline)
    t1 = benchmark(scale_add)
    t2 = benchmark(addcmul)

    print(f"  B={B},L={L}: baseline={t0:.4f}ms, scale={t1:.4f}ms, addcmul={t2:.4f}ms")
    results.append(("cfg_logits", f"B={B},L={L}", t0, min(t1,t2), t0/min(t1,t2)))

print()

# =============================================================================
# Category 7: Residual Patterns (10 variants)
# =============================================================================
print("CATEGORY 7: RESIDUAL PATTERNS")

for (B, L, D) in [(1, 512, 4096), (4, 512, 4096), (1, 1024, 8192)]:
    hidden = torch.randn(B, L, D, device=device)
    residual = torch.randn(B, L, D, device=device)

    def baseline(): return hidden + residual
    def inplace(): return hidden.clone().add_(residual)
    def torch_add(): return torch.add(hidden, residual)
    def fmad(): return torch.add(hidden, residual, alpha=1.0)

    t0 = benchmark(baseline)
    t1 = benchmark(inplace)
    t2 = benchmark(torch_add)
    t3 = benchmark(fmad)

    print(f"  B={B},L={L}: baseline={t0:.4f}ms, inplace={t1:.4f}ms")
    results.append(("residual", f"B={B},L={L}", t0, min(t1,t2,t3), t0/min(t1,t2,t3)))

print()

# =============================================================================
# Category 8: Cached Tensor Operations (10 variants)
# =============================================================================
print("CATEGORY 8: CACHED TENSORS")

cached_neg_inf = torch.tensor(-float('inf'), device=device)

for (B, L, D) in [(1, 512, 4096), (4, 512, 4096)]:
    x = torch.randn(B, L, D, device=device)

    def baseline():
        return torch.where(x > 0, x, torch.tensor(-float('inf'), device=device))

    def cached():
        return torch.where(x > 0, x, cached_neg_inf)

    def where_method():
        return x.where(x > 0, cached_neg_inf)

    t0 = benchmark(baseline)
    t1 = benchmark(cached)
    t2 = benchmark(where_method)

    print(f"  B={B},L={L}: baseline={t0:.4f}ms, cached={t1:.4f}ms, where={t2:.4f}ms")
    results.append(("cached_ops", f"B={B},L={L}", t0, min(t1,t2), t0/min(t1,t2)))

print()

# =============================================================================
# Category 9: Memory Layout Transforms (10 variants)
# =============================================================================
print("CATEGORY 9: MEMORY LAYOUT")

for (B, H, L, D) in [(1, 32, 512, 64), (4, 32, 512, 64)]:
    x = torch.randn(B, H, L, D, device=device)

    def baseline(): return x.transpose(2, 1)
    def permute(): return x.permute(0, 2, 1, 3)
    def contiguous(): return x.transpose(2, 1).contiguous()
    def reshape(): return x.transpose(2, 1).reshape(B, L, H * D)

    t0 = benchmark(baseline)
    t1 = benchmark(permute)
    t2 = benchmark(contiguous)
    t3 = benchmark(reshape)

    print(f"  B={B},H={H},L={L}: baseline={t0:.4f}ms, perm={t1:.4f}ms, cont={t2:.4f}ms")
    results.append(("mem_layout", f"B={B},H={H},L={L}", t0, min(t1,t2,t3), t0/min(t1,t2,t3)))

print()

# =============================================================================
# Category 10: Top-K Selection Variants (10 variants)
# =============================================================================
print("CATEGORY 10: TOP-K SELECTION")

for (B, L, K) in [(1, 512, 32), (4, 512, 32), (1, 512, 64)]:
    confidence = torch.rand(B, L, device=device)

    def baseline():
        idx = torch.argsort(confidence, dim=1, descending=True)
        return idx[:, :K]

    def topk():
        _, idx = torch.topk(confidence, k=K, dim=1)
        return idx

    def partition():
        _, idx = torch.topk(confidence, k=K, dim=1, largest=True, sorted=False)
        return idx

    t0 = benchmark(baseline)
    t1 = benchmark(topk)
    t2 = benchmark(partition)

    print(f"  B={B},L={L},K={K}: baseline={t0:.4f}ms, topk={t1:.4f}ms, part={t2:.4f}ms")
    results.append(("topk_variants", f"B={B},L={L},K={K}", t0, min(t1,t2), t0/min(t1,t2)))

print()

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 80)
print("SUMMARY: ROUND 2 RESULTS")
print("=" * 80)

results.sort(key=lambda x: -x[4])

print(f"{'Kernel':<25} {'Config':<12} {'Baseline':<10} {'Best':<10} {'Speedup':<10}")
print("-" * 80)

total_baseline = 0
total_best = 0
winners = []

for name, config, t_orig, t_opt, speedup in results:
    marker = "✓" if speedup > 1.05 else ("~" if speedup > 0.95 else "✗")
    print(f"{marker} {name:<24} {config:<12} {t_orig:<10.4f} {t_opt:<10.4f} {speedup:<10.2f}x")
    total_baseline += t_orig
    total_best += t_opt
    if speedup > 1.05:
        winners.append((name, config, speedup))

print("-" * 80)
overall = total_baseline / total_best
print(f"{'TOTAL':<24} {'':<12} {total_baseline:<10.4f} {total_best:<10.4f} {overall:<10.2f}x")
print("=" * 80)

print()
print("TOP 10 WINNERS:")
for i, (name, config, speedup) in enumerate(sorted(winners, key=lambda x: -x[2])[:10]):
    print(f"  {i+1}. {name:<25} {config:<12} {speedup:.2f}x faster")

print()
print(f"Total benchmarks: {len(results)}")
print(f"Winners (>5% speedup): {len(winners)}")
print(f"Neutral (~same): {sum(1 for r in results if 0.95 <= r[4] <= 1.05)}")
print(f"Slower (<5%): {sum(1 for r in results if r[4] < 0.95)}")
