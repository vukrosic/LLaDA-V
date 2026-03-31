"""
Master benchmark runner for all LLaDA-V kernels.
Runs key kernel variants and produces summary table.
"""
import torch
import time
import sys
import math
sys.path.insert(0, '/workspace/LLaDA-V/train')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def benchmark(fn, args, n_iters=500, warmup=50):
    """Benchmark a function."""
    # Warmup
    for _ in range(warmup):
        fn(*args)

    if device == 'cuda':
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iters):
            fn(*args)
        end.record()
        torch.cuda.synchronize()
        time_ms = start.elapsed_time(end) / n_iters
    else:
        start = time.perf_counter()
        for _ in range(n_iters):
            fn(*args)
        time_ms = (time.perf_counter() - start) / n_iters * 1000

    return time_ms

print("=" * 80)
print("LLaDA-V MASTER KERNEL BENCHMARK")
print(f"Device: {device}")
print("=" * 80)
print()

results = []

# =============================================================================
# 1. repeat_kv (from fast_dllm_hook)
# =============================================================================
print("1. repeat_kv")
print("-" * 60)
from llava.hooks.fast_dllm_hook import FastDLLMGenerationHook

for (B, H, L, D), n_rep in [((4, 8, 512, 64), 4), ((1, 8, 1024, 64), 4)]:
    hidden = torch.randn(B, H, L, D, device=device)

    def orig(): return hidden[:, :, None, :, :].expand(B, H, n_rep, L, D).reshape(B, H * n_rep, L, D)
    def opt(): return hidden.repeat_interleave(n_rep, dim=1)

    t_orig = benchmark(orig, ())
    t_opt = benchmark(opt, ())
    print(f"  repeat_kv B={B},H={H},L={L}: orig={t_orig:.4f}ms, opt={t_opt:.4f}ms, {t_orig/t_opt:.2f}x")
    results.append(("repeat_kv", f"B={B}", t_orig, t_opt, t_orig/t_opt))

print()

# =============================================================================
# 2. TopK Selection
# =============================================================================
print("2. topk_selection")
print("-" * 60)

def topk_loop(confidence, k):
    batch_size = confidence.shape[0]
    transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
    for j in range(batch_size):
        _, select_index = torch.topk(confidence[j], k=k[j])
        transfer_index[j, select_index] = True
    return transfer_index

def topk_vectorized(confidence, k):
    batch_size, seq_len = confidence.shape
    transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
    topk_values, topk_indices = torch.topk(confidence, k=max(k), dim=1)
    for j in range(batch_size):
        transfer_index[j, topk_indices[j, :k[j]]] = True
    return transfer_index

for (B, L), k_val in [((1, 512), 16), ((4, 512), 16), ((8, 512), 16)]:
    conf = torch.rand(B, L, device=device)
    k = torch.tensor([k_val] * B, device=device)

    t_orig = benchmark(topk_loop, (conf, k))
    t_vec = benchmark(topk_vectorized, (conf, k))
    print(f"  topk B={B},L={L},k={k_val}: loop={t_orig:.4f}ms, vec={t_vec:.4f}ms, {t_orig/t_vec:.2f}x")
    results.append(("topk_selection", f"B={B}", t_orig, t_vec, t_orig/t_vec))

print()

# =============================================================================
# 3. Gumbel Noise
# =============================================================================
print("3. gumbel_noise")
print("-" * 60)

def gumbel_original(logits, temperature):
    if temperature == 0:
        return logits
    logits_fp = logits.to(torch.float64)
    noise = torch.rand_like(logits_fp, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return (logits_fp.exp() / gumbel_noise).to(logits.dtype)

def gumbel_half(logits, temperature):
    if temperature == 0:
        return logits
    logits_fp = logits.to(torch.float64)
    noise = torch.rand(logits.shape, dtype=torch.float32, device=logits.device)
    noise = noise.to(torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return (logits_fp.exp() / gumbel_noise).to(logits.dtype)

for (B, L, V), temp in [((1, 128, 32000), 0.5), ((1, 512, 32000), 0.5), ((4, 128, 32000), 0.5)]:
    logits = torch.randn(B, L, V, device=device)

    t_orig = benchmark(gumbel_original, (logits, temp))
    t_half = benchmark(gumbel_half, (logits, temp))
    print(f"  gumbel B={B},L={L},V={V}: orig={t_orig:.4f}ms, half={t_half:.4f}ms, {t_orig/t_half:.2f}x")
    results.append(("gumbel_noise", f"B={B},L={L}", t_orig, t_half, t_orig/t_half))

print()

# =============================================================================
# 4. RoPE (Rotary Position Embedding)
# =============================================================================
print("4. rope")
print("-" * 60)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def rope_original(q, k, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

def rope_fused(q, k, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    mid = q.shape[-1] // 2
    q1, q2 = q[..., :mid], q[..., mid:]
    k1, k2 = k[..., :mid], k[..., mid:]
    return (torch.cat((-q2, q1), dim=-1) * sin + q * cos,
            torch.cat((-k2, k1), dim=-1) * sin + k * cos)

for (B, H, L, D) in [(1, 32, 512, 64), (4, 32, 512, 64), (4, 64, 1024, 64)]:
    q = torch.randn(B, H, L, D, device=device)
    k = torch.randn(B, H, L, D, device=device)
    cos = torch.randn(B, L, D, device=device)
    sin = torch.randn(B, L, D, device=device)

    t_orig = benchmark(rope_original, (q, k, cos, sin))
    t_fused = benchmark(rope_fused, (q, k, cos, sin))
    print(f"  rope B={B},H={H},L={L},D={D}: orig={t_orig:.4f}ms, fused={t_fused:.4f}ms, {t_orig/t_fused:.2f}x")
    results.append(("rope", f"B={B},H={H}", t_orig, t_fused, t_orig/t_fused))

print()

# =============================================================================
# 5. RMSNorm
# =============================================================================
print("5. rmsnorm")
print("-" * 60)

def rmsnorm_original(x, weight, eps=1e-6):
    variance = x.pow(2).mean(-1, keepdim=True)
    return weight * (x * torch.rsqrt(variance + eps))

def rmsnorm_fast(x, weight, eps=1e-6):
    var = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * var.rsqrt()
    return weight * x_norm

for (B, L, D) in [(1, 512, 4096), (4, 512, 4096), (1, 1024, 8192)]:
    x = torch.randn(B, L, D, device=device)
    weight = torch.randn(D, device=device)

    t_orig = benchmark(rmsnorm_original, (x, weight))
    t_fast = benchmark(rmsnorm_fast, (x, weight))
    print(f"  rmsnorm B={B},L={L},D={D}: orig={t_orig:.4f}ms, fast={t_fast:.4f}ms, {t_orig/t_fast:.2f}x")
    results.append(("rmsnorm", f"B={B},L={L}", t_orig, t_fast, t_orig/t_fast))

print()

# =============================================================================
# 6. Attention Score
# =============================================================================
print("6. attention_score")
print("-" * 60)

def attention_original(q, k, scale=None):
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    return torch.matmul(q, k.transpose(-2, -1)) * scale

def attention_fused_scale(q, k):
    scale = 1.0 / math.sqrt(q.shape[-1])
    q_scaled = q * scale
    return torch.matmul(q_scaled, k.transpose(-2, -1))

for (B, H, L, D) in [(1, 32, 512, 64), (4, 32, 512, 64), (1, 64, 1024, 64)]:
    q = torch.randn(B, H, L, D, device=device)
    k = torch.randn(B, H, L, D, device=device)

    t_orig = benchmark(attention_original, (q, k))
    t_fused = benchmark(attention_fused_scale, (q, k))
    print(f"  attn B={B},H={H},L={L},D={D}: orig={t_orig:.4f}ms, fused={t_fused:.4f}ms, {t_orig/t_fused:.2f}x")
    results.append(("attention_score", f"B={B},H={H}", t_orig, t_fused, t_orig/t_fused))

print()

# =============================================================================
# 7. KV Cache
# =============================================================================
print("7. kv_cache")
print("-" * 60)

def kv_cat(cache, new_kv):
    return torch.cat([cache, new_kv], dim=-2)

def kv_slice(cache, new_kv):
    B, H, L_max, D = cache.shape
    B2, H2, L_new, D2 = new_kv.shape
    result = torch.empty(B, H, L_max + L_new, D, device=cache.device, dtype=cache.dtype)
    result[..., :L_max, :] = cache
    result[..., L_max:, :] = new_kv
    return result

for (B, H, L_max, D), (B2, H2, L_new, D2) in [((1, 32, 1024, 64), (1, 32, 128, 64)), ((4, 32, 2048, 64), (4, 32, 32, 64))]:
    cache = torch.randn(B, H, L_max, D, device=device)
    new_kv = torch.randn(B2, H2, L_new, D2, device=device)

    t_cat = benchmark(kv_cat, (cache, new_kv))
    t_slice = benchmark(kv_slice, (cache, new_kv))
    print(f"  kv B={B},L={L_max}+{L_new}: cat={t_cat:.4f}ms, slice={t_slice:.4f}ms, {t_cat/t_slice:.2f}x")
    results.append(("kv_cache", f"B={B},L={L_max}", t_cat, t_slice, t_cat/t_slice))

print()

# =============================================================================
# 8. Softmax (stable)
# =============================================================================
print("8. softmax")
print("-" * 60)

def softmax_stable(x, dim=-1):
    exp_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def softmax_inplace(x, dim=-1):
    x_max = x.max(dim=dim, keepdim=True)[0]
    x.sub_(x_max)
    x.exp_()
    return x.div_(x.sum(dim=dim, keepdim=True))

for (B, H, L) in [(1, 32, 512), (4, 32, 512), (1, 64, 1024)]:
    x = torch.randn(B, H, L, L, device=device)

    t_stable = benchmark(softmax_stable, (x,))
    t_inplace = benchmark(softmax_inplace, (x,))
    print(f"  softmax B={B},H={H},L={L}: stable={t_stable:.4f}ms, inplace={t_inplace:.4f}ms, {t_stable/t_inplace:.2f}x")
    results.append(("softmax", f"B={B},H={H}", t_stable, t_inplace, t_stable/t_inplace))

print()

# =============================================================================
# 9. Conversation Mask
# =============================================================================
print("9. conversation_mask")
print("-" * 60)

def conv_mask_orig(conv_ids):
    ids_i = conv_ids.unsqueeze(-1)
    ids_j = conv_ids.unsqueeze(-2)
    return (ids_j <= ids_i).unsqueeze(1)

for (B, L) in [(1, 256), (1, 512), (4, 512)]:
    conv_ids = torch.randint(0, 100, (B, L), device=device)

    t = benchmark(conv_mask_orig, (conv_ids,))
    print(f"  conv_mask B={B},L={L}: {t:.4f}ms")
    results.append(("conversation_mask", f"B={B}", t, t, 1.0))

print()

# =============================================================================
# 10. get_num_transfer_tokens
# =============================================================================
print("10. get_num_transfer_tokens")
print("-" * 60)

def num_transfer_orig(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = base.expand(-1, steps).clone()
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens[mask] += 1
    return num_transfer_tokens.to(torch.int64)

def num_transfer_opt(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = base.expand(-1, steps)
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens = num_transfer_tokens + mask.to(num_transfer_tokens.dtype)
    return num_transfer_tokens.to(torch.int64)

for (B, L), steps in [((1, 512), 32), ((4, 512), 32)]:
    mask_index = torch.rand(B, L, device=device) > 0.3

    t_orig = benchmark(num_transfer_orig, (mask_index, steps))
    t_opt = benchmark(num_transfer_opt, (mask_index, steps))
    print(f"  num_transfer B={B},L={L},steps={steps}: orig={t_orig:.4f}ms, opt={t_opt:.4f}ms, {t_orig/t_opt:.2f}x")
    results.append(("get_num_transfer", f"B={B}", t_orig, t_opt, t_orig/t_opt))

print()

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 80)
print("SUMMARY: Speedups vs Baseline")
print("=" * 80)
print(f"{'Kernel':<22} {'Config':<12} {'Baseline':<10} {'Optimized':<10} {'Speedup':<10}")
print("-" * 80)

for name, config, t_orig, t_opt, speedup in sorted(results, key=lambda x: -x[4]):
    marker = "✓ WIN" if speedup > 1.05 else "  SAME"
    print(f"{marker} {name:<18} {config:<12} {t_orig:<10.4f} {t_opt:<10.4f} {speedup:<10.2f}x")

print("-" * 80)
total_orig = sum(r[2] for r in results)
total_opt = sum(r[3] for r in results)
print(f"{'OVERALL':<18} {'':<12} {total_orig:<10.4f} {total_opt:<10.4f} {total_orig/total_opt:<10.2f}x")
print("=" * 80)

print()
print("TOP OPTIMIZATIONS:")
for name, config, t_orig, t_opt, speedup in sorted(results, key=lambda x: -x[4])[:5]:
    print(f"  {name:<22} {config:<12} {speedup:.2f}x faster ({t_orig:.4f}ms -> {t_opt:.4f}ms)")
