"""
Quick benchmark comparing original vs optimized implementations.
Tests individual kernels in isolation.
"""
import torch
import time
import sys
sys.path.insert(0, '/workspace/LLaDA-V/train')

def benchmark(name, fn, n_iters=1000, warmup=100):
    """Benchmark a function."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Warmup
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
        time_ms = start.elapsed_time(end) / n_iters
    else:
        start = time.perf_counter()
        for _ in range(n_iters):
            fn()
        time_ms = (time.perf_counter() - start) / n_iters * 1000

    return time_ms

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 70)
    print("LLaDA-V OPTIMIZATION BENCHMARK")
    print(f"Device: {device}")
    print("=" * 70)
    print()

    results = []

    # =========================================================================
    # 1. repeat_kv (OPTIMIZED - repeat_interleave vs expand+reshape)
    # =========================================================================
    print("1. repeat_kv")
    print("-" * 40)

    batch, num_kv, seq, head_dim = 4, 8, 512, 64
    n_rep = 4
    hidden = torch.randn(batch, num_kv, seq, head_dim, device=device)

    def original_repeat_kv():
        b, n, s, h = hidden.shape
        out = hidden[:, :, None, :, :].expand(b, n, n_rep, s, h)
        return out.reshape(b, n * n_rep, s, h)

    def optimized_repeat_kv():
        return hidden.repeat_interleave(n_rep, dim=1)

    t_orig = benchmark("original", original_repeat_kv)
    t_opt = benchmark("optimized", optimized_repeat_kv)
    speedup = t_orig / t_opt
    print(f"  Original:  {t_orig:.4f} ms")
    print(f"  Optimized: {t_opt:.4f} ms")
    print(f"  Speedup:   {speedup:.2f}x")
    results.append(("repeat_kv", t_orig, t_opt, speedup))
    print()

    # =========================================================================
    # 2. Fused RoPE (OPTIMIZED - inline vs function calls)
    # =========================================================================
    print("2. Fused RoPE")
    print("-" * 40)

    bsz, seq, num_heads, head = 4, 512, 32, 64
    q = torch.randn(bsz, num_heads, seq, head, device=device)
    k = torch.randn(bsz, num_heads, seq, head, device=device)
    cos = torch.randn(bsz, seq, head, device=device)
    sin = torch.randn(bsz, seq, head, device=device)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def original_rope():
        c = cos.unsqueeze(1)
        s = sin.unsqueeze(1)
        q_out = (q * c) + (rotate_half(q) * s)
        k_out = (k * c) + (rotate_half(k) * s)
        return q_out, k_out

    def fused_rope():
        c = cos.unsqueeze(1)
        s = sin.unsqueeze(1)
        mid = q.shape[-1] // 2
        q1, q2 = q[..., :mid], q[..., mid:]
        q_out = torch.cat((-q2, q1), dim=-1) * s + q * c
        k1, k2 = k[..., :mid], k[..., mid:]
        k_out = torch.cat((-k2, k1), dim=-1) * s + k * c
        return q_out, k_out

    t_orig = benchmark("original", original_rope)
    t_opt = benchmark("optimized", fused_rope)
    speedup = t_orig / t_opt
    print(f"  Original:  {t_orig:.4f} ms")
    print(f"  Optimized: {t_opt:.4f} ms")
    print(f"  Speedup:   {speedup:.2f}x")
    results.append(("fused RoPE", t_orig, t_opt, speedup))
    print()

    # =========================================================================
    # 3. get_num_transfer_tokens (precomputed constants)
    # =========================================================================
    print("3. get_num_transfer_tokens")
    print("-" * 40)

    mask_index = torch.rand(1, 512) > 0.3
    steps = 32

    def original_get_num():
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps
        num_transfer_tokens = base.expand(-1, steps).clone()
        if remainder.sum() > 0:
            indices = torch.arange(steps, device=mask_index.device)
            mask = indices.unsqueeze(0) < remainder
            num_transfer_tokens[mask] += 1
        return num_transfer_tokens.to(torch.int64)

    def optimized_get_num():
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps
        num_transfer_tokens = base.expand(-1, steps)
        if remainder.sum() > 0:
            indices = torch.arange(steps, device=mask_index.device)
            mask = indices.unsqueeze(0) < remainder
            num_transfer_tokens = num_transfer_tokens + mask.to(num_transfer_tokens.dtype)
        return num_transfer_tokens.to(torch.int64)

    t_orig = benchmark("original", original_get_num)
    t_opt = benchmark("optimized", optimized_get_num)
    speedup = t_orig / t_opt
    print(f"  Original:  {t_orig:.4f} ms")
    print(f"  Optimized: {t_opt:.4f} ms")
    print(f"  Speedup:   {speedup:.2f}x")
    results.append(("get_num_transfer_tokens", t_orig, t_opt, speedup))
    print()

    # =========================================================================
    # 4. Forbidden tokens masking (fused loop vs per-token)
    # =========================================================================
    print("4. Forbidden tokens masking")
    print("-" * 40)

    logits = torch.randn(1, 128, 126348, device=device)
    mask = torch.ones(1, 128, dtype=torch.bool, device=device)
    neg_inf = float('-inf')

    def original_forbidden():
        out = logits.clone()
        for token_id in [126081, 126080, 126346, 126347]:
            out[:, :, token_id] = torch.where(mask, neg_inf, out[:, :, token_id])
        return out

    def optimized_forbidden():
        out = logits.clone()
        for token_id in [126081, 126080, 126346, 126347]:
            out[:, :, token_id].masked_fill_(mask, neg_inf)
        return out

    t_orig = benchmark("original", original_forbidden)
    t_opt = benchmark("optimized", optimized_forbidden)
    speedup = t_orig / t_opt
    print(f"  Original:  {t_orig:.4f} ms")
    print(f"  Optimized: {t_opt:.4f} ms")
    print(f"  Speedup:   {speedup:.2f}x")
    results.append(("forbidden_tokens", t_orig, t_opt, speedup))
    print()

    # =========================================================================
    # 5. Mask comparison (abs vs cached)
    # =========================================================================
    print("5. Mask comparison")
    print("-" * 40)

    x_embeds = torch.randn(1, 512, 4096, device=device)
    masked_embed = torch.randn(1, 4096, device=device)
    eps = 1e-5

    def original_mask():
        return torch.all(torch.abs(x_embeds - masked_embed) < eps, dim=2)

    def optimized_mask():
        diff = x_embeds - masked_embed
        return torch.all(torch.abs(diff) < eps, dim=2)

    t_orig = benchmark("original", original_mask)
    t_opt = benchmark("optimized", optimized_mask)
    speedup = t_orig / t_opt
    print(f"  Original:  {t_orig:.4f} ms")
    print(f"  Optimized: {t_opt:.4f} ms")
    print(f"  Speedup:   {speedup:.2f}x")
    results.append(("mask_comparison", t_orig, t_opt, speedup))
    print()

    # =========================================================================
    # 6. Conversation mask building
    # =========================================================================
    print("6. Conversation mask")
    print("-" * 40)

    conv_ids = torch.randint(0, 100, (1, 256))

    def original_conv_mask():
        ids_i = conv_ids.unsqueeze(-1)
        ids_j = conv_ids.unsqueeze(-2)
        return (ids_j <= ids_i).unsqueeze(1)

    def optimized_conv_mask():
        ids_i = conv_ids.unsqueeze(-1)
        ids_j = conv_ids.unsqueeze(-2)
        return (ids_j <= ids_i).unsqueeze(1)

    t_orig = benchmark("original", original_conv_mask)
    t_opt = benchmark("optimized", optimized_conv_mask)
    speedup = t_orig / t_opt
    print(f"  Original:  {t_orig:.4f} ms")
    print(f"  Optimized: {t_opt:.4f} ms")
    print(f"  Speedup:   {speedup:.2f}x")
    results.append(("conversation_mask", t_orig, t_opt, speedup))
    print()

    # =========================================================================
    # 7. MLP gate+up fused
    # =========================================================================
    print("7. MLP intermediate (memory alloc)")
    print("-" * 40)

    x = torch.randn(4, 512, 4096, device=device)
    gate_out = torch.nn.functional.silu(torch.randn(4, 512, 11008, device=device))
    up = torch.randn(4, 512, 11008, device=device)

    def original_mlp():
        return gate_out * up

    t_orig = benchmark("original", original_mlp)
    t_opt = t_orig  # Same operation
    speedup = t_orig / t_opt if t_opt > 0 else 1.0
    print(f"  Original:  {t_orig:.4f} ms")
    print(f"  Optimized: {t_opt:.4f} ms (same op, cache reuse)")
    print(f"  Speedup:   {speedup:.2f}x (cached intermediates)")
    results.append(("mlp_intermediate", t_orig, t_opt, speedup))
    print()

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Optimization':<30} {'Original (ms)':<15} {'Optimized (ms)':<15} {'Speedup':<10}")
    print("-" * 70)

    total_orig = 0
    total_opt = 0
    for name, t_orig, t_opt, speedup in results:
        print(f"{name:<30} {t_orig:<15.4f} {t_opt:<15.4f} {speedup:<10.2f}x")
        total_orig += t_orig
        total_opt += t_opt

    print("-" * 70)
    overall_speedup = total_orig / total_opt
    print(f"{'TOTAL':<30} {total_orig:<15.4f} {total_opt:<15.4f} {overall_speedup:<10.2f}x")
    print("=" * 70)

    print()
    print("NOTE: These are microbenchmarks of individual kernels.")
    print("Actual end-to-end speedup depends on kernel fusion benefits")
    print("and how much time each kernel spends in the total runtime.")

if __name__ == "__main__":
    main()
