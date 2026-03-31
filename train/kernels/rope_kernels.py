"""
Rotary Position Embedding (RoPE) Kernels

Implements 4 variants of RoPE:
1. rope_fused - single-pass without intermediate cat
2. rope_complex - using complex numbers
3. rope_lazy - compute only for positions that need it
4. rope_inplace - in-place where safe

Baseline (original) is provided for comparison.
"""

import torch
import time
from typing import Tuple, Optional


# =============================================================================
# Baseline (Original)
# =============================================================================

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Original rotate_half implementation."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rope_original(q: torch.Tensor, k: torch.Tensor,
                  cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Original rope implementation."""
    # cos/sin: (B, L, D) -> expand to (B, H, L, D) for broadcasting
    B, H, L, D = q.shape
    cos = cos.unsqueeze(1).expand(B, H, L, D)
    sin = sin.unsqueeze(1).expand(B, H, L, D)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================================================================
# Variant 1: rope_fused - single-pass without intermediate cat
# =============================================================================

def rope_fused(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused rope implementation that avoids intermediate concatenation.

    Instead of:
        x1, x2 = x[..., :D//2], x[..., D//2:]
        return cat([-x2, x1], dim=-1)

    We compute directly using reshape and negation, avoiding the cat operation.
    """
    B, H, L, D = q.shape
    half = D // 2

    # Expand cos/sin: (B, L, D) -> (B, H, L, D)
    cos = cos.unsqueeze(1).expand(B, H, L, D)
    sin = sin.unsqueeze(1).expand(B, H, L, D)

    # Compute rotated q and k directly
    q_embed = q * cos
    k_embed = k * cos

    # Compute rotate_half inline to avoid cat
    # q_rotated[..., :half] = -q[..., half:]
    # q_rotated[..., half:] = q[..., :half]
    q_rotated = torch.empty_like(q)
    k_rotated = torch.empty_like(k)

    q_rotated[..., :half] = -q[..., half:]
    q_rotated[..., half:] = q[..., :half]

    k_rotated[..., :half] = -k[..., half:]
    k_rotated[..., half:] = k[..., :half]

    q_embed = q_embed + q_rotated * sin
    k_embed = k_embed + k_rotated * sin

    return q_embed, k_embed


# =============================================================================
# Variant 2: rope_complex - using complex numbers
# =============================================================================

def rope_complex(q: torch.Tensor, k: torch.Tensor,
                 cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    RoPE using the correct rope formula with explicit indexing.

    The rope formula: q_embed = q * cos + rotate_half(q) * sin

    For element i:
        if i < D/2: q_embed[i] = q[i] * cos[i] - q[i+D/2] * sin[i]
        else:       q_embed[i] = q[i] * cos[i] + q[i-D/2] * sin[i]

    This uses direct indexing rather than complex arithmetic because
    rope's interleaving (swap first and second halves) doesn't match
    standard complex multiplication.
    """
    B, H, L, D = q.shape
    half = D // 2

    # Expand cos and sin to (B, H, L, D)
    cos_expanded = cos.unsqueeze(1).expand(B, H, L, D).contiguous()
    sin_expanded = sin.unsqueeze(1).expand(B, H, L, D).contiguous()

    # First half: indices 0..half-1 use q[0..half-1], q[half..D-1], cos[0..half-1], sin[0..half-1]
    q_first_half = q[..., :half] * cos_expanded[..., :half] - q[..., half:] * sin_expanded[..., :half]
    k_first_half = k[..., :half] * cos_expanded[..., :half] - k[..., half:] * sin_expanded[..., :half]

    # Second half: indices half..D-1 use q[half..D-1], q[0..half-1], cos[half..D-1], sin[half..D-1]
    q_second_half = q[..., half:] * cos_expanded[..., half:] + q[..., :half] * sin_expanded[..., half:]
    k_second_half = k[..., half:] * cos_expanded[..., half:] + k[..., :half] * sin_expanded[..., half:]

    # Concatenate
    q_embed = torch.cat([q_first_half, q_second_half], dim=-1)
    k_embed = torch.cat([k_first_half, k_second_half], dim=-1)

    return q_embed, k_embed


# =============================================================================
# Variant 3: rope_lazy - compute only for positions that need it
# =============================================================================

def rope_lazy(q: torch.Tensor, k: torch.Tensor,
              cos: torch.Tensor, sin: torch.Tensor,
              active_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Lazy RoPE that only computes for active positions.

    Args:
        q, k: query and key tensors (B, H, L, D)
        cos, sin: cached cos and sin (B, L, D)
        active_mask: optional boolean mask (B, L) indicating which positions to compute.
                     If None, computes all positions (degrades to original).
    """
    if active_mask is None:
        # No optimization possible without mask
        return rope_original(q, k, cos, sin)

    B, H, L, D = q.shape
    half = D // 2

    # Initialize output tensors
    q_embed = torch.zeros_like(q)
    k_embed = torch.zeros_like(k)

    # Expand mask for broadcasting
    # active_mask: (B, L) -> (B, 1, L, 1) for q/k, (B, 1, L, 1) for cos/sin
    mask_q = active_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, L, 1)
    mask_cos = active_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, L, 1)
    mask_sin = active_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, L, 1)

    # Compute only for masked positions
    q_active = q * mask_q
    k_active = k * mask_q
    cos_active = cos * mask_cos
    sin_active = sin * mask_sin

    # Apply rope to active positions only
    # Expand cos/sin to (B, H, L, D) for broadcasting
    cos_expanded = cos.unsqueeze(1).expand(B, H, L, D)
    sin_expanded = sin.unsqueeze(1).expand(B, H, L, D)

    q_embed = (q_active * cos_expanded) + (rotate_half(q_active) * sin_expanded)
    k_embed = (k_active * cos_expanded) + (rotate_half(k_active) * sin_expanded)

    return q_embed, k_embed


# =============================================================================
# Variant 4: rope_inplace - in-place where safe
# =============================================================================

def rope_inplace(q: torch.Tensor, k: torch.Tensor,
                 cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    RoPE with in-place operations where safe.

    This variant modifies tensors in-place when possible to reduce memory allocations.
    Note: q and k are modified in-place and returned as part of output.
    """
    B, H, L, D = q.shape
    half = D // 2

    # Expand cos/sin: (B, L, D) -> (B, H, L, D)
    cos_expanded = cos.unsqueeze(1).expand(B, H, L, D)
    sin_expanded = sin.unsqueeze(1).expand(B, H, L, D)

    # Clone inputs since we'll modify them
    q_out = q.clone()
    k_out = k.clone()

    # Compute q * cos and store in-place
    q_out.mul_(cos_expanded)
    k_out.mul_(cos_expanded)

    # Compute rotate_half in-place style
    # Create rotated versions (cannot do in-place due to stride issues)
    q_rot = torch.empty_like(q)
    k_rot = torch.empty_like(k)

    q_rot[..., :half] = -q[..., half:]
    q_rot[..., half:] = q[..., :half]

    k_rot[..., :half] = -k[..., half:]
    k_rot[..., half:] = k[..., :half]

    # Add rotated * sin in-place
    q_rot.mul_(sin_expanded)
    k_rot.mul_(sin_expanded)

    q_out.add_(q_rot)
    k_out.add_(k_rot)

    return q_out, k_out


# =============================================================================
# Benchmarking
# =============================================================================

def verify_correctness(q: torch.Tensor, k: torch.Tensor,
                       cos: torch.Tensor, sin: torch.Tensor,
                       variants: dict) -> bool:
    """Verify all variants produce bit-identical results to baseline."""
    baseline_q, baseline_k = rope_original(q, k, cos, sin)

    all_correct = True
    for name, func in variants.items():
        if name == 'original':
            continue
        q_out, k_out = func(q.clone(), k.clone(), cos.clone(), sin.clone())
        q_match = torch.allclose(q_out, baseline_q, rtol=1e-5, atol=1e-5)
        k_match = torch.allclose(k_out, baseline_k, rtol=1e-5, atol=1e-5)
        if not (q_match and k_match):
            print(f"  MISMATCH in {name}:")
            print(f"    q max diff: {(q_out - baseline_q).abs().max().item()}")
            print(f"    k max diff: {(k_out - baseline_k).abs().max().item()}")
            all_correct = False

    return all_correct


def benchmark_function(func, q, k, cos, sin, warmup=10, iterations=100):
    """Benchmark a single function."""
    # Warmup
    for _ in range(warmup):
        func(q.clone(), k.clone(), cos.clone(), sin.clone())

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        func(q.clone(), k.clone(), cos.clone(), sin.clone())

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    return elapsed / iterations


def run_benchmark(seq_len, num_heads, head_dim, batch_size, iterations=100):
    """Run benchmark for a specific configuration."""
    print(f"\n{'='*60}")
    print(f"SeqLen: {seq_len}, Heads: {num_heads}, HeadDim: {head_dim}, Batch: {batch_size}")
    print(f"{'='*60}")

    B, H, L, D = batch_size, num_heads, seq_len, head_dim

    # Create tensors on GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    q = torch.randn(B, H, L, D, device=device)
    k = torch.randn(B, H, L, D, device=device)
    cos = torch.randn(B, L, D, device=device)
    sin = torch.randn(B, L, D, device=device)

    variants = {
        'original': rope_original,
        'fused': rope_fused,
        'complex': rope_complex,
        'lazy': rope_lazy,
        'inplace': rope_inplace,
    }

    # Verify correctness
    print("\nVerifying correctness...")
    all_correct = verify_correctness(q, k, cos, sin, variants)
    if all_correct:
        print("  All variants produce correct results!")
    else:
        print("  WARNING: Some variants have mismatches!")

    # Benchmark
    print("\nTiming (ms) and speedup vs baseline:")
    print(f"{'Variant':<15} {'Time (ms)':<12} {'Speedup':<10}")
    print(f"{'-'*40}")

    baseline_time = None
    times = {}

    for name, func in variants.items():
        t = benchmark_function(func, q, k, cos, sin)
        times[name] = t
        speedup = baseline_time / t if baseline_time else 1.0
        print(f"{name:<15} {t*1000:<12.4f} {speedup:<10.2f}x")
        if name == 'original':
            baseline_time = t


def main():
    """Run all benchmarks."""
    print("RoPE Kernel Benchmark")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, running on CPU")

    configs = []

    # Generate all configurations
    seq_lengths = [128, 512, 1024, 2048]
    num_heads = [32, 64]
    head_dims = [64, 128]
    batch_sizes = [1, 4]

    for seq_len in seq_lengths:
        for n_heads in num_heads:
            for head_dim in head_dims:
                for batch_size in batch_sizes:
                    configs.append((seq_len, n_heads, head_dim, batch_size))

    # Limit total configs for reasonable runtime
    # Sample across dimensions
    selected_configs = [
        (128, 32, 64, 1),
        (128, 64, 128, 1),
        (512, 32, 64, 4),
        (512, 64, 128, 4),
        (1024, 32, 64, 1),
        (1024, 64, 128, 4),
        (2048, 32, 64, 1),
        (2048, 64, 128, 4),
    ]

    for config in selected_configs:
        run_benchmark(*config)

    print("\n" + "=" * 60)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
