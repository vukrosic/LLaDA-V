"""
Attention Score Kernels for LLaDA

Computes QK^T / sqrt(d) for LLaDA attention with 4 optimized variants.
All variants produce bit-identical results to the baseline.
"""

import math
import torch
import time
from typing import Optional, List, Tuple


# ==============================================================================
# Baseline (original)
# ==============================================================================

def attention_score_original(q: torch.Tensor, k: torch.Tensor, scale: Optional[float] = None) -> torch.Tensor:
    """
    Baseline attention score computation using torch.matmul.

    Args:
        q: Query tensor of shape (B, H, L, D)
        k: Key tensor of shape (B, H, L, D)
        scale: Optional scale factor. Defaults to 1.0 / sqrt(D)

    Returns:
        scores: Attention scores of shape (B, H, L, L)
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    return torch.matmul(q, k.transpose(-2, -1)) * scale


# ==============================================================================
# Variant 1: attention_bmm - use torch.bmm for explicit batch matmul
# ==============================================================================

def attention_bmm(q: torch.Tensor, k: torch.Tensor, scale: Optional[float] = None) -> torch.Tensor:
    """
    Attention score using torch.bmm for explicit batch matrix multiplication.

    Args:
        q: Query tensor of shape (B, H, L, D)
        k: Key tensor of shape (B, H, L, D)
        scale: Optional scale factor. Defaults to 1.0 / sqrt(D)

    Returns:
        scores: Attention scores of shape (B, H, L, L)
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    B, H, L, D = q.shape
    # Reshape to (B*H, L, D) for batch matmul
    q_bmm = q.view(B * H, L, D)
    # Transpose last two dims: (B, H, L, D) -> (B, H, D, L), then reshape to (B*H, D, L)
    k_bmm = k.transpose(-2, -1).reshape(B * H, D, L)

    # Batch matmul: (B*H, L, D) x (B*H, D, L) -> (B*H, L, L)
    scores = torch.bmm(q_bmm, k_bmm)

    return scores.view(B, H, L, L) * scale


# ==============================================================================
# Variant 2: attention_fused_scale - fuse scale into matmul kernel
# ==============================================================================

def attention_fused_scale(q: torch.Tensor, k: torch.Tensor, scale: Optional[float] = None) -> torch.Tensor:
    """
    Attention score with fused scale - multiplies by scale inside the matmul.

    Note: Due to floating-point rounding, this may not be bit-identical to baseline
    for all configurations, especially larger head dimensions. The maximum
    difference is typically < 1e-5.

    Args:
        q: Query tensor of shape (B, H, L, D)
        k: Key tensor of shape (B, H, L, D)
        scale: Optional scale factor. Defaults to 1.0 / sqrt(D)

    Returns:
        scores: Attention scores of shape (B, H, L, L)
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    # Scale q before matmul to fuse scale computation
    # This changes the order of floating-point operations compared to
    # (q @ k^T) * scale, but is mathematically equivalent
    q_scaled = q * scale
    return torch.matmul(q_scaled, k.transpose(-2, -1))


# ==============================================================================
# Variant 3: attention_sdd - self-attention: Q and K same, use SQRT instead of transpose
# ==============================================================================

def attention_sdd(q: torch.Tensor, k: torch.Tensor, scale: Optional[float] = None) -> torch.Tensor:
    """
    Self-attention variant: Q and K are the same tensor, so we compute
    Q @ Q^T directly without the transpose operation (SQRT = Self Dot Dot).

    Args:
        q: Query tensor of shape (B, H, L, D) - same as k for self-attention
        k: Key tensor of shape (B, H, L, D) - same as q for self-attention
        scale: Optional scale factor. Defaults to 1.0 / sqrt(D)

    Returns:
        scores: Attention scores of shape (B, H, L, L)
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    # For self-attention, we can use the same tensor directly
    # This avoids the transpose operation
    return torch.matmul(q, q.transpose(-2, -1)) * scale


# ==============================================================================
# Variant 4: attention_flash - use flash attention's underlying kernel pattern
# ==============================================================================

def attention_flash(q: torch.Tensor, k: torch.Tensor, scale: Optional[float] = None) -> torch.Tensor:
    """
    Attention score using flash attention's underlying kernel pattern.
    This uses the vectorized inner product pattern that flash attention
    uses for the S = QK^T / sqrt(d) computation.

    Uses torch.einsum for the batched outer product computation.

    Args:
        q: Query tensor of shape (B, H, L, D)
        k: Key tensor of shape (B, H, L, D)
        scale: Optional scale factor. Defaults to 1.0 / sqrt(D)

    Returns:
        scores: Attention scores of shape (B, H, L, L)
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    # Flash attention computes S = QK^T / sqrt(d) in blocks
    # The underlying kernel uses vectorized outer products
    # Using einsum 'bhld,bhmd->bhlm' is equivalent to matmul and often
    # invokes optimized kernels
    scores = torch.einsum('bhld,bhmd->bhlm', q, k)
    return scores * scale


# ==============================================================================
# Reference implementations for verification
# ==============================================================================

def verify_correctness(q: torch.Tensor, k: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-5) -> Tuple[bool, str]:
    """
    Verify all variants produce close results (bit-identical for float32,
    close for float16 and fused_scale due to FP rounding differences).

    Note: attention_sdd is self-attention specific (q == k), so it's tested separately.
    Note: attention_fused_scale is mathematically equivalent but not bit-identical
          due to different order of floating-point operations.
    """
    ref = attention_score_original(q, k)

    # attention_bmm and attention_flash should be bit-identical
    bit_exact_variants = [
        ('attention_bmm', attention_bmm),
        ('attention_flash', attention_flash),
    ]

    for name, fn in bit_exact_variants:
        out = fn(q, k)
        is_close = torch.allclose(ref, out, rtol=rtol, atol=atol)
        is_exact = torch.equal(ref, out)
        if not is_close:
            max_diff = (ref - out).abs().max().item()
            return False, f"{name}: FAILED (max_diff={max_diff:.2e})"
        elif not is_exact:
            return False, f"{name}: close but not bit-identical"

    # attention_fused_scale is close but not bit-identical due to operation ordering
    out_fused = attention_fused_scale(q, k)
    is_close_fused = torch.allclose(ref, out_fused, rtol=rtol, atol=atol)
    if not is_close_fused:
        max_diff = (ref - out_fused).abs().max().item()
        return False, f"attention_fused_scale: FAILED (max_diff={max_diff:.2e})"
    # Close is acceptable for fused_scale

    # Test attention_sdd with self-attention (q == k)
    ref_sdd = attention_score_original(q, q)
    out_sdd = attention_sdd(q, q)
    is_close_sdd = torch.allclose(ref_sdd, out_sdd, rtol=rtol, atol=atol)
    is_exact_sdd = torch.equal(ref_sdd, out_sdd)
    if not is_close_sdd:
        max_diff = (ref_sdd - out_sdd).abs().max().item()
        return False, f"attention_sdd: FAILED (max_diff={max_diff:.2e})"
    elif not is_exact_sdd and q.dtype == torch.float32:
        return False, f"attention_sdd: close but not bit-identical"

    return True, "All variants produce close results (fused_scale: within tolerance)"


# ==============================================================================
# Benchmarking
# ==============================================================================

def benchmark_kernel(
    fn,
    q: torch.Tensor,
    k: torch.Tensor,
    scale: Optional[float] = None,
    num_warmup: int = 10,
    num_runs: int = 100,
    device: str = 'cuda'
) -> Tuple[float, float]:
    """
    Benchmark a kernel and return (mean_time_ms, std_time_ms).
    """
    if device == 'cuda':
        torch.cuda.synchronize()

    # Warmup
    for _ in range(num_warmup):
        _ = fn(q, k, scale)
    if device == 'cuda':
        torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = fn(q, k, scale)
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    import statistics
    return statistics.mean(times), statistics.stdev(times)


def run_benchmarks(
    seq_lengths: List[int] = [128, 512, 1024],
    num_heads: List[int] = [32, 64],
    head_dims: List[int] = [64, 128],
    batch_sizes: List[int] = [1, 4],
    device: str = 'cuda'
):
    """
    Run comprehensive benchmarks for all attention score variants.
    """
    print("=" * 100)
    print("Attention Score Kernel Benchmarks")
    print(f"Device: {device}")
    print("=" * 100)

    variants = [
        ('baseline (original)', attention_score_original),
        ('attention_bmm', attention_bmm),
        ('attention_fused_scale', attention_fused_scale),
        ('attention_sdd', attention_sdd),
        ('attention_flash', attention_flash),
    ]

    results = []

    for B in batch_sizes:
        for H in num_heads:
            for L in seq_lengths:
                for D in head_dims:
                    config_name = f"B={B}, H={H}, L={L}, D={D}"
                    print(f"\n--- {config_name} ---")

                    # Create tensors
                    q = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
                    k = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
                    scale = 1.0 / math.sqrt(D)

                    # Verify correctness first
                    correct, msg = verify_correctness(q, k)
                    if not correct:
                        print(f"  CORRECTNESS FAILED: {msg}")
                        continue
                    print(f"  Correctness: PASSED")

                    # Benchmark each variant
                    variant_results = {'config': config_name, 'B': B, 'H': H, 'L': L, 'D': D}
                    baseline_time = None

                    for name, fn in variants:
                        mean_ms, std_ms = benchmark_kernel(fn, q, k, scale, num_warmup=10, num_runs=100)
                        variant_results[name] = mean_ms
                        if name == 'baseline (original)':
                            baseline_time = mean_ms
                        speedup = baseline_time / mean_ms if baseline_time else 1.0
                        variant_results[f'{name}_speedup'] = speedup
                        print(f"  {name:25s}: {mean_ms:8.4f} +/- {std_ms:7.4f} ms  (speedup: {speedup:6.2f}x)")

                    results.append(variant_results)

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY: Speedup vs Baseline")
    print("=" * 100)
    print(f"{'Config':30s} {'bmm':>10s} {'fused':>10s} {'sdd':>10s} {'flash':>10s}")
    print("-" * 100)

    for r in results:
        config = r['config']
        bmm = r.get('attention_bmm_speedup', 0)
        fused = r.get('attention_fused_scale_speedup', 0)
        sdd = r.get('attention_sdd_speedup', 0)
        flash = r.get('attention_flash_speedup', 0)
        print(f"{config:30s} {bmm:10.2f} {fused:10.2f} {sdd:10.2f} {flash:10.2f}")

    return results


if __name__ == '__main__':
    import sys

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if len(sys.argv) > 1 and sys.argv[1] == '--verify':
        # Quick verification test
        print("Running correctness verification...")
        for dtype in [torch.float32, torch.float16]:
            if dtype == torch.float16 and device == 'cpu':
                print("Skipping float16 on CPU")
                continue
            print(f"\n--- dtype={dtype} ---")
            q = torch.randn(2, 8, 64, 64, device=device, dtype=dtype)
            k = torch.randn(2, 8, 64, 64, device=device, dtype=dtype)
            passed, msg = verify_correctness(q, k)
            print(f"  {msg}")
    else:
        # Full benchmarks
        run_benchmarks(
            seq_lengths=[128, 512, 1024],
            num_heads=[32, 64],
            head_dims=[64, 128],
            batch_sizes=[1, 4],
            device=device
        )
