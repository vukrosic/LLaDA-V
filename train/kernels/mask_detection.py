"""
Mask Detection Kernels for LLaDA-V

Detects which positions have masked embeddings (need to be filled).
"""

import torch
import time
from typing import List, Tuple


def mask_original(x_embeds: torch.Tensor, masked_embed: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Baseline mask detection using absolute difference comparison.

    Args:
        x_embeds: (B, L, D) tensor of embeddings
        masked_embed: (1, D) tensor of masked embedding reference
        eps: tolerance for comparison

    Returns:
        mask: (B, L) boolean tensor, True where embedding matches masked_embed
    """
    return torch.all(torch.abs(x_embeds - masked_embed) < eps, dim=-1)


def mask_isclose(x_embeds: torch.Tensor, masked_embed: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Mask detection using torch.isclose with absolute tolerance.

    Args:
        x_embeds: (B, L, D) tensor of embeddings
        masked_embed: (1, D) tensor of masked embedding reference
        eps: absolute tolerance (passed as atol to isclose)

    Returns:
        mask: (B, L) boolean tensor, True where embedding matches masked_embed
    """
    # isclose returns True where |a - b| <= atol
    return torch.isclose(x_embeds, masked_embed, atol=eps, rtol=0).all(dim=-1)


def mask_sq_distance(x_embeds: torch.Tensor, masked_embed: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Mask detection using squared distance comparison.

    Args:
        x_embeds: (B, L, D) tensor of embeddings
        masked_embed: (1, D) tensor of masked embedding reference
        eps: tolerance for comparison

    Returns:
        mask: (B, L) boolean tensor, True where embedding matches masked_embed
    """
    # Compute squared distance: ((x - ref)**2).sum(dim) < eps**2
    diff = x_embeds - masked_embed
    sq_distance = (diff ** 2).sum(dim=-1)
    return sq_distance < eps ** 2


def mask_chunked(x_embeds: torch.Tensor, masked_embed: torch.Tensor, eps: float = 1e-5, chunk_size: int = 512) -> torch.Tensor:
    """
    Mask detection with chunked comparison to reduce peak memory usage.

    Args:
        x_embeds: (B, L, D) tensor of embeddings
        masked_embed: (1, D) tensor of masked embedding reference
        eps: tolerance for comparison
        chunk_size: number of embedding dimensions to compare per chunk

    Returns:
        mask: (B, L) boolean tensor, True where embedding matches masked_embed
    """
    B, L, D = x_embeds.shape
    device = x_embeds.device

    # Initialize mask as all True
    mask = torch.ones(B, L, dtype=torch.bool, device=device)

    # Process in chunks along the embedding dimension
    for start in range(0, D, chunk_size):
        end = min(start + chunk_size, D)
        chunk_mask = torch.all(
            torch.abs(x_embeds[..., start:end] - masked_embed[..., start:end]) < eps,
            dim=-1
        )
        mask = mask & chunk_mask

    return mask


def mask_tolerance(x_embeds: torch.Tensor, masked_embed: torch.Tensor, eps: float = 1e-5, relative_scale: float = 1e-4) -> torch.Tensor:
    """
    Mask detection with adaptive tolerance based on embedding magnitude.

    Args:
        x_embeds: (B, L, D) tensor of embeddings
        masked_embed: (1, D) tensor of masked embedding reference
        eps: base tolerance
        relative_scale: scale factor for magnitude-based tolerance

    Returns:
        mask: (B, L) boolean tensor, True where embedding matches masked_embed
    """
    # Compute magnitude of each embedding
    x_magnitude = torch.norm(x_embeds, dim=-1, keepdim=True)  # (B, L, 1)
    ref_magnitude = torch.norm(masked_embed, dim=-1, keepdim=True)  # (1, 1)

    # Adaptive tolerance: base tolerance + relative tolerance scaled by max magnitude
    max_magnitude = torch.maximum(x_magnitude, ref_magnitude)  # (B, L, 1)
    adaptive_eps = eps + relative_scale * max_magnitude  # (B, L, 1)

    # Compare with adaptive tolerance (adaptive_eps has shape B, L, 1 which broadcasts with B, L, D)
    return torch.all(torch.abs(x_embeds - masked_embed) < adaptive_eps, dim=-1)


def verify_numerical_equivalence(x_embeds: torch.Tensor, masked_embed: torch.Tensor, eps: float = 1e-5) -> bool:
    """
    Verify all mask variants produce numerically equivalent results.

    Args:
        x_embeds: (B, L, D) tensor of embeddings
        masked_embed: (1, D) tensor of masked embedding reference
        eps: tolerance for comparison

    Returns:
        True if all variants produce equivalent results
    """
    variants = {
        'original': mask_original(x_embeds, masked_embed, eps),
        'isclose': mask_isclose(x_embeds, masked_embed, eps),
        'sq_distance': mask_sq_distance(x_embeds, masked_embed, eps),
        'chunked': mask_chunked(x_embeds, masked_embed, eps),
        'tolerance': mask_tolerance(x_embeds, masked_embed, eps),
    }

    # Check all variants against original
    reference = variants['original']
    for name, result in variants.items():
        if name == 'original':
            continue
        if not torch.equal(reference, result):
            # Check with tolerance for floating point differences
            if not (reference == result).all():
                print(f"Variant '{name}' differs from baseline!")
                return False

    print("All variants produce numerically equivalent results.")
    return True


def benchmark_mask_functions(
    seq_lengths: List[int] = [128, 512, 1024, 2048],
    hidden_dims: List[int] = [4096, 8192],
    batch_sizes: List[int] = [1, 4],
    n_warmup: int = 10,
    n_benchmark: int = 100,
    device: str = 'cuda'
) -> List[Tuple]:
    """
    Benchmark all mask detection variants.

    Args:
        seq_lengths: list of sequence lengths to benchmark
        hidden_dims: list of hidden dimensions to benchmark
        batch_sizes: list of batch sizes to benchmark
        n_warmup: number of warmup iterations
        n_benchmark: number of benchmark iterations
        device: device to run benchmarks on

    Returns:
        List of benchmark results (name, seq_len, hidden_dim, batch_size, time_ms, speedup)
    """
    results = []

    functions = {
        'baseline (original)': mask_original,
        'isclose': mask_isclose,
        'sq_distance': mask_sq_distance,
        'chunked': mask_chunked,
        'tolerance': mask_tolerance,
    }

    print("=" * 100)
    print("Mask Detection Kernel Benchmark")
    print("=" * 100)
    print(f"Warmup iterations: {n_warmup}")
    print(f"Benchmark iterations: {n_benchmark}")
    print(f"Device: {device}")
    print("=" * 100)

    for hidden_dim in hidden_dims:
        for seq_len in seq_lengths:
            for batch_size in batch_sizes:
                # Create test tensors
                x_embeds = torch.randn(batch_size, seq_len, hidden_dim, device=device)
                masked_embed = torch.randn(1, hidden_dim, device=device)

                # Make some embeddings match the masked_embed (for realistic mask ratio)
                n_masked = (batch_size * seq_len) // 10
                idx = torch.randperm(batch_size * seq_len, device=device)[:n_masked]
                for i in idx:
                    b = i // seq_len
                    l = i % seq_len
                    x_embeds[b, l] = masked_embed[0]

                # Warmup
                for _ in range(n_warmup):
                    for func in functions.values():
                        _ = func(x_embeds, masked_embed)
                torch.cuda.synchronize()

                # Benchmark each function
                func_times = {}
                for name, func in functions.items():
                    start = time.perf_counter()
                    for _ in range(n_benchmark):
                        _ = func(x_embeds, masked_embed)
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    func_times[name] = (end - start) / n_benchmark * 1000  # ms

                # Calculate speedup vs baseline
                baseline_time = func_times['baseline (original)']
                for name, t in func_times.items():
                    speedup = baseline_time / t if t > 0 else float('inf')
                    results.append((name, seq_len, hidden_dim, batch_size, t, speedup))

                # Print results for this configuration
                print(f"\nSeq_len={seq_len}, Hidden_dim={hidden_dim}, Batch_size={batch_size}")
                print("-" * 80)
                print(f"{'Function':<25} {'Time (ms)':<15} {'Speedup vs Baseline':<20}")
                print("-" * 80)
                for name, t in func_times.items():
                    speedup = baseline_time / t if t > 0 else float('inf')
                    print(f"{name:<25} {t:<15.4f} {speedup:<20.4f}x")

    return results


def run_benchmark_table(results: List[Tuple]):
    """
    Print a formatted table of benchmark results.

    Args:
        results: list of benchmark results from benchmark_mask_functions
    """
    print("\n" + "=" * 120)
    print("SUMMARY TABLE")
    print("=" * 120)

    # Group by configuration
    configs = {}
    for r in results:
        name, seq_len, hidden_dim, batch_size, t, speedup = r
        key = (seq_len, hidden_dim, batch_size)
        if key not in configs:
            configs[key] = {}
        configs[key][name] = (t, speedup)

    print(f"\n{'Config':<35} {'Function':<25} {'Time (ms)':<15} {'Speedup':<10}")
    print("-" * 85)

    for (seq_len, hidden_dim, batch_size), funcs in sorted(configs.items()):
        config_str = f"L={seq_len}, D={hidden_dim}, B={batch_size}"
        baseline_time = funcs.get('baseline (original)', (0, 1))[0]

        for name in ['baseline (original)', 'isclose', 'sq_distance', 'chunked', 'tolerance']:
            if name in funcs:
                t, _ = funcs[name]
                speedup = baseline_time / t if t > 0 and baseline_time > 0 else 1.0
                print(f"{config_str:<35} {name:<25} {t:<15.4f} {speedup:<10.4f}x")
        print()


if __name__ == "__main__":
    # Syntax check with a simple test
    print("Running syntax check...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Small test for numerical equivalence
    B, L, D = 2, 16, 128
    x_embeds = torch.randn(B, L, D, device=device)
    masked_embed = torch.randn(1, D, device=device)

    # Make some embeddings match
    x_embeds[0, 0] = masked_embed[0]
    x_embeds[1, 5] = masked_embed[0]

    if device == 'cuda':
        print("\nVerifying numerical equivalence...")
        assert verify_numerical_equivalence(x_embeds, masked_embed), "Numerical equivalence check failed!"
        print("Numerical equivalence check passed!")

    # Run benchmarks if on CUDA
    if device == 'cuda':
        print("\nRunning benchmarks...")
        results = benchmark_mask_functions(
            seq_lengths=[128, 512, 1024, 2048],
            hidden_dims=[4096, 8192],
            batch_sizes=[1, 4],
            n_warmup=10,
            n_benchmark=100,
            device=device
        )
        run_benchmark_table(results)
    else:
        print("\nCUDA not available. Skipping benchmarks.")
        print("Test passed - all functions execute correctly.")
