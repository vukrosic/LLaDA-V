"""
Top-k Selection Kernels for LLaDA-V

Implements 4 variants of topk_selection kernels for selecting top-k tokens
based on confidence scores, plus benchmarking utilities.
"""

import torch
import time
from typing import Union, List, Tuple, Optional


# =============================================================================
# BASELINE (original loop-based implementation)
# =============================================================================

def topk_loop(confidence: torch.Tensor, k: Union[int, torch.Tensor]) -> torch.Tensor:
    """
    Baseline: Loop through batch and select top-k for each sample.

    Args:
        confidence: (B, L) tensor of confidence scores
        k: int or (B,) tensor of number of tokens to select

    Returns:
        transfer_index: (B, L) bool tensor, True for selected tokens
    """
    batch_size = confidence.shape[0]
    seq_len = confidence.shape[1]
    transfer_index = torch.zeros_like(confidence, dtype=torch.bool)

    if isinstance(k, int):
        k = torch.full((batch_size,), k, device=confidence.device, dtype=torch.long)

    for j in range(batch_size):
        _, select_index = torch.topk(confidence[j], k=k[j].item())
        transfer_index[j, select_index] = True

    return transfer_index


# =============================================================================
# VARIANT 1: Vectorized using scatter
# =============================================================================

def topk_vectorized(confidence: torch.Tensor, k: Union[int, torch.Tensor]) -> torch.Tensor:
    """
    Vectorized top-k selection using scatter operations.

    For uniform k: uses direct topk on 2D tensor
    For non-uniform k: uses argsort + scatter approach

    Args:
        confidence: (B, L) tensor of confidence scores
        k: int or (B,) tensor of number of tokens to select

    Returns:
        transfer_index: (B, L) bool tensor, True for selected tokens
    """
    batch_size, seq_len = confidence.shape

    if isinstance(k, int):
        # Fast path: uniform k across batch
        _, topk_indices = torch.topk(confidence, k=k, dim=1)
        transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
        transfer_index.scatter_(1, topk_indices, True)
        return transfer_index

    # Non-uniform k: vectorized approach using scatter
    # Get maximum k to compute topk for all rows
    max_k = k.max().item()

    # Compute topk indices for all rows (padded with extra indices)
    _, all_topk_indices = torch.topk(confidence, k=max_k, dim=1)

    # Create batch indices: [[0,0,0,...], [1,1,1,...], ...]
    batch_idx = torch.arange(batch_size, device=confidence.device)
    batch_idx = batch_idx[:, None].expand(-1, max_k)

    # Create mask: True for indices where j < k[batch]
    # k_expanded: (B, max_k) where each row is [0, 1, 2, ..., max_k-1]
    k_expanded = torch.arange(max_k, device=confidence.device)[None, :].expand(batch_size, -1)
    k_target = k[:, None].expand(-1, max_k)  # (B, max_k)
    valid_mask = k_expanded < k_target  # (B, max_k)

    # Scatter to set True only for valid topk positions
    transfer_index = torch.zeros((batch_size, seq_len), device=confidence.device, dtype=torch.bool)
    transfer_index[batch_idx[valid_mask], all_topk_indices[valid_mask]] = True

    return transfer_index


# =============================================================================
# VARIANT 2: Cumsum-based selection for uniform k
# =============================================================================

def topk_cumsum(confidence: torch.Tensor, k: Union[int, torch.Tensor]) -> torch.Tensor:
    """
    Cumsum-based top-k selection for uniform k.

    Uses sorting + cumsum to efficiently select top-k by computing
    the threshold value and marking all values above it.

    Args:
        confidence: (B, L) tensor of confidence scores
        k: int (must be uniform across batch)

    Returns:
        transfer_index: (B, L) bool tensor, True for selected tokens
    """
    batch_size, seq_len = confidence.shape

    if isinstance(k, int):
        k_val = k
    else:
        k_val = k[0].item()
        assert (k == k_val).all(), "topk_cumsum requires uniform k"

    # Sort along last dimension (descending)
    sorted_conf, _ = torch.sort(confidence, dim=1, descending=True)

    # Get the k-th highest value (threshold)
    # Handle edge case where k might be larger than seq_len
    k_safe = min(k_val, seq_len)
    threshold = sorted_conf[:, k_safe - 1]  # (B,)

    # Select all tokens with confidence >= threshold
    # But we need exactly k tokens, so we need to handle ties carefully
    # Use cumsum approach: count how many values in sorted order are >= threshold
    # Then adjust to get exactly k

    # Simple approach: mark all >= threshold
    transfer_index = confidence >= threshold[:, None]

    # If there are ties and we selected more than k, trim the extras
    # by keeping only the first k per row
    if transfer_index.sum(dim=1).max() > k_safe:
        # More tokens selected than needed due to ties
        # Use argsort to get original positions and keep only first k
        for b in range(batch_size):
            selected = transfer_index[b]
            indices = torch.nonzero(selected, as_tuple=True)[0]
            if len(indices) > k_safe:
                # Keep only the top k by original confidence
                topk_idx = torch.topk(confidence[b, indices], k=k_safe)[1]
                keep_indices = indices[topk_idx]
                mask = torch.zeros(seq_len, dtype=torch.bool, device=confidence.device)
                mask[keep_indices] = True
                transfer_index[b] = mask

    return transfer_index


# =============================================================================
# VARIANT 3: Partial sort using torch.topk with sorted=False
# =============================================================================

def topk_partial(confidence: torch.Tensor, k: Union[int, torch.Tensor]) -> torch.Tensor:
    """
    Partial sort top-k selection using torch.topk with sorted=False.

    Uses partial sort (quickselect algorithm) which is more efficient
    than full sort when only top-k elements are needed.

    Args:
        confidence: (B, L) tensor of confidence scores
        k: int or (B,) tensor of number of tokens to select

    Returns:
        transfer_index: (B, L) bool tensor, True for selected tokens
    """
    batch_size = confidence.shape[0]

    if isinstance(k, int):
        k = torch.full((batch_size,), k, device=confidence.device, dtype=torch.long)

    transfer_index = torch.zeros_like(confidence, dtype=torch.bool)

    # Use sorted=False for partial sort (more efficient)
    for j in range(batch_size):
        _, select_index = torch.topk(confidence[j], k=k[j].item(), sorted=False)
        transfer_index[j, select_index] = True

    return transfer_index


# =============================================================================
# VARIANT 4: Ray-based selective topk for non-uniform k
# =============================================================================

def topk_ray_selective(confidence: torch.Tensor, k: Union[int, torch.Tensor]) -> torch.Tensor:
    """
    Ray-based selective top-k for non-uniform k.

    Uses a selective approach with early termination for non-uniform k.
    Leverages torch.where and conditional operations for efficient selection.

    Args:
        confidence: (B, L) tensor of confidence scores
        k: int or (B,) tensor of number of tokens to select

    Returns:
        transfer_index: (B, L) bool tensor, True for selected tokens
    """
    batch_size, seq_len = confidence.shape

    if isinstance(k, int):
        # For uniform k, use simple vectorized topk
        _, topk_indices = torch.topk(confidence, k=k, dim=1)
        transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
        transfer_index.scatter_(1, topk_indices, True)
        return transfer_index

    # Non-uniform k case: selective approach
    max_k = k.max().item()
    transfer_index = torch.zeros((batch_size, seq_len), device=confidence.device, dtype=torch.bool)

    # Compute global ranking once using argsort
    # For each element, find its rank in descending order
    sorted_indices = torch.argsort(confidence, dim=1, descending=True)
    ranks = torch.argsort(sorted_indices, dim=1)

    # Select tokens where rank < k (i.e., top-k)
    # ranks: (B, L), k: (B,)
    k_expanded = k[:, None].expand(-1, seq_len)  # (B, L)
    transfer_index = ranks < k_expanded

    return transfer_index


# =============================================================================
# Aliases for convenience
# =============================================================================

def topk_selection_baseline(confidence, k):
    """Alias for topk_loop (baseline implementation)."""
    return topk_loop(confidence, k)


def topk_selection_vectorized(confidence, k):
    """Alias for topk_vectorized."""
    return topk_vectorized(confidence, k)


def topk_selection_cumsum(confidence, k):
    """Alias for topk_cumsum."""
    return topk_cumsum(confidence, k)


def topk_selection_partial(confidence, k):
    """Alias for topk_partial."""
    return topk_partial(confidence, k)


def topk_selection_ray_selective(confidence, k):
    """Alias for topk_ray_selective."""
    return topk_ray_selective(confidence, k)


# =============================================================================
# Benchmarking Utilities
# =============================================================================

def verify_correctness(
    confidence: torch.Tensor,
    k: Union[int, torch.Tensor],
    variants: List[str] = None
) -> Tuple[bool, dict]:
    """
    Verify all variants produce the same output as baseline.

    Args:
        confidence: (B, L) tensor of confidence scores
        k: int or (B,) tensor
        variants: list of variant names to verify (default: all)

    Returns:
        (all_match, dict of variant -> match_result)
    """
    if variants is None:
        variants = ['vectorized', 'cumsum', 'partial', 'ray_selective']

    variant_funcs = {
        'baseline': topk_loop,
        'vectorized': topk_vectorized,
        'cumsum': topk_cumsum,
        'partial': topk_partial,
        'ray_selective': topk_ray_selective,
    }

    # Compute baseline
    baseline_result = variant_funcs['baseline'](confidence, k)

    results = {'baseline': True}
    for name in variants:
        if name != 'baseline':
            try:
                result = variant_funcs[name](confidence, k)
                match = torch.equal(result, baseline_result)
                results[name] = match
            except Exception as e:
                results[name] = f"Error: {str(e)}"

    all_match = all(v is True for v in results.values())
    return all_match, results


def benchmark_function(
    func,
    confidence: torch.Tensor,
    k: Union[int, torch.Tensor],
    num_warmup: int = 10,
    num_runs: int = 100,
    device: str = 'cuda'
) -> float:
    """
    Benchmark a function over multiple runs.

    Args:
        func: function to benchmark
        confidence: input tensor
        k: k value
        num_warmup: number of warmup runs
        num_runs: number of timed runs

    Returns:
        average time per run in milliseconds
    """
    # Warmup
    for _ in range(num_warmup):
        _ = func(confidence, k)

    if device == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = func(confidence, k)
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return sum(times) / len(times)


def run_benchmark(
    seq_lens: List[int] = [128, 512, 1024],
    batch_sizes: List[int] = [1, 4, 8],
    k_values: List[int] = [4, 16, 64],
    device: str = 'cuda',
    num_warmup: int = 10,
    num_runs: int = 100,
    verbose: bool = True
) -> dict:
    """
    Run comprehensive benchmark across different configurations.

    Args:
        seq_lens: list of sequence lengths to test
        batch_sizes: list of batch sizes to test
        k_values: list of k values to test
        device: 'cuda' or 'cpu'
        num_warmup: warmup runs before timing
        num_runs: number of timed runs
        verbose: print results

    Returns:
        dict of benchmark results
    """
    variants = {
        'baseline': topk_loop,
        'vectorized': topk_vectorized,
        'cumsum': topk_cumsum,
        'partial': topk_partial,
        'ray_selective': topk_ray_selective,
    }

    results = {}

    for seq_len in seq_lens:
        for batch_size in batch_sizes:
            for k in k_values:
                # Skip impossible configurations
                if k > seq_len:
                    continue

                config_key = f"seq={seq_len}_bs={batch_size}_k={k}"

                # Generate random confidence scores
                confidence = torch.rand((batch_size, seq_len), device=device)

                # Create k tensor (uniform k for this benchmark)
                k_tensor = k

                if verbose:
                    print(f"\n{'='*60}")
                    print(f"Config: seq_len={seq_len}, batch_size={batch_size}, k={k}")
                    print(f"{'='*60}")

                config_results = {}

                # Benchmark baseline first
                baseline_time = benchmark_function(
                    topk_loop, confidence, k_tensor,
                    num_warmup=num_warmup, num_runs=num_runs, device=device
                )
                config_results['baseline'] = baseline_time

                # Verify correctness for all variants
                all_match, verify_results = verify_correctness(confidence, k_tensor)
                if not all_match:
                    if verbose:
                        print(f"WARNING: Variants don't match baseline!")
                        for name, match in verify_results.items():
                            if match is not True:
                                print(f"  {name}: {match}")

                # Benchmark each variant
                for name, func in variants.items():
                    if name == 'baseline':
                        continue

                    try:
                        t = benchmark_function(
                            func, confidence, k_tensor,
                            num_warmup=num_warmup, num_runs=num_runs, device=device
                        )
                        speedup = baseline_time / t if t > 0 else float('inf')
                        config_results[name] = t
                        config_results[f'{name}_speedup'] = speedup

                        if verbose:
                            print(f"  {name:20s}: {t:8.3f} ms  (speedup: {speedup:6.2f}x)")

                    except Exception as e:
                        config_results[name] = f"Error: {str(e)}"
                        if verbose:
                            print(f"  {name:20s}: ERROR - {str(e)}")

                results[config_key] = config_results

    return results


def print_benchmark_summary(results: dict):
    """Print a formatted summary of benchmark results."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    # Header
    print(f"{'Config':<30} {'Baseline':>10} {'Vectorized':>10} {'Cumsum':>10} {'Partial':>10} {'RaySel':>10}")
    print("-"*80)

    for config, data in results.items():
        baseline = data.get('baseline', 0)
        vec = data.get('vectorized', 0)
        cumsum = data.get('cumsum', 0)
        partial = data.get('partial', 0)
        ray = data.get('ray_selective', 0)

        if isinstance(baseline, (int, float)) and baseline > 0:
            print(f"{config:<30} {baseline:>9.3f}ms {vec:>9.3f}ms {cumsum:>9.3f}ms {partial:>9.3f}ms {ray:>9.3f}ms")
        else:
            print(f"{config:<30} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

    print("="*80)

    # Speedup summary
    print("\nSPEEDUP vs BASELINE")
    print("-"*80)
    print(f"{'Config':<30} {'Vectorized':>12} {'Cumsum':>12} {'Partial':>12} {'RaySel':>12}")
    print("-"*80)

    for config, data in results.items():
        baseline = data.get('baseline', 0)
        if not isinstance(baseline, (int, float)) or baseline <= 0:
            continue

        vec_s = data.get('vectorized_speedup', 0)
        cumsum_s = data.get('cumsum_speedup', 0)
        partial_s = data.get('partial_speedup', 0)
        ray_s = data.get('ray_selective_speedup', 0)

        print(f"{config:<30} {vec_s:>11.2f}x {cumsum_s:>11.2f}x {partial_s:>11.2f}x {ray_s:>11.2f}x")

    print("="*80)


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, running on CPU")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Run benchmark
    results = run_benchmark(
        seq_lens=[128, 512, 1024],
        batch_sizes=[1, 4, 8],
        k_values=[4, 16, 64],
        device=device,
        num_warmup=10,
        num_runs=100,
        verbose=True
    )

    # Print summary
    print_benchmark_summary(results)
