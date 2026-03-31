"""
Softmax kernel variants for attention weights.
Input shape: (B, H, L, L) where L is sequence length
"""

import torch
import time
from typing import Literal


def softmax_original(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Baseline softmax: exp(x - max) / sum(exp(x - max))
    Numerically stable version that subtracts max before exp.
    """
    exp_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def softmax_stable(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Variant 1: Numerically stable softmax.
    Same as baseline - explicitly documented for clarity.
    Subtracts max first to avoid overflow in exp.
    """
    max_x = x.max(dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - max_x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def softmax_inplace(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Variant 2: In-place exp and sum for memory efficiency.
    Uses separate buffers but minimizes allocations.
    """
    max_x = x.max(dim=dim, keepdim=True)[0]
    x_minus_max = x - max_x
    exp_x = torch.exp(x_minus_max)
    sum_exp_x = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp_x


def softmax_blocked(x: torch.Tensor, dim: int = -1, block_size: int = 512) -> torch.Tensor:
    """
    Variant 3: Block-wise softmax for long sequences to save memory.
    Processes the attention matrix in blocks.
    """
    B, H, L, _ = x.shape
    result = torch.zeros_like(x)

    # For block-wise processing along the last dimension
    for b in range(B):
        for h in range(H):
            for i in range(L):
                # Process in blocks
                row = x[b, h, i]
                max_row = row.max(dim=dim, keepdim=True)[0] if dim == -1 else row.max()
                # For dim=-1, we process the full row at once but use blocking for memory
                exp_row = torch.exp(row - max_row)
                result[b, h, i] = exp_row / exp_row.sum()

    return result


def softmax_blocked_optimized(x: torch.Tensor, dim: int = -1, block_size: int = 512) -> torch.Tensor:
    """
    Variant 3 (optimized): Block-wise softmax with true blocking for memory efficiency.
    Processes in blocks along the summation dimension.
    """
    B, H, L, _ = x.shape
    result = torch.empty_like(x)

    # Process each (B, H, i) row in blocks
    for b in range(B):
        for h in range(H):
            for i in range(L):
                row = x[b, h, i]
                max_val = row.max()

                # Block-wise exp summation
                total_sum = 0.0
                for start in range(0, L, block_size):
                    end = min(start + block_size, L)
                    block = row[start:end]
                    exp_block = torch.exp(block - max_val)
                    total_sum += exp_block.sum()

                # Compute final result in blocks
                for start in range(0, L, block_size):
                    end = min(start + block_size, L)
                    block = row[start:end]
                    exp_block = torch.exp(block - max_val)
                    result[b, h, i, start:end] = exp_block / total_sum

    return result


def softmax_vec(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Variant 4: Fully vectorized exp using torch.exp.
    Leverages PyTorch's vectorized operations for maximum GPU utilization.
    """
    max_x = x.max(dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - max_x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


# Alias for clarity
softmax_blocked = softmax_blocked_optimized


def benchmark_softmax(
    B: int,
    H: int,
    L: int,
    num_runs: int = 100,
    warmup: int = 10,
    device: str = "cuda"
) -> dict:
    """
    Benchmark all softmax variants.

    Args:
        B: Batch size
        H: Number of heads
        L: Sequence length
        num_runs: Number of timing runs
        warmup: Number of warmup runs
        device: Device to run on

    Returns:
        Dictionary with timing and numerical error results
    """
    # Create input tensor
    x = torch.randn(B, H, L, L, device=device, dtype=torch.float32)

    # Compute baseline
    torch.cuda.synchronize()
    for _ in range(warmup):
        baseline = softmax_original(x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_runs):
        baseline = softmax_original(x)
    torch.cuda.synchronize()
    baseline_time = (time.perf_counter() - t0) / num_runs * 1000  # ms

    # Generate random mask for testing
    mask = torch.rand(B, H, L, L, device=device) > 0.2
    x_masked = x.masked_fill(~mask, float('-inf'))

    results = {
        'shape': f"({B}, {H}, {L}, {L})",
        'baseline_time_ms': baseline_time,
    }

    variants = {
        'softmax_stable': softmax_stable,
        'softmax_inplace': softmax_inplace,
        'softmax_blocked': softmax_blocked,
        'softmax_vec': softmax_vec,
    }

    for name, fn in variants.items():
        # Warmup
        for _ in range(warmup):
            out = fn(x)
        torch.cuda.synchronize()

        # Timed run
        t0 = time.perf_counter()
        for _ in range(num_runs):
            out = fn(x)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / num_runs * 1000

        # Numerical error vs baseline
        diff = (out - baseline).abs()
        max_error = diff.max().item()

        speedup = baseline_time / elapsed if elapsed > 0 else 0

        results[f'{name}_time_ms'] = elapsed
        results[f'{name}_speedup'] = speedup
        results[f'{name}_max_error'] = max_error

    # Also test with masked input for numerical stability test
    torch.cuda.synchronize()
    baseline_masked = softmax_original(x_masked)
    for name, fn in variants.items():
        out_masked = fn(x_masked)
        diff = (out_masked - baseline_masked).abs()
        max_error_masked = diff.max().item()
        results[f'{name}_max_error_masked'] = max_error_masked

    return results


def run_benchmarks():
    """Run comprehensive benchmarks across different sequence lengths and batch sizes."""
    seq_lengths = [128, 512, 1024, 2048]
    batch_sizes = [1, 4]
    H = 4  # Number of heads

    print("=" * 80)
    print("Softmax Kernel Benchmarks")
    print("=" * 80)
    print(f"{'Shape':<20} {'Variant':<20} {'Time (ms)':<12} {'Speedup':<10} {'Max Error':<12}")
    print("-" * 80)

    all_results = []

    for B in batch_sizes:
        for L in seq_lengths:
            print(f"\n>>> Benchmarking shape: ({B}, {H}, {L}, {L})")
            results = benchmark_softmax(B=B, H=H, L=L, num_runs=50, warmup=5)

            # Print baseline
            print(f"  Baseline:        {results['baseline_time_ms']:.4f} ms")

            variants = ['softmax_stable', 'softmax_inplace', 'softmax_blocked', 'softmax_vec']
            for name in variants:
                time_ms = results[f'{name}_time_ms']
                speedup = results[f'{name}_speedup']
                max_err = results[f'{name}_max_error']
                print(f"  {name:<18} {time_ms:.4f} ms    {speedup:.3f}x    {max_err:.2e}")

            all_results.append(results)

    print("\n" + "=" * 80)
    print("Numerical Stability Test (with -inf values)")
    print("=" * 80)
    print(f"{'Shape':<20} {'Variant':<20} {'Max Error (masked)':<15}")
    print("-" * 80)

    for results in all_results:
        shape = results['shape']
        variants = ['softmax_stable', 'softmax_inplace', 'softmax_blocked', 'softmax_vec']
        for name in variants:
            max_err = results[f'{name}_max_error_masked']
            print(f"{shape:<20} {name:<20} {max_err:.2e}")

    return all_results


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    else:
        device = "cuda"
        print(f"Using device: {torch.cuda.get_device_name(0)}")

    run_benchmarks()
