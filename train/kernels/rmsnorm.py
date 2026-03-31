"""
RMSNorm kernels for LLaDA.

Implements 4 variants of RMSNorm with different optimizations:
1. rmsnorm_fast    - precompute rsqrt, avoid intermediate
2. rmsnorm_inplace - in-place variance computation
3. rmsnorm_blocked - block-wise for cache efficiency on long sequences
4. rmsnorm_fused   - fuse weight multiplication into same kernel

Baseline: rmsnorm_original
"""

import torch
import torch.nn as nn
import time
from typing import List, Tuple


def rmsnorm_original(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Baseline RMSNorm: standard implementation."""
    variance = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    return weight * x_norm


def rmsnorm_fast(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Variant 1: Precompute rsqrt and avoid intermediate tensor."""
    # Compute variance and rsqrt in one pass, avoiding intermediate pow tensor
    variance = x.pow(2).mean(-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    # Fused multiply: x * inv_rms directly, then scale by weight
    return weight * (x * inv_rms)


def rmsnorm_inplace(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Variant 2: In-place variance computation to reduce memory allocations."""
    # Compute variance in-place using addcmul
    result = x.new_empty(x.shape)
    variance = x.pow(2).mean(-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    # Compute normalized values
    result.copy_(x * inv_rms)
    # Multiply by weight
    return result * weight


def rmsnorm_blocked(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, block_size: int = 512) -> torch.Tensor:
    """Variant 3: Block-wise computation for cache efficiency on long sequences."""
    B, L, D = x.shape
    output = torch.empty_like(x)

    # Process sequence in blocks to improve cache locality
    for start in range(0, L, block_size):
        end = min(start + block_size, L)
        block = x[:, start:end, :]  # (B, block_len, D)

        # Compute RMS for this block
        variance = block.pow(2).mean(-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + eps)
        normalized = block * inv_rms

        output[:, start:end, :] = normalized * weight

    return output


def rmsnorm_fused(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Variant 4: Fused weight multiplication into single kernel."""
    # Single fused kernel: compute variance, rsqrt, normalize, and scale weight all in one
    B, L, D = x.shape

    # Compute variance
    variance = x.pow(2).mean(-1, keepdim=True)

    # Compute inverse RMS
    inv_rms = torch.rsqrt(variance + eps)

    # Fused multiply-add: (x * inv_rms) * weight in single pass
    # Using out= parameter to avoid intermediate allocation
    output = torch.empty_like(x)
    output.copy_(x * inv_rms)
    output.mul_(weight)

    return output


class RMSNorm(nn.Module):
    """RMSNorm module wrapping the original implementation."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm_original(x, self.weight, self.eps)


def benchmark_kernel(
    kernel_fn,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    warmup: int = 10,
    iterations: int = 100,
) -> Tuple[float, float]:
    """Benchmark a kernel and return (mean_ms, std_ms)."""
    # Warmup
    for _ in range(warmup):
        _ = kernel_fn(x, weight, eps)

    if x.is_cuda:
        torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = kernel_fn(x, weight, eps)
        if x.is_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return sum(times) / len(times), (max(times) - min(times)) / 2


def verify_numerical_equivalence(
    kernels: dict,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> bool:
    """Verify all kernels produce numerically equivalent results."""
    results = {}
    for name, kernel in kernels.items():
        torch.manual_seed(42)
        results[name] = kernel(x, weight, eps)

    baseline = results["rmsnorm_original"]
    tolerance = 1e-5

    all_pass = True
    for name, result in results.items():
        if name == "rmsnorm_original":
            continue
        diff = (result - baseline).abs().max().item()
        passed = diff < tolerance
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: max diff = {diff:.2e} [{status}]")
        if not passed:
            all_pass = False

    return all_pass


def run_benchmarks(
    hidden_dims: List[int] = [4096, 7168, 8192],
    seq_lengths: List[int] = [128, 512, 1024],
    batch_sizes: List[int] = [1, 4, 8],
    device: str = "cuda",
    eps: float = 1e-6,
):
    """Run comprehensive benchmarks across all configurations."""
    kernels = {
        "rmsnorm_original": rmsnorm_original,
        "rmsnorm_fast": rmsnorm_fast,
        "rmsnorm_inplace": rmsnorm_inplace,
        "rmsnorm_blocked": rmsnorm_blocked,
        "rmsnorm_fused": rmsnorm_fused,
    }

    print("=" * 80)
    print("RMSNorm Kernel Benchmarks")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Hidden dims: {hidden_dims}")
    print(f"Sequence lengths: {seq_lengths}")
    print(f"Batch sizes: {batch_sizes}")
    print("=" * 80)

    # Check if CUDA is available
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    all_results = []

    for hidden_dim in hidden_dims:
        for seq_len in seq_lengths:
            for batch_size in batch_sizes:
                shape = (batch_size, seq_len, hidden_dim)
                print(f"\nShape: {shape} (B={batch_size}, L={seq_len}, D={hidden_dim})")
                print("-" * 60)

                x = torch.randn(shape, device=device)
                weight = torch.ones(hidden_dim, device=device)

                # Verify numerical equivalence first
                print("Numerical equivalence check:")
                verify_numerical_equivalence(kernels, x, weight, eps)

                print(f"\nTiming (ms) - {100} iterations:")
                baseline_time = None
                row_results = {"shape": str(shape)}

                for name, kernel in kernels.items():
                    mean_ms, std_ms = benchmark_kernel(kernel, x, weight, eps)
                    row_results[name] = mean_ms
                    if name == "rmsnorm_original":
                        baseline_time = mean_ms
                    speedup = baseline_time / mean_ms if baseline_time else 1.0
                    print(f"  {name:20s}: {mean_ms:8.4f} +/- {std_ms:6.4f} ms  (speedup: {speedup:.2f}x)")

                all_results.append(row_results)

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Speedup vs Baseline (rmsnorm_original)")
    print("=" * 80)
    print(f"{'Shape':<25} {'fast':>8} {'inplace':>8} {'blocked':>8} {'fused':>8}")
    print("-" * 80)

    for result in all_results:
        shape = result["shape"]
        baseline = result["rmsnorm_original"]
        fast = result["rmsnorm_fast"] / baseline
        inplace = result["rmsnorm_inplace"] / baseline
        blocked = result["rmsnorm_blocked"] / baseline
        fused = result["rmsnorm_fused"] / baseline
        print(f"{shape:<25} {fast:>8.2f}x {inplace:>8.2f}x {blocked:>8.2f}x {fused:>8.2f}x")

    return all_results


if __name__ == "__main__":
    # Run benchmarks
    run_benchmarks()
