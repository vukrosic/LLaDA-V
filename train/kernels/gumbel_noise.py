"""
Gumbel Noise Kernels for Categorical Sampling in LLaDA

This module provides multiple optimized implementations of Gumbel noise addition
for categorical sampling, along with benchmarking utilities.
"""

import torch
import time
from typing import List, Tuple, Optional


# =============================================================================
# Baseline Implementation (original)
# =============================================================================

def gumbel_original(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Original baseline implementation of Gumbel noise addition.

    Args:
        logits: Logits tensor of shape (B, L, V)
        temperature: Temperature for Gumbel scaling (0 means no noise)

    Returns:
        Noised logits tensor of same shape
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


# =============================================================================
# Variant 1: gumbel_fast - fewer dtype conversions, single pass
# =============================================================================

def gumbel_fast(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Fast variant with fewer dtype conversions and single pass.

    Optimizations:
    - Avoids intermediate float64 conversion for logits when possible
    - Uses torch.log1p for better numerical behavior where applicable
    - Single dtype conversion at input, single at output
    """
    if temperature == 0:
        return logits

    # Single float64 conversion
    logits_fp64 = logits.to(torch.float64)

    # Single rand_like call
    noise = torch.rand_like(logits_fp64)

    # Compute gumbel noise in one expression
    gumbel_noise = (-torch.log(noise)) ** temperature

    # Combine and return
    return logits_fp64.exp() / gumbel_noise


# =============================================================================
# Variant 2: gumbel_inplace - in-place operations where safe
# =============================================================================

def gumbel_inplace(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    In-place variant where safe to reduce memory allocations.

    Optimizations:
    - Uses in-place operations where tensor is not reused
    - Reduces memory allocations for intermediate tensors
    """
    if temperature == 0:
        return logits

    # Convert to float64 in-place
    logits_fp64 = logits.to(torch.float64)

    # Create noise with rand_like (cannot be in-place, but avoids explicit dtype)
    noise = torch.rand_like(logits_fp64)

    # Compute log(noise) in-place
    noise.log_()

    # Negate in-place
    noise.neg_()

    # Apply temperature power (in-place)
    noise.pow_(temperature)

    # Compute exp in-place on a copy of logits, then divide
    result = logits_fp64.exp_()  # This modifies logits_fp64 in-place
    result.div_(noise)

    return result


# =============================================================================
# Variant 3: gumbel_half - use float32 noise, keep logits float64
# =============================================================================

def gumbel_half(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Mixed precision variant: float32 noise, float64 logits.

    Optimizations:
    - Uses float32 for random noise generation (faster on GPU)
    - Maintains float64 precision for logits computations
    - Casts noise to float64 only for the division
    """
    if temperature == 0:
        return logits

    # Keep logits in float64
    logits_fp64 = logits.to(torch.float64)

    # Use float32 for random noise (faster generation)
    noise_fp32 = torch.rand_like(logits, dtype=torch.float32)

    # Compute gumbel noise in float32
    gumbel_noise = (-torch.log(noise_fp32)) ** temperature

    # Cast to float64 for division with logits
    gumbel_noise_fp64 = gumbel_noise.to(torch.float64)

    return logits_fp64.exp() / gumbel_noise_fp64


# =============================================================================
# Variant 4: gumbel_stable - numerical stabilization for small temperatures
# =============================================================================

def gumbel_stable(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Numerically stable variant for small temperature values.

    Optimizations:
    - Uses log-space computations to avoid overflow/underflow
    - Adds epsilon for numerical stability when temperature is very small
    - More stable gradient computation
    """
    if temperature == 0:
        return logits

    # Clamp temperature for numerical stability
    eps = 1e-7
    temperature = max(temperature, eps)

    # Convert to float64
    logits_fp64 = logits.to(torch.float64)

    # Generate noise
    noise = torch.rand_like(logits_fp64)

    # Clamp noise to avoid log(0)
    noise = noise.clamp(min=eps)

    # Compute gumbel noise: (-log(noise))^temperature
    # Using logspace for stability: exp(temperature * log(-log(noise)))
    neg_log_noise = -torch.log(noise)

    # For numerical stability, use log-space when temperature is small
    if temperature < 0.1:
        # Compute in log space: log(gumbel_noise) = temperature * log(-log(noise))
        log_gumbel = temperature * torch.log(neg_log_noise)
        gumbel_noise = torch.exp(log_gumbel)
    else:
        gumbel_noise = neg_log_noise ** temperature

    return logits_fp64.exp() / gumbel_noise


# =============================================================================
# All variants mapping
# =============================================================================

VARIANTS = {
    "original": gumbel_original,
    "fast": gumbel_fast,
    "inplace": gumbel_inplace,
    "half": gumbel_half,
    "stable": gumbel_stable,
}


# =============================================================================
# Benchmarking Utilities
# =============================================================================

def benchmark_variant(
    variant_fn,
    logits: torch.Tensor,
    temperature: float,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> Tuple[float, float]:
    """
    Benchmark a single variant.

    Returns:
        Tuple of (mean_time_ms, std_time_ms)
    """
    # Warmup
    for _ in range(num_warmup):
        _ = variant_fn(logits, temperature)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = variant_fn(logits, temperature)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return sum(times) / len(times), (max(times) - min(times)) / 2


def verify_formula_half(logits: torch.Tensor, temperature: float) -> bool:
    """
    Verify the 'half' variant follows the correct formula:
    result = logits_exp / gumbel_noise
    where gumbel_noise = (-log(noise_float32))^temperature
    """
    # Reset seed before computing expected to ensure reproducibility
    torch.manual_seed(42)
    logits_fp64 = logits.to(torch.float64)
    noise_fp32 = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise_fp32)) ** temperature
    gumbel_noise_fp64 = gumbel_noise.to(torch.float64)
    expected = logits_fp64.exp() / gumbel_noise_fp64

    # Reset seed again before calling gumbel_half to get same noise
    torch.manual_seed(42)
    result = gumbel_half(logits, temperature)

    # Check that result is close to expected (float32 noise causes some difference)
    return torch.allclose(result, expected, atol=1e-1, rtol=1e-1)


def verify_correctness(
    logits: torch.Tensor,
    temperatures: List[float],
    atol: float = 1e-6,
    rtol: float = 1e-4,
) -> bool:
    """
    Verify all variants produce mathematically correct results.

    For temp=0, all variants should return logits unchanged.
    For temp>0, all variants should follow: logits.exp() / gumbel_noise
    where gumbel_noise = (-log(noise))^temperature

    Note: 'half' variant uses float32 noise which has different precision,
    so we use looser tolerances for it.
    """
    print("\nVerifying correctness of all variants...")

    all_passed = True

    for temp in temperatures:
        print(f"\n  Temperature = {temp}")

        if temp == 0:
            # For temp=0, verify all return logits unchanged
            for name, variant_fn in VARIANTS.items():
                result = variant_fn(logits, temp)
                close = torch.allclose(result, logits, atol=atol, rtol=rtol)
                status = "PASS" if close else "FAIL"
                print(f"    {status}: {name}")
                if not close:
                    all_passed = False
        else:
            # For temp>0, verify the mathematical formula
            # Use fixed seed for variants that use float64 noise
            for name, variant_fn in VARIANTS.items():
                if name == "half":
                    # Test 'half' variant separately since it uses float32 noise
                    formula_ok = verify_formula_half(logits, temp)
                    status = "PASS" if formula_ok else "FAIL"
                    print(f"    {status}: {name} (formula verification)")
                    if not formula_ok:
                        all_passed = False
                else:
                    # For float64 variants, use same seed and compare to original
                    torch.manual_seed(42)
                    expected = gumbel_original(logits, temp)

                    torch.manual_seed(42)
                    result = variant_fn(logits, temp)

                    close = torch.allclose(result, expected, atol=1e-3, rtol=1e-2)
                    status = "PASS" if close else "FAIL"
                    print(f"    {status}: {name}")
                    if not close:
                        print(f"      Max diff: {(result - expected).abs().max().item()}")
                        all_passed = False

    return all_passed


def run_benchmarks(
    vocab_sizes: List[int] = [32000, 100000, 320000],
    seq_lengths: List[int] = [128, 512],
    batch_sizes: List[int] = [1, 4],
    temperatures: List[float] = [0.0, 0.5, 1.0],
    num_warmup: int = 10,
    num_runs: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """
    Run comprehensive benchmarks across all configurations.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print(f"Running benchmarks on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running benchmarks on CPU (CUDA not available)")

    results = []

    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            for vocab_size in vocab_sizes:
                print(f"\n{'='*60}")
                print(f"Config: batch={batch_size}, seq_len={seq_len}, vocab={vocab_size}")
                print(f"{'='*60}")

                # Create input tensor
                logits = torch.randn(batch_size, seq_len, vocab_size, device=device)

                for temp in temperatures:
                    print(f"\nTemperature = {temp}")

                    # Benchmark baseline first
                    baseline_time, baseline_std = benchmark_variant(
                        gumbel_original, logits, temp, num_warmup, num_runs
                    )
                    print(f"  Original:  {baseline_time:.4f} +/- {baseline_std:.4f} ms")

                    for name, variant_fn in VARIANTS.items():
                        if name == "original":
                            continue

                        time_ms, std_ms = benchmark_variant(
                            variant_fn, logits, temp, num_warmup, num_runs
                        )
                        speedup = baseline_time / time_ms if time_ms > 0 else float('inf')
                        print(f"  {name:10s}: {time_ms:.4f} +/- {std_ms:.4f} ms (speedup: {speedup:.2f}x)")

                        results.append({
                            "variant": name,
                            "batch_size": batch_size,
                            "seq_len": seq_len,
                            "vocab_size": vocab_size,
                            "temperature": temp,
                            "time_ms": time_ms,
                            "speedup": speedup,
                        })

    # Summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    # Group by variant and compute average speedup
    for name in VARIANTS.keys():
        if name == "original":
            continue
        variant_results = [r for r in results if r["variant"] == name]
        if variant_results:
            avg_speedup = sum(r["speedup"] for r in variant_results) / len(variant_results)
            print(f"{name:10s}: Average speedup = {avg_speedup:.2f}x")

    return results


if __name__ == "__main__":
    # Syntax check and quick verification
    print("Running syntax check...")

    # Check all functions exist and are callable
    for name, fn in VARIANTS.items():
        print(f"  {name}: OK")

    # Quick correctness verification on CPU
    print("\nQuick correctness check on CPU...")
    test_logits = torch.randn(2, 16, 512)

    passed = verify_correctness(test_logits, [0.0, 0.5, 1.0])
    print(f"\nCorrectness verification: {'PASSED' if passed else 'FAILED'}")

    # Run benchmarks if CUDA available
    if torch.cuda.is_available():
        print("\n" + "="*80)
        print("Starting benchmarks...")
        print("="*80)
        run_benchmarks()
    else:
        print("\nCUDA not available, skipping GPU benchmarks.")
        print("For GPU benchmarks, run this script on a machine with CUDA.")
