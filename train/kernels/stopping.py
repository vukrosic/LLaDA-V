"""
Stopping criteria kernels for detecting stop tokens in generated sequences.

Implements 4 optimized variants alongside the baseline for comparison.
"""

import torch
import time
from typing import List, Tuple, Optional


def stop_loop(generated: torch.Tensor, stop_tokens: List[torch.Tensor]) -> Tuple[bool, int]:
    """
    Baseline: nested loop checking for stop sequences.

    Args:
        generated: (B, L) tensor of generated tokens
        stop_tokens: list of 1D tensors representing stop sequences

    Returns:
        (found: bool, position: int) - whether stop was found and first position
    """
    B, L = generated.shape
    for seq_idx in range(B):
        seq = generated[seq_idx]
        for stop_seq in stop_tokens:
            stop_len = len(stop_seq)
            if stop_len > len(seq):
                continue
            for start in range(len(seq) - stop_len + 1):
                if torch.all(seq[start:start + stop_len] == stop_seq):
                    return True, seq_idx * L + start
    return False, -1


def stop_vectorized(generated: torch.Tensor, stop_tokens: List[torch.Tensor]) -> Tuple[bool, int]:
    """
    Vectorized comparison using broadcasting.

    Iteration order: (batch_idx, stop_token_idx, position)
    For each batch, checks all stop_tokens and returns first match found.
    """
    if len(stop_tokens) == 0:
        return False, -1

    B, L = generated.shape

    # For each batch, find the earliest match across all stop_tokens
    for batch_idx in range(B):
        seq = generated[batch_idx]

        for stop_idx, stop_seq in enumerate(stop_tokens):
            stop_len = stop_seq.shape[0]
            if stop_len == 0 or stop_len > L:
                continue

            num_positions = L - stop_len + 1

            # Get windows for this batch: (num_positions, stop_len)
            windows = seq.unfold(0, stop_len, 1)  # (num_positions, stop_len)

            # Compare with stop_seq using broadcasting
            stop_expanded = stop_seq.view(stop_len)  # (stop_len,)
            matches = torch.all(windows == stop_expanded, dim=1)  # (num_positions,)

            # Find first match
            if matches.any():
                first_pos = torch.where(matches)[0][0].item()
                return True, batch_idx * L + first_pos

    return False, -1


def stop_conv(generated: torch.Tensor, stop_tokens: List[torch.Tensor]) -> Tuple[bool, int]:
    """
    Use unfold for sliding window extraction.

    Same iteration order as baseline: (batch_idx, stop_token_idx, position).
    """
    if len(stop_tokens) == 0:
        return False, -1

    B, L = generated.shape

    for batch_idx in range(B):
        seq = generated[batch_idx]

        for stop_idx, stop_seq in enumerate(stop_tokens):
            stop_len = stop_seq.shape[0]
            if stop_len == 0 or stop_len > L:
                continue

            num_positions = L - stop_len + 1

            # Get windows for this batch
            windows = seq.unfold(0, stop_len, 1)  # (num_positions, stop_len)

            # Compare with broadcasting
            stop_expanded = stop_seq.view(stop_len)
            matches = torch.all(windows == stop_expanded, dim=1)  # (num_positions,)

            if matches.any():
                first_pos = torch.where(matches)[0][0].item()
                return True, batch_idx * L + first_pos

    return False, -1


def stop_strided(generated: torch.Tensor, stop_tokens: List[torch.Tensor]) -> Tuple[bool, int]:
    """
    Use strided operations for efficient comparison.

    Uses as_strided for memory-efficient window extraction without copying.
    """
    if len(stop_tokens) == 0:
        return False, -1

    B, L = generated.shape

    for batch_idx in range(B):
        seq = generated[batch_idx]

        for stop_idx, stop_seq in enumerate(stop_tokens):
            stop_len = stop_seq.shape[0]
            if stop_len == 0 or stop_len > L:
                continue

            num_positions = L - stop_len + 1

            # Use as_strided for efficient window extraction
            # Note: storage_offset defaults to seq.storage_offset()
            windows = seq.as_strided((num_positions, stop_len), (1, 1))

            # Compare with broadcasting
            stop_expanded = stop_seq.view(stop_len)
            matches = torch.all(windows == stop_expanded, dim=1)  # (num_positions,)

            if matches.any():
                first_pos = torch.where(matches)[0][0].item()
                return True, batch_idx * L + first_pos

    return False, -1


def stop_index(generated: torch.Tensor, stop_tokens: List[torch.Tensor]) -> Tuple[bool, int]:
    """
    Use torch.index with precomputed positions.

    Extracts windows at specific positions using advanced indexing.
    """
    if len(stop_tokens) == 0:
        return False, -1

    B, L = generated.shape
    device = generated.device

    for batch_idx in range(B):
        seq = generated[batch_idx]

        for stop_idx, stop_seq in enumerate(stop_tokens):
            stop_len = stop_seq.shape[0]
            if stop_len == 0 or stop_len > L:
                continue

            num_positions = L - stop_len + 1

            # Create position indices
            pos_base = torch.arange(num_positions, device=device).view(-1, 1).expand(num_positions, stop_len)
            offset = torch.arange(stop_len, device=device).view(1, -1).expand(num_positions, stop_len)
            col_idx = pos_base + offset

            # Extract windows using indexing
            windows = seq[col_idx]  # (num_positions, stop_len)

            # Compare
            stop_expanded = stop_seq.view(stop_len)
            matches = torch.all(windows == stop_expanded, dim=1)  # (num_positions,)

            if matches.any():
                first_pos = torch.where(matches)[0][0].item()
                return True, batch_idx * L + first_pos

    return False, -1


def verify_correctness():
    """Verify all variants produce same results as baseline."""
    print("=" * 60)
    print("Verifying correctness across variants...")
    print("=" * 60)

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_cases = [
        # (batch, seq_len, stop_token_lens)
        (4, 128, [1, 2, 4]),
        (1, 512, [2, 4]),
        (4, 1024, [1, 4]),
        (1, 128, [1, 2]),
    ]

    variants = {
        'baseline': stop_loop,
        'vectorized': stop_vectorized,
        'conv': stop_conv,
        'strided': stop_strided,
        'index': stop_index,
    }

    all_passed = True
    for B, L, stop_lens in test_cases:
        # Generate random stop tokens
        stop_tokens = [torch.randint(0, 50000, (sl,), device=device) for sl in stop_lens]

        # Generate random sequences with some matches guaranteed
        generated = torch.randint(0, 50000, (B, L), device=device)

        # Inject stop sequences at random positions
        for stop_seq in stop_tokens:
            b = torch.randint(0, B, (1,)).item()
            pos = torch.randint(0, L - len(stop_seq) + 1, (1,)).item()
            generated[b, pos:pos + len(stop_seq)] = stop_seq

        # Get baseline result
        baseline_found, baseline_pos = stop_loop(generated, stop_tokens)

        print(f"\nTest case: B={B}, L={L}, stop_lens={stop_lens}")
        print(f"  Baseline: found={baseline_found}, pos={baseline_pos}")

        for name, func in variants.items():
            if name == 'baseline':
                continue
            found, pos = func(generated, stop_tokens)
            match = (found == baseline_found) and (pos == baseline_pos or (not found and not baseline_found))
            status = "PASS" if match else "FAIL"
            print(f"  {name}: found={found}, pos={pos} [{status}]")
            if not match:
                all_passed = False

    print("\n" + "=" * 60)
    print("All tests passed!" if all_passed else "SOME TESTS FAILED!")
    print("=" * 60)
    return all_passed


def benchmark():
    """Run benchmarks across different configurations."""
    print("=" * 60)
    print("Benchmarking stopping criteria kernels")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("WARNING: CUDA not available, running on CPU")

    # Benchmark configuration
    seq_lengths = [128, 512, 1024]
    num_stops = [1, 4, 16]
    batch_sizes = [1, 4]
    stop_token_lens = [1, 2, 4]

    variants = {
        'baseline': stop_loop,
        'vectorized': stop_vectorized,
        'conv': stop_conv,
        'strided': stop_strided,
        'index': stop_index,
    }

    results = []

    # Warm up
    print("\nWarming up...")
    for _ in range(10):
        dummy_gen = torch.randint(0, 1000, (4, 128), device=device)
        dummy_stops = [torch.tensor([1, 2], device=device)]
        for v in variants.values():
            v(dummy_gen, dummy_stops)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print("\nRunning benchmarks...")
    print("-" * 60)

    for L in seq_lengths:
        for num_stop in num_stops:
            for B in batch_sizes:
                for stop_len in stop_token_lens:
                    # Skip expensive combos on CPU
                    if device.type == 'cpu' and (L > 512 or num_stop > 4 or B > 1):
                        continue

                    # Create stop tokens
                    stop_tokens = [torch.randint(0, 50000, (stop_len,), device=device) for _ in range(num_stop)]

                    # Generate sequence and inject stop sequences
                    generated = torch.randint(0, 50000, (B, L), device=device)

                    # Inject first stop sequence at position 0 to ensure at least one match
                    if num_stop > 0 and stop_len <= L:
                        generated[0, 0:stop_len] = stop_tokens[0]

                    # Number of iterations for timing
                    n_iters = 100 if L <= 256 else 50

                    print(f"\nConfig: B={B}, L={L}, num_stops={num_stop}, stop_len={stop_len}")

                    variant_times = {}

                    for name, func in variants.items():
                        # Time multiple iterations
                        times = []
                        for _ in range(n_iters):
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            start = time.perf_counter()

                            found, pos = func(generated, stop_tokens)

                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            end = time.perf_counter()
                            times.append(end - start)

                        avg_time = sum(times) / len(times)
                        variant_times[name] = avg_time
                        speedup = variant_times['baseline'] / avg_time if avg_time > 0 else float('inf')
                        print(f"  {name:12s}: {avg_time*1000:.4f} ms  (speedup: {speedup:.2f}x)")

                    results.append({
                        'B': B, 'L': L, 'num_stops': num_stop, 'stop_len': stop_len,
                        'times': variant_times
                    })

    print("\n" + "=" * 60)
    print("Benchmark Summary (speedup vs baseline)")
    print("=" * 60)

    # Print summary table
    print(f"\n{'B':>3} {'L':>5} {'#stops':>7} {'slen':>5} | {'Baseline':>10} {'Vector':>10} {'Conv':>10} {'Stride':>10} {'Index':>10}")
    print("-" * 80)

    for r in results:
        base = r['times']['baseline'] * 1000
        vec = r['times']['vectorized'] * 1000 / base if base > 0 else 0
        conv = r['times']['conv'] * 1000 / base if base > 0 else 0
        stride = r['times']['strided'] * 1000 / base if base > 0 else 0
        index = r['times']['index'] * 1000 / base if base > 0 else 0
        print(f"{r['B']:>3} {r['L']:>5} {r['num_stops']:>7} {r['stop_len']:>5} | "
              f"{base:>9.3f}ms {vec:>9.2f}x {conv:>9.2f}x {stride:>9.2f}x {index:>9.2f}x")

    return results


if __name__ == "__main__":
    # First verify correctness
    verify_correctness()

    # Then run benchmark
    benchmark()
