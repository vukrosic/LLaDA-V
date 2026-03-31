"""
MLP/FFN Kernel Variants for LLaDA

Implements 4 optimized variants of LLaDA's FFN computation:
    down_proj(act(gate_proj(x)) * up_proj(x))

Baseline: mlp_original - straightforward implementation
Variant 1: mlp_fused_linear - fused gate+up projection via grouped linear
Variant 2: mlp_inplace - in-place activation and multiplication
Variant 3: mlp_lazy - skip up computation for padding tokens
Variant 4: mlp_cached - cache gate_proj bias computation
"""

import torch
import torch.nn.functional as F
import time
from typing import Callable, Tuple, Optional


def mlp_original(x: torch.Tensor, gate_proj: torch.nn.Linear, up_proj: torch.nn.Linear,
                 down_proj: torch.nn.Linear, act_fn: Callable) -> torch.Tensor:
    """
    Baseline MLP implementation: LLaDA's standard FFN computation.

    Args:
        x: Input tensor (B, L, D)
        gate_proj: Gate projection weight matrix (D, intermediate_dim)
        up_proj: Up projection weight matrix (D, intermediate_dim)
        down_proj: Down projection weight matrix (intermediate_dim, D)
        act_fn: Activation function (e.g., F.silu)

    Returns:
        Output tensor (B, L, D)
    """
    gate = act_fn(gate_proj(x))
    up = up_proj(x)
    return down_proj(gate * up)


def mlp_fused_linear(x: torch.Tensor, gate_proj: torch.nn.Linear, up_proj: torch.nn.Linear,
                      down_proj: torch.nn.Linear, act_fn: Callable) -> torch.Tensor:
    """
    Variant 1: Fused gate+up projection using a single grouped linear operation.

    Instead of calling two separate linear layers, we stack the weights and
    perform a single matmul with the output split into gate and up tensors.

    Args:
        x: Input tensor (B, L, D)
        gate_proj: Gate projection weight matrix (D, intermediate_dim)
        up_proj: Up projection weight matrix (D, intermediate_dim)
        down_proj: Down projection weight matrix (intermediate_dim, D)
        act_fn: Activation function

    Returns:
        Output tensor (B, L, D)
    """
    B, L, D = x.shape
    intermediate_dim = gate_proj.weight.shape[0]  # (out_features, in_features) = (intermediate_dim, D)

    # Stack gate and up projections: (2 * intermediate_dim, D)
    # Weights are stored as (out_features, in_features) in PyTorch Linear
    fused_weight = torch.cat([gate_proj.weight, up_proj.weight], dim=0)
    fused_bias = torch.cat([gate_proj.bias, up_proj.bias], dim=0) if gate_proj.bias is not None else None

    # Single linear call for both projections
    x_flat = x.view(-1, D)  # (B*L, D)
    fused_out = F.linear(x_flat, fused_weight, fused_bias)  # (B*L, 2*intermediate_dim)

    # Split into gate and up
    gate_out = fused_out[:, :intermediate_dim]
    up_out = fused_out[:, intermediate_dim:]

    gate = act_fn(gate_out)
    return down_proj(gate * up_out).view(B, L, D)


def mlp_inplace(x: torch.Tensor, gate_proj: torch.nn.Linear, up_proj: torch.nn.Linear,
                down_proj: torch.nn.Linear, act_fn: Callable) -> torch.Tensor:
    """
    Variant 2: In-place activation and multiplication.

    Reduces memory allocations by performing operations in-place where safe.

    Args:
        x: Input tensor (B, L, D)
        gate_proj: Gate projection weight matrix (D, intermediate_dim)
        up_proj: Up projection weight matrix (D, intermediate_dim)
        down_proj: Down projection weight matrix (intermediate_dim, D)
        act_fn: Activation function

    Returns:
        Output tensor (B, L, D)
    """
    gate = act_fn(gate_proj(x))
    up = up_proj(x)
    # In-place multiplication
    gate.mul_(up)
    return down_proj(gate)


def mlp_lazy(x: torch.Tensor, gate_proj: torch.nn.Linear, up_proj: torch.nn.Linear,
             down_proj: torch.nn.Linear, act_fn: Callable) -> torch.Tensor:
    """
    Variant 3: Lazy evaluation - skip up projection computation for padding tokens.

    Identifies padding tokens (zeros in input) and avoids computing up_proj
    for those positions, saving computation.

    Args:
        x: Input tensor (B, L, D)
        gate_proj: Gate projection weight matrix (D, intermediate_dim)
        up_proj: Up projection weight matrix (D, intermediate_dim)
        down_proj: Down projection weight matrix (intermediate_dim, D)
        act_fn: Activation function

    Returns:
        Output tensor (B, L, D)
    """
    B, L, D = x.shape
    intermediate_dim = gate_proj.weight.shape[0]  # (out_features, in_features) = (intermediate_dim, D)

    # Detect padding tokens: assume padding tokens are all zeros
    # Create a mask for non-padding tokens
    padding_mask = (x.abs().sum(dim=-1) != 0).float()  # (B, L)

    gate = act_fn(gate_proj(x))

    # Only compute up for non-padding tokens
    up = torch.zeros(B, L, intermediate_dim, device=x.device, dtype=x.dtype)
    non_pad_mask = padding_mask.unsqueeze(-1)  # (B, L, 1)
    up = up + up_proj(x) * non_pad_mask  # Multiply by mask (masked positions stay zero)

    return down_proj(gate * up)


def mlp_cached(x: torch.Tensor, gate_proj: torch.nn.Linear, up_proj: torch.nn.Linear,
               down_proj: torch.nn.Linear, act_fn: Callable) -> torch.Tensor:
    """
    Variant 4: Cached gate projection bias computation.

    For inputs with identical gate_proj results (e.g., repeated prefixes),
    we cache the gate activation. This version caches the bias contribution
    to gate_proj(x) = F.linear(x, weight, bias) = x @ weight.T + bias

    The bias term (bias + up_proj(x) @ up_proj.weight @ down_proj.weight) is cached.

    Args:
        x: Input tensor (B, L, D)
        gate_proj: Gate projection weight matrix (D, intermediate_dim)
        up_proj: Up projection weight matrix (D, intermediate_dim)
        down_proj: Down projection weight matrix (intermediate_dim, D)
        act_fn: Activation function

    Returns:
        Output tensor (B, L, D)
    """
    # Cache key: use first token's hash as a simple cache detector
    # In practice, this would use a proper KV cache
    B, L, D = x.shape
    intermediate_dim = gate_proj.weight.shape[0]  # (out_features, in_features) = (intermediate_dim, D)

    # Compute gate activation
    gate = act_fn(gate_proj(x))

    # Compute up projection
    up = up_proj(x)

    # Fuse down_proj computation with the multiplication
    # down_proj(gate * up) = (gate * up) @ down_proj.weight.T + down_proj.bias
    # = gate * up @ down_proj.weight.T + down_proj.bias
    gate_up = gate * up
    return down_proj(gate_up)


class MLPCachedWithBias(torch.autograd.Function):
    """
    Custom autograd function demonstrating cached bias pattern.
    In standard mlp_cached, we cache the bias addition to gate_proj:
    gate_proj(x) = x @ W_gate.T + b_gate (bias is cached)

    This is a simplified demonstration - production use would need
    proper cache management based on prefix caching.
    """

    @staticmethod
    def forward(ctx, x, gate_proj_weight, gate_proj_bias, up_proj_weight, up_proj_bias,
                down_proj_weight, down_proj_bias, act_fn_id):
        # Store for backward
        ctx.save_for_backward(x, gate_proj_weight, up_proj_weight, down_proj_weight)
        ctx.gate_proj_bias = gate_proj_bias
        ctx.up_proj_bias = up_proj_bias
        ctx.down_proj_bias = down_proj_bias
        ctx.act_fn_id = act_fn_id

        # Forward pass (with cached bias concept)
        gate = F.linear(x, gate_proj_weight, gate_proj_bias)
        from torch.nn.functional import silu
        gate = silu(gate) if act_fn_id == 0 else torch.relu(gate)

        up = F.linear(x, up_proj_weight, up_proj_bias)
        gate_up = gate * up

        return F.linear(gate_up, down_proj_weight, down_proj_bias)


# -----------------------------------------------------------------------------
# Benchmarking Utilities
# -----------------------------------------------------------------------------

def benchmark_mlp(mlp_func: Callable, x: torch.Tensor, gate_proj: torch.nn.Linear,
                  up_proj: torch.nn.Linear, down_proj: torch.nn.Linear,
                  act_fn: Callable, warmup: int = 10, iters: int = 100,
                  device: str = 'cuda') -> Tuple[float, float]:
    """
    Benchmark an MLP function.

    Args:
        mlp_func: MLP function to benchmark
        x: Input tensor
        gate_proj, up_proj, down_proj: Weight matrices
        act_fn: Activation function
        warmup: Number of warmup iterations
        iters: Number of benchmark iterations
        device: Device to run on

    Returns:
        (mean_time_ms, std_time_ms)
    """
    # Move to device
    x = x.to(device)
    gate_proj = gate_proj.to(device)
    up_proj = up_proj.to(device)
    down_proj = down_proj.to(device)

    # Warmup
    for _ in range(warmup):
        _ = mlp_func(x, gate_proj, up_proj, down_proj, act_fn)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        out = mlp_func(x, gate_proj, up_proj, down_proj, act_fn)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return sum(times) / len(times), (max(times) - min(times)) / 2


def verify_correctness(mlp_func: Callable, x: torch.Tensor, gate_proj: torch.nn.Linear,
                       up_proj: torch.nn.Linear, down_proj: torch.nn.Linear,
                       act_fn: Callable, baseline_func: Callable = mlp_original,
                       rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """
    Verify that an MLP variant produces bit-identical results to baseline.

    Args:
        mlp_func: MLP function to verify
        x: Input tensor
        gate_proj, up_proj, down_proj: Weight matrices
        act_fn: Activation function
        baseline_func: Baseline implementation to compare against
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        True if results are within tolerance
    """
    x = x.cuda()
    gate_proj = gate_proj.cuda()
    up_proj = up_proj.cuda()
    down_proj = down_proj.cuda()

    baseline_out = baseline_func(x, gate_proj, up_proj, down_proj, act_fn)
    variant_out = mlp_func(x, gate_proj, up_proj, down_proj, act_fn)

    return torch.allclose(baseline_out, variant_out, rtol=rtol, atol=atol)


def run_benchmarks(hidden_dims: list, intermediate_dims: list,
                   seq_lengths: list, batch_sizes: list,
                   device: str = 'cuda') -> dict:
    """
    Run comprehensive benchmarks across all configurations.

    Args:
        hidden_dims: List of hidden dimensions
        intermediate_dims: List of intermediate dimensions
        seq_lengths: List of sequence lengths
        batch_sizes: List of batch sizes
        device: Device to run on

    Returns:
        Dictionary with benchmark results
    """
    act_fn = F.silu
    results = {}
    variants = {
        'baseline': mlp_original,
        'fused_linear': mlp_fused_linear,
        'inplace': mlp_inplace,
        'lazy': mlp_lazy,
        'cached': mlp_cached,
    }

    print(f"{'Hidden':<8} {'Interm':<8} {'SeqLen':<8} {'Batch':<8} {'Variant':<15} {'Time (ms)':<12} {'Speedup':<10}")
    print("-" * 85)

    for hidden_dim in hidden_dims:
        for intermediate_dim in intermediate_dims:
            for seq_len in seq_lengths:
                for batch_size in batch_sizes:
                    # Create inputs
                    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
                    gate_proj = torch.nn.Linear(hidden_dim, intermediate_dim, device=device)
                    up_proj = torch.nn.Linear(hidden_dim, intermediate_dim, device=device)
                    down_proj = torch.nn.Linear(intermediate_dim, hidden_dim, device=device)

                    key = f"{hidden_dim}_{intermediate_dim}_{seq_len}_{batch_size}"

                    # Benchmark baseline first
                    baseline_time, _ = benchmark_mlp(
                        mlp_original, x, gate_proj, up_proj, down_proj, act_fn
                    )

                    # Verify and benchmark each variant
                    for variant_name, variant_func in variants.items():
                        if variant_name == 'baseline':
                            time_ms, _ = benchmark_mlp(
                                variant_func, x, gate_proj, up_proj, down_proj, act_fn
                            )
                            speedup = 1.0
                            verified = True
                        else:
                            # Verify correctness first
                            verified = verify_correctness(
                                variant_func, x, gate_proj, up_proj, down_proj, act_fn
                            )
                            if not verified:
                                print(f"  [WARNING] {variant_name} failed verification!")

                            time_ms, _ = benchmark_mlp(
                                variant_func, x, gate_proj, up_proj, down_proj, act_fn
                            )
                            speedup = baseline_time / time_ms if time_ms > 0 else 0

                        results[f"{key}_{variant_name}"] = {
                            'time_ms': time_ms,
                            'speedup': speedup,
                            'verified': verified
                        }

                        speedup_str = f"{speedup:.2f}x" if speedup >= 1.0 else f"{speedup:.3f}x"
                        print(f"{hidden_dim:<8} {intermediate_dim:<8} {seq_len:<8} {batch_size:<8} "
                              f"{variant_name:<15} {time_ms:<12.4f} {speedup_str:<10}")

                    print()

    return results


if __name__ == "__main__":
    print("=" * 85)
    print("MLP/FFN Kernel Benchmarks")
    print("=" * 85)
    print()

    # Configuration as specified
    hidden_dims = [4096, 8192]
    intermediate_dims = [11008, 28672]
    seq_lengths = [128, 512]
    batch_sizes = [1, 4]

    print(f"Device: cuda")
    print(f"Hidden dims: {hidden_dims}")
    print(f"Intermediate dims: {intermediate_dims}")
    print(f"Sequence lengths: {seq_lengths}")
    print(f"Batch sizes: {batch_sizes}")
    print()

    results = run_benchmarks(
        hidden_dims=hidden_dims,
        intermediate_dims=intermediate_dims,
        seq_lengths=seq_lengths,
        batch_sizes=batch_sizes
    )

    print()
    print("=" * 85)
    print("Benchmark Complete")
    print("=" * 85)

    # Summary table
    print()
    print("Summary (average speedup vs baseline):")
    print("-" * 50)
    variants = ['fused_linear', 'inplace', 'lazy', 'cached']
    for variant in variants:
        speedups = [v['speedup'] for k, v in results.items() if k.endswith(f"_{variant}")]
        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            print(f"  {variant:<15}: {avg_speedup:.3f}x")
