"""MLP/FFN kernel variants for LLaDA."""

from .mlp_ffn import (
    mlp_original,
    mlp_fused_linear,
    mlp_inplace,
    mlp_lazy,
    mlp_cached,
    benchmark_mlp,
    verify_correctness,
    run_benchmarks,
)

__all__ = [
    "mlp_original",
    "mlp_fused_linear",
    "mlp_inplace",
    "mlp_lazy",
    "mlp_cached",
    "benchmark_mlp",
    "verify_correctness",
    "run_benchmarks",
]
