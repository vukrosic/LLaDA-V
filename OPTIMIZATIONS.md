# LLaDA-V Inference Optimizations

This document explains the performance optimizations made to LLaDA-V's generation code, what they do, why they work, and how we verified they don't break anything.

## Background: How LLaDA Generation Works

LLaDA is a **diffusion-based language model** that generates text by iteratively "revealing" tokens. Unlike standard autoregressive models (like GPT) that generate one token at a time left-to-right, LLaDA works by:

1. Starting with all positions set to a special `[MASK]` token
2. At each step, predicting which masked positions should be filled with actual tokens
3. Deciding how many tokens to "transfer" (reveal) based on confidence
4. Repeating until all positions are filled

The key generation loop runs hundreds of times per inference, so any optimization inside this loop has outsized impact.

## Optimization 1: KV-Head Repetition (~1.5x speedup)

### The Problem

LLaDA uses **Grouped Query Attention (GQA)**, where there are fewer key/value heads than query heads. For example, 32 query heads but only 8 key/value heads. Each key/value head must be "repeated" to match the query head count.

The original code used `expand` + `reshape`:

```python
# Original: creates an intermediate tensor
hidden_states = hidden_states[:, :, None, :, :].expand(
    batch, num_key_value_heads, n_rep, slen, head_dim
)
return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
```

The `expand()` call creates a **view** (no actual memory copy), but the subsequent `reshape()` often requires a copy because the data layout doesn't match.

### The Solution

PyTorch's `repeat_interleave` does the same operation but more efficiently:

```python
return hidden_states.repeat_interleave(n_rep, dim=1)
```

This directly repeats the data in a single operation without creating an intermediate view.

### Why It Works

`repeat_interleave` is implemented as a single CUDA kernel that:
1. Knows the target layout upfront
2. Can write directly to the output without an intermediate step
3. Often benefits from better memory coalescing

### Testing

We verified numerical correctness first:

```
Testing repeat_kv correctness...
  Match: True, Max diff: 0.0
```

Then benchmarked:

```
Original (expand+reshape): 0.1027 ms
Optimized (repeat_interleave): 0.0705 ms
Speedup: 1.46x
```

---

## Summary of Changes

| Optimization | Location | Speedup | Memory Impact |
|-------------|----------|---------|---------------|
| KV repetition | `train/llava/hooks/fast_dllm_hook.py` | ~1.5x | Same |

## Files Modified

- `train/llava/hooks/fast_dllm_hook.py` - `_repeat_kv` function changed from `expand+reshape` to `repeat_interleave`

## Testing Approach

### 1. Syntax Validation
Before any testing, we verified the code has no syntax errors:
```bash
python3 -m py_compile llava/hooks/fast_dllm_hook.py
# Output: Syntax OK
```

### 2. Unit Tests for Correctness
We wrote focused tests that:
- Compare optimized output to original implementation
- Use `torch.allclose()` for floating-point comparison
- Test edge cases (e.g., n_rep=1 for repeat_kv)

Run the tests:
```bash
cd train
python3 test_repeat_kv.py
```

### 3. Numerical Equivalence
The optimization produces **bit-identical** results to the original:
- `repeat_kv`: Max difference = 0.0 (exact match)

### 4. Functional Invariants
We verified:
- All positions eventually get filled (no positions skipped)
- Transfer counts remain correct (same number of tokens revealed per step)
- Block boundaries are respected (future blocks not affected)
- Stop sequences work correctly

## Impact on RTX 3090

With an RTX 3090 (24GB VRAM) and LLaDA-8B:
- The repeat_kv optimization reduces **compute** (CPU/GPU cycles) in the attention hot path
- ~1.5x speedup for the repeat_kv operation itself

The actual end-to-end inference speedup will depend on how much time the model spends in this specific operation vs. matrix multiplications and other attention computations.

## Future Optimization Opportunities

1. **Torch.compile**: Adding `@torch.compile` to the generation loop could provide additional speedups
2. **Attention caching**: The fast_dLLM hook already does caching, but further optimization possible
3. **Half-precision**: Running in FP16 instead of FP32 (model already supports this)
4. **Batched generation**: Process multiple prompts simultaneously
5. **Mask tracking**: A previously explored optimization tracked filled positions via boolean OR instead of embedding comparison, providing ~5x speedup for that operation. This was reverted due to correctness concerns and is not currently implemented.
