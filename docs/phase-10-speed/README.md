# Phase 10 — Speed Optimization (3.5h → ~21min target)

## Problem

10-second video generation took **3.5 hours** on RTX 4090 Laptop 16GB.
User requirement: **≤20 minutes**.

### Root Causes (from log analysis)

1. **VRAM thrashing in swap blocks**: 23 persistent blocks used 11.78 GiB
   during swap compute. With ~2.5 GiB intermediates, peak reached 14.3 GiB.
   CUDA reserved pool grew to 14.5–15.1 GiB, leaving only 0.9–1.5 GiB free.
   Windows WDDM spilled swap-block intermediates to system RAM → **5.1× slowdown**
   (8.1s/blk vs 1.58s/blk persistent).

2. **Redundant forward passes**: With `text_guide_scale=1.0` and
   `audio_guide_scale=2.5`, every denoising step ran 2 full model forward
   passes (cond + drop_audio). For 8 chunks × 4 steps = **64 total passes**.
   Even at ideal speed (1.0s/blk), 64 × 40s = 2,560s = 42.7 min.

3. **Per-swap empty_cache() overhead**: Each swap block called
   `torch.cuda.empty_cache()` adding ~0.15s × 17 blocks = 2.5s per pass.

### Full Calculation (before fixes)

```
Steady-state per forward pass:
  23 persistent × 1.58s/blk = 36.3s
  17 swapped   × 8.10s/blk = 137.7s
  Total = 174.5s

Per chunk: 4 steps × 2 passes × 174.5s = 1,396s
8 chunks: 8 × 1,396 = 11,168s = 186 min (steady state)
+ cuBLAS warmup penalty on chunk 0: ~1,400s
Grand total: ~210 min = 3.5 hours
```

## Fixes Applied

### Fix 1: Reduce persistent blocks (VRAM headroom)

`config.py`: `activation_reserve` 5.0 → **7.0 GiB**

Result: ~17 persistent / ~23 swapped (was 23/17).

```
17 persistent × 470MB = 7,990 MB
+ non-block (500MB) + swap buffer (470MB) = 8,960 MB
+ intermediates (~2,500 MB) = 11,460 MB peak
Headroom: 16,000 - 11,460 = 4,540 MB ← safe, no WDDM spill
```

### Fix 2: Single-pass denoising (audio_guide_scale=1.0)

`config.py`: `VIDEO_LORA_AUDIO_GUIDE_SCALE` 2.5 → **1.0**

When both `text_guide_scale≈1.0` AND `audio_guide_scale≈1.0`, the CFG
formula reduces to `noise_pred = noise_pred_cond` (proved for both APG
and vanilla modes). The second forward pass is mathematically redundant.

`multitalk.py`: Added `_skip_cfg` check — when true, skips all extra
forward passes. **Halves total passes: 64 → 32.**

The LoRA (`cfg_step_distill`) is specifically designed for low/no guidance.
Audio conditioning is still present in the cond pass; CFG only amplifies
the difference vs the no-audio baseline.

### Fix 3: Remove per-swap empty_cache()

`multitalk_model.py`: Removed `torch.cuda.empty_cache()` inside
`_run_blocks_with_swap` loop. The between-forward-pass `empty_cache()`
in the denoising loop is sufficient. Saves ~2.5s per pass.

## Projected Performance

```
All blocks at ~0.9-1.0s/blk (no WDDM spill):
  Per pass: 17×0.9 + 23×(0.9+0.08) = 15.3 + 22.5 = 37.8s
  32 passes: 32 × 37.8 = 1,210s
  + overhead (VAE, chunk prep): ~120s
  Total: ~1,330s ≈ 22 min

At best observed speed (0.86s/blk):
  Per pass: 17×0.86 + 23×0.94 = 14.6 + 21.6 = 36.2s
  32 × 36.2 + 120 = 1,278s ≈ 21.3 min
```

## Files Modified

- `config.py` — activation_reserve 5.0→7.0 GiB, audio_guide_scale 2.5→1.0, ETA 35→42s
- `vendor/infinitetalk/wan/multitalk.py` — skip_cfg single-pass optimization
- `vendor/infinitetalk/wan/modules/multitalk_model.py` — remove per-swap empty_cache()
- `video_engine.py` — log passes/step diagnostic

## Quality Notes

- Audio conditioning is still present in the cond pass (model receives audio embeddings)
- CFG amplification removed (was 2.5×) — lip sync emphasis will be softer
- If quality is insufficient, user can increase `--audio-cfg` (e.g. 1.5 or 2.0)
  which re-enables the 2nd forward pass but adds ~50% more time
