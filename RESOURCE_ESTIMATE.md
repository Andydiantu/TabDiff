# TabDiff Resource Estimate (VRAM + Runtime)

This document estimates the hardware needed to reproduce TabDiff results, based on the current repository configuration and training loop behavior.

## What the repo is configured to do

- Training uses `batch_size = 4096` and `steps = 8000` by default.  
- The model backbone is `UniModMLP` with `dim_t = 1024`, `num_layers = 2`, `d_token = 4`.  
- The training loop iterates over the full dataloader **inside each epoch**, so total optimizer updates are:

```text
updates = steps * len(train_loader)
```

(Here, `steps` acts like epoch count, not gradient-step count.)

## Why VRAM requirement is modest

The network is relatively small:
- `d_token = 4` keeps transformer token width tiny.
- Most parameters live in the MLP block (`dim_t=1024`) and are still only on the order of ~10M parameters for the largest tabular schema in this repo.

A rough worst-case memory budget for training (FP32) is:
- Model params + grads + Adam states: typically a few hundred MB.
- EMA copy of denoiser and schedules: additional tens of MB.
- Activations with `batch_size=4096`: usually under ~1–2 GB for this architecture.
- Framework + CUDA workspace + dataloader overhead: additional margin needed.

## Practical VRAM recommendation

- **Minimum likely workable**: **~6 GB VRAM** (tight, may need to reduce batch size if spikes happen).
- **Recommended**: **8 GB VRAM** for stable training at default settings.
- **Comfortable**: **12+ GB VRAM** if you want extra headroom for concurrent processes or profiling.

If you hit OOM, reduce:
1. `train.main.batch_size` (first lever), and then
2. `sample.batch_size` during evaluation/report runs.

## Runtime estimate to reproduce results

Runtime depends heavily on GPU class, but because the model is small, wall-clock is mainly from the total number of updates:

```text
updates = 8000 * len(train_loader)
```

For typical dataset sizes in this benchmark family, `len(train_loader)` is often in the low single digits to ~10 at `batch_size=4096`, so total updates commonly land around **25k–80k**.

Rule-of-thumb runtime for one full training run at defaults:
- **High-end datacenter GPU (A100/H100 class)**: roughly **4–12 hours**.
- **Prosumer GPU (RTX 3090/4090 class)**: roughly **8–24 hours**.
- **Older midrange GPU**: can extend to **1–2+ days**.

Additional evaluation/reporting (`--mode test --report`) and DCR/imputation experiments add extra time beyond the core training run.

## Fast local sanity-check before full reproduction

Before full runs, launch one dataset training and monitor utilization:

```bash
python main.py --dataname adult --mode train --no_wandb
```

Then watch GPU memory and throughput with:

```bash
nvidia-smi -l 2
```

This gives an immediate empirical bound for your exact hardware.
