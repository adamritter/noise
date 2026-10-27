# Experimental partial raw-key offset

This optional Track 3 experiment improves a **small-backbone pilot** with the
record's full-size hashed n-gram tables. It is **not a leaderboard submission**:
only one GH200 was available for these experiments, and the full 12 × 1024,
30M-token run on the required 8 × H100 node has not been run. The published
record defaults are unchanged.

## What changes

With `partial_rope=True, key_offset=True`, half of each head's raw-key channels
come from the previous token, with zero padding at the sequence start. For a
64-dimensional head these are channels `16:32` and `48:64`; their RoPE phase is
zero. The other channels retain the current token. Normalization follows this
assembly.

This is inspired by [modded-nanogpt's Partial Key Offset](https://github.com/KellerJordan/modded-nanogpt/pull/169),
but deliberately shifts **before** key normalization. It is not a bit-identical
port of the post-normalization shift. This placement keeps the local raw-key
finite-difference experiment at one assembled key. The measured raw-key credit
is then transported to its source positions by the transpose of the linear
shift before the key weight outer product. No parameters, autograd, or full
handwritten backward pass are added.

Independent input-key and future-token probes have **not** been adapted in this
patch: an input move would affect two adjacent keys and their shared attention
denominator. Unsupported combinations fail explicitly. The tested recipe is
Cut with `switch=0`, `key_path_draws=0`, and `future_draws=0`; both comparison
arms use these settings. Local raw-key finite differences remain enabled.

## Saved final-checkpoint results

Each run: one GH200, 4 blocks, width 128, 2 heads of 64, 512-wide ReLU² MLP,
vocabulary 4096, sequence length 2048. Training uses the same 512 distinct
sequences once: **1,048,576 prediction tokens / 512 updates**, without an end
merge. Seeds 0 and 1 change initialization and estimator randomness, not data
order. The three tables are bigram, trigram and skip-bigram value embeddings.

The following means use two initializations. All scores are **teacher-forced
next-token cross-entropy**, natural logarithms, measured from saved final
weights. The official metric uses `utils.eval.evaluate`; the extra held-out
slice contains 65,536 predictions.

| Table size, each of 3 | Variant | New held-out CE | Full official validation CE |
| --- | --- | ---: | ---: |
| 4096 × 32 | Base | 5.138423 | 5.123848 |
| 4096 × 32 | Partial RoPE + key offset | 4.918433 | 4.859301 |
| 1,048,576 × 128 | Base | 5.071707 | 5.051540 |
| 1,048,576 × 128 | Partial RoPE + key offset | 4.909240 | 4.851346 |

With full-size tables, official-validation improvements are 0.231265 and
0.169123 for seeds 0 and 1. Both arms have approximately 265.021 TMAC per run
under the same ATen forward-operation estimate, which excludes optimizer,
memory traffic, and custom Triton operations. Mean measured training times are
128.16 s for base and 129.28 s for offset, excluding warmup and evaluation.
These are not speedrun-record timings. Small-table timings are historical,
not a simultaneous hardware comparison.

The full-size hash tables contain 402,653,184 parameters. The historical pilot
had 406,651,145 total parameters including one unused, zero-valued `skip_alpha`
scalar. This focused port omits that unrelated scalar (406,651,144 parameters).
It also omits the unhelpful skip experiment. A separate small-table,
seed-0 partial-RoPE-only control scored 5.166359 on official validation versus
the base's 5.123193; simply reducing RoPE was worse. A large-table partial-only
control has not been run, so the large-table result concerns the combined
partial-RoPE + offset change.

## Data and evaluation discipline

The official hash-verified training and validation tensors and tokenizer are
unchanged. With `torch.randperm(TRAIN_ROWS, generator=torch.Generator().manual_seed(1))`,
training uses the first 512 rows, development uses `[-96:-80]`, and the new
confirmation slice uses `[-192:-160]`. The latter was checked against all 18
prior local routing/architecture runs and was unused for training or evaluation.
All four full-size-table runs finished before confirmation evaluation. The
older small-table checkpoints were re-evaluated on the same new slice without
retraining. Official validation had been used before, hence the separate new
confirmation slice. These are token-window splits, not a new document-level
deduplication claim.

There was no checkpoint selection by validation loss. Every model passed
saved/reloaded loss parity and a future-input intervention causality check.
Each final checkpoint was evaluated in its own process to avoid a compiler
cache limit changing the attention implementation across models; the maximum
CE difference from the initial shared-process evaluation was 0.000052.

The focused port also passed a four-update training regression against the
historical runner for both base and offset. With deterministic algorithms
enabled, all common final weights and the teacher-forced development losses
were bit-identical. Forward-only numerical checks verify the shift's transpose,
the key-weight directional finite difference, and causal behavior. The measured
pilot and port checks used PyTorch 2.10.0+cu130 on GH200; this does not establish
8-H100 timing or bit-identical reproduction with a different software stack.

Raw configurations, per-update training losses, final evaluations, checkpoint
hashes, the frozen protocol, and the negative control are in
[`log/key_offset_pilot/`](log/key_offset_pilot/). These are historical pilot logs
reformatted into `config.json`, `metrics.jsonl`, `result.json`, and
`evaluation.json`; per-update timestamps are not reconstructed. No checkpoints
or datasets are bundled in git. Configuration flags for the inactive skip
experiment remain in the historical logs for provenance.

## Reproduce the pilot

Prepare the official data as described in the repository README. Launch each
run in a separate process (default table dimensions are already full size):

```bash
uv run train/arch/check_key_offset.py
uv run train/arch/key_offset_pilot.py --arm base --seed 0 --count-macs --confirm --official-validation --output logs/key_offset/base0
uv run train/arch/key_offset_pilot.py --arm shift --seed 0 --count-macs --confirm --official-validation --output logs/key_offset/shift0
uv run train/arch/key_offset_pilot.py --arm base --seed 1 --count-macs --confirm --official-validation --output logs/key_offset/base1
uv run train/arch/key_offset_pilot.py --arm shift --seed 1 --count-macs --confirm --official-validation --output logs/key_offset/shift1
```

For the small-table comparison add `--ngram-rows 4096 --ngram-dim 32`; for the
partial-RoPE-only control use `--arm partial`. Every output directory contains
its own standalone `model.py` with the configuration baked in. Re-score it with:

```bash
uv run utils/eval.py logs/key_offset/shift0/final.pt --model logs/key_offset/shift0/model.py
```

The stock `train/arch/run.py` still reproduces the existing record by default.
Exploring the offset at the full architecture requires enabling both flags in
`GPTConfig` and setting `key_path_draws=0`; that changes the record recipe and
has not been validated at scale. A full-scale, rules-compliant comparison is
needed before proposing a leaderboard entry or changing the defaults.
