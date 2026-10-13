# Evaluation Tools

This directory contains evaluation code and scoring-protocol resources, not videos, models, caches, or historical scores. Run all commands from the project root.

The tools use different protocols; do not reuse arbitrary video sets across all three:

| Tool | Input and naming | Purpose |
| :--- | :--- | :--- |
| Standard VBench | Official prompts; `<full-prompt>-<sample-index>.mp4` | Standard dimensions and aggregate scores |
| Dynamic Degree | 100 MP4 files, with no other images or GIFs in the directory | Dynamic degree in `custom_input` mode |
| VisionReward / IF | 100 prompts and matching `<first-100-prompt-characters>.mp4` files | Fixed 29-question score and separate IF |

## 1. Standard VBench

Install compatible PyTorch / torchvision versions and official VBench in a separate evaluation environment:

```bash
python -m pip install -r vbench/requirements.txt
```

Follow the installed VBench version's requirements for evaluation models, licenses, and caches. Use `VBENCH_CACHE_DIR`, `HF_HOME`, and `TORCH_HOME` to select cache locations. This repository does not configure private proxies or link or modify model caches in your HOME directory.

### Sampling

Generate videos in the **RepForcing training/inference environment**, not an environment containing only VBench.

Place the official `VBench_full_info.json`, available from the installed VBench package, at `dataset/VBench_full_info.json`. Do not silently substitute expanded prompts for the official input. The sampler also accepts corresponding original/expanded prompt files in the same order when explicitly requested.

```bash
sh vbench/run_sample.sh \
  --config logs/dmd_framewise/runtime_config.yaml \
  --ckpt checkpoints/dmd_framewise.pt \
  --prompts_path dataset/VBench_full_info.json \
  --output_dir output/vbench_framewise \
  --num_samples 5 \
  --use_ema
```

`--config` reads the complete YAML saved during training, not a registered configuration name. Use the matching configuration and checkpoint for chunk-wise models. `--rank` and `--world_size` can partition prompts manually; set visible GPUs externally for each process. `--limit` supports small checks, whose outputs must not be reported as a full standard evaluation.

### Scoring

Switch to the VBench evaluation environment:

```bash
python vbench/scripts/score.py \
  --videos_dir output/vbench_framewise \
  --output_dir vbench/results/framewise \
  --name framewise
```

By default, all standard dimensions are evaluated to produce Quality, Semantic, and Total scores. To evaluate selected dimensions, pass `--allow_partial`; incomplete dimensions cannot produce a full aggregate score:

```bash
python vbench/scripts/score.py \
  --videos_dir output/vbench_framewise \
  --output_dir vbench/results/framewise_dynamic \
  --dimensions dynamic_degree \
  --allow_partial
```

Use `scripts/merge_scores.py` to aggregate existing per-dimension results:

```bash
python vbench/scripts/merge_scores.py \
  --partial_root vbench/results/partial \
  --output_dir vbench/results/merged \
  --name framewise
```

Sampling skips existing videos with the same filename. Use a new output directory to avoid mixing checkpoints, prompts, or sampling settings.

## 2. Standalone Dynamic Degree

This entry point uses a **100-video** protocol, not a full standard VBench evaluation. It can score videos generated from `prompts/demos.txt` by the main inference entry point.

Prepare the official VBench RAFT weights under:

```text
.cache/vbench/raft_model/models/raft-things.pth
```

```bash
VBENCH_CACHE_DIR=.cache/vbench sh vbench/run_dynamic_degree.sh \
  output/framewise vbench/results/dynamic_framewise
```

The script checks the input count, RAFT weights, and output directory. Missing weights cause an error rather than an automatic download. It uses VBench `custom_input` mode and verifies coverage of every input video, binary per-video decisions, and their agreement with the aggregate score. The output directory must not already exist.

## 3. VisionReward and Instruction Following

Use another separate environment. Install compatible GPU dependencies first, then:

```bash
python -m pip install -r vbench/scripts/requirements-visionreward.txt
```

Its Transformers version differs from the training environment; do not merge the two installations. Prepare the official VisionReward-Video model and tokenizer at `checkpoints/VisionReward-Video`, or select another location with `--model-path`. Review the model's custom code and obtain it from a trusted source.

```bash
python vbench/scripts/score_vision_reward.py \
  output/framewise vbench/results/vision_framewise \
  --model-path checkpoints/VisionReward-Video \
  --prompts prompts/demos.txt
```

Protocol rules:

- Inputs must correspond one-to-one with the 100 prompts. Video filenames are the first 100 prompt characters followed by `.mp4`.
- The score uses the fixed 29 questions and coefficients identified by the source manifest. IF is a separate question, excluded from those 29 terms.
- An exact first-token `yes` maps to +1; other answers map to -1 under the original protocol. Answers other than `yes` or `no` are recorded and stop evaluation.
- `vision_reward_pct` is the raw weighted score multiplied by 100, not an accuracy percentage. IF is not ordinary answer accuracy either.
- `--limit N` scores only the first N entries but still requires the complete 100-video input set. The output is marked as partial.
- Resuming requires explicit `--resume` and the original output directory. Changes to inputs, models, protocols, or the environment are rejected.
- The manifest also records the scoring script's SHA-256. Editing the script, including its comments, invalidates an older run's resume manifest. Use a new output directory rather than bypassing the check.

`vbench/scripts/vision_reward/weights.json` contains **29 scoring coefficients**, not neural-network weights. It is retained with the question files, `source.json`, and `LICENSE` to verify protocol provenance and file hashes.

## Tests Without Model Weights

```bash
PYTHONDONTWRITEBYTECODE=1 python -B vbench/scripts/test_score_vision_reward.py
```

These tests cover the protocol, answer mapping, input correspondence, and resume validation. They do not validate real model outputs or a full GPU evaluation.
