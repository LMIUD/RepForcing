#!/usr/bin/env python3
# Modified for the RepForcing source release; see NOTICE.md.
"""Generate VBench-standard samples with a RepForcing DMD checkpoint."""

import argparse
import json
import os
import time

import torch
from omegaconf import OmegaConf
from torchvision.io import write_video

from pipeline import CausalInferencePipeline
from utils.misc import set_seed


def load_vbench_prompts(path):
    """Load and deduplicate VBench prompts while preserving their first-seen order."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)

    items = []
    if isinstance(data, dict):
        for dimension, prompt_list in data.items():
            for entry in prompt_list:
                prompt = entry if isinstance(entry, str) else entry.get(
                    "prompt_en", entry.get("prompt", "")
                )
                if prompt:
                    items.append({"prompt": prompt, "dimension": dimension})
    elif isinstance(data, list):
        for entry in data:
            prompt = entry.get("prompt_en", entry.get("prompt", ""))
            dimension = entry.get("dimension", ["unknown"])
            if isinstance(dimension, list):
                dimension = dimension[0] if dimension else "unknown"
            if prompt:
                items.append({"prompt": prompt, "dimension": dimension})
    else:
        raise TypeError(f"Unsupported VBench prompt format: {type(data).__name__}")

    unique = []
    seen = set()
    for item in items:
        if item["prompt"] not in seen:
            seen.add(item["prompt"])
            unique.append(item)
    return unique


def attach_extended_prompts(items, raw_path, extended_path):
    with open(raw_path, encoding="utf-8") as handle:
        raw = [line.rstrip("\n") for line in handle if line.strip()]
    with open(extended_path, encoding="utf-8") as handle:
        extended = [line.rstrip("\n") for line in handle if line.strip()]
    if len(raw) != len(extended):
        raise ValueError(
            f"Raw and extended prompt counts differ: {len(raw)} != {len(extended)}"
        )

    mapping = {short.strip(): long for short, long in zip(raw, extended)}
    missing = 0
    for item in items:
        generated_prompt = mapping.get(item["prompt"].strip())
        if generated_prompt is None:
            missing += 1
            generated_prompt = item["prompt"]
        item["generation_prompt"] = generated_prompt
    return missing


def load_generator(pipeline, checkpoint_path, use_ema):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    requested_key = "generator_ema" if use_ema else "generator"
    if requested_key not in checkpoint:
        available = [key for key in ("generator", "generator_ema") if key in checkpoint]
        if not available:
            raise KeyError("Checkpoint contains neither generator nor generator_ema")
        requested_key = available[0]

    fixed_state = {}
    for key, value in checkpoint[requested_key].items():
        if key.startswith("model._fsdp_wrapped_module."):
            key = key.replace("model._fsdp_wrapped_module.", "model.", 1)
        fixed_state[key] = value

    missing, unexpected = pipeline.generator.load_state_dict(fixed_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    del checkpoint, fixed_state
    return requested_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompts_path", required=True)
    parser.add_argument("--extended_prompts_path", default="")
    parser.add_argument("--raw_prompts_path", default="")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_ema", action="store_true")
    args = parser.parse_args()

    if (args.extended_prompts_path and not args.raw_prompts_path) or (
        args.raw_prompts_path and not args.extended_prompts_path
    ):
        raise ValueError("Extended and raw prompt paths must be supplied together")

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed + args.rank)
    torch.set_grad_enabled(False)
    device = torch.device("cuda")

    config = OmegaConf.merge(
        OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(args.config)
    )
    pipeline = CausalInferencePipeline(config, device=device)
    loaded_key = load_generator(pipeline, args.ckpt, args.use_ema)
    print(f"[rank {args.rank}] loaded checkpoint key: {loaded_key}", flush=True)

    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to(device)
    pipeline.generator.to(device)
    pipeline.vae.to(device)

    _, num_frames, channels, height, width = list(config.image_or_video_shape)
    prompts = load_vbench_prompts(args.prompts_path)
    if args.extended_prompts_path:
        missing = attach_extended_prompts(
            prompts, args.raw_prompts_path, args.extended_prompts_path
        )
        print(f"[rank {args.rank}] extended prompt misses: {missing}", flush=True)
    else:
        for item in prompts:
            item["generation_prompt"] = item["prompt"]

    if args.limit > 0:
        prompts = prompts[: args.limit]
    shard = prompts[args.rank :: args.world_size]
    print(
        f"[rank {args.rank}] total_prompts={len(prompts)} shard={len(shard)} "
        f"samples_per_prompt={args.num_samples} shape="
        f"[{num_frames},{channels},{height},{width}]",
        flush=True,
    )

    generated = 0
    skipped = 0
    started = time.time()
    for prompt_index, item in enumerate(shard):
        for sample_index in range(args.num_samples):
            output_path = os.path.join(
                args.output_dir, f"{item['prompt']}-{sample_index}.mp4"
            )
            if os.path.isfile(output_path):
                skipped += 1
                continue

            noise = torch.randn(
                [1, num_frames, channels, height, width],
                device=device,
                dtype=torch.bfloat16,
            )
            video = pipeline.inference(
                noise=noise,
                text_prompts=[item["generation_prompt"]],
                return_latents=False,
                initial_latent=None,
            )
            frames = (
                video[0]
                .detach()
                .float()
                .clamp(0, 1)
                .mul(255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .cpu()
            )
            write_video(output_path, frames, fps=args.fps)
            pipeline.vae.model.clear_cache()
            generated += 1

        if (prompt_index + 1) % 5 == 0 or prompt_index + 1 == len(shard):
            seconds_per_video = (time.time() - started) / max(generated, 1)
            remaining = (len(shard) - prompt_index - 1) * args.num_samples
            print(
                f"[rank {args.rank}] prompts={prompt_index + 1}/{len(shard)} "
                f"generated={generated} skipped={skipped} "
                f"seconds_per_video={seconds_per_video:.2f} "
                f"eta_minutes={remaining * seconds_per_video / 60:.1f}",
                flush=True,
            )
        torch.cuda.empty_cache()

    print(
        f"[rank {args.rank}] DONE generated={generated} skipped={skipped}", flush=True
    )


if __name__ == "__main__":
    main()
