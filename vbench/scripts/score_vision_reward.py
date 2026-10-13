#!/usr/bin/env python3
"""Evaluate VisionReward and separate Instruction Following using the fixed official protocol."""

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = Path(__file__).resolve().parent / "vision_reward"
OFFICIAL_COMMIT = "511960d61a777b2008cd486a85a090ba96792a32"
IF_QUESTION = 'Does the video meet some of the requirements stated in the text "[[prompt]]"?'
GENERATION_KWARGS = {
    "max_new_tokens": 2048,
    "pad_token_id": 128002,
    "top_k": 1,
    "do_sample": False,
    "top_p": 0.1,
    "temperature": 0.1,
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_json(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_protocol(asset_dir=ASSET_DIR):
    source = json.loads((asset_dir / "source.json").read_text())
    if source["commit"] != OFFICIAL_COMMIT:
        raise ValueError("Official protocol version does not match this script")
    for name, info in source["files"].items():
        if sha256(asset_dir / name) != info["sha256"]:
            raise ValueError(f"Official protocol file checksum mismatch: {name}")
    # Preserve trailing newlines to match the official readlines input tokens.
    questions = (asset_dir / "questions.txt").read_text().splitlines(keepends=True)
    weights = json.loads((asset_dir / "weights.json").read_text())
    if len(questions) != 29 or len(weights) != 29 or not all(q.strip() for q in questions):
        raise ValueError("The official protocol requires exactly 29 questions and weights")
    if not all(isinstance(w, (float, int)) and math.isfinite(w) for w in weights):
        raise ValueError("Weights must be finite numbers")
    candidates = (asset_dir / "all_questions.txt").read_text().splitlines(keepends=True)
    instruction = [q for q in candidates if q.rstrip("\r\n") == IF_QUESTION]
    if len(instruction) != 1 or any(q.rstrip("\r\n") == IF_QUESTION for q in questions):
        raise ValueError("The IF question must be separate from the 29 overall-score questions")
    return questions, weights, instruction[0], source


def encode_answer(answer):
    # Match the official first-token rule without probability or text normalization.
    return 1 if answer == "yes" else -1


def aggregate_answers(answers, weights, instruction_answer):
    if len(answers) != len(weights) or not answers:
        raise ValueError("Answer count does not match the weights")
    values = [encode_answer(answer) for answer in answers]
    weighted = np.asarray(values, dtype=np.float64) * np.asarray(weights, dtype=np.float64)
    return {
        "answer_values": values,
        "weighted_values": weighted.tolist(),
        "vision_reward": float(np.mean(weighted)),
        "instruction_following": encode_answer(instruction_answer),
        "unexpected_answers": [i for i, a in enumerate(answers) if a not in ("yes", "no")],
        "unexpected_instruction_answer": instruction_answer not in ("yes", "no"),
    }


def resolve_inputs(videos_dir, prompts_path):
    prompts = [line.rstrip() for line in prompts_path.read_text(encoding="utf-8").splitlines()]
    names = [prompt[:100] + ".mp4" for prompt in prompts]
    if len(prompts) != 100 or not all(prompts) or len(set(names)) != 100:
        raise ValueError("Expected 100 nonempty full prompts with distinct filename prefixes")
    if not videos_dir.is_dir():
        raise FileNotFoundError(videos_dir)
    actual = {p.name for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"}
    if actual != set(names):
        raise ValueError(f"Videos do not match prompts: {len(set(names) - actual)} missing, {len(actual - set(names))} extra")
    return [{"video_path": str((videos_dir / name).resolve()), "prompt": prompt}
            for name, prompt in zip(names, prompts)]


def file_state(path):
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def model_state(model_path):
    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    shards = sorted(set(index["weight_map"].values()))
    states = [file_state(model_path / name) for name in shards]
    metadata = [p for p in model_path.iterdir() if p.suffix in (".json", ".py")]
    return {
        "path": str(model_path),
        "metadata_sha256": {p.name: sha256(p) for p in sorted(metadata)},
        "weight_shards_size_mtime": states,
    }


def environment_versions():
    names = ("torch", "torchvision", "triton", "transformers", "tokenizers",
             "decord", "pytorchvideo", "numpy", "accelerate", "sentencepiece")
    return {name: importlib.metadata.version(name) for name in names}


def build_manifest(args, items, source, instruction):
    return {
        "schema_version": 1,
        "scorer_sha256": sha256(Path(__file__)),
        "model": model_state(args.model_path),
        "videos_dir": str(args.videos_dir),
        "inputs": [dict(item, file=file_state(Path(item["video_path"]))) for item in items],
        "prompts_path": str(args.prompts),
        "prompts_sha256": sha256(args.prompts),
        "requested_videos": len(items),
        "dataset_videos": 100,
        "protocol": source,
        "instruction_question": instruction,
        "generation_kwargs": GENERATION_KWARGS,
        "frame_sampling": "official_chat_nearest_each_second_max_24",
        "answer_rule": "first_token_exact_yes=1;otherwise=-1",
        "aggregate_rule": "mean_over_videos(mean_over_29(answer*weight))",
        "torch_dtype": "bfloat16",
        "python": sys.executable,
        "python_version": sys.version,
        "package_versions": environment_versions(),
    }


def read_records(path, items, weights):
    if not path.exists():
        return {}
    expected = {item["video_path"]: item["prompt"] for item in items}
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.endswith("\n"):
                raise ValueError("Truncated per-video record; refusing to discard data automatically")
            record = json.loads(line)
            key = record["video_path"]
            if key in records or key not in expected or record["prompt"] != expected[key]:
                raise ValueError("Resume records contain duplicate videos or do not match the current inputs")
            calculated = aggregate_answers(record["answers"], weights, record["instruction_answer"])
            for name, value in calculated.items():
                if record[name] != value:
                    raise ValueError(f"Resume record score validation failed: {key} / {name}")
            if record["unexpected_answers"] or record["unexpected_instruction_answer"]:
                raise ValueError("Resume records contain unexpected answers; inspect the model outputs first")
            records[key] = record
    return records


def ensure_video_transform_compatibility():
    import torchvision.transforms

    legacy_name = "torchvision.transforms.functional_tensor"
    if importlib.util.find_spec(legacy_name) is None:
        from torchvision.transforms import _functional_tensor

        # Restore the pytorchvideo 0.1.5 import path without modifying installed files.
        sys.modules[legacy_name] = _functional_tensor
        torchvision.transforms.functional_tensor = _functional_tensor


def load_model(model_path):
    ensure_video_transform_compatibility()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA device is required; precision is not changed automatically")
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    required = index["metadata"]["total_size"] + 8 * 1024**3
    free, _ = torch.cuda.mem_get_info(0)
    if free < required:
        raise RuntimeError(
            f"Insufficient free VRAM: {free / 1024**3:.1f} GiB; "
            f"at least {required / 1024**3:.1f} GiB is required before loading (weights plus 8 GiB temporary space)"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, low_cpu_mem_usage=True,
    ).eval().to("cuda:0")
    return model, tokenizer


def load_video(video_path):
    from decord import VideoReader, bridge, cpu

    bridge.set_bridge("torch")
    reader = VideoReader(io.BytesIO(video_path.read_bytes()), ctx=cpu(0))
    total = len(reader)
    if not total:
        raise ValueError(f"Empty video: {video_path}")
    timestamps = [value[0] for value in reader.get_frame_timestamp(np.arange(total))]
    frame_ids = []
    for second in range(round(max(timestamps)) + 1):
        closest = min(timestamps, key=lambda timestamp: abs(timestamp - second))
        frame_ids.append(timestamps.index(closest))
        if len(frame_ids) >= 24:
            break
    frames = reader.get_batch(frame_ids).permute(3, 0, 1, 2)
    return frames, {
        "total_frames": total,
        "fps": reader.get_avg_fps(),
        "sampled_frame_ids": frame_ids,
        "sampled_shape_cthw": list(frames.shape),
    }


def answer_question(model, tokenizer, video, query):
    import torch

    conversation = model.build_conversation_input_ids(
        tokenizer=tokenizer, query=query, images=[video], history=[], template_version="chat",
    )
    inputs = {
        "input_ids": conversation["input_ids"].unsqueeze(0).to("cuda:0"),
        "token_type_ids": conversation["token_type_ids"].unsqueeze(0).to("cuda:0"),
        "attention_mask": conversation["attention_mask"].unsqueeze(0).to("cuda:0"),
        "images": [[conversation["images"][0].to(device="cuda:0", dtype=torch.bfloat16)]],
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **GENERATION_KWARGS)
        first_token = outputs[:, inputs["input_ids"].shape[1]]
    return tokenizer.decode(first_token[0])


def score_video(model, tokenizer, item, questions, weights, instruction):
    started = time.monotonic()
    video, metadata = load_video(Path(item["video_path"]))
    answers = [answer_question(model, tokenizer, video, q.replace("[[prompt]]", item["prompt"]))
               for q in questions]
    instruction_answer = answer_question(
        model, tokenizer, video, instruction.replace("[[prompt]]", item["prompt"]),
    )
    return dict(
        item, video_metadata=metadata, answers=answers, instruction_answer=instruction_answer,
        elapsed_seconds=time.monotonic() - started,
        **aggregate_answers(answers, weights, instruction_answer),
    )


def make_summary(records, requested, error_events=0):
    values = list(records.values())
    unexpected = sum(len(r["unexpected_answers"]) + r["unexpected_instruction_answer"] for r in values)
    done = len(values) == requested
    status = "complete" if done and requested == 100 and not unexpected else "partial"
    if unexpected:
        status = "invalid_answers"
    vr = float(np.mean([r["vision_reward"] for r in values])) if values else None
    instruction = float(np.mean([r["instruction_following"] for r in values])) if values else None
    return {
        "status": status,
        "complete_requested_subset": done,
        "requested_videos": requested,
        "dataset_videos": 100,
        "num_videos": len(values),
        "num_failed_or_pending": requested - len(values),
        "num_error_events": error_events,
        "num_unexpected_answers": unexpected,
        "vision_reward": vr,
        "vision_reward_pct": 100 * vr if vr is not None else None,
        "instruction_following": instruction,
        "instruction_following_pct": 100 * instruction if instruction is not None else None,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the official VisionReward score and separate IF on 100 videos")
    parser.add_argument("videos_dir", type=Path)
    parser.add_argument("output_dir", type=Path, nargs="?")
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "checkpoints/VisionReward-Video")
    parser.add_argument("--prompts", type=Path, default=PROJECT_ROOT / "prompts/demos.txt")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N videos for a small test; output is marked partial")
    parser.add_argument("--resume", action="store_true", help="Resume an output directory with a matching manifest")
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 100:
        parser.error("--limit must be between 1 and 100")
    if args.resume and args.output_dir is None:
        parser.error("--resume requires an existing output directory")
    for name in ("videos_dir", "model_path", "prompts"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        args.output_dir = PROJECT_ROOT / "vbench/results" / f"vision_reward_{args.videos_dir.name}_{stamp}"
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def main():
    args = parse_args()
    questions, weights, instruction, source = load_protocol()
    all_items = resolve_inputs(args.videos_dir, args.prompts)
    items = all_items[:args.limit] if args.limit else all_items
    manifest = build_manifest(args, items, source, instruction)
    manifest_path = args.output_dir / "run_manifest.json"
    if args.resume:
        if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume manifest does not match the model, inputs, protocol, or environment")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json(manifest_path, manifest)
    results_path = args.output_dir / "video_results.jsonl"
    records = read_records(results_path, items, weights)
    errors_path = args.output_dir / "errors.jsonl"
    error_events = len(errors_path.read_text().splitlines()) if errors_path.exists() else 0
    print(f"Input: {args.videos_dir}\nOutput: {args.output_dir}\nRequested: {len(items)}/100, completed: {len(records)}", flush=True)
    model = tokenizer = None
    current = None
    try:
        if len(records) != len(items):
            print(f"Loading local model: {args.model_path}", flush=True)
            model, tokenizer = load_model(args.model_path)
            for i, current in enumerate(items, 1):
                if current["video_path"] in records:
                    continue
                print(f"[{i}/{len(items)}] {Path(current['video_path']).name}", flush=True)
                record = score_video(model, tokenizer, current, questions, weights, instruction)
                append_json(results_path, record)
                records[current["video_path"]] = record
                write_json(args.output_dir / "summary.json", make_summary(records, len(items), error_events))
                if record["unexpected_answers"] or record["unexpected_instruction_answer"]:
                    raise RuntimeError("Unexpected first token other than yes/no; raw answers saved, evaluation stopped")
                print(f"VR={100 * record['vision_reward']:.4f}, IF={100 * record['instruction_following']}", flush=True)
    except Exception as exc:
        error_events += 1
        append_json(errors_path, {"video": current, "type": type(exc).__name__, "message": str(exc)})
        write_json(args.output_dir / "summary.json", make_summary(records, len(items), error_events))
        raise
    summary = make_summary(records, len(items), error_events)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
