import numpy as np
import random
import torch


def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def merge_dict_list(dict_list):
    if len(dict_list) == 1:
        return dict_list[0]

    merged_dict = {}
    for k, v in dict_list[0].items():
        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                merged_dict[k] = torch.stack([d[k] for d in dict_list], dim=0)
            else:
                merged_dict[k] = torch.cat([d[k] for d in dict_list], dim=0)
        else:
            # for non-tensor values, we just copy the value from the first item
            merged_dict[k] = v
    return merged_dict


def resolve_grad_accum(config, world_size):
    """Micro-batches per optimizer step, derived so the global batch stays fixed.

    `total_batch_size` is the invariant of an experiment; the accumulation factor
    adapts to however many ranks the job actually got. This lets the same config
    run on 8 or 16 GPUs without touching the learning rate.
    """
    batch_size = int(getattr(config, "batch_size", 1))
    total = int(getattr(config, "total_batch_size", batch_size * world_size))
    denom = batch_size * world_size
    if denom <= 0:
        raise ValueError(f"batch_size*world_size must be positive, got {denom}")
    if total % denom != 0:
        raise ValueError(
            f"total_batch_size={total} is not divisible by batch_size*world_size={denom} "
            f"(batch_size={batch_size}, world_size={world_size}). "
            f"Pick a combination that divides evenly."
        )
    return max(total // denom, 1)


def merge_scalar_logs(dict_list):
    """Stack 0-d tensors across micro-batches; keep the last value for the rest.

    Callers already reduce with `.mean()` at log time, so stacking is enough to
    make a logged scalar the average over the whole accumulated batch. Non-scalar
    entries (latents kept around for visualization) are deliberately not
    concatenated -- one micro-batch's worth is all the visualizers use, and
    concatenating would grow memory linearly with the accumulation factor.
    """
    if len(dict_list) == 1:
        return dict_list[0]
    merged = dict(dict_list[-1])
    for k, v in dict_list[0].items():
        if isinstance(v, torch.Tensor) and v.ndim == 0:
            merged[k] = torch.stack([d[k] for d in dict_list if k in d], dim=0)
    return merged


def resolve_guidance(args):
    """(real, fake) guidance scales in the c + w(c-u) parameterization.

    Single source of truth for the fallback: configs may set the pair explicitly,
    otherwise `guidance_scale` drives the real side and the fake side is 0.0,
    which means "no guidance". Used both by model/dmd.py and by the trainer for
    naming, so the two cannot drift apart.
    """
    if hasattr(args, "real_guidance_scale"):
        return float(args.real_guidance_scale), float(args.fake_guidance_scale)
    return float(args.guidance_scale), 0.0
