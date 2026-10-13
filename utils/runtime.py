"""Public configuration handling without importing CUDA model code."""
import os
from pathlib import Path

from omegaconf import OmegaConf

from configs.registry import get_config


def public_config(config):
    """Return a copy suitable for logging, without authentication fields."""
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    if isinstance(config, dict):
        return {
            key: public_config(value)
            for key, value in config.items()
            if str(key).lower() not in {"key", "token", "password", "secret", "authorization", "wandb_key"}
            and not str(key).lower().endswith(("_api_key", "api_key", "_token", "_password", "_secret"))
        }
    if isinstance(config, (list, tuple)):
        return [public_config(value) for value in config]
    return config


def load_training_config(args):
    """Load a registered config or YAML, then apply explicit CLI overrides."""
    if bool(args.config) == bool(args.config_path):
        raise ValueError("provide exactly one of --config or --config_path")
    if args.config:
        config = OmegaConf.create(get_config(args.config).to_dict())
        name = args.config
    else:
        default_path = Path(__file__).resolve().parents[1] / "configs/default_config.yaml"
        config = OmegaConf.merge(OmegaConf.load(default_path), OmegaConf.load(args.config_path))
        name = Path(args.config_path).stem
    config = OmegaConf.create(public_config(config))
    for key in ("data_path", "generator_ckpt"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if config.trainer == "score_distillation" and not config.get("generator_ckpt"):
        raise ValueError("DMD requires --generator_ckpt pointing to a matching Stage-1 checkpoint")
    checkpoint = config.get("generator_ckpt")
    if checkpoint and not Path(checkpoint).is_file():
        raise FileNotFoundError(f"Generator checkpoint not found: {checkpoint}")
    if not config.get("data_path") or not Path(config.data_path).exists():
        raise FileNotFoundError(f"Training data not found: {config.get('data_path')}")
    config.config_name = name
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize
    config.tf = args.tf
    default_logdir = "logs/stage1" if config.trainer == "rft" else "logs/dmd"
    config.logdir = args.logdir if args.logdir is not None else default_logdir
    config.wandb_save_dir = args.wandb_save_dir or config.logdir
    config.disable_wandb = not getattr(args, "enable_wandb", False) or getattr(args, "disable_wandb", False)
    return config


def dump_runtime_config(config, logdir):
    """Save non-secret configuration on rank zero, before NCCL initialization."""
    if not logdir or int(os.environ.get("RANK", "0")) != 0:
        return
    Path(logdir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(public_config(config)), Path(logdir) / "runtime_config.yaml")
