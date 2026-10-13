# Modified for the RepForcing source release; see NOTICE.md.
"""Train RepForcing with Stage-1 representation supervision or Stage-2 DMD."""
import argparse

from utils.runtime import dump_runtime_config, load_training_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="", help="Registered variant: rft.framewise, rft.chunkwise, dmd.framewise, or dmd.chunkwise")
    parser.add_argument("--config_path", default="", help="Full YAML configuration instead of --config")
    parser.add_argument("--data_path", default=None, help="Stage 1: clean-latent LMDB; Stage 2: one training prompt per line")
    parser.add_argument("--generator_ckpt", default=None, help="Generator initialization: optional for Stage 1; matching Stage-1 weights required for Stage 2")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", default=None, help="Default: logs/stage1 for Stage 1, logs/dmd for Stage 2")
    parser.add_argument("--wandb-save-dir", default="")
    logging = parser.add_mutually_exclusive_group()
    logging.add_argument("--enable-wandb", action="store_true", help="Use external W&B environment credentials")
    logging.add_argument("--disable-wandb", action="store_true", help="Disable W&B (the default)")
    parser.add_argument("--tf", action="store_true")
    args = parser.parse_args()
    try:
        config = load_training_config(args)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.error(str(error))

    if config.trainer == "score_distillation":
        from trainer import ScoreDistillationTrainer as Trainer
    elif config.trainer == "rft":
        if config.get("perceptual_type") != "dit":
            parser.error("Stage 1 only supports representation supervision with perceptual_type=dit")
        from trainer.rft import Trainer
    else:
        parser.error(f"Unsupported trainer: {config.trainer}")

    import wandb

    dump_runtime_config(config, config.logdir)
    trainer = Trainer(config)
    trainer.train()
    if config.trainer == "score_distillation":
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    main()
