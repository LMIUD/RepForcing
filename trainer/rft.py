# Modified for the RepForcing source release; see NOTICE.md.
import gc
import logging
import os
import random
import time

import torch
import torch.distributed as dist
import wandb
from torchvision.io import write_video

from model.diffusion_rft import CausalDiffusionRFT
from utils.runtime import public_config
from pipeline import CausalDiffusionInferencePipeline, CausalInferencePipeline
from utils.dataset import cycle, LatentLMDBDataset
from utils.distributed import EMA_FSDP, barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import merge_scalar_logs, resolve_grad_accum, set_seed


RFT_VARIANTS = {
    "rft": CausalDiffusionRFT,
}


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        run_id_list = [time.strftime("%Y%m%d_%H%M%S") if self.is_main_process else None]
        dist.broadcast_object_list(run_id_list, src=0)
        self.run_id = run_id_list[0]
        if getattr(config, "resume_run_id", ""):
            self.run_id = str(config.resume_run_id)

        set_seed(config.seed + global_rank)
        self.cfg_dropout_rng = random.Random(config.seed + global_rank + 1)

        if self.is_main_process and not self.disable_wandb:
            wandb.init(
                config=public_config(config),
                name=f"{config.config_name}_{self.run_id}",
                project=os.environ.get("WANDB_PROJECT", "RepForcing"),
                dir=config.wandb_save_dir,
            )

        self.output_path = config.logdir

        self.variant = str(getattr(config, "trainer", ""))
        if self.variant not in RFT_VARIANTS:
            raise ValueError(
                f"Unknown RFT variant '{self.variant}'. Expected one of {sorted(RFT_VARIANTS)}."
            )
        model_class = RFT_VARIANTS[self.variant]
        self.model = model_class(config, device=self.device)

        perceptual_type = str(config.perceptual_type).lower()
        if perceptual_type != "dit":
            raise ValueError("RepForcing Stage 1 requires perceptual_type='dit'")

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
        )

        if perceptual_type == "dit":
            assert self.model.base_generator is not None, (
                f"base_generator must be constructed in {model_class.__name__} when perceptual_type=dit."
            )
            base_dtype = torch.bfloat16 if config.mixed_precision else torch.float32
            self.model.base_generator = self.model.base_generator.to(
                device=self.device, dtype=base_dtype
            )
            self.model.base_generator.eval()
            for p in self.model.base_generator.parameters():
                p.requires_grad_(False)
            self.model.attach_base_generator(self.model.base_generator)

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
        )

        self.model.vae = self.model.vae.to(
            device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32
        )

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters() if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )

        dataset = LatentLMDBDataset(config.data_path, max_pair=int(1e8))
        self.dataset = dataset
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8,
        )
        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue
            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
                fixed = {}
                for k, v in state_dict.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            elif "generator_ema" in state_dict:
                gen_sd = state_dict["generator_ema"]
                fixed = {}
                for k, v in gen_sd.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            self.model.generator.load_state_dict(state_dict, strict=True)

        self.step = int(getattr(config, "resume_step", 0))

        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm = 10.0
        self.previous_time = None
        self.grad_accum = resolve_grad_accum(config, dist.get_world_size())
        if self.is_main_process:
            print(
                f"Gradient accumulation: {self.grad_accum} micro-batches x "
                f"batch_size {config.batch_size} x {dist.get_world_size()} ranks "
                f"= global batch {config.batch_size * dist.get_world_size() * self.grad_accum}",
                flush=True,
            )
        self.cfg_rate = float(getattr(config, "cfg_rate", 0.1))
        self.negative_prompt = config.negative_prompt

        self.eval_interval = getattr(self.config, "eval_interval", 0)
        self.visualize_interval = int(getattr(self.config, "visualize_interval", 0))
        self.eval_sampler = str(getattr(self.config, "eval_sampler", "cps")).lower()
        if self.eval_interval or self.visualize_interval:
            # "few_step" walks the DMD stage's `denoising_step_list` through the same
            # pipeline trainer/distillation.py uses, so stage-1 and stage-2 eval videos
            # come out of the same sampler. "cps" and "consistent" build their own grid
            # inside CausalDiffusionInferencePipeline instead.
            pipeline_class = (
                CausalInferencePipeline
                if self.eval_sampler == "few_step"
                else CausalDiffusionInferencePipeline
            )
            self.pipeline = pipeline_class(
                config,
                device=self.device,
                generator=self.model.generator,
                text_encoder=self.model.text_encoder,
                vae=self.model.vae,
            )

        self.visualize_num_prompts = int(getattr(self.config, "visualize_num_prompts", 1))
        self.visualize_fps = int(getattr(self.config, "visualize_fps", 16))
        self.realscore_vis_interval = int(getattr(self.config, "realscore_vis_interval", 0))
        self.cps_num_steps = int(getattr(self.config, "cps_num_steps", 8))
        self.cps_noise_level = float(getattr(self.config, "cps_noise_level", 0.7))
        self.cps_deterministic = bool(getattr(self.config, "cps_deterministic", False))
        self.consistent_num_steps = int(getattr(self.config, "consistent_num_steps", 8))
        self.consistent_deterministic = bool(getattr(self.config, "consistent_deterministic", False))
        self.visualize_prompts = []
        if self.visualize_interval:
            prompt_path = getattr(self.config, "visualize_prompt_path", None) or getattr(
                self.config, "prompt_path", None
            )
            pool_size = dist.get_world_size() * max(self.visualize_num_prompts, 1)
            if prompt_path and os.path.isfile(prompt_path):
                with open(prompt_path, "r") as f:
                    lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                self.visualize_prompts = lines[:pool_size]
            if not self.visualize_prompts:
                self.visualize_prompts = ["a cat is walking on the grass"]

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(self.model.generator)
        if self.config.ema_start_step < self.step and self.generator_ema is not None:
            state_dict = {
                "generator": generator_state_dict,
                "generator_ema": self.generator_ema.full_state_dict(self.model.generator),
            }
        else:
            state_dict = {"generator": generator_state_dict}

        if self.is_main_process:
            ckpt_dir = os.path.join(
                self.output_path, f"run_{self.run_id}", f"checkpoint_model_{self.step:06d}"
            )
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(state_dict, os.path.join(ckpt_dir, "model.pt"))
            print("Model saved to", os.path.join(ckpt_dir, "model.pt"))

    def _visualize_wandb(self):
        if not self.visualize_interval or not self.visualize_prompts:
            return
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        prompt_pool = self.visualize_prompts
        per_rank = max(self.visualize_num_prompts, 1)
        total_slots = world_size * per_rank
        my_indices = list(range(rank * per_rank, (rank + 1) * per_rank))

        torch.cuda.empty_cache()
        self.model.generator.eval()
        image_or_video_shape = list(self.config.image_or_video_shape)
        _, num_frames, num_channels, height, width = image_or_video_shape

        eval_dir = os.path.join(self.output_path, "eval", f"step_{self.step:06d}")
        if self.is_main_process:
            os.makedirs(eval_dir, exist_ok=True)
        barrier()

        with torch.no_grad():
            for slot in my_indices:
                prompt = prompt_pool[slot % len(prompt_pool)]
                noise = torch.randn(
                    [1, num_frames, num_channels, height, width],
                    device=self.device,
                    dtype=self.dtype,
                )
                if self.eval_sampler == "few_step":
                    video = self.pipeline.inference(noise=noise, text_prompts=[prompt])
                elif self.eval_sampler == "consistent":
                    video, _ = self.pipeline.inference_consistent(
                        noise=noise,
                        text_prompts=[prompt],
                        consistent_num_steps=self.consistent_num_steps,
                        consistent_deterministic=self.consistent_deterministic,
                        return_latents=True,
                        initial_latent=None,
                    )
                else:
                    video, _ = self.pipeline.inference_cps(
                        noise=noise,
                        text_prompts=[prompt],
                        cps_num_steps=self.cps_num_steps,
                        cps_noise_level=self.cps_noise_level,
                        cps_deterministic=self.cps_deterministic,
                        return_latents=True,
                        initial_latent=None,
                    )
                v = video[0].detach().float().clamp(0.0, 1.0)
                v = (v * 255.0).to(torch.uint8).permute(0, 2, 3, 1).cpu()
                video_path = os.path.join(eval_dir, f"slot{slot:03d}.mp4")
                write_video(video_path, v, fps=self.visualize_fps)
                self.pipeline.vae.model.clear_cache()

        barrier()

        if self.is_main_process and not self.disable_wandb:
            log_dict = {}
            for slot in range(total_slots):
                p = prompt_pool[slot % len(prompt_pool)]
                path_s = os.path.join(eval_dir, f"slot{slot:03d}.mp4")
                if os.path.isfile(path_s):
                    log_dict[f"samples/slot{slot:03d}"] = wandb.Video(
                        path_s, fps=self.visualize_fps, format="mp4", caption=p
                    )
            if log_dict:
                wandb.log(log_dict, step=self.step)

        self.model.generator.train()
        torch.cuda.empty_cache()

    def _visualize_realscore(self, log_dict):
        if not self.is_main_process:
            return
        keys = [k for k in ("pred_video", "real_x0") if log_dict.get(k, None) is not None]
        if not keys:
            return
        vis_dir = os.path.join(self.output_path, "realscore_vis", f"step_{self.step:06d}")
        os.makedirs(vis_dir, exist_ok=True)
        wandb_dict = {}
        for key in keys:
            latent = log_dict[key]
            with torch.no_grad():
                pixels = self.model.vae.decode_to_pixel(
                    latent[:1].to(device=self.device, dtype=self.dtype)
                )
                pixels = (pixels * 0.5 + 0.5).clamp(0.0, 1.0)
            self.model.vae.model.clear_cache()
            v = pixels[0].float().clamp(0.0, 1.0)
            v = (v * 255.0).to(torch.uint8).permute(0, 2, 3, 1).cpu()
            path = os.path.join(vis_dir, f"{key}.mp4")
            write_video(path, v, fps=self.visualize_fps)
            if not self.disable_wandb:
                wandb_dict[f"realscore/{key}"] = wandb.Video(
                    path, fps=self.visualize_fps, format="mp4"
                )
        if wandb_dict and not self.disable_wandb:
            wandb.log(wandb_dict, step=self.step)

    def _apply_caption_dropout(self, text_prompts):
        if self.cfg_rate <= 0.0:
            return list(text_prompts)
        out = []
        for p in text_prompts:
            if self.cfg_dropout_rng.random() < self.cfg_rate:
                out.append(self.negative_prompt)
            else:
                out.append(p)
        return out

    def _fwdbwd_micro(self, batch):
        """One micro-batch: forward and scaled backward, no optimizer interaction."""
        text_prompts = batch["prompts"]
        if not self.config.load_raw_video:
            clean_latent = batch["clean_latent"].to(device=self.device, dtype=self.dtype)
        else:
            frames = batch["frames"].to(device=self.device, dtype=self.dtype)
            with torch.no_grad():
                clean_latent = self.model.vae.encode_to_latent(frames).to(
                    device=self.device, dtype=self.dtype
                )
        image_latent = clean_latent[:, 0:1]

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        text_prompts_cond = self._apply_caption_dropout(text_prompts)

        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts_cond)
            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.negative_prompt] * batch_size
                )
                unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        generator_loss, log_dict = self.model.generator_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if getattr(self.config, "i2v", False) else None,
        )
        # Scale so the accumulated gradient is the mean over the global batch.
        (generator_loss / self.grad_accum).backward()
        log_dict["generator_loss"] = generator_loss.detach()
        return log_dict

    def train_one_step(self):
        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        self.generator_optimizer.zero_grad()
        micro_logs = [
            self._fwdbwd_micro(next(self.dataloader)) for _ in range(self.grad_accum)
        ]
        # Clip once, on the fully accumulated gradient.
        generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm)
        self.generator_optimizer.step()
        log_dict = merge_scalar_logs(micro_logs)

        self.step += 1

        wandb_loss_dict = {
            "generator_loss": log_dict["generator_loss"].mean().item(),
            "generator_grad_norm": generator_grad_norm.item(),
            "beta": log_dict["beta"].mean().item(),
            "timesteps_min": log_dict["timesteps_min"].min().item(),
            "timesteps_max": log_dict["timesteps_max"].max().item(),
        }

        if self.is_main_process:
            print(
                f"step: {self.step} | generator_loss: {wandb_loss_dict['generator_loss']:.6f} | "
                f"grad_norm: {wandb_loss_dict['generator_grad_norm']:.6f} | "
                f"beta: {wandb_loss_dict['beta']:.6f} | "
                f"timesteps: [{wandb_loss_dict['timesteps_min']:.1f}, {wandb_loss_dict['timesteps_max']:.1f}]",
                flush=True
            )
            if not self.disable_wandb:
                wandb.log(wandb_loss_dict, step=self.step)

        if self.step % self.config.gc_interval == 0:
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()

        if self.realscore_vis_interval and self.step % self.realscore_vis_interval == 0:
            self._visualize_realscore(log_dict)

    def train(self):
        if self.visualize_interval:
            self._visualize_wandb()
        while True:
            self.train_one_step()

            if (not self.config.no_save) and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            if self.visualize_interval and self.step % self.visualize_interval == 0:
                self._visualize_wandb()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log(
                            {"per iteration time": current_time - self.previous_time},
                            step=self.step,
                        )
                    self.previous_time = current_time
