# Modified for the RepForcing source release; see NOTICE.md.
import gc
import logging
from pipeline import CausalInferencePipeline
from utils.dataset import cycle
from utils.dataset import TextDataset
from utils.distributed import EMA_FSDP, barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import merge_scalar_logs, resolve_grad_accum, set_seed
import torch.distributed as dist
from utils.runtime import public_config
from model import DMD
from torchvision.io import write_video
import torch
import wandb
import time
import os


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        # `config_name` is the registry variant name (train.py:19), which already
        # encodes teacher, guidance, schedule and step count. A timestamp keeps
        # repeat runs of one variant apart. Archived runs launched through
        # `--config_path` carry a `run_tag` instead; honour it when present.
        run_tag = str(getattr(config, "run_tag", "") or time.strftime("%Y%m%d_%H%M%S"))
        self.run_name = f"{config.config_name}_{run_tag}"

        if self.is_main_process and not self.disable_wandb:
            wandb.init(
                config=public_config(config),
                name=self.run_name,
                project=os.environ.get("WANDB_PROJECT", "RepForcing"),
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        if config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=False
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            cpu_offload=False
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            cpu_offload=False
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
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

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
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
            self.model.generator.load_state_dict(
                state_dict, strict=True
            )

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None
        self.grad_accum = resolve_grad_accum(config, self.world_size)
        if self.is_main_process:
            print(
                f"Gradient accumulation: {self.grad_accum} micro-batches x "
                f"batch_size {config.batch_size} x {self.world_size} ranks "
                f"= global batch {config.batch_size * self.world_size * self.grad_accum}",
                flush=True,
            )

        self.eval_pipeline = None
        self.visualize_num_prompts = int(getattr(config, "visualize_num_prompts", 2))
        self.visualize_fps = int(getattr(config, "visualize_fps", 16))
        self.visualize_prompts = []
        prompt_path = getattr(config, "visualize_prompt_path", None) or getattr(config, "prompt_path", None)
        pool_size = dist.get_world_size() * max(self.visualize_num_prompts, 1)
        if prompt_path and os.path.isfile(prompt_path):
            with open(prompt_path, "r") as f:
                self.visualize_prompts = [ln.strip() for ln in f.readlines() if ln.strip()][:pool_size]

    def _visualize_wandb(self):
        """Sample videos with the few-step inference pipeline and log them to disk and wandb."""
        if not self.visualize_prompts:
            return

        if self.eval_pipeline is None:
            self.eval_pipeline = CausalInferencePipeline(
                self.config,
                device=self.device,
                generator=self.model.generator,
                text_encoder=self.model.text_encoder,
                vae=self.model.vae,
            )

        rank = dist.get_rank()
        per_rank = max(self.visualize_num_prompts, 1)
        total_slots = dist.get_world_size() * per_rank
        my_slots = list(range(rank * per_rank, (rank + 1) * per_rank))

        _, num_frames, num_channels, height, width = list(self.config.image_or_video_shape)
        eval_dir = os.path.join(self.output_path, "eval", f"step_{self.step:06d}")
        if self.is_main_process:
            os.makedirs(eval_dir, exist_ok=True)
        barrier()

        # Official framewise inference uses the EMA weights (README `--use_ema`).
        live_state = None
        if self.generator_ema is not None:
            live_state = {
                n: p.detach().clone()
                for n, p in self.model.generator.module.named_parameters()
            }
            self.generator_ema.copy_to(self.model.generator)

        torch.cuda.empty_cache()
        self.model.generator.eval()
        try:
            with torch.no_grad():
                for slot in my_slots:
                    prompt = self.visualize_prompts[slot % len(self.visualize_prompts)]
                    noise = torch.randn(
                        [1, num_frames, num_channels, height, width],
                        device=self.device,
                        dtype=self.dtype,
                    )
                    video = self.eval_pipeline.inference(noise=noise, text_prompts=[prompt])
                    v = video[0].detach().float().clamp(0.0, 1.0)
                    v = (v * 255.0).to(torch.uint8).permute(0, 2, 3, 1).cpu()
                    write_video(os.path.join(eval_dir, f"slot{slot:03d}.mp4"), v, fps=self.visualize_fps)
                    self.model.vae.model.clear_cache()
        finally:
            self.model.generator.train()
            if live_state is not None:
                with torch.no_grad():
                    for n, p in self.model.generator.module.named_parameters():
                        p.data.copy_(live_state[n].to(dtype=p.dtype, device=p.device))
            torch.cuda.empty_cache()

        barrier()

        if self.is_main_process and not self.disable_wandb:
            log_dict = {}
            for slot in range(total_slots):
                path = os.path.join(eval_dir, f"slot{slot:03d}.mp4")
                if os.path.isfile(path):
                    log_dict[f"samples/slot{slot:03d}"] = wandb.Video(
                        path, fps=self.visualize_fps, format="mp4",
                        caption=self.visualize_prompts[slot % len(self.visualize_prompts)],
                    )
            if log_dict:
                wandb.log(log_dict, step=self.step)

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator_ema": self.generator_ema.full_state_dict(self.model.generator),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
            }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def save_critic(self):
        print("Start gathering distributed model states...")
        
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        
        state_dict = critic_state_dict

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))
            
    def fwdbwd_one_step(self, batch, train_generator, clean_latent=None):
        self.model.eval()

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        text_prompts = batch["prompts"]
        batch_size = len(text_prompts)

        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        if self.config.i2v:
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            image_latent = None

        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None
            )
            (generator_loss / self.grad_accum).backward()
            generator_log_dict.update({"generator_loss": generator_loss.detach()})
            return generator_log_dict

        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )
        (critic_loss / self.grad_accum).backward()
        critic_log_dict.update({"critic_loss": critic_loss.detach()})
        return critic_log_dict


    def train(self):
        start_step = self.step
       
        while True:
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            if self.is_main_process:
                update_branch = (
                    "fake_score+dmd_generator" if TRAIN_GENERATOR else "fake_score_only"
                )
                print(
                    f"Update schedule | step={self.step} | fake_score_update=True | "
                    f"dmd_generator_update={TRAIN_GENERATOR} | "
                    f"fake_score:dmd_ratio={self.config.dfake_gen_update_ratio}:1 | "
                    f"branch={update_branch}",
                    flush=True,
                )

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                
                micro_logs = [
                    self.fwdbwd_one_step(next(self.dataloader), True)
                    for _ in range(self.grad_accum)
                ]
                generator_log_dict = merge_scalar_logs(micro_logs)
                # Clip once, on the fully accumulated gradient.
                generator_log_dict["generator_grad_norm"] = \
                    self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)

                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)
                
                
                

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            micro_logs = [
                self.fwdbwd_one_step(next(self.dataloader), False)
                for _ in range(self.grad_accum)
            ]
            critic_log_dict = merge_scalar_logs(micro_logs)
            critic_log_dict["critic_grad_norm"] = \
                self.model.fake_score.clip_grad_norm_(self.max_grad_norm_critic)
                
            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            periodic_save = (
                not self.config.no_save
                and (self.step - start_step) > 0
                and self.step % self.config.log_iters == 0
            )
            if periodic_save:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()
                if not self.config.no_visualize:
                    self._visualize_wandb()
                    torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                        }
                    )
                    for key, value in generator_log_dict.items():
                        if key.startswith("dmdtrain_") and torch.is_tensor(value):
                            wandb_loss_dict[key] = value.float().mean().item()

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time
