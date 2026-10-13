from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from model.base import BaseModel
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class CausalDiffusionRFT(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True

        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.guidance_scale = args.guidance_scale
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.teacher_forcing = getattr(args, "teacher_forcing", True)
        self.noise_augmentation_max_timestep = getattr(args, "noise_augmentation_max_timestep", 0)

        self.teacher_guidance_scale = float(getattr(args, "teacher_guidance_scale", 3.0))
        assert self.teacher_guidance_scale > 0, "teacher_guidance_scale must be > 0"
        self.flow_matching_loss_weight = float(getattr(args, "flow_matching_loss_weight", 1.0))
        self.lpips_train_frames = int(getattr(args, "lpips_train_frames", 8))

        self.perceptual_type = str(getattr(args, "perceptual_type", "internvid")).lower()
        self.perceptual_layer = int(getattr(args, "perceptual_layer", 22))
        self.perceptual_timestep = float(getattr(args, "perceptual_timestep", 500.0))

        self.perceptual_model: Optional[nn.Module] = None

    def _initialize_models(self, args, device):
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

        self.base_generator: Optional[nn.Module] = None
        if str(getattr(args, "perceptual_type", "internvid")).lower() == "dit":
            base_model_name = getattr(args, "real_name", "Wan2.1-T2V-1.3B")
            self.base_generator = WanDiffusionWrapper(
                model_name=base_model_name,
                **getattr(args, "model_kwargs", {}),
                is_causal=True,
            )
            # The perceptual feature is read from blocks[perceptual_layer] via a
            # forward hook; blocks past it never contribute to the loss. Drop
            # them so a large extractor does not hold dead weights on every GPU
            # (base_generator is not FSDP-sharded: 14B full = ~28 GB bf16,
            # truncated at block 8 = ~8 GB).
            perceptual_layer = int(getattr(args, "perceptual_layer", 22))
            self.base_generator.model.blocks = (
                self.base_generator.model.blocks[: perceptual_layer + 1]
            )
            self.base_generator.model.requires_grad_(False)
            self.base_generator.eval()

    def attach_perceptual_model(self, perceptual_model: nn.Module):
        self.perceptual_model = perceptual_model.to(self.device)
        self.perceptual_model.requires_grad_(False)
        self.perceptual_model.eval()

    def attach_base_generator(self, base_generator: nn.Module):
        self.base_generator = base_generator

    def _decode_to_pixel_ckpt(self, latent: torch.Tensor) -> torch.Tensor:
        """VAE decode with gradient checkpointing. Returns (B, C, T, H, W) in [-1, 1]."""

        def _decode(l):
            pixels = self.vae.decode_to_pixel(l)
            return pixels.permute(0, 2, 1, 3, 4).contiguous()

        return ckpt.checkpoint(_decode, latent, use_reentrant=False)

    def _decode_to_pixel_no_grad(self, latent: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pixels = self.vae.decode_to_pixel(latent)
        return pixels.permute(0, 2, 1, 3, 4).contiguous()

    def _resolve_inner_model(self, module: nn.Module) -> nn.Module:
        current = module
        while hasattr(current, "_fsdp_wrapped_module"):
            current = current._fsdp_wrapped_module
        inner = current.model
        while hasattr(inner, "_fsdp_wrapped_module"):
            inner = inner._fsdp_wrapped_module
        return inner

    def _resolve_dit_block(self, module: nn.Module) -> nn.Module:
        return self._resolve_inner_model(module).blocks[self.perceptual_layer]

    def _dit_extract_feat(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        inner = self._resolve_inner_model(module)
        block = inner.blocks[self.perceptual_layer]

        # Keep the extractor's block mask cached across calls. Nulling it forces
        # `create_block_mask` to re-materialize a dense [padded_len, padded_len]
        # int64 mask on every call -- 8.0 GiB at 21 frames -- which OOMs once the
        # extractor is large enough to eat the headroom. The mask only depends on
        # the frame count here (this path always passes clean_x=None, so it is
        # always the block-wise causal mask), so it is rebuilt only when that
        # count changes.
        num_frame = hidden_states.shape[1]
        if getattr(self, "_perc_mask_num_frame", None) != num_frame:
            inner.block_mask = None
            self._perc_mask_num_frame = num_frame
        cache = {}

        def hook(_m, _inp, out):
            cache["feat"] = out

        handle = block.register_forward_hook(hook)
        try:
            module(
                noisy_image_or_video=hidden_states,
                conditional_dict=conditional_dict,
                timestep=timestep,
                clean_x=None,
                aug_t=None,
            )
        finally:
            handle.remove()
        feat = cache["feat"]
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        return feat

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        if self.perceptual_type == "dit":
            assert self.base_generator is not None, (
                "base_generator must be attached before training when perceptual_type=dit."
            )
        else:
            assert self.perceptual_model is not None, (
                "InternVid perceptual model must be attached via attach_perceptual_model before training."
            )

        noise = torch.randn_like(clean_latent)
        batch_size, num_frame = image_or_video_shape[:2]

        index = self._get_timestep(
            0,
            self.scheduler.num_train_timesteps,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=False,
        )
        timestep = self.scheduler.timesteps[index].to(dtype=self.dtype, device=self.device)
        noisy_latents = self.scheduler.add_noise(
            clean_latent.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, (batch_size, num_frame))

        if self.noise_augmentation_max_timestep > 0:
            index_clean_aug = self._get_timestep(
                self.noise_augmentation_max_timestep,
                1000,
                image_or_video_shape[0],
                image_or_video_shape[1],
                self.num_frame_per_block,
                uniform_timestep=False,
            )
            timestep_clean_aug = self.scheduler.timesteps[index_clean_aug].to(
                dtype=self.dtype, device=self.device
            )
            clean_latent_aug = self.scheduler.add_noise(
                clean_latent.flatten(0, 1),
                noise.flatten(0, 1),
                timestep_clean_aug.flatten(0, 1),
            ).unflatten(0, (batch_size, num_frame))
        else:
            clean_latent_aug = clean_latent
            timestep_clean_aug = None

        _, x0_pred_cond = self.generator(
            noisy_image_or_video=noisy_latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
            clean_x=clean_latent_aug if self.teacher_forcing else None,
            aug_t=timestep_clean_aug if self.teacher_forcing else None,
        )

        x0_pred = x0_pred_cond

        if self.perceptual_type == "dit":
            perc_timestep = torch.full(
                (batch_size, num_frame),
                self.perceptual_timestep,
                device=self.device,
                dtype=timestep.dtype,
            )
            feat_pred = self._dit_extract_feat(
                self.base_generator,
                x0_pred,
                conditional_dict,
                perc_timestep,
            )
            with torch.no_grad():
                feat_target = self._dit_extract_feat(
                    self.base_generator, clean_latent, conditional_dict, perc_timestep
                ).detach()
            loss = (
                1.0 - F.cosine_similarity(feat_pred.float(), feat_target.float(), dim=-1)
            ).mean() * self.flow_matching_loss_weight
        else:
            num_frames_total = x0_pred.shape[1]
            num_frames_sample = min(self.lpips_train_frames, num_frames_total)
            if num_frames_sample < num_frames_total:
                perm = torch.randperm(num_frames_total, device=x0_pred.device)[:num_frames_sample]
                perm, _ = perm.sort()
                x0_pred_sub = x0_pred[:, perm]
                clean_sub = clean_latent[:, perm]
            else:
                x0_pred_sub = x0_pred
                clean_sub = clean_latent

            pred_pixels = self._decode_to_pixel_ckpt(x0_pred_sub.to(self.vae.model.dtype if hasattr(self.vae.model, "dtype") else self.dtype))
            target_pixels = self._decode_to_pixel_no_grad(clean_sub.to(self.dtype))

            pred_pixels = pred_pixels.to(torch.bfloat16)
            target_pixels = target_pixels.to(torch.bfloat16)

            loss = self.perceptual_model(pred_pixels, target_pixels) * self.flow_matching_loss_weight

        log_dict = {
            "x0": clean_latent.detach(),
            "x0_pred": x0_pred.detach(),
            "beta": torch.tensor(1.0, device=self.device),
            "timesteps_min": timestep.detach().min(),
            "timesteps_max": timestep.detach().max(),
        }
        return loss, log_dict
