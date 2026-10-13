# Modified for the RepForcing source release; see NOTICE.md.
"""Stage-2 four-step distillation from a user-provided Stage-1 checkpoint."""
from . import base


def _dmd(num_frame_per_block, batch_size):
    config = base.get_config()
    config.trainer = "score_distillation"
    config.distribution_loss = "dmd"
    config.gradient_checkpointing = True
    config.load_raw_video = False
    config.data_path = "dataset/train_prompts.txt"
    config.real_name = "Wan2.1-T2V-14B"
    config.fake_name = "Wan2.1-T2V-1.3B"
    config.real_score_fsdp_wrap_strategy = "size"
    config.fake_score_fsdp_wrap_strategy = "size"
    # Guidance uses c + w(c-u), as in the upstream DMD implementation.
    config.real_guidance_scale = 3.0
    config.fake_guidance_scale = 0.0
    config.guidance_scale = 3.0
    config.lr_critic = 4.0e-07
    config.beta1_critic = 0.0
    config.beta2_critic = 0.999
    config.dfake_gen_update_ratio = 5
    config.denoising_loss_type = "flow"
    config.num_train_timestep = 1000
    config.denoising_step_list = [1000, 750, 500, 250]
    config.ts_schedule = False
    config.ema_weight = 0.99
    config.ema_start_step = 200
    config.num_frame_per_block = num_frame_per_block
    config.batch_size = batch_size
    config.image_or_video_shape = [1, 21, 16, 60, 104]
    config.negative_prompt = base.NEGATIVE_PROMPT
    config.generator_ckpt = ""
    config.log_iters = 50
    config.visualize_prompt_path = "prompts/demos.txt"
    config.visualize_num_prompts = 1
    config.visualize_fps = 16
    return config


def framewise():
    """Distill the framewise Stage-1 generator with a frozen 14B teacher."""
    return _dmd(num_frame_per_block=1, batch_size=2)


def chunkwise():
    """Distill the chunkwise Stage-1 generator with a frozen 14B teacher."""
    return _dmd(num_frame_per_block=3, batch_size=1)
