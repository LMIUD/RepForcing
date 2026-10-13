# Modified for the RepForcing source release; see NOTICE.md.
"""Stage-1 representation supervision for RepForcing."""
from . import base


def _rft(num_frame_per_block, perceptual_timestep):
    config = base.get_config()
    config.trainer = "rft"
    config.gradient_checkpointing = True
    config.load_raw_video = False
    config.data_path = "dataset/clean_data"
    config.real_name = "Wan2.1-T2V-14B"
    config.real_score_fsdp_wrap_strategy = "size"
    config.fake_score_fsdp_wrap_strategy = "size"
    config.distribution_loss = "dmd"
    config.denoising_loss_type = "flow"
    config.num_train_timestep = 1000
    config.ts_schedule = False

    # Stage 1 uses conditional representation supervision without CFG targets.
    config.guidance_scale = 3.0
    config.teacher_guidance_scale = 3.0
    config.teacher_forcing = True
    config.cfg_rate = 0.1
    config.flow_matching_loss_weight = 1.0
    config.perceptual_type = "dit"
    config.perceptual_layer = 8
    config.perceptual_timestep = perceptual_timestep
    config.lpips_train_frames = 8
    config.lr_critic = 4.0e-07
    config.beta1_critic = 0.0
    config.beta2_critic = 0.999
    config.dfake_gen_update_ratio = 5

    # Keep the original Stage-1 EMA schedule; use non-EMA checkpoints here.
    config.ema_weight = 0.99
    config.ema_start_step = 2000000000
    config.num_frame_per_block = num_frame_per_block
    config.image_or_video_shape = [1, 21, 16, 60, 104]
    config.negative_prompt = base.NEGATIVE_PROMPT
    config.eval_sampler = "cps"
    config.cps_num_steps = 8
    config.cps_noise_level = 0.7
    config.cps_deterministic = False
    config.consistent_num_steps = 8
    config.consistent_deterministic = False
    config.visualize_interval = 200
    config.visualize_prompt_path = "prompts/demos.txt"
    config.visualize_num_prompts = 1
    config.visualize_fps = 16
    return config


def framewise():
    """One latent frame per block; clean features from Wan-14B block index 8."""
    return _rft(num_frame_per_block=1, perceptual_timestep=0)


def chunkwise():
    """Three latent frames per block; Wan-14B features at timestep 0."""
    return _rft(num_frame_per_block=3, perceptual_timestep=0)
