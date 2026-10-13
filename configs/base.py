# Modified for the RepForcing source release; see NOTICE.md.
"""Shared defaults for every training variant.

`get_config()` returns only the keys that every trainer reads. Variant modules
(`dmd.py`, `rft.py`) add the keys their own trainer needs.

Key *presence* is semantic in this codebase, not just key value:

  - `utils/misc.py:89` (`resolve_guidance`) returns the explicit
    (real_guidance_scale, fake_guidance_scale) pair when `real_guidance_scale`
    exists, and falls back to (guidance_scale, 0.0) when it does not.
  - `model/base.py:20` only builds `denoising_step_list` when the key exists.
  - `model/base.py:31` and `pipeline/causal_inference.py:37` branch on
    `denoising_step_list_first_chunk`.

So a variant that historically omitted a key must keep omitting it. Do not
promote those keys into this function; add them in the variant that needs them.
"""

import ml_collections



NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def get_config():
    """Defaults shared by every variant. Returns a fresh instance per call."""
    config = ml_collections.ConfigDict()

    # Distributed / precision
    config.mixed_precision = True
    config.sharding_strategy = "hybrid_full"
    config.generator_fsdp_wrap_strategy = "size"
    config.text_encoder_fsdp_wrap_strategy = "size"

    # Optimizer
    config.lr = 2.0e-06
    config.beta1 = 0.0
    config.beta2 = 0.999
    config.weight_decay = 0.01

    # Batch
    config.batch_size = 1
    config.total_batch_size = 64

    # Causal-forcing structure
    config.causal = True
    config.num_frame_per_block = 1
    config.num_training_frames = 21
    config.independent_first_frame = False
    config.same_step_across_blocks = True
    config.last_step_only = False
    config.context_noise = 0
    config.i2v = False

    # Schedule
    config.warp_denoising_step = True
    config.timestep_shift = 5.0
    config.model_kwargs = ml_collections.ConfigDict({"timestep_shift": 5.0})

    # Logging
    config.seed = 0
    config.log_iters = 200
    config.gc_interval = 100

    # Fallback prompt file for the training-time visualisation. Variants that set
    # `visualize_prompt_path` take precedence (trainer/distillation.py:213).
    config.prompt_path = "prompts/demos.txt"

    return config
