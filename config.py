"""
Global configuration for the layer decomposition pipeline.

All machine-specific settings (model paths, GPU cards, output directory)
live here so the rest of the repo stays portable. Every value can also be
overridden through the environment (the env var name is shown next to each
entry), so you never have to edit this file to run on a new machine.

Before the first run, point at least these at your local resources:
  * QWEN_MODEL_ID   -- local directory (or HuggingFace repo id) of the VLM.
  * JOYAI_CKPT_ROOT -- Diffusers-format checkpoint dir of the JoyAI image-edit
                       model (loaded via diffusers.JoyImageEditPipeline).
"""

import os


def _from_env(name: str, default: str) -> str:
    """Return an environment variable, falling back to a default value."""
    return os.environ.get(name, default)


# ---------------------------------------------------------------------------
# Retry budgets
# ---------------------------------------------------------------------------
ELEMENT_RETRIES = 3
BACKGROUND_RETRIES = 3
GLOBAL_ATTEMPTS = 3
MAX_ENUM_REOPENINGS = 3
MAX_ELEMENTS = 20

# ---------------------------------------------------------------------------
# VLM configuration (Qwen3.8 multimodal LLM)
# ---------------------------------------------------------------------------
# Local model directory or a HuggingFace repo id. Env: QWEN_MODEL_ID
QWEN_MODEL_ID = _from_env("QWEN_MODEL_ID", "/remote-home/Zhangkaile/models/Qwen3.8-27B")

# GPU card for the VLM, e.g. "cuda:0". Leave None to auto-detect the freest
# card at runtime. Env: QWEN_DEVICE
QWEN_DEVICE = os.environ.get("QWEN_DEVICE")

VLM_SYSTEM_PROMPT = ""  # filled per role
VLM_MAX_TOKENS_PLANNER = 15000
VLM_MAX_TOKENS_CHECKER = 2560
VLM_MAX_TOKENS_PROMPT_WRITER = 5120
VLM_MAX_TOKENS_DESCRIBER = 15000

# ---------------------------------------------------------------------------
# JoyAI configuration (image edit model, diffusers JoyImageEditPipeline)
# ---------------------------------------------------------------------------
# Diffusers-format checkpoint directory. Env: JOYAI_CKPT_ROOT
JOYAI_CKPT_ROOT = _from_env("JOYAI_CKPT_ROOT", "/remote-home/Zhangkaile/models/JoyAI-Image-Edit-Diffusers/")

# GPU card for JoyAI, e.g. "cuda:1". Leave None to auto-detect the freest
# card at runtime. MUST be a different card than the VLM when both models run
# together (each is ~80G). Env: JOYAI_DEVICE
JOYAI_DEVICE = os.environ.get("JOYAI_DEVICE")

JOYAI_BASE_SEED = 42
JOYAI_STEPS = 40
JOYAI_GUIDANCE_SCALE = 4.0

# ---------------------------------------------------------------------------
# Bounding box normalization
# ---------------------------------------------------------------------------
BBOX_NORM = 1000  # bbox coords are 0-1000

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = "runs"
