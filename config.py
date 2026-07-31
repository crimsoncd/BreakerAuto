"""
Global configuration for the layer decomposition pipeline.

All machine-specific settings (model paths, GPU cards, output directory)
live here so the rest of the repo stays portable. Every value can also be
overridden through the environment (the env var name is shown next to each
entry), so you never have to edit this file to run on a new machine.

Before the first run, point at least these at your local resources:
  * QWEN_MODEL_ID   -- local directory (or HuggingFace repo id) of Qwen3-VL.
  * JOYAI_CKPT_ROOT -- checkpoint directory of the JoyAI image-edit model.
  * JOYAI_SRC_DIR   -- root of your local clone of the JoyAI-Image repo.
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
# VLM configuration (Qwen3-VL)
# ---------------------------------------------------------------------------
# Local model directory or a HuggingFace repo id. Env: QWEN_MODEL_ID
QWEN_MODEL_ID = _from_env("QWEN_MODEL_ID", "Qwen/Qwen3-VL-32B-Instruct")

# GPU card for the VLM, e.g. "cuda:0". Leave None to auto-detect the freest
# card at runtime. Env: QWEN_DEVICE
QWEN_DEVICE = os.environ.get("QWEN_DEVICE")

VLM_SYSTEM_PROMPT = ""  # filled per role
VLM_MAX_TOKENS_PLANNER = 1500
VLM_MAX_TOKENS_CHECKER = 256
VLM_MAX_TOKENS_PROMPT_WRITER = 512

# ---------------------------------------------------------------------------
# JoyAI configuration (image edit model)
# ---------------------------------------------------------------------------
# Checkpoint root for the JoyAI image-edit model. Env: JOYAI_CKPT_ROOT
JOYAI_CKPT_ROOT = _from_env("JOYAI_CKPT_ROOT", "models/JoyAI-Image-Edit/")

# Root of the JoyAI-Image source repository. Its `src` subdirectory is added
# to sys.path so `infer_runtime` / `modules` resolve. Env: JOYAI_SRC_DIR
JOYAI_SRC_DIR = _from_env("JOYAI_SRC_DIR", "path/to/JoyAI-Image")

# GPU card for JoyAI, e.g. "cuda:1". Leave None to auto-detect the freest
# card at runtime. MUST be a different card than the VLM when both models run
# together (each is ~80G). Env: JOYAI_DEVICE
JOYAI_DEVICE = os.environ.get("JOYAI_DEVICE")

JOYAI_BASE_SEED = 42
JOYAI_STEPS = 30
JOYAI_GUIDANCE_SCALE = 5.0

# ---------------------------------------------------------------------------
# Bounding box normalization
# ---------------------------------------------------------------------------
BBOX_NORM = 1000  # bbox coords are 0-1000

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = "runs"
