"""
Global configuration for the layer decomposition pipeline.

All machine-specific settings (model paths, GPU cards, output directory)
live here so the rest of the repo stays portable. Every value can also be
overridden through the environment (the env var name is shown next to each
entry), so you never have to edit this file to run on a new machine.

Before the first run, point at least these at your resources:
  * BAILIAN_API_KEY -- in the git-ignored .env file (key for the Bailian VLM API).
  * QWEN_EDIT_CKPT_ROOT -- checkpoint directory of the Qwen-Image-2.1
                           image-edit model (loaded via diffusers -- see
                           call_qwen_edit.py).
"""

import os
from pathlib import Path


def _load_env_file(path: str = ".env") -> None:
    """Populate os.environ from a KEY=VALUE .env file (existing env wins).

    Used to keep secrets like BAILIAN_API_KEY out of the repo: the file is
    listed in .gitignore and loaded once at import time.
    """
    env_path = Path(path)
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file()


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
# VLM configuration (Bailian / DashScope OpenAI-compatible API)
# ---------------------------------------------------------------------------
# The VLM runs remotely via API (default model: qwen3.7-flash), so it needs
# no local GPU and no local checkpoint. The API key lives in the git-ignored
# .env file as BAILIAN_API_KEY.

# OpenAI-compatible endpoint base URL. Env: VLM_BASE_URL
VLM_BASE_URL = _from_env(
    "VLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# API key. Env: BAILIAN_API_KEY (normally loaded from .env at import time)
VLM_API_KEY = _from_env("BAILIAN_API_KEY", "")

# Model name served by the endpoint. Env: VLM_MODEL_NAME
VLM_MODEL_NAME = _from_env("VLM_MODEL_NAME", "qwen3.7-flash-2026-07-15")

VLM_SYSTEM_PROMPT = ""  # filled per role
VLM_MAX_TOKENS_PLANNER = 15000
VLM_MAX_TOKENS_CHECKER = 2560
VLM_MAX_TOKENS_PROMPT_WRITER = 5120
VLM_MAX_TOKENS_DESCRIBER = 15000

# ---------------------------------------------------------------------------
# Qwen-Image-2.1 configuration (image edit model, loaded via diffusers
# QwenImage21Pipeline -- see call_qwen_edit.py)
# ---------------------------------------------------------------------------
# Checkpoint root for the Qwen-Image-2.1 image-edit model.
# Env: QWEN_EDIT_CKPT_ROOT
QWEN_EDIT_CKPT_ROOT = _from_env(
    "QWEN_EDIT_CKPT_ROOT", "/remote-home/Zhangkaile/models/Qwen-Image-2.1/"
)

# GPU card for the edit model, e.g. "cuda:2". Leave None to auto-detect the
# freest card at runtime. This is the only local model — the VLM is remote —
# so no co-residence constraint applies. Env: QWEN_EDIT_DEVICE
QWEN_EDIT_DEVICE = os.environ.get("QWEN_EDIT_DEVICE")

QWEN_EDIT_BASE_SEED = 42
QWEN_EDIT_STEPS = 40

# NOTE: the diffusion model may emit an output size different from the input;
# call_qwen_edit.QwenEdit always resizes the result back to the exact input
# size before returning it.

# ---------------------------------------------------------------------------
# Bounding box normalization
# ---------------------------------------------------------------------------
BBOX_NORM = 1000  # bbox coords are 0-1000

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = "runs"
