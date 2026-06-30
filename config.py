"""
Budget constants and configuration for the layer decomposition pipeline.
"""
# Retry budgets
ELEMENT_RETRIES = 3
BACKGROUND_RETRIES = 3
GLOBAL_ATTEMPTS = 3
MAX_ENUM_REOPENINGS = 3
MAX_ELEMENTS = 20

# VLM configuration
VLM_SYSTEM_PROMPT = ""  # filled per role
VLM_MAX_TOKENS_PLANNER = 1500
VLM_MAX_TOKENS_CHECKER = 256
VLM_MAX_TOKENS_PROMPT_WRITER = 512

# JoyAI configuration
JOYAI_BASE_SEED = 42
JOYAI_STEPS = 30
JOYAI_GUIDANCE_SCALE = 5.0

# Bounding box normalization
BBOX_NORM = 1000  # bbox coords are 0-1000

# Output
DEFAULT_OUTPUT_DIR = "runs"