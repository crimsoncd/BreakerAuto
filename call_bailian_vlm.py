"""
Bailian (DashScope) OpenAI-compatible API wrapper for the pipeline's VLM role.

This module replaces the former locally-hosted Qwen3.8-27B wrapper
(call_Qwen3VL.py, removed) with remote API calls, WITHOUT changing the
pipeline logic: the exposed function keeps the same call signature as the
old `Qwen3VL_inference`, so `pipeline_tools.vlm` / `pipeline_tools.vlm_json`
work unchanged.

Notes
-----
* The API key is read from the `BAILIAN_API_KEY` environment variable, which
  is normally populated from the repo-local `.env` file (git-ignored) by
  `config._load_env_file()` at import time.
* `device` is accepted for signature compatibility and ignored — the model
  runs remotely, so the VLM no longer occupies a local GPU.
* Images are passed as base64 data URLs. PIL images, local paths, and lists
  of either are all accepted (the verifier roles pass [original, result]).
"""

from __future__ import annotations

import base64
import io
import re
import time
from pathlib import Path

import config
from PIL import Image

# Defaults (config can be overridden via environment variables)
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds between API retries

try:  # the openai client is present in the JoyNew conda env
    from openai import OpenAI
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "The 'openai' package is required for the Bailian VLM API. "
        "It should already be installed in the JoyNew conda env."
    ) from _e

# Module-level client, created lazily so importing this module never touches
# the network and never requires the API key to be present (e.g. --fake mode).
_client = None


def get_client():
    """Return a cached OpenAI-compatible client pointed at Bailian."""
    global _client
    if _client is None:
        api_key = config.VLM_API_KEY
        if not api_key:
            raise RuntimeError(
                "VLM_API_KEY is empty — set BAILIAN_API_KEY in .env or in the "
                "environment before running the pipeline in non-fake mode."
            )
        _client = OpenAI(api_key=api_key, base_url=config.VLM_BASE_URL)
    return _client


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".tiff": "image/tiff",
}


def _encode_pil_image(img: Image.Image) -> str:
    """Encode a PIL image to a base64 PNG data URL."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _encode_path(path: str | Path) -> str:
    """Encode an image file to a base64 data URL, preserving the mime type."""
    path = Path(path)
    mime = _MIME_BY_SUFFIX.get(path.suffix.lower(), "image/png")
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _normalize_images(image_input):
    """Accept None / a single image (path or PIL) / a list of them; return a list."""
    if image_input is None:
        return []
    if isinstance(image_input, (list, tuple)):
        return list(image_input)
    return [image_input]


def _to_data_url(image) -> str:
    """Coerce one image entry (path / PIL) into a base64 data URL."""
    if isinstance(image, Image.Image):
        return _encode_pil_image(image)
    return _encode_path(str(image))


# ---------------------------------------------------------------------------
# Response post-processing
# ---------------------------------------------------------------------------
def _strip_think(text: str) -> str:
    """Remove reasoning/think blocks the model may emit before the answer."""
    return re.sub(
        r"(<think>.*?</think>|^.*?</think>|<think>.*$)",
        "",
        text,
        flags=re.DOTALL,
    ).strip()


# ---------------------------------------------------------------------------
# Main entrypoint — VLM inference via the Bailian API
# ---------------------------------------------------------------------------
def BailianVLM_inference(
    image_input,
    prompt,
    system_prompt=None,
    use_flash_attn=False,
    max_new_tokens=65536,
    deterministic=True,
    device=None,
):
    """
    Runs inference on the Bailian-hosted VLM with one or MORE images and a
    text prompt. Accepts the same arguments as the former local-model
    wrapper (call_Qwen3VL.Qwen3VL_inference, removed).

    Args:
        image_input: a single image (local path / PIL.Image) OR a list of
            images, or None for a text-only call. Multiple images are required
            for the verifier roles, which see the ORIGINAL as a reference
            alongside the RESULT. When a list is passed, images appear in the
            prompt in the given order; refer to them in the text as "the first
            image", "the second image", etc.
        prompt (str): the user text / question.
        system_prompt (str|None): optional system role text. Our agent roles
            are written as system prompts; pass them here.
        use_flash_attn (bool): ignored, kept for API compatibility.
        max_new_tokens (int): mapped to the API's max_tokens.
        deterministic (bool): if True, temperature=0 for reproducible,
            parseable structured output; if False, temperature=0.7/top_p=0.8.
        device (str|None): ignored, kept for API compatibility (remote model).

    Returns:
        str: the generated text response (think blocks stripped).
    """
    client = get_client()
    model = config.VLM_MODEL_NAME

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content = [
        {"type": "image_url", "image_url": {"url": _to_data_url(img)}}
        for img in _normalize_images(image_input)
    ]
    user_content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": user_content})

    gen_kwargs = {"max_tokens": max_new_tokens}
    if deterministic:
        gen_kwargs.update(temperature=0, top_p=1)
    else:
        gen_kwargs.update(temperature=0.7, top_p=0.8)

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                **gen_kwargs,
            )
            output_text = completion.choices[0].message.content or ""
            return _strip_think(output_text.strip())
        except Exception as e:  # noqa: BLE001 — retry any API/network error
            last_err = e
            print(f"[BailianVLM] Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    raise RuntimeError(
        f"BailianVLM_inference failed after {MAX_RETRIES} attempts: {last_err}"
    )


if __name__ == "__main__":
    # Small sanity CLI, mirroring call_Qwen3VL.py's usage:
    #   python call_bailian_vlm.py --image in.png --prompt "What is in this image?"
    import argparse

    parser = argparse.ArgumentParser(description="Query the Bailian VLM via CLI.")
    parser.add_argument("--image", type=str, nargs="+", default=None,
                        help="One or more image paths (space separated).")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--sample", action="store_true",
                        help="Enable sampling (default is greedy).")
    args = parser.parse_args()

    image_arg = args.image
    if image_arg is not None and len(image_arg) == 1:
        image_arg = image_arg[0]

    response = BailianVLM_inference(
        image_input=image_arg,
        prompt=args.prompt,
        system_prompt=args.system,
        max_new_tokens=args.max_tokens,
        deterministic=not args.sample,
    )
    print("\n--- Model Response ---")
    print(response)
