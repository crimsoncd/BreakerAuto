"""
OpenRouter Decisions API wrapper — the pipeline's fast verification gate.

A \"decider\" (decision model) answers cheap, low-latency yes/no (noul)
questions about images. The pipeline uses it as the default verification
gate for Stage-2 element cutouts and Stage-3 backgrounds: one call, several
questions, numeric scores. Only when the gate FAILS does the pipeline spend
an expensive VLM call to diagnose the failure and rewrite the prompt (see
prompts.P9/P10 and pipeline.run_decider()).

Notes
-----
* The API key is read from the `OPENROUTER_API_KEY` environment variable,
  normally populated from the repo-local `.env` file (git-ignored) by
  `config._load_env_file()` at import time.
* Images are sent as base64 data URLs — PIL images, local paths, or lists
  of either are accepted. A single image is the default usage; the wrapper
  supports multiple images per state for future use.
* Question format (per question):
      {"instructions": str, "criteria": {"true": str, "false": str}}
  The response score `answers[qid]["noul"]` is a float in [0, 1]; higher =
  more likely the "true" criteria applies.
* Transport errors are retried `DECIDER_MAX_RETRIES` times, then the caller
  is expected to treat the call as \"unavailable\" (decide() returns None) —
  a flaky API must never block the pipeline.
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Optional, Union

import requests
from PIL import Image

import config

ImageLike = Union[str, Path, Image.Image]

_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".tiff": "image/tiff",
}

# --------------------------------------------------------------------------
# Image helpers
# --------------------------------------------------------------------------
def _image_to_data_url(image: ImageLike) -> str:
    """Encode a PIL image or image path as a base64 data URL."""
    if isinstance(image, (str, Path)):
        suffix = Path(image).suffix.lower()
        mime = _MIME_BY_SUFFIX.get(suffix, "image/png")
        with open(image, "rb") as f:
            return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"
    # PIL image — always re-encoded as PNG (safe for any mode)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def _state_from_images(state_text: str, image_input) -> list:
    """Build the Decisions API `state` list: text first, then image parts."""
    state: list = [state_text]
    if image_input is None:
        return state
    if not isinstance(image_input, (list, tuple)):
        image_input = [image_input]
    for img in image_input:
        state.append({"type": "image_url", "image_url": {"url": _image_to_data_url(img)}})
    return state


def _headers() -> dict:
    """Auth + optional OpenRouter attribution headers."""
    headers = {
        "Authorization": f"Bearer {config.DECIDER_API_KEY}",
        "Content-Type": "application/json",
    }
    if config.DECIDER_SITE_URL:
        headers["HTTP-Referer"] = config.DECIDER_SITE_URL
    if config.DECIDER_SITE_NAME:
        headers["X-OpenRouter-Title"] = config.DECIDER_SITE_NAME
    return headers


# --------------------------------------------------------------------------
# Main entrypoint
# --------------------------------------------------------------------------
def decide(state_text: str, questions: dict, image_input=None,
           model: Optional[str] = None) -> Optional[dict[str, float]]:
    """Ask the decider a set of yes/no (noul) questions about image(s).

    Args:
        state_text: free text describing the task/context (shown first in
            the state).
        questions: {qid: {"instructions": str, "true": str, "false": str}}
            — one entry per question; "true"/"false" are the criteria
            descriptions for each side of the judgment.
        image_input: a single image (path or PIL Image) or a list of them.
            Default usage in the pipeline is a single image.
        model: model override; defaults to config.DECIDER_MODEL.

    Returns:
        {qid: score} with score in [0, 1] (higher = the "true" criteria
        applies), or None if the API could not be reached after retries —
        callers should treat None as "verification unavailable" (pass).
    """
    if not config.DECIDER_API_KEY:
        raise RuntimeError(
            "DECIDER_API_KEY is empty — set OPENROUTER_API_KEY in .env or "
            "in the environment before running the decider in non-fake mode."
        )

    payload_questions = {}
    for qid, spec in questions.items():
        payload_questions[qid] = {
            "type": "noul",
            "instructions": spec["instructions"],
            "criteria": {
                "true": spec["true"],
                "false": spec["false"],
            },
        }

    payload = {
        "model": model or config.DECIDER_MODEL,
        "state": _state_from_images(state_text, image_input),
        "questions": payload_questions,
    }

    last_err: Exception | None = None
    for attempt in range(1, config.DECIDER_MAX_RETRIES + 1):
        try:
            resp = requests.post(
                config.DECIDER_API_URL,
                headers=_headers(),
                json=payload,
                timeout=config.DECIDER_TIMEOUT,
            )
            resp.raise_for_status()
            answers = resp.json()["answers"]
            scores = {}
            for qid in questions:
                entry = answers.get(qid) or {}
                score = entry.get("noul")
                if score is None:
                    raise ValueError(f"decider answer missing for question '{qid}'")
                scores[qid] = float(score)
            return scores
        except Exception as e:  # noqa: BLE001 — network/HTTP/parse errors
            last_err = e
            print(f"[Decider] request failed (attempt {attempt}/{config.DECIDER_MAX_RETRIES}): {e}")
            if attempt < config.DECIDER_MAX_RETRIES:
                time.sleep(config.DECIDER_RETRY_DELAY)

    print(f"[Decider] UNAVAILABLE after {config.DECIDER_MAX_RETRIES} attempts "
          f"({last_err}) — treating verification as unavailable (pass)")
    return None
