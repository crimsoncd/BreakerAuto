"""Judge whether an image contains exactly one <item>, via the OpenRouter Decisions API.

Usage:
    python judge_single_item.py path/to/image.jpg "red mug"
"""

import base64
import json
import mimetypes
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "perplexity/pplx-decider-v1.1-27b"  # text + image capable; swap for any multimodal Decisions model
# MODEL = "openai/gpt-6-luna-decisions"   # This model cannot be reached 
SUPPORTED_MIME = {"image/png", "image/jpeg", "image/webp"}


def image_data_url(path: str) -> str:
    """Return a base64 data URL for a local png/jpeg/webp image."""
    mime = mimetypes.guess_type(path)[0]
    if mime not in SUPPORTED_MIME:
        raise ValueError(f"Unsupported image type {mime!r}; use png, jpeg or webp.")
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


def _headers() -> dict:
    """Auth + optional ranking headers read from .env."""
    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
    }
    if site := os.getenv("OPENROUTER_SITE_URL"):
        headers["HTTP-Referer"] = site
    if title := os.getenv("OPENROUTER_SITE_NAME"):
        headers["X-OpenRouter-Title"] = title
    return headers


def contains_exactly_one(image_path: str, item: str) -> bool:
    """True iff the image shows exactly {item} ."""
    payload = {
        "model": MODEL,
        "state": [
            f"Does this image contain only {item} and nothing else?",
            {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
        ],
        "questions": {
            "exactly_one": {
                "type": "noul",
                "instructions": f"Does the image contain only {item} and nothing else?",
                "criteria": {
                    "true": f"The image shows exactly only {item} and nothing else.",
                    "false": f"The image shows no {item}, or shows something more than one {item}.",
                },
            }
        },
    }
    resp = requests.post(API_URL, headers=_headers(), data=json.dumps(payload), timeout=60)
    resp.raise_for_status()
    return resp.json()["answers"]["exactly_one"]["noul"] > 0.5


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <image_path> <item>")
    print(contains_exactly_one(sys.argv[1], sys.argv[2]))