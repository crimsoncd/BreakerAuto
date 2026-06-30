"""
Pipeline tools — the 6 core functions that all stages use.

Tools:
  vlm(system_prompt, image(s), user_text) -> str
  joyai(image, prompt) -> image
  crop(image, bbox, pad=0.1) -> image
  matte_to_alpha(image_on_plain_bg) -> RGBA
  resize(image, size) -> image
  composite(background, [layers_in_z_order]) -> image

Plus JSON parsing utilities for the VLM outputs.
"""

from __future__ import annotations

import json
import re
import subprocess
import traceback
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image, ImageFilter

from call_Qwen3VL import Qwen3VL_inference
from call_JoyAI import JoyEdit
from config import BBOX_NORM, JOYAI_BASE_SEED

import rembg

# Runtime-resolved device assignments
# QWEN_DEVICE = _QWEN_DEFAULT
# JOYAI_DEVICE = "cuda:2"

# global QWEN_DEVICE, JOYAI_DEVICE

QWEN_DEVICE, JOYAI_DEVICE = None, None
# GPU_SET = False

def auto_detect_gpu_devices(force_qwen: str = None, force_joyai: str = None) -> tuple[str, str]:
    """Detect free GPUs and assign VLM to one, JoyAI to another.

    Strategy: Query `nvidia-smi` for free memory. Pick the two cards with
    the most available memory as candidates.
    """

    global QWEN_DEVICE, JOYAI_DEVICE

    if QWEN_DEVICE is not None and JOYAI_DEVICE is not None:
        print("Using current devices")
        print("Qwen:", QWEN_DEVICE)
        print("JoyAI:", JOYAI_DEVICE)
        return QWEN_DEVICE, JOYAI_DEVICE

    if force_qwen and force_joyai:
        QWEN_DEVICE = force_qwen
        JOYAI_DEVICE = force_joyai
        print(f"[GPU] Manual override: VLM={QWEN_DEVICE}, JoyAI={JOYAI_DEVICE}")
        return QWEN_DEVICE, JOYAI_DEVICE

    try:
        import torch
        if not torch.cuda.is_available():
            print("[GPU] No CUDA available — using CPU fallback")
            QWEN_DEVICE = "cpu"
            JOYAI_DEVICE = "cpu"
            return QWEN_DEVICE, JOYAI_DEVICE

        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            print(f"[GPU] Only {n_gpus} GPU(s) detected — both models on cuda:0")
            QWEN_DEVICE = "cuda:0"
            JOYAI_DEVICE = "cuda:0"
            return QWEN_DEVICE, JOYAI_DEVICE

        # Query nvidia-smi for utilization and memory
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True
        )
        
        gpu_stats = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 4:
                idx = int(parts[0])
                
                # 如果设置了 CUDA_VISIBLE_DEVICES，物理索引可能大于 torch 的可用数量
                # 为了防止报错，如果索引超出了 torch 的检测范围，这里做个安全过滤
                if idx >= n_gpus:
                    continue
                    
                util = float(parts[1])
                mem_used = float(parts[2])
                mem_total = float(parts[3])
                mem_free = mem_total - mem_used
                
                gpu_stats.append((idx, util, mem_free))

        # 核心改动：优先按照【剩余显存】从大到小排序 (x[2]降序)，如果剩余一样，再看【利用率】从小到大 (x[1]升序)
        gpu_stats.sort(key=lambda x: (-x[2], x[1]))

        # 如果过滤后找不到足够的卡，降级处理
        if not gpu_stats:
            raise RuntimeError("No matching GPUs found after filtering.")

        if len(gpu_stats) >= 2:
            vlm_idx = gpu_stats[0][0]
            joyai_idx = gpu_stats[1][0]
            vlm_util = gpu_stats[0][1]
            joyai_util = gpu_stats[1][1]
        else:
            vlm_idx = gpu_stats[0][0]
            joyai_idx = gpu_stats[0][0]
            vlm_util = gpu_stats[0][1]
            joyai_util = gpu_stats[0][1]

        QWEN_DEVICE = f"cuda:{vlm_idx}"
        JOYAI_DEVICE = f"cuda:{joyai_idx}"
        
        print(f"[GPU] Auto-detected: VLM={QWEN_DEVICE} (util={vlm_util:.0f}%), "
              f"JoyAI={JOYAI_DEVICE} (util={joyai_util:.0f}%)")

        # 反射同步到目标模块
        try:
            import call_Qwen3VL
            call_Qwen3VL.QWEN_DEVICE = QWEN_DEVICE
        except ImportError:
            print("[GPU] Warning: call_Qwen3VL module not found, skip variable sync.")

        print("Using Device:")
        print("Qwen:", QWEN_DEVICE)
        print("JoyAI:", JOYAI_DEVICE)

    except Exception as e:
        # 如果全局变量未定义，设定安全兜底值
        if 'QWEN_DEVICE' not in globals(): QWEN_DEVICE = "cuda:2"
        if 'JOYAI_DEVICE' not in globals(): JOYAI_DEVICE = "cuda:3"
        print(f"[GPU] Error occurred ({e}) — fallback to defaults QWEN={QWEN_DEVICE} JOYAI={JOYAI_DEVICE}")

    GPU_SET = True
    return QWEN_DEVICE, JOYAI_DEVICE




ImageLike = Union[str, Path, Image.Image]


# ---------------------------------------------------------------------------
# JSON parsing utilities
# ---------------------------------------------------------------------------
def parse_json_strict(text: str) -> Optional[dict]:
    """Attempt to parse text as strict JSON. Returns dict or None on failure."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_json_relaxed(text: str) -> Optional[dict]:
    """Parse JSON with one retry: strip markdown fences and try again."""
    if not text:
        return None
    # First try strict
    result = parse_json_strict(text)
    if result is not None:
        return result
    # Strip markdown code fences and retry
    cleaned = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    return parse_json_strict(cleaned)


def vlm_json(system_prompt: str, image_input, user_text: str,
             max_new_tokens: int = 512,
             role: str = "vlm",
             logger=None) -> dict:
    """Call VLM and parse JSON response with one retry on parse failure.

    Returns parsed dict (empty dict on total failure).
    Logs the call if logger is provided.
    """
    response = Qwen3VL_inference(
        image_input=image_input,
        prompt=user_text,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        deterministic=True,
        device=QWEN_DEVICE
    )
    if logger:
        logger.log_vlm_call(
            role=role,
            system_prompt=system_prompt,
            user_text=user_text,
            response=response,
            image_label=role,
        )
    result = parse_json_relaxed(response)
    if result is not None:
        return result
    # One explicit reformat-retry
    retry_prompt = f"{user_text}\n\nIMPORTANT: Return ONLY valid JSON, no markdown fences, no other text."
    response2 = Qwen3VL_inference(
        image_input=image_input,
        prompt=retry_prompt,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        deterministic=True,
        device=QWEN_DEVICE
    )
    if logger:
        logger.log_vlm_call(
            role=f"{role}_retry",
            system_prompt=system_prompt,
            user_text=retry_prompt,
            response=response2,
        )
    result = parse_json_relaxed(response2)
    if result is not None:
        return result
    # Total failure — log loudly and return empty
    if logger:
        logger.log_text(
            f"JSON parse failure after retry.\n"
            f"Response 1:\n{response}\n\nResponse 2:\n{response2}",
            label=f"{role}_parse_failure",
        )
    print(f"[WARNING] VLM JSON parse failure for role '{role}' after retry. Returning empty dict.")
    return {}


# ---------------------------------------------------------------------------
# Tool 1: VLM (the only VLM entrypoint for text responses)
# ---------------------------------------------------------------------------
def vlm(system_prompt: str, image_input, user_text: str,
        max_new_tokens: int = 512,
        logger=None) -> str:
    """Call the VLM and return raw text response. Logs if logger provided."""
    response = Qwen3VL_inference(
        image_input=image_input,
        prompt=user_text,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        deterministic=True,
        device=QWEN_DEVICE
    )
    if logger:
        logger.log_vlm_call(
            role="vlm_text",
            system_prompt=system_prompt,
            user_text=user_text,
            response=response,
        )
    return response


# ---------------------------------------------------------------------------
# Tool 2: JoyAI (the only edit entrypoint)
# ---------------------------------------------------------------------------
def joyai(image: ImageLike, prompt: str, output_path: Optional[str | Path] = None,
          seed: int = JOYAI_BASE_SEED, device: Optional[str] = None,
          logger=None) -> Optional[Image.Image]:
    """Run JoyAI edit. Returns PIL Image or None on failure.

    Uses the auto-detected JOYAI_DEVICE by default; pass device= to override.
    """
    if device is None:
        device = JOYAI_DEVICE  # use globally-resolved device
    if logger and prompt:
        logger.log_text(prompt, label="joyai_prompt")
    try:
        result = JoyEdit(
            image=image,
            prompt=prompt,
            output_path=str(output_path) if output_path else None,
            device=device,
            seed=seed,
        )
        if result.ok and result.image is not None:
            return result.image
        else:
            print(f"[JoyAI] Edit failed: {result.error}")
            return None
    except Exception as e:
        print(f"[JoyAI] Exception: {e}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Tool 3: Crop
# ---------------------------------------------------------------------------
def _denorm_bbox(bbox: list[int], img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Convert normalized 0-1000 bbox to pixel coords."""
    xmin = int(bbox[0] / BBOX_NORM * img_w)
    ymin = int(bbox[1] / BBOX_NORM * img_h)
    xmax = int(bbox[2] / BBOX_NORM * img_w)
    ymax = int(bbox[3] / BBOX_NORM * img_h)
    return (xmin, ymin, xmax, ymax)


def crop(image: ImageLike, bbox: list[int], pad: float = 0.1) -> Image.Image:
    """Crop an image to bbox with optional padding.

    Args:
        image: PIL Image or path.
        bbox: [xmin, ymin, xmax, ymax] normalized to 0-1000.
        pad: padding ratio (default 0.1 = 10% on each side).
    """
    if isinstance(image, (str, Path)):
        img = Image.open(image)
    else:
        img = image
    w, h = img.size

    xmin, ymin, xmax, ymax = _denorm_bbox(bbox, w, h)

    # Apply padding
    box_w = xmax - xmin
    box_h = ymax - ymin
    pad_x = int(box_w * pad)
    pad_y = int(box_h * pad)

    xmin = max(0, xmin - pad_x)
    ymin = max(0, ymin - pad_y)
    xmax = min(w, xmax + pad_x)
    ymax = min(h, ymax + pad_y)

    return img.crop((xmin, ymin, xmax, ymax)).convert('RGB')


def denorm_bbox_pixels(bbox: list[int], img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Utility: convert normalized bbox to pixel coords (public version)."""
    return _denorm_bbox(bbox, img_w, img_h)


# ---------------------------------------------------------------------------
# Tool 4: Matte to Alpha
# ---------------------------------------------------------------------------

# New function: use rembg to cut to matte
def matte_to_alpha(image_on_plain_bg: ImageLike) -> Image.Image:

    img = image_on_plain_bg
    if isinstance(img, (str, Path)):
        img = Image.open(img)
    img = img.convert('RGB')

    output = rembg.remove(img)

    return output




# ---------------------------------------------------------------------------
# Tool 5: Resize
# ---------------------------------------------------------------------------
def resize(image: ImageLike, size: tuple[int, int]) -> Image.Image:
    """Resize image to (W, H)."""
    if isinstance(image, (str, Path)):
        img = Image.open(image)
    else:
        img = image
    return img.resize(size, Image.LANCZOS)


def resize_layer_to_bbox(layer_rgba: Image.Image, bbox: list[int],
                         image_w: int, image_h: int) -> Image.Image:
    """Resize a generated layer to the pixel dimensions of its bbox."""
    xmin, ymin, xmax, ymax = _denorm_bbox(bbox, image_w, image_h)
    target_w = xmax - xmin
    target_h = ymax - ymin
    return layer_rgba.resize((target_w, target_h), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Tool 6: Composite
# ---------------------------------------------------------------------------
def composite(background: ImageLike,
              layers: list[tuple[Image.Image, list[int]]], pad = 0.1) -> Image.Image:
    """Composite layers over background.

    Args:
        background: PIL Image or path (RGB or RGBA).
        layers: list of (layer_rgba, bbox_normalized) tuples.
                Layers are drawn in list order (first = bottom-most).
    """
    if isinstance(background, (str, Path)):
        bg = Image.open(background)
    else:
        bg = background

    bg_w, bg_h = bg.size
    canvas = bg.convert('RGBA')

    for layer_img, bbox in layers:
        if layer_img is None:
            continue

        # Ensure layer is RGBA
        if layer_img.mode != 'RGBA':
            layer_img = layer_img.convert('RGBA')

        # Resize layer to bbox dimensions
        # xmin, ymin, xmax, ymax = _denorm_bbox(bbox, bg_w, bg_h)
        # target_w = xmax - xmin
        # target_h = ymax - ymin

        xmin, ymin, xmax, ymax = _denorm_bbox(bbox, bg_w, bg_h)

        # Apply padding
        box_w = xmax - xmin
        box_h = ymax - ymin
        pad_x = int(box_w * pad)
        pad_y = int(box_h * pad)

        xmin = max(0, xmin - pad_x)
        ymin = max(0, ymin - pad_y)
        xmax = min(bg_w, xmax + pad_x)
        ymax = min(bg_h, ymax + pad_y)

        target_w = xmax - xmin
        target_h = ymax - ymin

        if target_w <= 0 or target_h <= 0:
            continue

        layer_resized = layer_img.resize((target_w, target_h), Image.LANCZOS)

        # Paste onto canvas at bbox origin
        canvas.paste(layer_resized, (xmin, ymin), layer_resized)

    # Convert back to RGB for output
    return canvas.convert('RGB')


# ---------------------------------------------------------------------------
# Fakes/stubs for testing (Step 1 of build order)
# ---------------------------------------------------------------------------
FAKE_MODE = False


def set_fake_mode(on: bool = True) -> None:
    """Enable stub mode for testing end-to-end without real models."""
    global FAKE_MODE
    FAKE_MODE = on


def _fake_vlm(system_prompt: str, image_input, user_text: str,
              max_new_tokens: int = 512, logger=None) -> str:
    """Stub VLM that returns canned JSON."""
    # Return different canned responses based on what the system prompt contains
    if "scene analyst" in system_prompt.lower():
        return json.dumps({
            "elements": [
                {"name": "girl", "bbox": [200, 50, 450, 600], "depth_rank": 1, "overlaps": ["grass"]},
                {"name": "tree", "bbox": [600, 30, 900, 500], "depth_rank": 2, "overlaps": ["grass"]},
                {"name": "grass", "bbox": [0, 400, 1000, 1000], "depth_rank": 3, "overlaps": []},
            ]
        })
    if "occupancy" in system_prompt.lower() or "intrude" in system_prompt.lower():
        return json.dumps({"target_present": True, "contaminants": []})
    if "foreground objects to remove" in system_prompt:
        return json.dumps({"prompt": "remove all foreground objects, fill background"})
    if "editing instruction" in system_prompt.lower():
        return json.dumps({"prompt": "isolate the object on plain background"})
    if "quality checker" in system_prompt.lower():
        return json.dumps({"ok": True, "defects": [], "notes": "looks clean"})
    if "final auditor" in system_prompt.lower():
        return json.dumps({"ok": True, "missing": [], "bad_layers": [], "reorder": [], "notes": "acceptable"})
    if "clean background" in system_prompt.lower() or "foreground objects" in system_prompt.lower():
        return json.dumps({"ok": True, "defects": [], "notes": "clean"})
    return "{}"


def _fake_joyai(image: ImageLike, prompt: str, output_path=None, seed=42,
                device="cuda:1", logger=None) -> Optional[Image.Image]:
    """Stub JoyAI that returns the input image unchanged."""
    if isinstance(image, (str, Path)):
        img = Image.open(image)
    else:
        img = image
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
    return img.copy()