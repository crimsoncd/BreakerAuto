"""
Pipeline tools — the 6 core functions that all stages use.

Tools:
  vlm(system_prompt, image(s), user_text) -> str
  qwen_edit(image, prompt) -> image
  crop(image, bbox, pad=0.1) -> image
  matte_to_alpha(image_on_plain_bg) -> RGBA
  resize(image, size) -> image
  composite(background, [layers_in_z_order]) -> image

Plus JSON parsing utilities for the VLM outputs.

The VLM is a remote Bailian (DashScope) API model — see call_bailian_vlm.py.
It keeps the same call signature as the old local Qwen3.8 wrapper, so the
pipeline logic is unchanged; only the transport differs.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
import traceback
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

from call_bailian_vlm import BailianVLM_inference
from call_qwen_edit import QwenEdit
from config import (
    BBOX_NORM, QWEN_EDIT_BASE_SEED,
    QWEN_EDIT_DEVICE as _QWEN_EDIT_DEVICE,
    BG_COLOR_TOL, BG_FLAT_MIN_FRAC, BG_FILL_DILATE_IT, BG_FILL_PAD,
)

import rembg

# Edit model's device. Seeded from config/env; resolved at runtime by
# resolve_edit_device() when None. The VLM is remote (Bailian API), so the
# Qwen-Image-2.1 edit model is the only local model and a single GPU suffices.
EDIT_DEVICE = _QWEN_EDIT_DEVICE


def resolve_edit_device(force: str = None) -> str:
    """Resolve the GPU card for the Qwen-Image-2.1 edit model — the only local
    model in the pipeline.

    The VLM runs remotely via the Bailian API, so no second card is needed
    and there is no co-residence constraint anymore.

    Priority: explicit `force` argument > config.QWEN_EDIT_DEVICE (env
    QWEN_EDIT_DEVICE) > auto-detect the card with the most free memory.
    """
    global EDIT_DEVICE

    if EDIT_DEVICE is not None:
        print(f"[GPU] Edit-model device (pre-set): {EDIT_DEVICE}")
        return EDIT_DEVICE

    force = force or _QWEN_EDIT_DEVICE
    if force:
        EDIT_DEVICE = force
        print(f"[GPU] Edit-model device (manual override): {EDIT_DEVICE}")
        return EDIT_DEVICE

    try:
        import torch
        if not torch.cuda.is_available():
            print("[GPU] No CUDA available — edit model falling back to CPU")
            EDIT_DEVICE = "cpu"
            return EDIT_DEVICE

        # Query nvidia-smi for memory; pick the card with the most free memory.
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True
        )

        n_gpus = torch.cuda.device_count()
        gpu_stats = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 4:
                idx = int(parts[0])
                # If CUDA_VISIBLE_DEVICES is set, the physical index may exceed
                # what torch sees. Filter out such indices safely.
                if idx >= n_gpus:
                    continue
                util = float(parts[1])
                mem_used = float(parts[2])
                mem_total = float(parts[3])
                gpu_stats.append((idx, util, mem_total - mem_used))

        if not gpu_stats:
            raise RuntimeError("No matching GPUs found after filtering.")

        # Most free memory first, then lowest utilization.
        gpu_stats.sort(key=lambda x: (-x[2], x[1]))
        EDIT_DEVICE = f"cuda:{gpu_stats[0][0]}"
        print(f"[GPU] Auto-detected edit-model device: {EDIT_DEVICE}")

    except Exception as e:
        EDIT_DEVICE = "cuda:0"
        print(f"[GPU] Error occurred ({e}) — edit-model fallback device: {EDIT_DEVICE}")

    return EDIT_DEVICE




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
    n_imgs = 0 if image_input is None else (
        len(image_input) if isinstance(image_input, (list, tuple)) else 1)
    print(f"[VLM] {role}: sending request ({n_imgs} image(s), max_tokens={max_new_tokens})...")
    t0 = time.time()
    response = BailianVLM_inference(
        image_input=image_input,
        prompt=user_text,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        deterministic=True,
    )
    print(f"[VLM] {role}: response received in {time.time() - t0:.1f}s")
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
    print(f"[VLM] {role}: JSON parse failed — reformat retry...")
    retry_prompt = f"{user_text}\n\nIMPORTANT: Return ONLY valid JSON, no markdown fences, no other text."
    t1 = time.time()
    response2 = BailianVLM_inference(
        image_input=image_input,
        prompt=retry_prompt,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        deterministic=True,
    )
    print(f"[VLM] {role}: reformat retry done in {time.time() - t1:.1f}s")
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
    print(f"[VLM] vlm_text: sending request (max_tokens={max_new_tokens})...")
    t0 = time.time()
    response = BailianVLM_inference(
        image_input=image_input,
        prompt=user_text,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        deterministic=True,
    )
    print(f"[VLM] vlm_text: response received in {time.time() - t0:.1f}s")
    if logger:
        logger.log_vlm_call(
            role="vlm_text",
            system_prompt=system_prompt,
            user_text=user_text,
            response=response,
        )
    return response


# ---------------------------------------------------------------------------
# Tool 2: Qwen-Image-2.1 edit (the only edit entrypoint)
# ---------------------------------------------------------------------------
def qwen_edit(image: ImageLike, prompt: str, output_path: Optional[str | Path] = None,
              seed: int = QWEN_EDIT_BASE_SEED, device: Optional[str] = None,
              logger=None) -> Optional[Image.Image]:
    """Run the Qwen-Image-2.1 edit. Returns PIL Image or None on failure.

    The pipeline is loaded lazily on the first call and kept resident for the
    rest of the task. The model may emit a different output size than the
    input; the result is always resized back to the exact input size (done
    inside QwenEdit).

    Uses the auto-detected EDIT_DEVICE by default; pass device= to override.
    """
    if device is None:
        device = EDIT_DEVICE  # use globally-resolved device
    if logger and prompt:
        logger.log_text(prompt, label="edit_prompt")
    print(f"[QwenEdit] editing started (device={device}, seed={seed}): {prompt[:80]}")
    t0 = time.time()
    try:
        result = QwenEdit(
            image=image,
            prompt=prompt,
            output_path=str(output_path) if output_path else None,
            device=device,
            seed=seed,
        )
        if result.ok and result.image is not None:
            print(f"[QwenEdit] editing done in {time.time() - t0:.1f}s")
            return result.image
        else:
            print(f"[QwenEdit] editing FAILED after {time.time() - t0:.1f}s: {result.error}")
            return None
    except Exception as e:
        print(f"[QwenEdit] editing EXCEPTION after {time.time() - t0:.1f}s: {e}")
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

    t0 = time.time()
    output = rembg.remove(img)
    print(f"[Matte] rembg matting done in {time.time() - t0:.1f}s")

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
# Tool 7: Classical background analysis & fill (Stage 3, default route)
# ---------------------------------------------------------------------------
def _bbox_union_mask(bboxes: list[list[int]], img_w: int, img_h: int,
                     pad: float = BG_FILL_PAD) -> np.ndarray:
    """Binary uint8 mask (255 inside) of the padded union of element bboxes.

    bboxes are [xmin, ymin, xmax, ymax] normalized to 0-1000.
    """
    m = np.zeros((img_h, img_w), np.uint8)
    for b in bboxes:
        x0, y0 = int(b[0] / BBOX_NORM * img_w), int(b[1] / BBOX_NORM * img_h)
        x1, y1 = int(b[2] / BBOX_NORM * img_w), int(b[3] / BBOX_NORM * img_h)
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)
        x0, y0 = max(0, x0 - int(bw * pad)), max(0, y0 - int(bh * pad))
        x1, y1 = min(img_w, x1 + int(bw * pad)), min(img_h, y1 + int(bh * pad))
        m[y0:y1, x0:x1] = 255
    return m


def dilate_mask(mask: np.ndarray, iterations: int = BG_FILL_DILATE_IT) -> np.ndarray:
    """Binary dilation of a 0/255 uint8 mask with a 3x3 square element.

    Pure numpy (no cv2 dependency); good enough for small kernels.
    """
    m = (mask > 0).astype(np.uint8)
    for _ in range(int(iterations)):
        p = np.pad(m, 1, mode="constant")
        m = np.maximum.reduce([p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:],
                               p[1:-1, :-2], p[1:-1, 1:-1], p[1:-1, 2:],
                               p[2:, :-2], p[2:, 1:-1], p[2:, 2:]]).astype(np.uint8)
    return m * 255


def estimate_background(image: ImageLike, bboxes: list[list[int]],
                        pad: float = BG_FILL_PAD, tol: int = BG_COLOR_TOL,
                        flat_min_frac: float = BG_FLAT_MIN_FRAC) -> dict:
    """Classically analyze the background of an illustration.

    Estimates the background color as the median of all pixels OUTSIDE the
    padded union of the element bboxes, and measures how flat the background is.

    Returns dict:
        color        [r, g, b] or None (not measurable / nothing unmasked)
        flat_frac    fraction of unmasked pixels within `tol` of the color
        unmasked_frac fraction of the image left outside the padded boxes
        is_flat      True if flat_frac >= flat_min_frac
    """
    img = Image.open(image) if isinstance(image, (str, Path)) else image
    img = img.convert("RGB")
    w, h = img.size
    arr = np.array(img)

    if bboxes:
        bmask = _bbox_union_mask(bboxes, w, h, pad)
        keep = bmask == 0
    else:
        keep = np.ones((h, w), bool)

    unmasked_frac = float(keep.mean())
    if unmasked_frac < 0.01:
        # No reliable background region to sample — cannot analyze.
        return {"color": None, "flat_frac": 0.0, "unmasked_frac": unmasked_frac,
                "is_flat": False}

    px = arr[keep].reshape(-1, 3).astype(np.int32)
    color = np.median(px, axis=0)
    dev = np.abs(px - color).max(axis=1)
    flat_frac = float((dev < tol).mean())

    return {"color": [int(c) for c in color],
            "flat_frac": round(flat_frac, 4),
            "unmasked_frac": round(unmasked_frac, 4),
            "is_flat": bool(flat_frac >= flat_min_frac)}


def classic_background_fill(image: ImageLike, bboxes: list[list[int]],
                            bg_color: list[int],
                            pad: float = BG_FILL_PAD, tol: int = BG_COLOR_TOL,
                            dilate_it: int = BG_FILL_DILATE_IT
                            ) -> tuple[Image.Image, np.ndarray]:
    """Classical background extraction: replace foreground with the bg color.

    Fill mask = dilated color-outlier mask  ∪  padded bbox union:
      - the color-outlier part catches object pixels anywhere in the image
        (including fragments outside their planned bboxes);
      - the dilation removes anti-aliasing ghost fringes;
      - the bbox union catches objects whose color is close to the background.

    Every fill-mask pixel is set to `bg_color`; all other pixels are passed
    through unchanged — the result is pixel-exact outside the filled regions.

    Returns (filled RGB image, fill mask ndarray 0/255).
    """
    img = Image.open(image) if isinstance(image, (str, Path)) else image
    img = img.convert("RGB")
    w, h = img.size
    arr = np.array(img)

    color = np.array(bg_color, dtype=np.int32)
    dev = np.abs(arr.astype(np.int32) - color).max(axis=2)
    outlier = ((dev > tol) * 255).astype(np.uint8)

    fill_mask = np.maximum(dilate_mask(outlier, dilate_it),
                           _bbox_union_mask(bboxes, w, h, pad))

    filled = arr.copy()
    filled[fill_mask > 0] = color.astype(np.uint8)
    return Image.fromarray(filled), fill_mask


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
    # Layout describer — must be checked first: its prompt mentions several
    # phrases shared with other roles (background, foreground objects).
    if "scene describer" in system_prompt.lower():
        # Element ids arrive inside the {element_summaries} JSON payload.
        ids = re.findall(r'"id":\s*"([^"]+)"', system_prompt)
        return json.dumps({
            "description": "A flat-style illustration of a simple outdoor scene: a girl stands on a grassy field beside a tree under a bright open sky.",
            "background": {
                "name": "bright sky and open field",
                "description": "A plain outdoor backdrop with soft sky tones above an empty grassy field and no foreground objects.",
            },
            "global_style": {"color_scheme": "Bright natural tones", "mood": "Calm and cheerful"},
            "elements": [
                {"id": i, "description": f"Layout item {i} placed in the scene."}
                for i in ids
            ],
        })
    # Return different canned responses based on what the system prompt contains
    if "scene analyst" in system_prompt.lower():
        return json.dumps({
            "layout": [
                {"order": 0, "name": "bright_sky_background", "bbox": [0, 0, 1000, 1000],
                 "description": "A flat-style illustration of a simple outdoor scene: a girl stands on a grassy field beside a tree under a bright open sky."},
                {"order": 1, "name": "grassy_field", "bbox": [0, 400, 1000, 1000],
                 "description": "The grassy field covering the lower part of the scene."},
                {"order": 2, "name": "tree", "bbox": [600, 30, 900, 500],
                 "description": "A leafy tree on the right side of the scene."},
                {"order": 3, "name": "girl", "bbox": [200, 50, 450, 600],
                 "description": "A girl standing on the grassy field in the foreground."},
            ]
        })
    if "occupancy" in system_prompt.lower() or "intrude" in system_prompt.lower():
        return json.dumps({"target_present": True, "contaminants": []})
    if "measured background color" in system_prompt.lower():
        # background prompt writer (anchored)
        return json.dumps({"prompt": "remove all foreground objects and fill with the flat background color"})
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


def _fake_qwen_edit(image: ImageLike, prompt: str, output_path=None, seed=42,
                    device="cuda:1", logger=None) -> Optional[Image.Image]:
    """Stub edit model that returns the input image unchanged."""
    if isinstance(image, (str, Path)):
        img = Image.open(image)
    else:
        img = image
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
    return img.copy()