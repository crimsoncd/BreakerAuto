"""Callable inference API for the JoyAI-Image release.

Importable wrapper around the original ``JoyAI-inference.py`` CLI script so
that other code can run image editing / text-to-image generation directly:

    from call_JoyAI import JoyEdit, JoyEditBatch

    # Single edit (or pure T2I if image=None)
    result = JoyEdit(
        image='input.png',
        prompt='make the sky pink',
        output_path='out/edited.png',
        device='cuda:1',          # keep OFF the card Qwen3VL occupies
    )

The checkpoint defaults to DEFAULT_CKPT_ROOT
('/remote-home/Zhangkaile/models/JoyAI-Image-Edit/'); pass ckpt_root=...
to override.

The heavy model is built lazily on the first call and cached per
(ckpt_root, config, device) so repeated calls are cheap. Prompt rewriting and
multi-GPU/FSDP are intentionally not supported here.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup (mirrors JoyAI-inference.py so `infer_runtime` / `modules`
# resolve when this file lives next to the original script).
# ---------------------------------------------------------------------------
TARGET_SRC_DIR = '/remote-home/Zhangkaile/dev/JoyAI-Image/src'

if TARGET_SRC_DIR not in sys.path:
    sys.path.insert(0, TARGET_SRC_DIR)

warnings.filterwarnings('ignore')

ImageLike = Union[str, Path, Image.Image, None]

# Default checkpoint location (override per-call via ckpt_root=...)
DEFAULT_CKPT_ROOT = '/remote-home/Zhangkaile/models/JoyAI-Image-Edit/'

# Default card for JoyAI. MUST differ from the VLM's card (Qwen is on cuda:0).
DEFAULT_DEVICE = 'cuda:2'


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------
@dataclass
class JoyResult:
    """Outcome of one generation call."""
    image: Optional[Image.Image]          # the generated PIL image (None on failure)
    output_path: Optional[Path]           # where it was saved (None if not saved / failed)
    prompt: str
    elapsed: float = 0.0                  # seconds spent in model.infer
    ok: bool = True
    error: Optional[str] = None

    def __bool__(self) -> bool:           # allows `if result:`
        return self.ok


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict = {}


def _resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve the target device.

    Priority: explicit `device` arg  ->  DEFAULT_DEVICE  ->  cpu.
    We deliberately do NOT fall back to LOCAL_RANK/cuda:0, because cuda:0 is
    occupied by the VLM and co-locating the two 80G-class models OOMs.
    """
    if not torch.cuda.is_available():
        return torch.device('cpu')
    if device is not None:
        dev = torch.device(device)
    else:
        dev = torch.device(DEFAULT_DEVICE)
    if dev.type == 'cuda':
        torch.cuda.set_device(dev.index or 0)
    return dev


def get_model(ckpt_root: Union[str, Path],
              config: Optional[Union[str, Path]] = None,
              default_seed: int = 42,
              device: Optional[Union[str, torch.device]] = None,
              verbose: bool = True):
    """Build (or fetch from cache) the JoyAI model for the given checkpoint.

    Returns the model object exposing ``.infer(InferenceParams)``.
    Cached per (ckpt_root, config, device).
    """
    from infer_runtime.model import build_model
    from infer_runtime.settings import load_settings
    from modules.models.attention import describe_attention_backend

    resolved = _resolve_device(device)
    key = (str(Path(ckpt_root).resolve()), str(config) if config else None, str(resolved))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    settings = load_settings(
        ckpt_root=str(ckpt_root),
        config_path=str(config) if config else None,
        rewrite_model=None,          # prompt rewriting not supported here
        default_seed=default_seed,
    )

    if verbose:
        print(f'[JoyAI] Device: {resolved}')
        print(f'[JoyAI] Attention backend: {describe_attention_backend()}')
        print(f'[JoyAI] Config path: {settings.config_path}')
        print(f'[JoyAI] Checkpoint path: {settings.ckpt_path}')

    model = build_model(settings, device=resolved)
    _MODEL_CACHE[key] = model
    return model


def clear_model_cache() -> None:
    """Drop cached models and free GPU memory."""
    _MODEL_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_image(image: ImageLike) -> Optional[Image.Image]:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert('RGB')
    return Image.open(str(image)).convert('RGB')


def _save(img: Image.Image, output_path: Union[str, Path]) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# Public API: single generation
# ---------------------------------------------------------------------------
def JoyEdit(image: ImageLike,
            prompt: str,
            output_path: Optional[Union[str, Path]] = None,
            *,
            ckpt_root: Union[str, Path] = DEFAULT_CKPT_ROOT,
            config: Optional[Union[str, Path]] = None,
            device: Optional[Union[str, torch.device]] = None,
            height: int = 512,
            width: int = 512,
            steps: int = 30,
            guidance_scale: float = 5.0,
            seed: int = 42,
            neg_prompt: str = '',
            basesize: int = 512,
            model=None,
            verbose: bool = True) -> JoyResult:
    """Edit an image (or generate one from text if ``image`` is None).

    Args:
        image: input image path / PIL.Image for editing, or None for T2I.
        prompt: edit instruction or T2I prompt.
        output_path: where to save the result; if None the image is only
            returned in memory.
        ckpt_root: checkpoint root directory (required, keyword-friendly).
        config: optional config path (defaults to <ckpt_root>/infer_config.py).
        device: which GPU to run on, e.g. 'cuda:1'. Defaults to DEFAULT_DEVICE
            (cuda:1) so JoyAI stays OFF the VLM's card. Only used when `model`
            is built here (ignored if a pre-built `model` is passed).
        height / width: output size, only used for text-to-image.
        steps, guidance_scale, seed, neg_prompt, basesize: sampler params.
            NOTE on retries: bump `seed` per attempt — re-running the same prompt
            with the same seed reproduces the same bad output and wastes the try.
        model: pass a pre-built model (from ``get_model``) to skip the cache.
        verbose: print progress info.

    Returns:
        JoyResult with the PIL image, save path, and timing.
    """
    from infer_runtime.model import InferenceParams

    if model is None:
        model = get_model(ckpt_root, config=config, default_seed=seed,
                          device=device, verbose=verbose)

    input_image = _load_image(image)

    print("\n[JoyAI] Processing Image on Joy AI")
    print(f"[JoyAI] Prompt: {prompt}")

    start = time.time()
    output_image = model.infer(
        InferenceParams(
            prompt=prompt,
            image=input_image,
            height=height,
            width=width,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            neg_prompt=neg_prompt,
            basesize=basesize,
        )
    )
    elapsed = time.time() - start

    print(f"[JoyAI] Inference completed in {elapsed:.2f}s.")

    saved_path: Optional[Path] = None
    if output_path is not None:
        saved_path = _save(output_image, output_path)

    if verbose:
        if saved_path:
            print(f'[JoyAI] Saved: {saved_path}')
        print(f'[JoyAI] Time: {elapsed:.2f}s')

    return JoyResult(image=output_image, output_path=saved_path,
                     prompt=prompt, elapsed=elapsed)


# ---------------------------------------------------------------------------
# Public API: batch generation
# ---------------------------------------------------------------------------
def JoyEditBatch(image_path_list: Sequence[ImageLike],
                 prompt_list: Union[str, Sequence[str]],
                 output_path: Union[str, Path, Sequence[Union[str, Path]], None] = None,
                 *,
                 ckpt_root: Union[str, Path] = DEFAULT_CKPT_ROOT,
                 config: Optional[Union[str, Path]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 height: int = 512,
                 width: int = 512,
                 steps: int = 30,
                 guidance_scale: float = 5.0,
                 seed: int = 42,
                 vary_seed: bool = False,
                 neg_prompt: str = '',
                 basesize: int = 512,
                 skip_errors: bool = True,
                 verbose: bool = True) -> List[JoyResult]:
    """Run a batch of edits/generations sequentially with one model load.

    Args:
        image_path_list: list of input images (path / PIL.Image / None for T2I).
        prompt_list: one prompt per image, or a single prompt applied to all.
        output_path: directory, list of paths, or None (in-memory only).
        device: which GPU to run on (e.g. 'cuda:1'); shared across the batch.
        seed: base seed; if ``vary_seed`` is True, item i uses seed + i.
        skip_errors: if True, a failing item yields JoyResult(ok=False) and the
            batch continues; if False the exception propagates.

    Returns:
        list of JoyResult, in input order.
    """
    n = len(image_path_list)

    # Normalize prompts
    if isinstance(prompt_list, str):
        prompts = [prompt_list] * n
    else:
        prompts = list(prompt_list)
        if len(prompts) != n:
            raise ValueError(
                f'prompt_list length ({len(prompts)}) != image list length ({n})')

    # Normalize output paths
    out_paths: List[Optional[Path]]
    if output_path is None:
        out_paths = [None] * n
    elif isinstance(output_path, (str, Path)):
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_paths = [out_dir / f'{i:04d}.png' for i in range(n)]
    else:
        out_paths = [Path(p) for p in output_path]
        if len(out_paths) != n:
            raise ValueError(
                f'output path list length ({len(out_paths)}) != image list length ({n})')

    # Load the model once for the whole batch, on the chosen card.
    model = get_model(ckpt_root, config=config, default_seed=seed,
                      device=device, verbose=verbose)

    results: List[JoyResult] = []
    for i, (img, prompt) in enumerate(zip(image_path_list, prompts)):
        item_seed = seed + i if vary_seed else seed
        if verbose:
            print(f'[JoyAI] ({i + 1}/{n}) {prompt[:80]}')
        try:
            res = JoyEdit(
                img, prompt, out_paths[i],
                ckpt_root=ckpt_root, config=config, model=model,
                height=height, width=width, steps=steps,
                guidance_scale=guidance_scale, seed=item_seed,
                neg_prompt=neg_prompt, basesize=basesize,
                verbose=verbose,
            )
        except Exception as exc:  # noqa: BLE001
            if not skip_errors:
                raise
            if verbose:
                print(f'[JoyAI] item {i} failed: {exc}')
            res = JoyResult(image=None, output_path=None, prompt=prompt,
                            ok=False, error=str(exc))
        results.append(res)

    if verbose:
        ok = sum(1 for r in results if r.ok)
        print(f'[JoyAI] Batch done: {ok}/{n} succeeded.')
    return results


# ---------------------------------------------------------------------------
# Optional: tiny CLI for quick sanity checks
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Quick test of call_JoyAI.')
    parser.add_argument('--ckpt-root', default=DEFAULT_CKPT_ROOT)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--image')
    parser.add_argument('--output', default='example.png')
    parser.add_argument('--device', default=None, help="e.g. cuda:1 (defaults to cuda:1)")
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    r = JoyEdit(args.image, args.prompt, args.output,
                ckpt_root=args.ckpt_root, device=args.device,
                steps=args.steps, seed=args.seed)
    print('OK' if r.ok else f'FAILED: {r.error}')
