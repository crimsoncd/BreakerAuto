"""Callable inference API for the Qwen-Image-2.1 image-edit model.

Importable wrapper around the diffusers ``QwenImage21Pipeline`` so that the
pipeline code can run image editing directly:

    from call_qwen_edit import QwenEdit

    result = QwenEdit(
        image='input.png',
        prompt='isolate the girl on a plain background',
        output_path='out/edited.png',
        device='cuda:2',          # keep OFF busy cards
    )

The checkpoint defaults to config.QWEN_EDIT_CKPT_ROOT. The heavy pipeline is
built lazily on the first call and cached per (ckpt_root, device) so repeated
calls are cheap — the model stays resident for the whole task and is never
reloaded per step.

Size contract: Qwen-Image-2.1 is a diffusion model that internally resizes
any input to multiple-of-32 dimensions with area ~= output_resolution^2
(aspect preserved), so the raw output size generally differs from the input
size. Every call therefore resizes the generated image back to the exact
original input size (LANCZOS) before returning it.

NOTE on retries: bump ``seed`` per attempt — re-running the same prompt with
the same seed reproduces the same bad output and wastes the try.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

from PIL import Image

import config

ImageLike = Union[str, Path, Image.Image]

# Default checkpoint location (override per-call via ckpt_root=...)
DEFAULT_CKPT_ROOT = config.QWEN_EDIT_CKPT_ROOT

# Default GPU card. None -> resolved by the caller (pipeline_tools) or
# 'cuda:0' at the last resort. Env: QWEN_EDIT_DEVICE.
DEFAULT_DEVICE = config.QWEN_EDIT_DEVICE or 'cuda:0'

# Default sampler settings. Steps follow the reference sample script.
DEFAULT_STEPS = config.QWEN_EDIT_STEPS
DEFAULT_BASE_SEED = config.QWEN_EDIT_BASE_SEED


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------
@dataclass
class QwenEditResult:
    """Outcome of one generation call."""
    image: Optional[Image.Image]          # generated PIL image, resized to the ORIGINAL input size (None on failure)
    output_path: Optional[Path]           # where it was saved (None if not saved / failed)
    prompt: str
    elapsed: float = 0.0                  # seconds spent in the pipeline call
    ok: bool = True
    error: Optional[str] = None

    def __bool__(self) -> bool:           # allows `if result:`
        return self.ok


# ---------------------------------------------------------------------------
# Pipeline cache (model stays resident across calls)
# ---------------------------------------------------------------------------
_PIPELINE_CACHE: dict = {}


def _resolve_device(device: Optional[str] = None) -> str:
    """Resolve the target device string.

    Priority: explicit `device` arg -> DEFAULT_DEVICE -> cpu when CUDA is
    unavailable.
    """
    import torch

    if not torch.cuda.is_available():
        return 'cpu'
    if device is not None:
        return str(device)
    return DEFAULT_DEVICE


def get_pipeline(ckpt_root: Union[str, Path] = DEFAULT_CKPT_ROOT,
                 device: Optional[str] = None,
                 verbose: bool = True):
    """Build (or fetch from cache) the QwenImage21Pipeline.

    The heavy model is loaded ONCE and kept resident; subsequent calls reuse
    the cached pipeline so no reload happens per task step. Cached per
    (ckpt_root, device).
    """
    import torch
    from diffusers import QwenImage21Pipeline

    resolved = _resolve_device(device)
    key = (str(Path(ckpt_root).resolve()), resolved)
    if key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[key]

    if verbose:
        print(f'[QwenEdit] Loading Qwen-Image-2.1 pipeline (this happens ONCE)...')
        print(f'[QwenEdit] Checkpoint: {ckpt_root}')
        print(f'[QwenEdit] Device: {resolved}')

    t0 = time.time()
    pipe = QwenImage21Pipeline.from_pretrained(
        str(ckpt_root),
        torch_dtype=torch.bfloat16,
    ).to(resolved)
    elapsed = time.time() - t0

    _PIPELINE_CACHE[key] = pipe
    if verbose:
        print(f'[QwenEdit] Pipeline loaded and cached in {elapsed:.1f}s (stays resident)')
    return pipe


def clear_pipeline_cache() -> None:
    """Drop cached pipelines and free GPU memory."""
    _PIPELINE_CACHE.clear()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_image(image: ImageLike) -> Image.Image:
    if image is None:
        raise ValueError('QwenEdit requires an input image (no text-to-image support).')
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
def QwenEdit(image: ImageLike,
             prompt: str,
             output_path: Optional[Union[str, Path]] = None,
             *,
             ckpt_root: Union[str, Path] = DEFAULT_CKPT_ROOT,
             device: Optional[str] = None,
             steps: int = DEFAULT_STEPS,
             guidance_scale: Optional[float] = None,
             seed: int = DEFAULT_BASE_SEED,
             model=None,
             verbose: bool = True) -> QwenEditResult:
    """Edit an image with Qwen-Image-2.1.

    Args:
        image: input image path / PIL.Image (required).
        prompt: edit instruction.
        output_path: where to save the result; if None the image is only
            returned in memory.
        ckpt_root: checkpoint root directory of Qwen-Image-2.1.
        device: which GPU to run on, e.g. 'cuda:1'. Only used when `model`
            is built here (ignored if a pre-built `model` is passed).
        steps: number of inference steps (default from config.QWEN_EDIT_STEPS).
        guidance_scale: optional classifier-free guidance; None keeps the
            pipeline's own default.
        seed: sampler seed. NOTE on retries: bump `seed` per attempt — the
            same prompt + seed reproduces the same bad output.
        model: pass a pre-built pipeline (from ``get_pipeline``) to skip the
            cache lookup.
        verbose: print progress info.

    Returns:
        QwenEditResult with the PIL image **resized back to the exact
        original input size** (the diffusion model may emit a different
        resolution), the save path, and timing.
    """
    import torch

    if model is None:
        model = get_pipeline(ckpt_root, device=device, verbose=verbose)
        resolved_device = _resolve_device(device)
    else:
        # Generator must live on the same device as the (pre-built) pipeline.
        resolved_device = str(getattr(model, 'device', _resolve_device(device)))

    input_image = _load_image(image)
    orig_size = input_image.size  # (w, h)

    print("\n[QwenEdit] Processing image with Qwen-Image-2.1")
    print(f"[QwenEdit] Prompt: {prompt}")

    kwargs = dict(
        prompt=prompt,
        image=input_image,
        num_inference_steps=steps,
        generator=torch.Generator(resolved_device).manual_seed(seed),
    )
    if guidance_scale is not None:
        kwargs['guidance_scale'] = guidance_scale

    start = time.time()
    output_image = model(**kwargs).images[0]
    elapsed = time.time() - start

    # --- Restore exact original size (diffusion output size may differ) ---
    if output_image.size != orig_size:
        if verbose:
            print(f"[QwenEdit] Resizing output {output_image.size} -> original {orig_size}")
        output_image = output_image.resize(orig_size, Image.LANCZOS)

    print(f"[QwenEdit] Inference completed in {elapsed:.2f}s.")

    saved_path: Optional[Path] = None
    if output_path is not None:
        saved_path = _save(output_image, output_path)

    if verbose:
        if saved_path:
            print(f'[QwenEdit] Saved: {saved_path}')
        print(f'[QwenEdit] Time: {elapsed:.2f}s')

    return QwenEditResult(image=output_image, output_path=saved_path,
                          prompt=prompt, elapsed=elapsed)


# ---------------------------------------------------------------------------
# Public API: batch generation
# ---------------------------------------------------------------------------
def QwenEditBatch(image_path_list: Sequence[ImageLike],
                  prompt_list: Union[str, Sequence[str]],
                  output_path: Union[str, Path, Sequence[Union[str, Path]], None] = None,
                  *,
                  ckpt_root: Union[str, Path] = DEFAULT_CKPT_ROOT,
                  device: Optional[str] = None,
                  steps: int = DEFAULT_STEPS,
                  guidance_scale: Optional[float] = None,
                  seed: int = DEFAULT_BASE_SEED,
                  vary_seed: bool = False,
                  skip_errors: bool = True,
                  verbose: bool = True) -> List[QwenEditResult]:
    """Run a batch of edits sequentially with ONE model load (resident).

    Args:
        image_path_list: list of input images (path / PIL.Image).
        prompt_list: one prompt per image, or a single prompt applied to all.
        output_path: directory, list of paths, or None (in-memory only).
        device: which GPU to run on (e.g. 'cuda:1'); shared across the batch.
        seed: base seed; if `vary_seed` is True, item i uses seed + i.
        skip_errors: if True, a failing item yields QwenEditResult(ok=False)
            and the batch continues; if False the exception propagates.

    Returns:
        list of QwenEditResult, in input order.
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

    # Load the model ONCE for the whole batch, on the chosen card.
    model = get_pipeline(ckpt_root, device=device, verbose=verbose)

    results: List[QwenEditResult] = []
    for i, (img, prompt) in enumerate(zip(image_path_list, prompts)):
        item_seed = seed + i if vary_seed else seed
        if verbose:
            print(f'[QwenEdit] ({i + 1}/{n}) {prompt[:80]}')
        try:
            res = QwenEdit(
                img, prompt, out_paths[i],
                ckpt_root=ckpt_root, device=device, model=model,
                steps=steps, guidance_scale=guidance_scale,
                seed=item_seed, verbose=verbose,
            )
        except Exception as exc:  # noqa: BLE001
            if not skip_errors:
                raise
            if verbose:
                print(f'[QwenEdit] item {i} failed: {exc}')
            res = QwenEditResult(image=None, output_path=None, prompt=prompt,
                                 ok=False, error=str(exc))
        results.append(res)

    if verbose:
        ok = sum(1 for r in results if r.ok)
        print(f'[QwenEdit] Batch done: {ok}/{n} succeeded.')
    return results


# ---------------------------------------------------------------------------
# Optional: tiny CLI for quick sanity checks
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Quick test of call_qwen_edit.')
    parser.add_argument('--ckpt-root', default=DEFAULT_CKPT_ROOT)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', default='example.png')
    parser.add_argument('--device', default=None,
                        help="e.g. cuda:1 (defaults to config.QWEN_EDIT_DEVICE)")
    parser.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    parser.add_argument('--seed', type=int, default=DEFAULT_BASE_SEED)
    args = parser.parse_args()

    try:
        r = QwenEdit(args.image, args.prompt, args.output,
                     ckpt_root=args.ckpt_root, device=args.device,
                     steps=args.steps, seed=args.seed)
        print('OK' if r.ok else f'FAILED: {r.error}')
    except Exception as exc:  # noqa: BLE001
        print(f'FAILED: {exc}')
