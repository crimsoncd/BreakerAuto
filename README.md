# Layer Decomposition Pipeline

An agentic pipeline that takes a single artistic illustration and decomposes it into a **background** layer plus N **element** layers (object-level cutouts), then reassembles them into a reconstruction of the original.

- Perception & judgment (planning, prompt-writing, verification escalation) → **qwen3.7-flash** via the Bailian (DashScope) OpenAI-compatible API — no local GPU needed
- Fast verification gate (element/background checks) → a **decider** (OpenRouter Decisions API, default `perplexity/pplx-decider-v1.1-27b`): one cheap call, a few yes/no questions per image
- Generative isolation / amodal completion / background fill (model route) → **Qwen-Image-2.1** (diffusers `QwenImage21Pipeline`, local GPU, stays resident)
- Background extraction (default route), matting, resizing, compositing → classical CV (Pillow / numpy / rembg), no ML

For the full design spec (scene-graph data contract, stage breakdown, budgets, known pitfalls), see [`AGENT_BUILD_INSTRUCTIONS.md`](AGENT_BUILD_INSTRUCTIONS.md). The system prompts for every agent role live in [`prompts.py`](prompts.py) (originally [`SYSTEM_PROMPTS.md`](SYSTEM_PROMPTS.md)).

---

## Pipeline at a glance

```
Stage 1  Planning        VLM returns layout -> SceneGraph (id=element_XX, name, bbox, order, overlaps)
Stage 2  Extraction      per element: occupancy check -> isolation prompt -> Qwen edit
                         -> decider gate (only_target / clean_rendering) -> matte to alpha
                         -> resize -> on gate FAIL: multimodal VLM diagnose-and-rewrite
                         prompt -> retry (seed bumped), up to ELEMENT_RETRIES
Stage 3  Background      classic: measure bg color; if flat -> classical fill (instant, default)
                         + decider gate; fallback/model: color-ANCHORED VLM prompt ->
                         Qwen edit, decider gate + retry-writer retry, up to 3x
Stage 4  Reassembly      composite layers over background -> [optional global verify + routing loop, 3x]
```

Every stage reads and writes one shared [`SceneGraph`](scene_graph.py) object; every intermediate image, prompt, and response is logged to a timestamped run folder (see [`logger.py`](logger.py)).

---

## Requirements

- **Hardware**: one GPU with enough VRAM for Qwen-Image-2.1 (7B DiT + 8B text encoder; ~80G-class card is comfortable). The VLM runs remotely through the Bailian API, so it needs no local GPU. Spare cards can host a second pipeline worker to parallelize element extraction.
- **API keys** (git-ignored `.env` file, loaded at import time; see `config.py`):
  `BAILIAN_API_KEY` for the VLM, `OPENROUTER_API_KEY` for the decider gate.
- **Model / code**, pointed at from `config.py`:
  - The **Qwen-Image-2.1** checkpoint (`QWEN_EDIT_CKPT_ROOT`), loaded through diffusers' `QwenImage21Pipeline` — no separate source repo and no `flash_attn` needed (that was the old JoyAI deployment).
- **Python 3.10+** with the packages in [`requirements.txt`](requirements.txt) (or install this repo as a package via `pip install -e .`).
- **Reference environment**: the conda env `JoyNew` (`/remote-home/Zhangkaile/miniconda3/envs/JoyNew/bin/python`) — it ships the `openai` client and the torch/diffusers stack the wrappers rely on (see [`MODELS_AND_RESOURCES.md`](MODELS_AND_RESOURCES.md)).

---

## Setup

### 1. Install this repo

```bash
git clone https://github.com/crimsoncd/BreakerAuto.git
cd BreakerAuto
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt       # or: pip install -e .   (builds the package, adds the `decompose` CLI)
```

### 2. Configure your local resources

All machine-specific settings live in [`config.py`](config.py). **Either** set
the environment variables **or** edit `config.py` directly — don't do both:

```bash
# .env (git-ignored) — Bailian API key for the VLM:
#   BAILIAN_API_KEY=sk-...
export QWEN_EDIT_CKPT_ROOT=/path/to/Qwen-Image-2.1      # checkpoint root
export QWEN_EDIT_DEVICE=cuda:3                          # optional, else auto-detected
```

Editing `config.py` instead looks like this:

```python
QWEN_EDIT_CKPT_ROOT = "/path/to/Qwen-Image-2.1"
```

   > Only Qwen-Image-2.1 runs locally (lazily on first use, then resident), so a single GPU suffices. If you don't set `QWEN_EDIT_DEVICE`, the pipeline picks the card with the most free memory at startup.

---

## Usage

> If you installed this repo as a package (`pip install -e .`), the `decompose`
> command is equivalent to `python main.py` below.

### Single image

```bash
python main.py --image images/009.png
```

### Batch

```bash
python main.py --dir images/ --output runs
```

### Fake / stub mode (no GPUs, no models)

Useful for wiring the control flow and inspecting the logging:

```bash
python main.py --image images/009.png --fake
```

### Verification flags

The verification gate is **on by default**: each generated element cutout and
background is checked by a cheap decider call (two yes/no questions); only a
FAIL spends an expensive VLM call (to diagnose the failure and rewrite the
retry prompt). Global reconstruction verification remains VLM-based and off
by default:

| Flag | Effect |
|------|--------|
| `--use_verify` | (Default: on) Decider gate: for each element, asks "exactly one {name} and nothing else?" and "any large distortion / bizarre colors?" on the generated image; for backgrounds, "no foreground object left?". Scores in the 0.35–0.65 band count as pass (tolerance for ambiguity). A fail feeds the failed image into a multimodal VLM diagnose-and-rewrite of the next prompt (seed also bumped). Retries up to `ELEMENT_RETRIES` / `BACKGROUND_RETRIES`; the attempt with the highest gate confidence ships. |
| `--skip_verify` | Turn the gate off entirely (no verification, single generation attempt). |
| `--use_global` | After reassembly, the VLM compares the reconstruction to the original and routes: re-extract a bad layer, add a missing element, or fix z-order. Retries up to `GLOBAL_ATTEMPTS`. |
| `--bg_method` | Stage-3 background method: `classic` (default) or `model`. See below. |

```bash
python main.py --image images/009.png --skip_verify   # no verification
python main.py --image images/009.png --use_global    # + global VLM check
```

### Full CLI reference

```
--image PATH      single illustration to process
--dir PATH        batch: process every image in a directory
--output PATH     run-folder root (default: runs)
--fake            stub the VLM, decider, and Qwen edit calls (testing only)
--use_verify      (default) decider verification gate + retry-writer retries
--skip_verify     disable the verification gate entirely
--use_global      enable final global reconstruction verification
--bg_method M     background extraction: classic (default) or model
```

### Background extraction methods (Stage 3)

- **`classic` (default)**: the background color is measured classically
  (median of the pixels outside the padded element bboxes); if the background
  is verifiably flat, the foreground (dilated color-outlier mask ∪ padded
  bbox union) is filled with that color — instant, pixel-exact, no model. If
  the background is not flat, or the decider gate rejects the fill, the
  pipeline falls back to the model route.
- **`model`**: a VLM-written, color-**anchored** prompt (the measured bg
  color, or "keep the visible background unchanged" for textured bgs) is run
  through Qwen-Image-2.1 with the verify-retry loop. Anchoring matters: an
  un-anchored prompt makes the edit model re-invent the background.

---

## Configuration reference

Everything below lives in [`config.py`](config.py). Env vars (if set) override the file values.

| Key | Env var | Default | Meaning |
|-----|---------|---------|---------|
| `VLM_BASE_URL` | `VLM_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | OpenAI-compatible endpoint for the VLM |
| `VLM_API_KEY` | `BAILIAN_API_KEY` | — (from `.env`) | API key for the VLM endpoint |
| `VLM_MODEL_NAME` | `VLM_MODEL_NAME` | `qwen3.7-flash` | Model name served by the endpoint |
| `DECIDER_API_URL` | `DECIDER_API_URL` | `https://openrouter.ai/api/alpha/decisions` | OpenRouter Decisions API endpoint (fast verification gate) |
| `DECIDER_API_KEY` | `OPENROUTER_API_KEY` | — (from `.env`) | API key for the decider |
| `DECIDER_MODEL` | `DECIDER_MODEL` | `perplexity/pplx-decider-v1.1-27b` | Decisions model (must be multimodal) |
| `DECIDER_TIMEOUT` / `DECIDER_MAX_RETRIES` / `DECIDER_RETRY_DELAY` | — | `60` / `3` / `5` | transport: per-request timeout, retries, delay; after final failure the gate is treated as unavailable (pass) |
| `DECIDER_FAIL_MAX` / `DECIDER_OK_MIN` | — | `0.35` / `0.65` | tolerance band: scores inside it count as "didn't fail" |
| `QWEN_EDIT_CKPT_ROOT` | `QWEN_EDIT_CKPT_ROOT` | `/remote-home/Zhangkaile/models/Qwen-Image-2.1/` | Qwen-Image-2.1 checkpoint dir (diffusers) |
| `QWEN_EDIT_DEVICE` | `QWEN_EDIT_DEVICE` | auto-detect | GPU card for the edit model (the only local model) |
| `QWEN_EDIT_STEPS` | — | `40` | inference steps for the edit model |
| `QWEN_EDIT_BASE_SEED` | — | `42` | base seed; bumped on retries |
| `BACKGROUND_METHOD` | `BACKGROUND_METHOD` | `classic` | Stage-3 route: `classic` (flat-fill + fallback) or `model` |
| `BG_COLOR_TOL` | — | `24` | max channel delta for "background-colored" pixels |
| `BG_FLAT_MIN_FRAC` | — | `0.97` | min fraction of matching pixels for a "flat" verdict |
| `BG_FILL_DILATE_IT` | — | `5` | dilation (px) of the outlier mask before filling |
| `BG_FILL_PAD` | — | `0.05` | padding fraction for the bbox union in the fill mask |
| `ELEMENT_RETRIES` | — | `3` | max per-element generation attempts |
| `BACKGROUND_RETRIES` | — | `3` | max background attempts |
| `GLOBAL_ATTEMPTS` | — | `3` | max global-verification rounds |
| `MAX_ELEMENTS` | — | `20` | sanity cap on elements per image |
| `DEFAULT_OUTPUT_DIR` | — | `runs` | default `--output` |

---

## Output layout

Each image produces a timestamped run folder, e.g. `runs/009_20260731_153000/`:

```
009_20260731_153000/
├── 0001_stage1_plan.json              # scene graph after planning
├── 0002_crop_element_01.png             # cropped input per element (labelled by id)
├── 0003_vlm_occupancy_element_01.json   # every VLM call logged (prompt + response)
├── ... edit_*.png, gen_*, matte_*, layer_*.png   # edit-model input/output per attempt
├── ... decider_*.json                     # every decider gate call (questions + scores + verdict)
├── bg_analysis.json                     # classical background color / flatness measurements
├── background_classic.png               # Stage 3 classic fill (when the bg is flat)
├── background_final.png               # Stage 3 result (model route)
├── reconstruction.png                 # Stage 4 composite
└── 00XX_final_shipped.json            # final scene graph (incl. bg `method` field)
```

Every VLM prompt+response, every edit-model input/output, and every intermediate matte is saved so any bad output is traceable to the stage and call that produced it.

---

## Utility scripts

- [`call_bailian_vlm.py`](call_bailian_vlm.py) — the VLM wrapper (Bailian API, qwen3.7-flash); also a small `__main__` sanity CLI: `python call_bailian_vlm.py --image in.png --prompt "..."`.
- [`call_decider.py`](call_decider.py) — the decider wrapper (OpenRouter Decisions API, `pplx-decider-v1.1-27b`): `decide(state_text, questions, image)` → `{qid: score}`; transport retries, then returns None (= treat as pass) so a flaky gate never blocks the pipeline. `requests` is the only dependency.
- [`call_qwen_edit.py`](call_qwen_edit.py) — importable `QwenEdit` / `QwenEditBatch` wrappers around the diffusers `QwenImage21Pipeline` (lazy load, resident cache, output resized back to the exact input size); also a small `__main__` sanity CLI.
- [`cleaner.py`](cleaner.py) — sort finished run folders into a flat, metadata-annotated dataset: `python cleaner.py --input runsreal/<collection> --output cleaned/ --code C`.
- [`gather_reconstruction.py`](gather_reconstruction.py) — copy the final reconstruction from many run folders into one directory.
- [`collage_svg.py`](collage_svg.py) — rebuild SVG collages from layerwise `metadata.json` files.

---

## Notes & caveats

- Only Qwen-Image-2.1 loads locally and stays resident; run the whole batch in a single invocation rather than one process per image (the VLM is a remote API call).
- On edit-model retries the seed is bumped automatically — the same prompt + seed reproduces the same output.
- The VLM grading its own pipeline tends to be lenient; the decider gate is a separate, cheap model family, and its tolerance band (0.35–0.65 = pass) deliberately leans lenient to avoid burning retries on ambiguous verdicts. It judges the single generated image (pre-matting); the VLM only sees failures, as the multimodal retry-writer (P9/P10 in `prompts.py`).
- **Background extraction is classical by default** (see the method section above): instant and pixel-exact for the flat-style backgrounds this dataset is full of; the model route with the anchored prompt is the fallback. An un-anchored prompt makes the edit model re-invent the background color (measured: purple output for a teal input).
- **Verification is cheap now**: the decider gate is ~2–3 s per image and only a FAIL spends a VLM call, so it is on by default. The dominant real failure modes from actual runs (contamination by other objects; distortion/bizarre colors) are exactly what the two gate questions test. Run `--skip_verify` only for maximum throughput.
- See [`AGENT_BUILD_INSTRUCTIONS.md`](AGENT_BUILD_INSTRUCTIONS.md) §10 for the known failure modes (missed elements, identity drift, local-vs-global threshold conflicts, heavy occlusion).
